import os  # Filename handling for the no-info-file fallback path
import re  # Regular expressions for parsing metadata
import copy  # Object copying utilities
import numpy as np # type: ignore
import matplotlib.pyplot as plt  # type: ignore
from scipy.optimize import curve_fit # type: ignore
from scipy.signal import find_peaks # type: ignore
import xarray as xr # type: ignore

"""
Module ramantools
=============
Tools to analize Raman spectroscopy data, measured using the Witec 300rsa+ confocal Raman spectrometer.
"""

# NOTE: Indentation uses spaces exclusively to avoid TabError and maintain readability.
# Helper utilities -----------------------------------------------------------

def _midpoint(values: np.ndarray) -> float:
        """Return the midpoint of an array of values.

        This function centralizes the midpoint calculation used throughout the
        module to determine the center of width and height coordinates. Having
        a single helper avoids repeated formulae and clarifies intent.
        """
        vmin = np.min(values)
        vmax = np.max(values)
        return (vmax - vmin) / 2 + vmin


# Sentinels used when no info file is available. Strings use "N/A" so printed
# metadata stays self-describing; numeric fields use NaN so downstream math
# that touches them propagates NaN rather than a silent zero.
_NO_INFO_STR = 'N/A'
_NO_INFO_NUM = float('nan')


def _parse_witec_datafile_header(header_text: str) -> dict:
        """Extract metadata from a Witec "table-export" data-file header.

        Handles both the Witec Project **v5** ("old table export") and **v7**
        ("new table export") header formats. Their ``[Header]`` blocks share
        the same set of key=value lines; the only observed difference is that
        v7 leaves ``FileName =`` empty while v5 fills in the source ``.wip``
        path. Maps include ``SizeX`` / ``SizeY`` (pixel counts),
        ``ScanWidth`` / ``ScanHeight`` (physical size in the unit given by
        ``ScanUnit``) and ``ScanOriginX/Y/Z``. Single spectra include
        ``PositionX/Y/Z`` and trivial ``SizeX = SizeY = 1``. This helper
        returns a dict of whatever fields it finds; missing fields are simply
        absent from the dict so callers can decide how to fall back.

        :param header_text: raw text of the header block as captured by
                :py:meth:`ramanmap._load_map` / :py:meth:`singlespec._load_singlespec`
                into the ``metadata_datafile`` attribute.
        :type header_text: str
        :return: dict keyed by attribute name (``pixel_x``, ``pixel_y``,
                ``size_x``, ``size_y``, ``positioner_x``, ``positioner_y``,
                ``positioner_z``, ``wipfilename``, ``graphname``)
        :rtype: dict
        """
        out: dict = {}
        if not header_text:
                # No header captured (bare data file). Nothing to extract.
                return out

        # Small helper that pulls the first regex match and casts it; returns
        # None (rather than raising) if the pattern is absent or the cast
        # fails — callers handle missing keys via dict.get / "in" checks.
        def _first(pattern: str, text: str, cast=str):
                matches = re.findall(pattern, text)
                if not matches:
                        return None
                try:
                        return cast(matches[0])
                except (ValueError, TypeError):
                        return None

        # Both map and spec headers carry FileName and GraphName.
        file_name = _first(r'(?<=FileName = ).+', header_text)
        if file_name is not None:
                # Strip in case the line ends with whitespace or \r.
                out['wipfilename'] = file_name.strip()
        graph_name = _first(r'(?<=GraphName = ).+', header_text)
        if graph_name is not None:
                out['graphname'] = graph_name.strip()

        # Integer pixel dimensions. Map files set these to real pixel counts;
        # spec files set them to 1. Either way they're correct to propagate.
        pix_x = _first(r'(?<=SizeX = )-?\d+', header_text, int)
        if pix_x is not None:
                out['pixel_x'] = pix_x
        pix_y = _first(r'(?<=SizeY = )-?\d+', header_text, int)
        if pix_y is not None:
                out['pixel_y'] = pix_y

        # Map-only: physical scan dimensions and scan origin. We don't guard
        # with a "map vs spec" flag — spec headers simply don't have these
        # keys, so the regex returns None and the dict keys are skipped.
        scan_width = _first(r'(?<=ScanWidth = )-?\d+(?:\.\d+)?', header_text, float)
        if scan_width is not None:
                out['size_x'] = scan_width
        scan_height = _first(r'(?<=ScanHeight = )-?\d+(?:\.\d+)?', header_text, float)
        if scan_height is not None:
                out['size_y'] = scan_height
        scan_origin_x = _first(r'(?<=ScanOriginX = )-?\d+(?:\.\d+)?', header_text, float)
        if scan_origin_x is not None:
                out['positioner_x'] = scan_origin_x
        scan_origin_y = _first(r'(?<=ScanOriginY = )-?\d+(?:\.\d+)?', header_text, float)
        if scan_origin_y is not None:
                out['positioner_y'] = scan_origin_y

        # Spec-only: absolute sample-positioner coordinates. Overwrites any
        # map-origin values if both appear (they shouldn't in practice).
        pos_x = _first(r'(?<=PositionX = )-?\d+(?:\.\d+)?', header_text, float)
        if pos_x is not None:
                out['positioner_x'] = pos_x
        pos_y = _first(r'(?<=PositionY = )-?\d+(?:\.\d+)?', header_text, float)
        if pos_y is not None:
                out['positioner_y'] = pos_y
        pos_z = _first(r'(?<=PositionZ = )-?\d+(?:\.\d+)?', header_text, float)
        if pos_z is not None:
                out['positioner_z'] = pos_z

        return out


def _graphname_to_name(graphname: str) -> str:
        """Normalize a Witec GraphName into a mapname/specname.

        GraphName lines in data headers look like
        ``MK_FLG_ABC_111--Spectrum--092--Spec.Data 1``; the info-file first line
        omits the trailing ``--Spec.Data N`` suffix. Strip it so attributes
        match between the info-file and header-only load paths.
        """
        # Remove the trailing "--Spec.Data <N>" token plus any trailing whitespace.
        return re.sub(r'--Spec\.Data\s*\d+\s*$', '', graphname).strip()

class ramanmap:
        """
        Container for Raman maps, imported from a text file.
        The text file needs to be exported as a “table” from Witec Project or Witec Control.
        Additional info also needs to be exported, containing the metadata for the measurement.
        This is the text next to the map data in the Witec software.

        :return: object containing the data and metadata
        :rtype: :class:`ramanmap` instance

        :param map_path: Path to the text file, containing the Raman map, exported from Witec
        :type map_path: str
        :param info_path: Path to the info file, containing the metadata, exported from Witec.
                Optional — when omitted (``None``), metadata is populated from the data file's
                own ``[Header]`` block. The parser recognizes both the Witec Project v5
                "old table export" format (``FileName`` carries the source ``.wip`` path) and
                the v7 "new table export" format (``FileName =`` is left empty); the keys
                ``SizeX`` / ``SizeY`` (pixel counts), ``ScanWidth`` / ``ScanHeight`` (physical
                size), ``ScanOriginX`` / ``Y`` and ``GraphName`` are identical between the two.
                Fields not present in the header fall back to ``'N/A'`` (strings) or
                ``NaN`` (numerics). If the data file has no ``[Header]`` block either,
                constructing the map fails with a clear error because pixel dimensions
                cannot be inferred.
        :type info_path: str, optional

        Most important variables of the :class:`ramanmap` instance:

        :var mapxr: (type :py:mod:`xarray` DataArray) all data, coordinates and metadata
        :var map: (type :py:mod:`numpy` array) Raman intensity values
        :var ramanshift: (type :py:mod:`numpy` array) Raman shift values for the datapoints stored in `map`
        :var mask: (type: :py:mod:`numpy` array) A boolean array of the same length as the ``ramanshift``. It's only available if :py:meth:`singlespec.remove_bg` or :py:meth:`ramanmap.remove_bg` is called.
        :var samplename: (type: str) name of the sample, as shown in the Witec software.
        :var mapname: (type: str) contains the name of the Raman map, as shown in the Witec software.

        For a compete list see example below.

        :Example:

        .. code-block:: python

                import ramantools as rt

                map_path = r'data path on you machine'
                info_path = r'metadata path on your machine'
                # use raw strings, starting with `r'` to escape special characters, such as backslash

                map = rt.ramanmap(map_path, info_path)
                # list of the variables stored in the `ramanmap` instance
                print(list(map.__dict__))

                # Load without a metadata file — the data file's [Header]
                # block supplies pixel_x / pixel_y / size_x / size_y:
                map_noinfo = rt.ramanmap(map_path)

        """

        def history(self):
                """Display the notes accumulated in the 'comments' attribute of the :class:`ramanmap.mapxr` `xarray` variable.
                """                
                print('Data modification history:\n')
                print(self.mapxr.attrs['comments'])

        def print_metadata(self):
                """
                Prints the metadata of the :class:`ramanmap` instance, imported from the info file.

                :return: none
                """
                print('Comments of the `xarray` DataArray \n')
                print(self.mapxr.attrs['comments'])
                print('------------------')
                print(self.metadata)

        def plotspec(self, width, height, shift):
                """
                Plots a Raman map at a given Raman shift and displays alongside a selected spectrum at a specified width and height.
                Needs width and height coordinates for the single spectrum and the Raman shift where we want to plot the Raman intensity in the map.
                
                :return: none

                :param width: 'width' coordinate in um, from :class:`ramanmap.mapxr`
                :type width: float
                :param height: 'height' coordinate in um, from :class:`ramanmap.mapxr`
                :type height: float
                :param shift: 'ramanshift' coordinate in um, from :class:`ramanmap.mapxr`
                :type shift: float

                """
                spec = self.mapxr.sel(width = width, height = height, method = 'nearest')
                ramanintensity = self.mapxr.sel(ramanshift = shift, method = 'nearest')

                # Creating two subplots; the figure handle is intentionally
                # discarded ("_" variable) because it is not used later.
                _, [ax0, ax1] = plt.subplots(1, 2, figsize = (9, 4))
                # plotting the density plot of the 2D peak area ratio
                ramanintensity.plot(
                        ax = ax0,
                        cmap = 'plasma') # type: ignore
                # plotting the spectrum at the given coordinates on the right plot
                spec.plot(ax = ax1, color = 'lime') # type: ignore
                # plotting a dot on the density map, where the selected spectrum was measured
                ax0.scatter(
                        spec.coords['width'].data,
                        spec.coords['height'].data,
                        color = 'lime', marker = 'x')
                ax0.set_aspect('equal', 'box')
                ax0.axes.title.set_size(10)
                ax1.axes.title.set_size(10)
                plt.tight_layout()  # Adjust subplot spacing for readability

        def remove_bg(self, mode = 'const', fitmask = None, height = None, width = None, **kwargs):
                """Remove the background of Raman maps.
                It takes the same optional arguments as :func:`bgsubtract`.
                Default fit function is a first order polynomial. This can be changed by the ``polyorder`` parameter.

                There are several modes of background fitting, as a function of the optional variables: ``mode`` and ``fitmask``:
                
                * ``mode`` = `'const'`: (this is the default) Subtracts the same background from all spectra in the map. The background is determined by either a fitmask calculated by :func:`bgsubtract` or passed to the method using the ``fitmask`` parameter. Additionally, a ``height`` and ``width`` parameter can be passed in which case, the spectrum at those coordinates is used to determine the background. If a fitmask is supplied, it is used instead of the coordinates.
                * ``mode`` = `'individual'`: An individual background is subtracted from all spectra.

                        * If ``fitmask`` = `None`, :func:`bgsubtract` does a peak search for each Raman spectrum and the fitmask is determined based on the parameters passed to :func:`bgsubtract`.
                        * If a ``fitmask`` is passed to :py:meth:`~ramanmap.remove_bg`, :func:`bgsubtract` only does the polynomial fit.

                :param mode: Values can be: `'const'`, `'individual'`
                :type mode: str, optional, default: `'const'`
                :param fitmask: fitmask as returned by :func:`bgsubtract` or as contained in the `singlespec.mask` variable of the :class:`singlespec` class, or in :class:`ramanmap`
                :type fitmask: :py:mod:`numpy` array, optional
                :param height: height coordinate of the :class:`ramanmap.mapxr` `xarray`
                :type height: float, optional
                :param width: width coordinate of the :class:`ramanmap.mapxr` `xarray`
                :type width: float, optional
                
                :return:
                        
                        * ``map_mod``: new :class:`ramanmap` instance, containing the data with background removed.
                        * ``coeff``: the coefficients and
                        * ``covar``: covariance of the polynomial fit, as supplied by :func:`polynomial_fit`.
                
                :rtype: tuple: (:class:`ramanmap`, :py:mod:`numpy`, :py:mod:`numpy`)

                .. note::
                        If the peaks and background are complicated, it is advised to test the background fit, by selecting the most difficult spectrum from the map and tweaking the fit parameters directly, using :func:`bgsubtract`.

                        Metadata is copied over to the returned :class:`ramanmap` instance, because there are no unit changes, in removing the background.
                        After running, the 'comments' attribute of the new `xarray` instance is updated with the background fit information.

                .. seealso::
                        ``mode = 'individual'`` is not implemented yet.

                :Example:
                
                .. code-block:: python

                        import ramantools as rt

                        spec_path = r'data path on you machine'
                        info_path = r'metadata path on your machine'
                        # use raw strings, starting with `r'` to escape special characters, such as backslash

                        # Loading a single spectrum from files
                        m = rt.ramanmap(spec_path, info_path)

                        # create a new `ramanmap` instance `new_m`, with the background removed
                        new_m, coeff, covar = m.remove_bg(fitmask = mask, toplot = True)
                        # plot the spectrum in the middle of the map
                        new_m.mapxr.sel(width = self.size_x/2, height = self.size_y/2, method = 'nearest').plot()

                """
                # Validate optional parameters
                if mode not in ['const', 'individual']:
                        raise ValueError(f"mode must be 'const' or 'individual', got '{mode}'")

                if fitmask is not None:
                        if not isinstance(fitmask, (np.ndarray, list)):
                                raise TypeError(f"fitmask must be a numpy array or list, got {type(fitmask)}")
                        fitmask = np.asarray(fitmask, dtype=bool)

                if width is not None:
                        if not isinstance(width, (int, float, np.number)):
                                raise TypeError(f"width must be a number, got {type(width)}")
                        width = float(width)

                if height is not None:
                        if not isinstance(height, (int, float, np.number)):
                                raise TypeError(f"height must be a number, got {type(height)}")
                        height = float(height)

                # create a lightweight copy of the instance; we avoid deepcopy to
                # reduce memory churn and instead copy only the data array below
                map_mod = copy.copy(self)
                map_mod.mapxr = self.mapxr.copy()

                if mode == 'const':
                        # Remove the same background for all spectra in the map
                        # Set the spectra selected for background fitting
                        middle = self.mapxr.sel(width = self.size_x/2, height = self.size_y/2, method = 'nearest')

                        # check if width and height are specified
                        if (width is None) or (height is None):
                                spectofit = middle
                        else:
                                spectofit = self.mapxr.sel(width = width, height = height, method = 'nearest')

                        # check for fitmask
                        if fitmask is None:
                                # take the spectrum at the middle of the map and fit the background
                                spectofit = middle
                                _, bg_values, coeff, _, mask, covar = bgsubtract(spectofit.ramanshift.data, spectofit.data, **kwargs)
                                # add the mask to the new `ramanmap` instance
                                map_mod.mask = mask
                        else:
                                map_mod.mask = fitmask
                                _, bg_values, coeff, _, mask, covar = bgsubtract(spectofit.ramanshift.data, spectofit.data, fitmask = map_mod.mask, **kwargs)

                        # extending bg_values to have two more axes, to allow adding to bg
                        bg_values = bg_values[:, np.newaxis, np.newaxis]
                        # subtract these along the ramanshift dimension of bg
                        map_mod.mapxr[:] -= bg_values
                        # update the comments attribute if it exists
                        if hasattr(map_mod.mapxr, 'comments'):
                                map_mod.mapxr.attrs['comments'] += 'background subtracted - mode == const, background fit: middle spectrum \n'

                elif mode == 'individual':
                        # TODO(future): implement per-pixel background subtraction.
                        # Intended behavior: fit a polynomial to each (width, height)
                        # spectrum independently — likely via xr.apply_ufunc for speed.
                        # Raising loudly (instead of printing and returning an
                        # unmodified map with coeff=0, covar=0) prevents silent
                        # data-corruption bugs where callers believe the background
                        # was removed but it wasn't.
                        raise NotImplementedError(
                                "remove_bg(mode='individual') is not implemented yet — "
                                "use mode='const' or supply a pre-computed fitmask."
                        )

                else:
                        coeff = 0
                        covar = 0
                        pass

                return map_mod, coeff, covar

        def calibrate(self, peakshift, calibfactor = 0, width = None, height = None, **kwargs):
                """Calibrating a Raman map.
                It uses :func:`peakfit` to find the Raman peak closest to ``peakshift`` and returns a new :class:`ramanmap` instance, with the ramanshift coordinate offset to have the position of the peak at ``peakshift``.
                If the optional argument ``calibfactor`` is passed, ``peakshift`` is ignored and the data is shifted by the given value.
                All possible keyword arguments of :func:`peakfit` can be passed.

                :param peakshift: expected position, in 1/cm, of the peak used for calibration
                :type peakshift: float
                :param calibfactor: If the calibration factor is known it can be passed directly. In this case ``peakshift`` is ignored, defaults to 0
                :type calibfactor: int, optional
                :param width: width coordinate of the spectrum, which will be used for calibration. Defaults to the middle of the map.
                :type width: float, optional
                :param height: height coordinate of the spectrum, which will be used for calibration. Defaults to the middle of the map.
                :type height: float, optional

                :return: calibrated :class:`ramanmap` instance
                :rtype: :class:`ramanmap`

                .. note::
                        Only single-peak shape functions are supported (``gaussian``, ``lorentz``).
                        Passing ``func=lorentz2`` raises ``ValueError`` because the method
                        hardcodes ``.sel(param='x0')`` on the fit result — ``lorentz2`` emits
                        ``x01`` / ``x02`` parameter names instead.
                """
                # Guard against double-peak shapes. calibrate relies on a single
                # ``x0`` parameter existing in the fit result; lorentz2 breaks
                # that contract. Fail loudly rather than hit a confusing KeyError
                # deep in xarray.
                if kwargs.get('func') is lorentz2:
                        raise ValueError(
                                "calibrate() requires a single-peak shape function "
                                "(use func=gaussian or func=lorentz); lorentz2 is not supported."
                        )

                # Validate parameters
                if not isinstance(peakshift, (int, float, np.number)):
                        raise TypeError(f"peakshift must be a number, got {type(peakshift)}")
                peakshift = float(peakshift)

                if not isinstance(calibfactor, (int, float, np.number)):
                        raise TypeError(f"calibfactor must be a number, got {type(calibfactor)}")
                calibfactor = float(calibfactor)

                if width is not None:
                        if not isinstance(width, (int, float, np.number)):
                                raise TypeError(f"width must be a number, got {type(width)}")
                        width = float(width)

                if height is not None:
                        if not isinstance(height, (int, float, np.number)):
                                raise TypeError(f"height must be a number, got {type(height)}")
                        height = float(height)

                # Get the middle spectrum
                middle = self.mapxr.sel(width = self.size_x/2, height = self.size_y/2, method = 'nearest')

                # check if width and height are specified
                if (width is None) or (height is None):
                        spectofit = middle
                else:
                        spectofit = self.mapxr.sel(width = width, height = height, method = 'nearest')

                # if a calibration factor is specified, don't fit just shift the values by calibfactor
                if calibfactor == 0:
                        # before fitting crop to the area around the peak
                        fitrange = [peakshift - 100, peakshift + 100]
                        
                        # if one of the ranges out of bounds with the data
                        if fitrange[0] < self.mapxr.ramanshift.min().data:
                                fitrange[0] = self.mapxr.ramanshift.min().data
                        if fitrange[1] > self.mapxr.ramanshift.max().data:
                                fitrange[1] = self.mapxr.ramanshift.max().data
                        # crop the spectrum
                        spectofit_crop = spectofit.sel(ramanshift = slice(fitrange[0], fitrange[1]))

                        # fit to the peak around `peakshift`
                        fit = peakfit(spectofit_crop, stval = {'x0': peakshift}, **kwargs)
                        # correction factor relative to the expected value: peakshift
                        calibshift = peakshift - fit['curvefit_coefficients'].sel(param = 'x0').data
                else:
                        calibshift = calibfactor

                # create a shallow copy of the instance; the new coordinates are
                # assigned below so a deep copy is unnecessary
                map_mod = copy.copy(self)

                # shift the ramanshift values by the correction factor in the new singlespec instance
                map_mod.mapxr = self.mapxr.assign_coords(ramanshift = self.mapxr['ramanshift'] + calibshift)

                # add to the comments attribute of the new instance if it exists
                if hasattr(map_mod.mapxr, 'comments'):
                        map_mod.mapxr.attrs['comments'] += 'calibrated Raman shift by adding ' + f'{calibshift:.2f}' + ' cm^-1 to the raw ramanshift \n'

                return map_mod

        def normalize(self, peakshift, width = None, height = None, mode = 'const', **kwargs):
                """Normalize the Raman spectrum to the peak at ``peakshift``.
                Returns a normalized :class:`ramanmap` instance.
                An exception will be raised if the background has not been removed.
                It uses :func:`peakfit` to find the amplitude of the peak to be normalized. It accepts all keyword arguments accepted by :func:`peakfit`.

                :param peakshift: rough position of the peak in :class:`ramanmap.mapxr.ramanshift` dimension
                :type peakshift: float
                :param mode: Has two modes: 'const' and 'individual'. defaults to 'const'.
                :type mode: str, optional
                :param width: width coordinate of the spectrum, which will be used for normalization in 'const' mode, defaults to the middle of the map.
                :type width: float, optional
                :param height: height coordinate of the spectrum, which will be used for normalization in 'const' mode, defaults to the middle of the map.
                :type height: float, optional

                :return: normalized :class:`ramanmap` instance
                :rtype: :class:`ramanmap`

                :raises ValueError: Background needs to be removed for normalization to make sense.
                :raises ValueError: `mode` parameter must be either: 'const' or 'individual'.

                .. note::
                        Attributes of :class:`ramanmap.mapxr` are updated to reflect the fact that the normalized peak intensities are dimensionless, with a new `long_name`.

                        In ``mode == 'individual'``, each spectrum in the map will be normalized to the local peak amplitude. In ``mode == 'const'``, the peak at the position specified by ``width`` and ``height`` is used for normalization.
                        If ``mode == 'individual'``, the ``width`` and ``height`` parameters are ignored.

                        Only single-peak shape functions are supported (``gaussian``, ``lorentz``).
                        Passing ``func=lorentz2`` raises ``ValueError`` because the method
                        hardcodes ``.sel(param='x0')`` and ``.sel(param='ampl')`` on the fit
                        result.
                """
                # Guard against double-peak shapes — see :py:meth:`calibrate`.
                if kwargs.get('func') is lorentz2:
                        raise ValueError(
                                "normalize() requires a single-peak shape function "
                                "(use func=gaussian or func=lorentz); lorentz2 is not supported."
                        )

                # Validate parameters
                if not isinstance(peakshift, (int, float, np.number)):
                        raise TypeError(f"peakshift must be a number, got {type(peakshift)}")
                peakshift = float(peakshift)

                if mode not in ['const', 'individual']:
                        raise ValueError(f"mode must be 'const' or 'individual', got '{mode}'")

                if width is not None:
                        if not isinstance(width, (int, float, np.number)):
                                raise TypeError(f"width must be a number, got {type(width)}")
                        width = float(width)

                if height is not None:
                        if not isinstance(height, (int, float, np.number)):
                                raise TypeError(f"height must be a number, got {type(height)}")
                        height = float(height)

                # Determine reference coordinates. When none are supplied we use
                # the midpoint of the map, computed via the helper above to
                # avoid duplicated midpoint logic.
                if (width is not None) and (height is not None):
                        mapwidth = width
                        mapheight = height
                else:
                        mapwidth = _midpoint(self.mapxr.width.data)
                        mapheight = _midpoint(self.mapxr.height.data)

                # crop the data in ramanshift to around the peak specified
                cropregion = 100
                cropped = self.mapxr.sel(ramanshift = slice(peakshift - cropregion, peakshift + cropregion))

                # take the offset value, as the intensity value near the peak edge
                cropped_middle = cropped.sel(width = mapwidth, height = mapheight, method = 'nearest')
                bgoffset_low = cropped_middle[0].data
                bgoffset_high = cropped_middle[-1].data

                # check to see of if the background was removed for the spectrum
                if ((bgoffset_high + bgoffset_low)/2 > 500) or ('background subtracted' not in self.mapxr.attrs['comments']):
                        raise ValueError("The background was not removed, or the peak selected is not suitable for normalization. This should be done in case of normalizing to a peak amplitude")
                        return

                if mode == 'const':
                        # pick the spectrum to normalize to
                        cropped = cropped.sel(width = mapwidth, height = mapheight, method = 'nearest')

                        # fit to the cropped region
                        fit = peakfit(cropped, stval = {'x0': peakshift, 'offset': (bgoffset_high + bgoffset_low)/2}, **kwargs)
                        peakampl = fit['curvefit_coefficients'].sel(param = 'ampl').data
                        peakpos = fit['curvefit_coefficients'].sel(param = 'x0').data

                        # normalize to the peak amplitde
                        normalized = self.mapxr / peakampl

                        # copy attributes and change them acccordingly
                        normalized.attrs = self.mapxr.attrs.copy()
                        normalized.attrs['units'] = ' '
                        normalized.attrs['long_name'] = 'normalized Raman intensity'
                        normalized.attrs['comments'] += 'normalized to peak at: ' + f'{peakpos:.2f}' + ' in mode == const' + ' by a factor of ' + f'{peakampl:.2f}' + '\n'

                elif mode == 'individual':
                        # fit to the cropped region
                        fit = peakfit(cropped, stval = {'x0': peakshift, 'offset': (bgoffset_high + bgoffset_low)/2}, **kwargs)
                        peakampl = fit['curvefit_coefficients'].sel(param = 'ampl').data
                        peakpos = fit['curvefit_coefficients'].sel(param = 'x0').sel(width = mapwidth, height = mapheight).data

                        # normalize to the peak amplitde
                        normalized = self.mapxr / peakampl

                        # copy attributes and change them acccordingly
                        normalized.attrs = self.mapxr.attrs.copy()
                        normalized.attrs['units'] = ' '
                        normalized.attrs['long_name'] = 'normalized Raman intensity'
                        normalized.attrs['comments'] += 'normalized to peak at: ' + f'{peakpos:.2f}' + ' in mode == individual' + ' by a factor of ' + f'{peakampl:.2f}' + '\n'
                        
                else:
                        raise ValueError('`mode` parameter must be either: \'const\' or \'individual\'')
                        return
                
                # create a shallow copy of the instance and attach the new
                # normalized data array computed above
                map_norm = copy.copy(self)
                map_norm.mapxr = normalized.copy()

                # add the normalization factor to the ramanmap instance
                map_norm.normfactor = peakampl

                return map_norm


        def crr(self, cutoff = 2, window = 2, **kwargs):
                """Tool for removing cosmic rays from a spectroscopy maps.
                The CRR peaks are determined as the standard deviation of the data: `std` times the `cutoff` value, in the `window` sized vicinity of each pixel.

                :param cutoff: multiplication factor for the data's standard deviation; defaults to 2.
                :type cutoff: int, optional
                :param window: size of the neighborhood to consider; defaults to 2.
                :type window: int, optional
                
                :return: instance of the :class:`ramanmap` class with the cosmic-ray peaks removed.
                :rtype: :class:`ramanmap`

                .. note::
                        If CRR is not satisfactory, keep reducing the `cutoff` value and compare to the original data.
                """

                # Compute rolling statistics to estimate local noise and mean.
                # Using xarray's vectorized rolling operations clarifies intent
                # and avoids manual shifting of data arrays.
                rolling = self.mapxr.rolling(ramanshift=2*window+1, center=True)
                local_mean = rolling.mean()
                local_std = rolling.std()

                # Identify positions where the signal significantly exceeds the
                # local mean by ``cutoff`` standard deviations.
                crrpos = (self.mapxr - local_mean) > (cutoff * local_std)

                # Replace cosmic-ray spikes with the local mean while preserving
                # untouched data elsewhere.
                map_crr_removed = xr.where(crrpos, local_mean, self.mapxr)

                # Explicitly copy attributes since xr.where may not preserve them
                map_crr_removed.attrs = self.mapxr.attrs.copy()

                # Make a lightweight copy of the instance and attach the cleaned data
                map_crr = copy.copy(self)
                map_crr.mapxr = map_crr_removed

                # add comment to attributes
                map_crr.mapxr.attrs['comments'] += 'replaced cosmic ray values with local mean at ' + f'{crrpos.sum().data}' + ' coordinates.\n'

                return map_crr


        def peakmask(self, peakpos, cutoff = 0.1, width = None, height = None, **kwargs):
                """Create a boolean mask for the map, where the mean Raman intensity of the peak at ``peakpos`` is larger than the peak mean in the selected spectrum by the ``cutoff`` value.
                The method also returns the cropped :py:mod:`xarray` DataArray, with the values that are cropped replaced by NaNs.
                The method needs a reference spectrum for  determining the "typical" mean of the peak amplitude.
                This is also used to determine the cutoff value for the rest of the map as:
                
                .. code-block:: python

                        'selected spectrum mean value' * cutoff
                
                The optional ``width`` and ``height`` parameters can be passed, which selects that spectrum for reference.
                If these are not passed, the spectrum in the middle of the map is taken as reference.

                :param peakpos: position in 1/cm of the peak we want to create the mask for
                :type peakpos: float
                :param cutoff: cutoff value, interpreted as a percentage. Values between 0 and 1. Defaults to 0.1
                :type cutoff: float, optional
                :param width: width parameter of the spectrum in the map we want to have as a reference, defaults to None
                :type width: float, optional
                :param height: height parameter of the spectrum in the map we want to have as a reference, defaults to None
                :type height: float, optional
                
                :return:
                        - ``mapmasked``: :class:`ramanmap` instance containing the cropped map
                        - ``peakmask``: :py:mod:`xarray` DataArray containing the mask
                :rtype: tuple: (:class:`ramanmap`, :py:mod:`xarray`)

                .. note::
                        The peak position specified by ``peakpos`` must not be exact.
                        The method uses :func:`peakfit` to find the peak near ``peakpos``.
                        Keyword arguments used by :func:`peakfit` can be passed to the method.
                """
                # Validate parameters
                if not isinstance(peakpos, (int, float, np.number)):
                        raise TypeError(f"peakpos must be a number, got {type(peakpos)}")
                peakpos = float(peakpos)

                if not isinstance(cutoff, (int, float, np.number)):
                        raise TypeError(f"cutoff must be a number, got {type(cutoff)}")
                cutoff = float(cutoff)
                if not (0 <= cutoff <= 1):
                        raise ValueError(f"cutoff must be between 0 and 1, got {cutoff}")

                if width is not None:
                        if not isinstance(width, (int, float, np.number)):
                                raise TypeError(f"width must be a number, got {type(width)}")
                        width = float(width)

                if height is not None:
                        if not isinstance(height, (int, float, np.number)):
                                raise TypeError(f"height must be a number, got {type(height)}")
                        height = float(height)

                # Determine reference coordinates using supplied values or the
                # midpoint of the map when absent. The helper function above
                # encapsulates midpoint math for clarity.
                if (width is not None) and (height is not None):
                        mapwidth = width
                        mapheight = height
                else:
                        mapwidth = _midpoint(self.mapxr.width.data)
                        mapheight = _midpoint(self.mapxr.height.data)

                # first we fit a Lorentzian to the peak at peakpos to determine its width
                spectofit = self.mapxr.sel(width = mapwidth, height = mapheight, method = 'nearest')
                fit = peakfit(spectofit, stval = {'x0': peakpos}, **kwargs)
                peakx0 = fit['curvefit_coefficients'].sel(param = 'x0').data
                peakwidth = fit['curvefit_coefficients'].sel(param = 'width').data

                # vicinity of the peak
                # this needs to be larger than the exclusion_factor used by bgsubtract()
                peakvicinity = slice(peakx0 - 5*peakwidth, peakx0 + 5*peakwidth)

                # crop the data around the peak
                cropped = self.mapxr.sel(ramanshift = peakvicinity)

                # determine the background, by using a 0th order polynomial
                xx = cropped.ramanshift.data
                yy = cropped.sel(width = mapwidth, height = mapheight, method = 'nearest').data
                _, _, background, _, _, _ = bgsubtract(xx, yy, polyorder = 0, peak_pos = [peakx0], exclusion_factor = 3)

                # get the mean value of the peak in the region we cropped and subtract the background
                selected_peakmean = spectofit.sel(ramanshift = peakvicinity).mean(dim = 'ramanshift').data - background
                peakmean = cropped.mean(dim = 'ramanshift') - background

                # make the boolean mask where the value of the mean is more than the cutoff times the selected peak mean
                peakmask = peakmean > cutoff*selected_peakmean

                # crop the data with the mask
                mapmasked = copy.copy(self)  # shallow copy of container
                mapmasked.mapxr = self.mapxr.where(peakmask)

                # update the comment attributes if it exists
                if hasattr(mapmasked.mapxr, 'comments'):
                        mapmasked.mapxr.attrs['comments'] += 'cropped regions, where mean Raman int. of peak: ' + f'{peakx0:.2f}' + ' is less than ' + f'{cutoff:.2f}' + ' of selected peak\n'

                return mapmasked, peakmask

        # internal functions --------------------------

        def __init__(self, map_path, info_path=None):
                """Constructor for :class:`ramanmap`.

                ``info_path`` is optional. When ``None``, metadata defaults to
                sentinels and :py:meth:`_load_map` parses the data file's
                ``[Header]`` block to fill in dimensional attributes.
                """
                # filename
                self.filename = map_path
                # fitmask
                self.mask = None
                # normalization factor, factor by which the raman intensity values are divided during normalize
                self.normfactor = None
                # Track whether an info file was supplied. Used by _load_map to
                # decide whether to parse the data header, and by _toxarray to
                # decide the comment text and coord unit label.
                self._info_loaded = info_path is not None
                # Populate metadata: either from info file or from sentinel defaults
                # that the data-file header may later override in _load_map.
                if self._info_loaded:
                        self._load_info(info_path)
                else:
                        self._load_defaults()
                # Load the Raman map — if no info file, this also parses the data
                # header to fill in pixel_x / pixel_y / size_x / size_y before
                # attempting the reshape.
                self._load_map(map_path)
                # load the data into an xarray container
                self._toxarray()


        def _load_defaults(self):
                """Populate metadata attributes with placeholder values.

                Called by :py:meth:`__init__` when no info file is supplied.
                Dimensional attributes (``pixel_x``, ``pixel_y``, ``size_x``,
                ``size_y``) are set to ``None`` so :py:meth:`_load_map` knows
                they still need to be filled — from the data-file header if
                present, or via a clear error if not. All other metadata
                attributes get sentinels (``'N/A'`` for strings, ``NaN`` for
                floats) so the rest of the pipeline — especially
                :py:meth:`_toxarray` — can run unchanged.
                """
                # No raw metadata text yet; print_metadata then shows just the
                # processing comments, which is what we want.
                self.metadata = ''
                # Best-effort mapname derived from the data file's stem. If a
                # GraphName appears in the data header, _apply_datafile_header
                # will overwrite this.
                self.mapname = os.path.splitext(os.path.basename(self.filename))[0]
                # Dimensional attrs stay None until the data header is parsed
                # (or an error is raised if the header doesn't supply them).
                self.pixel_x = None
                self.pixel_y = None
                self.size_x = None
                self.size_y = None
                # Info-file-only fields; headers don't carry them.
                self.date = _NO_INFO_STR
                self.time = _NO_INFO_STR
                self.samplename = _NO_INFO_STR
                self.laser = _NO_INFO_NUM
                self.itime = _NO_INFO_NUM
                self.grating = _NO_INFO_STR
                self.objname = _NO_INFO_STR
                self.objmagn = _NO_INFO_STR
                # Positioner values: defaults may be overridden by the data
                # header (ScanOriginX / PositionX etc).
                self.positioner_x = _NO_INFO_NUM
                self.positioner_y = _NO_INFO_NUM


        def _apply_datafile_header(self):
                """Override sentinel metadata with values parsed from the data header.

                Called by :py:meth:`_load_map` after ``self.metadata_datafile``
                has been captured. Mutates attributes directly. Missing header
                fields leave the corresponding attribute at its sentinel value.
                """
                # Parse whatever key=value pairs the header contains — unknown
                # or missing keys are simply absent from the result dict.
                fields = _parse_witec_datafile_header(self.metadata_datafile)
                # Apply only the keys that were found, so sentinels remain for
                # the rest. The mapname override uses _graphname_to_name to
                # strip the trailing "--Spec.Data N" token.
                if 'graphname' in fields:
                        self.mapname = _graphname_to_name(fields['graphname'])
                if 'wipfilename' in fields:
                        self.wipfilename = fields['wipfilename']
                if 'pixel_x' in fields:
                        self.pixel_x = fields['pixel_x']
                if 'pixel_y' in fields:
                        self.pixel_y = fields['pixel_y']
                if 'size_x' in fields:
                        self.size_x = fields['size_x']
                if 'size_y' in fields:
                        self.size_y = fields['size_y']
                if 'positioner_x' in fields:
                        self.positioner_x = fields['positioner_x']
                if 'positioner_y' in fields:
                        self.positioner_y = fields['positioner_y']


        def _load_info(self, info_path, **kwargs):
                """
                Load the file containing the metadata.
                The metadata will be filled by searching the info file for various patterns, using regular expressions.
                """
                with open(info_path, mode = 'r', encoding = 'latin1') as infofile:
                        metadata = infofile.read()

                self.metadata = metadata
                # find any character up to a newline, show the forst result `[0]`, this will be the name of the map.
                # Use raw strings: `r` to treat special characters as characters.
                self.mapname = re.findall(r'.*', metadata)[0]
                # find the pixel values in X and Y
                pixel_x_match = re.findall(r'(?<=Points per Line:\t)-?\d+', metadata)
                self.pixel_x = int(pixel_x_match[0])
                pixel_y_match = re.findall(r'(?<=Lines per Image:\t)-?\d+', metadata)
                self.pixel_y = int(pixel_y_match[0])
                # get the size of the scan in microns
                size_x_match = re.findall(r'(?<=Scan Width \[µm\]:\t)-?\d+\.\d+', metadata)
                size_y_match = re.findall(r'(?<=Scan Height \[µm\]:\t)-?\d+\.\d+', metadata)
                self.size_x = float(size_x_match[0])
                self.size_y = float(size_y_match[0])
                # date of measurement
                self.date = re.findall(r'(?<=Start Date:\t)-?.+', metadata)[0]
                # time of measurement
                self.time = re.findall(r'(?<=Start Time:\t)-?.+', metadata)[0]
                # sample name
                self.samplename = re.findall(r'(?<=Sample Name:\t).*', metadata)[0] # new regex to match also no characters after sample name
                # laser energy
                self.laser = float(re.findall(r'(?<=Excitation Wavelength \[nm\]:\t)-?.+', metadata)[0])
                # integration time
                self.itime = float(re.findall(r'(?<=Integration Time \[s\]:\t)-?.+', metadata)[0])
                # grating
                self.grating = re.findall(r'(?<=Grating:\t)-?.+', metadata)[0]
                # objective name and magnification
                self.objname = re.findall('(?<=Objective Name:\t)-?.+', metadata)[0]
                self.objmagn = re.findall('(?<=Objective Magnification:\t)-?.+', metadata)[0]
                # positioner position
                self.positioner_x = float(re.findall(r'(?<=Position X \[µm\]:\t)-?.+', metadata)[0])
                self.positioner_y = float(re.findall(r'(?<=Position Y \[µm\]:\t)-?.+', metadata)[0])

                return metadata

        def _load_map(self, map_path):
                """
                Load the Raman map data into a numpy array.
                """

                # Load the first part of the file to search for metadata
                with open(map_path, 'r', encoding='latin1') as file:
                        lines = []
                        for _ in range(40): # start reading the first 40 lines
                                line = file.readline()
                                if not line:  # stop if end of file is reached
                                        break
                                lines.append(line.strip())
                
                # define a regular expression to search for the start of the data. It looks for any number, followed by a dot, with more numbers after, then any character and a tab or space and more numbers
                data_pattern = r'(\d+\.\d+.+[\t ]+)+'
                # initialize lineskip parameter
                toskip = 0
                # Check each line for a match
                for idx, line in enumerate(lines, start=0):
                        if re.search(data_pattern, line):
                                toskip = idx
                                break  # Stop after finding the first match

                # save datafile metadata
                if toskip == 0:
                        # there is no header
                        self.wipfilename = map_path
                        self.metadata_datafile = ''
                else:
                        # add the data metadata to a class variable
                        with open(map_path, 'r', encoding = 'latin1') as file:
                                lines = [next(file).strip() for _ in range(toskip)]
                                self.metadata_datafile = '\n'.join(lines)
                        # Extract the WIP filename. ``[ \t]*`` after ``=`` (but
                        # NOT ``\s*``, which would swallow the line's ``\n``
                        # and capture the next line) tolerates stripped
                        # trailing whitespace: newer v7 exports emit
                        # ``FileName = \n`` that ``strip()`` reduces to
                        # ``FileName =``, breaking a literal ``FileName = ``
                        # match. The ``if matches`` guard defends against the
                        # FileName line being absent entirely.
                        matches = re.findall(r'FileName =[ \t]*(.*?)(?:\n|$)', self.metadata_datafile)
                        self.wipfilename = matches[0] if matches else ''

                # if 'Header' in lines[1]:
                #         # we have a header
                #         # load additional metadata from the data file itself, ie the first 19 lines we have skipped.
                #         with open(map_path, 'r', encoding = 'latin1') as file:
                #                 lines = [next(file).strip() for _ in range(17)]
                #                 self.metadata_datafile = '\n'.join(lines)
                #         # need to skip the header when loading
                #         toskip = 19

                #         # extract the WIP filename
                #         self.wipfilename = re.findall(r'FileName = (.*?)(?:\n|$)', self.metadata_datafile)[0]
                # else:
                #         # there is no header
                #         toskip = 0
                #         self.wipfilename = map_path
                
                # When no info file was supplied, the data-file header is our
                # only source for pixel_x / pixel_y / size_x / size_y. Parse
                # it now so the reshape below succeeds.
                if not self._info_loaded:
                        self._apply_datafile_header()
                # Guard: without dims, reshape would fail with an opaque error.
                # Raise a clear message instead pointing at the real cause.
                if self.pixel_x is None or self.pixel_y is None:
                        raise ValueError(
                                f"cannot determine pixel dimensions for '{map_path}': "
                                "no info file was supplied and the data file's header "
                                "does not contain SizeX / SizeY. Pass an info_path or "
                                "export the map with Witec's table option."
                        )
                # Load the data
                m = np.loadtxt(map_path, skiprows = toskip, encoding = 'latin1')
                # The raman shift is the first column in the exported table.
                self.ramanshift = m[:, 0]
                self.map = np.reshape(m[:, 1:], (m.shape[0], self.pixel_y, self.pixel_x))

                return self.map

        def _toxarray(self):
                """
                Load the raw numpy data, as well as the metadata into an xarray object.
                """
                width = np.linspace(0, self.size_x, num = self.pixel_x)
                height = np.linspace(0, self.size_y, num = self.pixel_y)
                # We need to flip the array along the height axis, so that the data show up in the same orientation as in Witec Project.
                self.mapxr = xr.DataArray(
                        np.flip(self.map, axis = 1),
                        dims = ['ramanshift', 'height', 'width'],
                        coords = {
                                'ramanshift': self.ramanshift,
                                'width': width,
                                'height': height
                                })
                # Comment text notes the load provenance so a downstream reader
                # can tell whether metadata came from an info file or purely
                # from the data-file header.
                if self._info_loaded:
                        self.mapxr.attrs['comments'] = 'raw data loaded \n'
                else:
                        self.mapxr.attrs['comments'] = 'raw data loaded (no info file; metadata from data header) \n'
                # adding attributes
                self.mapxr.name = 'Raman intensity' # this is needed if used with hvplot
                self.mapxr.attrs['wipfile name'] = self.wipfilename
                self.mapxr.attrs['units'] = 'au'
                self.mapxr.attrs['long_name'] = 'Raman intensity'
                self.mapxr.attrs['sample name'] = self.samplename
                self.mapxr.attrs['laser excitation'] = str(self.laser) + ' nm'
                self.mapxr.attrs['time of measurement'] = self.time
                self.mapxr.attrs['date of measurement'] = self.date
                self.mapxr.attrs['integration time'] = str(self.itime) + ' s'
                self.mapxr.attrs['map width'] = str(self.size_x) + ' um'
                self.mapxr.attrs['map height'] = str(self.size_y) + ' um'
                self.mapxr.attrs['sample positioner X'] = self.positioner_x
                self.mapxr.attrs['sample positioner Y'] = self.positioner_y
                self.mapxr.attrs['objective name'] = self.objname
                # Only append the conventional 'x' multiplier when a real
                # magnification value is present; otherwise the N/A sentinel
                # would produce an awkward 'N/Ax' string.
                if self._info_loaded:
                        self.mapxr.attrs['objective magnification'] = self.objmagn + 'x'
                else:
                        self.mapxr.attrs['objective magnification'] = self.objmagn
                self.mapxr.attrs['grating'] = self.grating
                # coordinate attributes
                self.mapxr.coords['ramanshift'].attrs['units'] = r'1/cm'
                self.mapxr.coords['ramanshift'].attrs['long_name'] = 'Raman shift'
                # self.mapxr.coords['width'].attrs['units'] = r'$\mathrm{\mu m}$' # type: ignore
                self.mapxr.coords['width'].attrs['units'] = 'um' # type: ignore
                self.mapxr.coords['width'].attrs['long_name'] = 'width'
                # self.mapxr.coords['height'].attrs['units'] = r'$\mathrm{\mu m}$' # type: ignore
                self.mapxr.coords['height'].attrs['units'] = 'um' # type: ignore
                self.mapxr.coords['height'].attrs['long_name'] = 'height'

## ----------------------------------------------------

class singlespec:
        """
        Container for Raman single spectra, imported from a text file.
        The text file needs to be exported as a "table" from Witec Project or Witec Control
        Additional info also needs to be exported, containing the metadata for the measurement.
        This is the text next to the map data in the Witec software.
        It takes two arguments, the first is the path to the file containing the spectroscopy data, the second is the path to the metadata.

        :param spec_path: Path to the text file, containing the Raman spectrum, exported from Witec
        :type spec_path: str
        :param info_path: Path to the info file, containing the metadata, exported from Witec.
                Optional — when omitted (``None``), metadata is populated from the data file's
                own ``[Header]`` block if present. Both the Witec Project v5 "old table
                export" (``FileName`` carries the source ``.wip`` path) and the v7 "new
                table export" (``FileName =`` empty) formats are recognized; the
                ``PositionX`` / ``PositionY`` / ``PositionZ`` and ``GraphName`` keys used to
                populate metadata are identical between the two. Fields not present in the
                header fall back to ``'N/A'`` (strings) or ``NaN`` (numerics); a bare data
                file (no header) still loads because spectra don't need dimensional metadata.
        :type info_path: str, optional

        Most important variables of the :class:`singlespec` instance:

        :var ssxr: (type :py:mod:`xarray` DataArray) all data, coordinates and metadata
        :var mask: (type: :py:mod:`numpy` array) A boolean array of the same length as the ``ramanshift``. It's only available if :py:meth:`singlespec.remove_bg` is called.
        :var counts: (type :py:mod:`numpy` array) Raman intensity values
        :var ramanshift: (type :py:mod:`numpy` array) Raman shift values for the datapoints stored in `map`
        :var samplename: (type: str) name of the sample, as shown in the Witec software.
        :var specname: (type: str) contains the name of the Raman single spectrum, as shown in the Witec software.

        For a complete list see example below.

        :return: :class:`singlespec` instance containing the data and metadata
        :rtype: :class:`singlespec` instance

        :Example:

        .. code-block:: python

                import ramantools as rt

                spec_path = r'data path on you machine'
                info_path = r'metadata path on your machine'
                # use raw strings, starting with `r'` to escape special characters, such as backslash

                single_spectrum = rt.singlespec(spec_path, info_path)
                # list of variables stored in the `singlespec` instance
                print(list(single_spectrum.__dict__))

                # Load without a metadata file — the data file's [Header]
                # block (if present) fills in positioner coordinates etc:
                spec_noinfo = rt.singlespec(spec_path)

        """

        def history(self):
                """Display the notes accumulated in the 'comments' attribute of the :class:`singlespec.ssxr` `xarray` variable.
                """                
                print('Data modification history:\n')
                print(self.ssxr.attrs['comments'])

        def print_metadata(self):
                """
                Prints the metadata of the :class:`singlespec` instance, imported from the info file.

                :return: none
                """
                print('Comments of the `xarray` DataArray \n')
                print(self.ssxr.attrs['comments'])
                print('------------------')
                print(self.metadata)

        def remove_bg(self, **kwargs):
                """Remove the background of Raman spectra.
                The :py:mod:`xarray` variable of the :class:`singlespec` instance is updated with the new dataset, with the background removed.
        
                It takes the same optional arguments as :func:`bgsubtract`.
                Default fit function is a first order polynomial.
                This can be changed by the ``polyorder`` parameter.

                :return:
                        
                        * ``singlesp_mod``: new :class:`singlespec` instance, containing the data with background removed.
                        * ``coeff``: the coefficients and
                        * ``covar``: covariance of the polynomial fit, as supplied by :func:`polynomial_fit`.

                :rtype: tuple: (:class:`singlespec`, :py:mod:`numpy`, :py:mod:`numpy`)

                .. note::
                        Metadata is copied over to the returned :class:`singlespec` instance, because there are no unit changes, in removing the background.
                        After running, the 'comments' attribute of the new `xarray` instance is updated with the background fit information.

                :Example:
                
                .. code-block:: python

                        import ramantools as rt

                        spec_path = r'data path on you machine'
                        info_path = r'metadata path on your machine'
                        # use raw strings, starting with `r'` to escape special characters, such as backslash

                        # Loading a single spectrum from files
                        single_spectrum = rt.singlespec(spec_path, info_path)

                        # Using remove_bg to fit and remove the background
                        # In this example, we let remove_bg() find the peaks automatically. In this case, if no options are passed, the fit is returned.
                        new_ss, coeff, covar = single_spectrum.remove_bg()
                        # `new_ss` is the new `singlespec` instance containing the ssxr `xarray` object, with the background removed

                        # In this example, we also want to plot the result and we select the peaks by hand, by using `peak_pos`.
                        new_ss, coeff, covar = single_spectrum.remove_bg(toplot = True, peak_pos = [520, 1583, 2700], wmin = 15)

                        # Fitting a third order polynomial
                        new_ss, coeff, covar = single_spectrum.remove_bg(polyorder = 3)

                        # Replot the `xarray` DataArray, which has the background removed
                        new_ss.ssxr.plot()

                """                
                data_nobg, bg_values, coeff, fitparams, mask, covar = bgsubtract(self.ssxr.coords['ramanshift'].data, self.ssxr.data, **kwargs)

                # create a lightweight copy of the instance and copy the data
                singlesp_mod = copy.copy(self)
                singlesp_mod.ssxr = self.ssxr.copy()

                # remove the background from ssxr
                singlesp_mod.ssxr -= bg_values
                # adding a note to `xarray` comments attribute if it exists
                if hasattr(singlesp_mod.ssxr, 'comments'):
                        singlesp_mod.ssxr.attrs['comments'] += 'background subtracted, with parameters: ' + str(fitparams) + '\n'
                # save the fitmask as a variable of `singlespec`
                singlesp_mod.mask = mask

                return singlesp_mod, coeff, covar

        def calibrate(self, peakshift, calibfactor = 0, **kwargs):
                """Calibrating a single Raman spectrum.
                It uses :func:`peakfit` to find the Raman peak closest to ``peakshift`` and returns a new :class:`singlespec` instance, with the ramanshift coordinate offset to have the position of the peak at ``peakshift``.
                If the optional argument ``calibfactor`` is passed, ``peakshift`` is ignored and the data is shifted by the given value.
                All possible keyword arguments of :func:`peakfit` can be passed.

                :param peakshift: expected position, in 1/cm, of the peak used for calibration
                :type peakshift: float
                :param calibfactor: If the calibration factor is known it can be passed directly. In this case ``peakshift`` is ignored, defaults to 0
                :type calibfactor: int, optional
                
                :return: calibrated :class:`singlespec` instance
                :rtype: :class:`singlespec`

                .. note::
                        Only single-peak shape functions are supported (``gaussian``, ``lorentz``).
                        Passing ``func=lorentz2`` raises ``ValueError``.
                """
                # Guard against double-peak shapes — see :py:meth:`ramanmap.calibrate`.
                if kwargs.get('func') is lorentz2:
                        raise ValueError(
                                "calibrate() requires a single-peak shape function "
                                "(use func=gaussian or func=lorentz); lorentz2 is not supported."
                        )

                # Validate parameters
                if not isinstance(peakshift, (int, float, np.number)):
                        raise TypeError(f"peakshift must be a number, got {type(peakshift)}")
                peakshift = float(peakshift)

                if not isinstance(calibfactor, (int, float, np.number)):
                        raise TypeError(f"calibfactor must be a number, got {type(calibfactor)}")
                calibfactor = float(calibfactor)

                # if a calibration factor is specified, don't fit just shift the values by calibfactor
                if calibfactor == 0:
                        # before fitting crop to the area around the peak
                        fitrange = [peakshift - 100, peakshift + 100]

                        # if one of the ranges out of bounds with the data
                        if fitrange[0] < self.ssxr.ramanshift.min().data:
                                fitrange[0] = self.ssxr.ramanshift.min().data
                        if fitrange[1] > self.ssxr.ramanshift.max().data:
                                fitrange[1] = self.ssxr.ramanshift.max().data
                        # crop the spectrum
                        spectofit_crop = self.ssxr.sel(ramanshift = slice(fitrange[0], fitrange[1]))

                        # fit to the peak around `peakshift`
                        fit = peakfit(spectofit_crop, stval = {'x0': peakshift}, **kwargs)

                        # correction factor relative to the expected value: peakshift
                        calibshift = peakshift - fit['curvefit_coefficients'].sel(param = 'x0').data
                else:
                        calibshift = calibfactor

                # create a shallow copy since assign_coords returns a new DataArray
                ss_mod = copy.copy(self)

                # shift the ramanshift values by the correction factor in the new singlespec instance
                ss_mod.ssxr = self.ssxr.assign_coords(ramanshift = self.ssxr['ramanshift'] + calibshift)

                # add to the comments attribute of the new instance if it exists
                if hasattr(ss_mod.ssxr, 'comments'):
                        ss_mod.ssxr.attrs['comments'] += 'calibrated Raman shift by adding ' + f'{calibshift:.2f}' + ' cm^-1 to the raw ramanshift\n'

                return ss_mod

        def normalize(self, peakshift, **kwargs):
                """Normalize the Raman spectrum to the peak at ``peakshift``.
                Returns a normalized :class:`singlespec` instance.
                An exception will be raised if the background has not been removed.
                It uses :func:`peakfit` to find the amplitude of the peak to be normalized. It accepts all keyword arguments accepted by :func:`peakfit`.

                :param peakshift: rough position of the peak in :class:`singlespec.ssxr.ramanshift` dimension
                :type peakshift: float

                :return: normalized :class:`singlespec` instance
                :rtype: :class:`singlespec`

                :raises ValueError: Background needs to be removed for normalization to make sense.

                .. note::
                        Attributes of :class:`singlespec.ssxr` are updated to reflect the fact that the normalized peak intensities are dimensionless, with a new `long_name`.

                        Only single-peak shape functions are supported (``gaussian``, ``lorentz``).
                        Passing ``func=lorentz2`` raises ``ValueError``.
                """
                # Guard against double-peak shapes — see :py:meth:`ramanmap.normalize`.
                if kwargs.get('func') is lorentz2:
                        raise ValueError(
                                "normalize() requires a single-peak shape function "
                                "(use func=gaussian or func=lorentz); lorentz2 is not supported."
                        )

                # Validate parameters
                if not isinstance(peakshift, (int, float, np.number)):
                        raise TypeError(f"peakshift must be a number, got {type(peakshift)}")
                peakshift = float(peakshift)

                # crop the data to around the peak specified
                cropregion = 100
                cropped = self.ssxr.sel(ramanshift = slice(peakshift - cropregion, peakshift + cropregion))
                # take the offset value, as the intensity value near the peak edge
                bgoffset_low = cropped[0].data
                bgoffset_high = cropped[-1].data

                # check to see of if the background was removed for the spectrum
                if ((bgoffset_high + bgoffset_low)/2 > 500) or ('background subtracted' not in self.ssxr.attrs['comments']):
                        raise ValueError("The background was not removed, or the peak selected is not suitable for normalization. This should be done in case of normalizing to a peak amplitude")
                        return

                # fit to the cropped region
                fit = peakfit(cropped, stval = {'x0': peakshift, 'offset': (bgoffset_high + bgoffset_low)/2}, **kwargs)
                peakampl = fit['curvefit_coefficients'].sel(param = 'ampl').data
                peakpos = fit['curvefit_coefficients'].sel(param = 'x0').data

                # normalize to the peak amplitde
                normalized = self.ssxr / peakampl

                # copy attributes and change them acccordingly
                normalized.attrs = self.ssxr.attrs.copy()
                normalized.attrs['units'] = ' '
                normalized.attrs['long_name'] = 'normalized Raman intensity'
                normalized.attrs['comments'] += 'normalized to peak at: ' + f'{peakpos:.2f}' + ' by a factor of ' + f'{peakampl:.2f}' + '\n'

                # copy the singlespec instance lightly and attach normalized data
                ss_norm = copy.copy(self)
                ss_norm.ssxr = normalized.copy()

                # add the normalization factor to the singlespec instance
                ss_norm.normfactor = peakampl

                return ss_norm

        def crr(self, cutoff = 2, window = 2, **kwargs):
                """Tool for removing cosmic rays from a single spectrum.
                The CRR peaks are determined as the standard deviation of the data: `std` times the `cutoff` value, in the `window` sized vicinity of each pixel.

                :param cutoff: multiplication factor for the data's standard deviation; defaults to 2.
                :type cutoff: int, optional
                :param window: size of the neighborhood to consider; defaults to 2.
                :type window: int, optional
                
                :return: instance of the :class:`singlespec` class with the cosmic-ray peaks removed.
                :rtype: :class:`singlespec`

                .. note::
                        If CRR is not satisfactory, keep reducing the `cutoff` value and compare to the original data.
                """

                # Use rolling window statistics to estimate local mean and
                # standard deviation, replacing manual shifts.
                rolling = self.ssxr.rolling(ramanshift=2*window+1, center=True)
                local_mean = rolling.mean()
                local_std = rolling.std()

                # Identify spikes that exceed the local mean by ``cutoff`` standard deviations
                crrpos = (self.ssxr - local_mean) > (cutoff * local_std)

                # Create a lightweight copy and replace spikes with the local mean
                ss_crr = copy.copy(self)
                ss_crr_data = xr.where(crrpos, local_mean, self.ssxr)

                # Explicitly copy attributes since xr.where may not preserve them
                ss_crr_data.attrs = self.ssxr.attrs.copy()
                ss_crr.ssxr = ss_crr_data

                # add comment to attributes
                ss_crr.ssxr.attrs['comments'] += 'replaced cosmic ray values with local mean at ' + f'{crrpos.sum().data}' + ' Ramanshift coordinates.\n'

                return ss_crr


        # internal functions ----------------------------------

        def __init__(self, spec_path, info_path=None):
                """Constructor for :class:`singlespec`.

                ``info_path`` is optional. When ``None``, metadata defaults to
                sentinels and :py:meth:`_load_singlespec` parses the data file's
                ``[Header]`` block if present to fill in positioner coordinates
                etc.
                """
                # filename
                self.filename = spec_path
                # fit mask
                self.mask = None
                # normalization factor, factor by which the raman intensity values are divided during normalize
                self.normfactor = None
                # Track whether an info file was supplied; consumed by
                # _load_singlespec (whether to parse the header) and by
                # _toxarray (comment text / objmagn suffix).
                self._info_loaded = info_path is not None
                # Populate metadata: info-file path takes full control; the
                # default path writes sentinels that the data header may later
                # override in _load_singlespec.
                if self._info_loaded:
                        self._load_info(info_path)
                else:
                        self._load_defaults()
                # Load the Raman single spectrum (ramanshift + counts columns).
                # When no info file was supplied this call also invokes
                # _apply_datafile_header to enrich metadata from the header.
                self._load_singlespec(spec_path)
                # load the data into an xarray container
                self._toxarray()


        def _load_defaults(self):
                """Populate metadata attributes with placeholder values.

                Mirror of :py:meth:`ramanmap._load_defaults` for single spectra.
                Unlike the map variant, there are no dimensional attributes to
                defer — a 1-D spectrum is fully described by the two columns
                in the data file.
                """
                self.metadata = ''
                # Spectrum name derived from the data file's stem; the data
                # header's GraphName, if present, will overwrite this via
                # _apply_datafile_header.
                self.specname = os.path.splitext(os.path.basename(self.filename))[0]
                self.date = _NO_INFO_STR
                self.time = _NO_INFO_STR
                self.samplename = _NO_INFO_STR
                self.laser = _NO_INFO_NUM
                self.itime = _NO_INFO_NUM
                self.grating = _NO_INFO_STR
                self.objname = _NO_INFO_STR
                self.objmagn = _NO_INFO_STR
                # Positioner values: defaults may be overridden by the data
                # header (PositionX / PositionY / PositionZ).
                self.positioner_x = _NO_INFO_NUM
                self.positioner_y = _NO_INFO_NUM
                self.positioner_z = _NO_INFO_NUM


        def _apply_datafile_header(self):
                """Override sentinel metadata with values parsed from the data header.

                Called by :py:meth:`_load_singlespec` after
                ``self.metadata_datafile`` has been captured. Overrides are
                limited to keys the parser actually finds; missing fields
                leave sentinels in place.
                """
                fields = _parse_witec_datafile_header(self.metadata_datafile)
                # GraphName (if any) becomes the spectrum name — strip the
                # trailing "--Spec.Data N" suffix for parity with info-file loads.
                if 'graphname' in fields:
                        self.specname = _graphname_to_name(fields['graphname'])
                if 'wipfilename' in fields:
                        self.wipfilename = fields['wipfilename']
                # Spectra don't use pixel_x / pixel_y / size_x / size_y even if
                # SizeX / SizeY = 1 appear in the header — skip those keys.
                if 'positioner_x' in fields:
                        self.positioner_x = fields['positioner_x']
                if 'positioner_y' in fields:
                        self.positioner_y = fields['positioner_y']
                if 'positioner_z' in fields:
                        self.positioner_z = fields['positioner_z']


        def _load_info(self, info_path):
                """
                Load the file containing the metadata.
                The metadata will be filled by searching the info file for various patterns, using regular expressions.
                """
                with open(info_path, mode='r', encoding = 'latin1') as infofile:
                        metadata = infofile.read()

                self.metadata = metadata
                # find any character up to a newline, show the forst result `[0]`, this will be the name of the map.
                # Use raw strings: `r` to treat special characters as characters.
                self.specname = re.findall(r'.*', metadata)[0]
                # date of measurement
                self.date = re.findall(r'(?<=Start Date:\t)-?.+', metadata)[0]
                # time of measurement
                self.time = re.findall(r'(?<=Start Time:\t)-?.+', metadata)[0]
                # sample name
                self.samplename = re.findall(r'(?<=Sample Name:\t).*', metadata)[0] # new regex to match also no characters after sample name
                # laser energy
                self.laser = float(re.findall(r'(?<=Excitation Wavelength \[nm\]:\t)-?.+', metadata)[0])
                # integration time
                self.itime = float(re.findall(r'(?<=Integration Time \[s\]:\t)-?.+', metadata)[0])
                # grating
                self.grating = re.findall(r'(?<=Grating:\t)-?.+', metadata)[0]
                # objective name and magnification
                self.objname = re.findall('(?<=Objective Name:\t)-?.+', metadata)[0]
                self.objmagn = re.findall('(?<=Objective Magnification:\t)-?.+', metadata)[0]
                # positioner position
                self.positioner_x = float(re.findall(r'(?<=Position X \[µm\]:\t)-?.+', metadata)[0])
                self.positioner_y = float(re.findall(r'(?<=Position Y \[µm\]:\t)-?.+', metadata)[0])
                self.positioner_z = float(re.findall(r'(?<=Position Z \[µm\]:\t)-?.+', metadata)[0])

                return metadata

        def _load_singlespec(self, spec_path):
                """
                Load the Raman map data into a numpy array.
                """

                # Load the first part of the file to search for metadata
                with open(spec_path, 'r', encoding='latin1') as file:
                        lines = []
                        for _ in range(40): # start reading the first 40 lines
                                line = file.readline()
                                if not line:  # stop if end of file is reached
                                        break
                                lines.append(line.strip())
                
                # define a regular expression to search for the start of the data. It looks for any number, followed by a dot, with more numbers after, then any character and a tab or space and more numbers
                data_pattern = r'(\d+\.\d+.+[\t ]+)+'
                # initialize lineskip parameter
                toskip = 0
                # Check each line for a match
                for idx, line in enumerate(lines, start=0):
                        if re.search(data_pattern, line):
                                toskip = idx
                                break  # Stop after finding the first match

                # save datafile metadata
                if toskip == 0:
                        # there is no header
                        self.wipfilename = spec_path
                        self.metadata_datafile = ''
                else:
                        # add the data metadata to a class variable
                        with open(spec_path, 'r', encoding = 'latin1') as file:
                                lines = [next(file).strip() for _ in range(toskip)]
                                self.metadata_datafile = '\n'.join(lines)
                        # See :py:meth:`ramanmap._load_map` for rationale on
                        # ``[ \t]*`` and the ``if matches`` fallback — newer
                        # v7 Witec exports emit an empty ``FileName =`` line.
                        matches = re.findall(r'FileName =[ \t]*(.*?)(?:\n|$)', self.metadata_datafile)
                        self.wipfilename = matches[0] if matches else ''

                # # Check to see if metadata are present in the data file
                # with open(spec_path, 'r', encoding = 'latin1') as file:
                #         lines = [next(file).strip() for _ in range(2)]
                
                # if 'Header' in lines[1]:
                #         # we have a header
                #         # load additional metadata from the data file itself, ie the first 19 lines we have skipped.
                #         with open(spec_path, 'r', encoding = 'latin1') as file:
                #                 lines = [next(file).strip() for _ in range(17)]
                #                 self.metadata_datafile = '\n'.join(lines)
                #         # need to skip the header when loading
                #         toskip = 17

                #         # extract the WIP filename
                #         self.wipfilename = re.findall(r'FileName = (.*?)(?:\n|$)', self.metadata_datafile)[0]
                # else:
                #         # there is no header
                #         toskip = 0
                #         self.wipfilename = spec_path
                
                # When no info file was supplied, the data-file header (if any)
                # is our only source for extra metadata such as PositionX/Y/Z.
                # A spectrum with no header still loads fine — the parser just
                # returns an empty dict and sentinels remain in place.
                if not self._info_loaded:
                        self._apply_datafile_header()
                # Load the data
                ss = np.loadtxt(spec_path, skiprows = toskip, encoding = 'latin1')
                self.ramanshift = ss[:, 0]
                self.counts = ss[:, 1]

                return self.ramanshift, self.counts

        def _toxarray(self):
                """
                Load the raw numpy data, as well as the metadata into an xarray object
                """
                self.ssxr = xr.DataArray(
                        self.counts,
                        dims = ['ramanshift'],
                        coords = {'ramanshift': self.ramanshift})
                
                # Comment text flags whether metadata came from an info file
                # or from the data-file header / defaults.
                if self._info_loaded:
                        self.ssxr.attrs['comments'] = 'raw data loaded \n'
                else:
                        self.ssxr.attrs['comments'] = 'raw data loaded (no info file; metadata from data header) \n'
                # adding attributes
                self.ssxr.name = 'Raman intensity' # this is needed if used with hvplot
                self.ssxr.attrs['wipfile name'] = self.wipfilename
                self.ssxr.attrs['units'] = 'au'
                self.ssxr.attrs['long_name'] = 'Raman intensity'
                self.ssxr.attrs['sample name'] = self.samplename
                self.ssxr.attrs['laser excitation'] = str(self.laser) + ' nm'
                self.ssxr.attrs['time of measurement'] = self.time
                self.ssxr.attrs['date of measurement'] = self.date
                self.ssxr.attrs['integration time'] = str(self.itime) + ' s'
                self.ssxr.attrs['sample positioner X'] = self.positioner_x
                self.ssxr.attrs['sample positioner Y'] = self.positioner_y
                self.ssxr.attrs['sample positioner Z'] = self.positioner_z
                self.ssxr.attrs['objective name'] = self.objname
                # Only append the 'x' multiplier when a real magnification
                # value is present — avoids awkward 'N/Ax' in the no-info path.
                if self._info_loaded:
                        self.ssxr.attrs['objective magnification'] = self.objmagn + 'x'
                else:
                        self.ssxr.attrs['objective magnification'] = self.objmagn
                self.ssxr.attrs['grating'] = self.grating
                # coordinate attributes
                self.ssxr.coords['ramanshift'].attrs['units'] = r'1/cm'
                self.ssxr.coords['ramanshift'].attrs['long_name'] = 'Raman shift'

## internal functions ------------------------------------------------------------

# nothing here yet

## Tools -----------------------------------------------------------------

def gaussian(x, x0 = 1580, ampl = 10, width = 15, offset = 0):
        """Gaussian function. Width and amplitude parameters have the same meaning as for :func:`lorentz`.

        :param x: values for the x coordinate
        :type x: float, :py:mod:`numpy` array, etc.
        :param x0: shift along the `x` corrdinate
        :type x0: float
        :param ampl: amplitude of the peak
        :type ampl: float
        :param width: FWHM of the peak
        :type width: float
        :param offset: offset along the function value
        :type offset: float

        :return: values of a Gaussian function
        :rtype: float, :py:mod:`numpy` array, etc.
        """        
        # using the FWHM for the width
        return offset + ampl * np.exp(-4*np.log(2)*(x - x0)**2 / (width**2))


def lorentz(x, x0 = 1580, ampl = 1, width = 14, offset = 0):
        """Single Lorentz function

        :param x: values for the x coordinate
        :type x: float, :py:mod:`numpy` array, etc.
        :param x0: shift along the `x` corrdinate
        :type x0: float
        :param ampl: amplitude of the peak
        :type ampl: float
        :param width: FWHM of the peak
        :type width: float
        :param offset: offset along the function value
        :type offset: float

        :return: values of a single Lorentz function
        :rtype: float, :py:mod:`numpy` array, etc.

        .. note::
                The area of the peak can be given by:
                
                .. code-block:: python

                        area = np.pi * amplitude * width / 2
        
        """
        area = np.pi * ampl * width / 2
        return offset + (2/np.pi) * (area * width) / (4*(x - x0)**2 + width**2)

def lorentz2(x, x01 = 2700, ampl1 = 1, width1 = 15, x02 = 2730, ampl2 = 1, width2 = 15, offset = 0):
        """Double Lorentz function

        :return: values of a double Lorentz function
        :rtype: float, :py:mod:`numpy` array, etc.

        :param x: values for the x coordinate
        :type x: float, :py:mod:`numpy` array, etc.
        :param x0: shift along the `x` corrdinate
        :type x0: float
        :param area: area of the peak
        :type area: float
        :param width: width (FWHM) of the peak
        :type width: float
        :param offset: offset along the function value
        :type offset: float
        
        """
        area1 = np.pi * ampl1 * width1 / 2
        area2 = np.pi * ampl2 * width2 / 2
        return offset + (2/np.pi) * (area1 * width1) / (4*(x - x01)**2 + width1**2) + (2/np.pi) * (area2 * width2) / (4*(x - x02)**2 + width2**2)

def polynomial_fit(order, x_data, y_data):
        """Polinomial fit to `x_data`, `y_data`

        :param order: order of the polinomial to be fit
        :type order: int
        :param x_data: x coordinate of the data, this would be Raman shift
        :type x_data: :py:mod:`numpy` array
        :param y_data: y coordinate of the data, this would be Raman intensity
        :type y_data: :py:mod:`numpy` array

        :return: coefficients of the polinomial ``coeff``, as used by :py:mod:`numpy.polyval`, covariance matrix ``covar``, as returned by :py:mod:`scipy.optimize.curve_fit`
        :rtype: tuple: (:py:mod:`numpy` array, :py:mod:`numpy` array)

        """    
        # Define polynomial function of given order
        def poly_func(x, *coeffs):
                y = np.polyval(coeffs, x)
                return y

        # Initial guess for the coefficients is all ones
        guess = np.ones(order + 1)

        # Use curve_fit to find best fit parameters
        coeff, covar = curve_fit(poly_func, x_data, y_data, p0 = guess)

        return coeff, covar

def bgsubtract(x_data, y_data, polyorder = 1, toplot = False, fitmask = None, hmin = 50, hmax = 10000, wmin = 4, wmax = 60, prom = 10, exclusion_factor = 6, peak_pos = None):
        """Takes Raman shift and Raman intensity data and automatically finds peaks in the spectrum, using :py:mod:`scipy.find_peaks`.
        These peaks are then used to define the areas of the background signal.
        In the areas with the peaks removed, the background is fitted by a polynomial of order given by the optional argument: ``polyorder``.
        The fit is performed by :py:mod:`scipy.optimize.curve_fit`.
        The function returns the Raman intensity counts with the background removed, the background polinomial values themselves and the coefficients of the background fit results, as used by :py:mod:`numpy.polyval`.

        In cases, where the automatic peak find is not functioning as expected, one can pass the values in ``x_data``, at which peaks appear.
        In this case, the ``wmin`` option determines the width of all peaks.

        If a ``fitmask`` is supplied for fitting, the fitmask is not calculated and only a polynomial fit is performed.
        This can decrease the runtime.

        :param x_data: Raman shift values
        :type x_data: :py:mod:`numpy` array
        :param y_data: Raman intensity values
        :type y_data: :py:mod:`numpy` array
        :param polyorder: order of polynomial used to fit the background, defaults to 1
        :type polyorder: int, optional
        :param toplot: if `True` a plot of: the fit, the background used and positions of the peaks is shown, defaults to False
        :type toplot: bool, optional
        :param fitmask: Fitmask to be used for polynomial fitting.
        :type fitmask: :py:mod:`numpy` array
        :param hmin: minimum height of the peaks passed to :py:mod:`scipy.signal.find_peaks`, defaults to 50
        :type hmin: float, optional
        :param hmax: maximum height of the peaks passed to :py:mod:`scipy.signal.find_peaks`, defaults to 10000
        :type hmax: float, optional
        :param wmin: minimum width of the peaks, passed to :py:mod:`scipy.signal.find_peaks`, defaults to 4
        :type wmin: float, optional
        :param wmax: maximum width of the peaks passed to :py:mod:`scipy.signal.find_peaks`, defaults to 60
        :type wmax: float, optional
        :param prom: prominence of the peaks, passed to :py:mod:`scipy.signal.find_peaks`, defaults to 10
        :type prom: float, optional
        :param exclusion_factor: this parameter multiplies the width of the peaks found by :py:mod:`scipy.signal.find_peaks`, or specified by ``wmin`` if the peak positions are passed by hand, using ``peak_pos``, defaults to 6
        :type exclusion_factor: float, optional
        :param peak_pos: list of the peak positions in ``x_data`` values used for exclusion, defaults to None
        :type peak_pos: list of floats, optional

        :return: ``y_data_nobg``, ``bg_values``, ``coeff``, ``params_used_at_run``, ``mask``, ``covar``
        :rtype: tuple: (:py:mod:`numpy` array, :py:mod:`numpy` array, :py:mod:`numpy` array, dictionary, :py:mod:`numpy` array, :py:mod:`numpy` array)

        * ``y_data_nobg``: Raman counts, with the background subtracted,
        * ``bg_values``: the polynomial values of the fit, at the Raman shift positions,
        * ``coeff``: coefficients of the polynomial fit, as used by: :py:mod:`numpy.polyval`,
        * ``params_used_at_run``: parameters used at runtime
        * ``mask``: the calculated fitmask
        * ``covar``: covariance of the fit parameters

        .. note::
                Using the option: ``peak_pos``, a ``wmin*exclusion_factor/2`` region (measured in datapoints) on both sides of the peaks is excluded from the background fit.
                If automatic peak finding is used, the exclusion area is calculated in a similar way, but the width of the individual peaks are used, as determined by :py:mod:`scipy.signal.find_peaks`.

        """
        # if a mask is passed to the function, don't calculate the peak positions
        if fitmask is None:
                if peak_pos is None:
                        # Find the peaks with specified minimum height and width
                        # re_height sets the width from the maximum at which value the width is evaluated
                        peak_properties = find_peaks(y_data, height = (hmin, hmax), width = (wmin, wmax), prominence = prom, rel_height = 0.5)

                        # Find the indices of the peaks
                        peak_indices = peak_properties[0]

                        # Get the properties of the peaks
                        peak_info = peak_properties[1]
                else:
                        # Use the provided peak positions
                        peak_indices = []
                        for peak_position in peak_pos:
                                # Find the index of the closest data point to the peak position
                                closest_index = np.argmin(np.abs(x_data - peak_position))
                                peak_indices.append(closest_index)

                        # Calculate the widths of the peaks using the data
                        peak_widths = [wmin] * len(peak_indices)  # Use the minimum width if peak widths cannot be calculated from the data
                        peak_info = {'widths': peak_widths}

                # Calculate the start and end indices of each peak with a larger exclusion area
                start_indices = peak_indices - (exclusion_factor * np.array(peak_info['widths'])).astype(int)
                end_indices = peak_indices + (exclusion_factor * np.array(peak_info['widths'])).astype(int)
                
                # Ensure indices are within data bounds
                start_indices = np.maximum(start_indices, 0)
                end_indices = np.minimum(end_indices, len(x_data) - 1)
                
                # Define the indices covered by the peaks
                covered_indices = []
                for start, end in zip(start_indices, end_indices):
                        covered_indices.extend(range(start, end + 1))

                # Remove these indices from the data
                mask = np.ones(x_data.shape[0], dtype = bool)
                mask[covered_indices] = False
        else:
                # if a mask was passed, use that
                mask = fitmask
                peak_indices = None
        
        # make the mask False for the region below the notch filter cutoff (~80 cm^{-1})
        x_data_notch = x_data[x_data < 95]
        mask[:x_data_notch.shape[0]] = False
        uncovered_x_data = x_data[mask]
        uncovered_y_data = y_data[mask]

        # Fit polynomial to the remaining data
        coeff, covar = polynomial_fit(polyorder, uncovered_x_data, uncovered_y_data)

        # Calculate the fitted polynomial values
        bg_values = np.polyval(coeff, x_data)

        # Line subtracted data
        y_data_nobg = y_data - bg_values

        if toplot == True:
                # Plot the data and peaks
                plt.plot(x_data, y_data, label = 'Raman spectrum')

                # Highlight the peaks
                if fitmask is None:
                        plt.scatter(x_data[peak_indices], y_data[peak_indices], color = 'green', label = 'peaks')
                else:
                        pass

                # Plot the fitted polynomial
                plt.plot(x_data, bg_values, color = 'k', ls = "dashed", label = 'fitted polynomial')

                # Highlight the background used for fitting
                plt.scatter(uncovered_x_data, uncovered_y_data, color = 'red', marker= 'o', alpha = 1, label = 'background used for fit') # type: ignore

                plt.xlabel('Raman shift (cm$^{-1}$)')
                plt.ylabel('Raman intensity (a.u.)')
                plt.title('Data plot with peaks, fitted line and background highlighted.')
                plt.legend()
        
        params_used_at_run = {'polyorder': polyorder, 'hmin': hmin, 'hmax': hmax, 'wmin': wmin, 'wmax': wmax, 'prom':prom, 'exclusion_factor': exclusion_factor, 'peak_pos': peak_pos}

        return y_data_nobg, bg_values, coeff, params_used_at_run, mask, covar

def peakfit(xrobj, func = lorentz, fitresult = None, stval = None, bounds = None, toplot = False, width = None, height = None, **kwargs):
        """Fitting a function to peaks in the data contained in ``xrobj``, which can be a single spectrum, a map or a selected spectrum from a map.

        :param xrobj: :py:mod:`xarray` DataArray, of a single spectrum or a map.
        :type xrobj: :py:mod:`xarray` DataArray
        :param func: function to be used for fitting, defaults to lorentz
        :type func: function, optional
        :param fitresult: an :py:mod:`xarray` Dataset of a previous fit calculation, with matching dimensions. If this is passed to :func:`peakfit`, the fit calculation in skipped and the passed Dataset is used.
        :type fitresult: :py:mod:`xarray` Dataset, optional
        :param stval: starting values for the fit parameters of ``func``. You are free to specify only some of the values, the rest will be filled by defaults. Defaults are given in the starting values for keyword arguments in ``func``.
        :type stval: dictionary of ``func`` parameters, optional
        :param bounds: bounds for the fit parameters, used by :py:mod:`xarray.curvefit`. Simlar dictionary, like ``stval``, but the values area a list, with lower and upper components. Defaults to None
        :type bounds: dictionary of ``func`` parameters, with tuples containing lower and upper bounds, optional
        :param toplot: plot the fit result, defaults to ``False``
        :type toplot: boolean, optional
        :param width: width parameter of an :py:mod:`xarray` map to be used in conjunction with ``toplot = True``
        :type width: `int` or `float`, optional
        :param height: height parameter of an :py:mod:`xarray` map to be used in conjunction with ``toplot = True``
        :type height: `int` or `float`, optional
        
        :return: fitted parameters of ``func`` and covariances in a Dataset
        :rtype: :py:mod:`xarray` Dataset

        :Example:

        .. code-block:: python

                import ramantools as rt

                map_path = r'data path on you machine'
                info_path = r'metadata path on your machine'
                # use raw strings, starting with `r'` to escape special characters, such as backslash

                # Load a map
                map = rt.ramanmap(map_path, info_path)

                # Creating the dictionary for the starting values and the bounds
                # The default function is `lorentz`, with parameters:
                p = {'x0': 2724, 'ampl': 313, 'width': 49, 'offset': 0}
                # passing the starting values contained in `p` and bounds: `b` to the `peakfit()` method. 
                b = {'x0': [2500, 2900], 'ampl': [0, 900], 'width': [20, 100], 'offset': [-10, 50]}
                mapfit = rt.peakfit(m_nobg.mapxr, stval = p, bounds = b, toplot = True)

        .. note::

                - Use ``toplot`` = `True` to tweak the starting values. If ``toplot`` = `True`, in case of a map, if no ``width`` and ``height`` are specified, the middle of the map is used for plotting.
                - Passing a ``bounds`` dictionary to :func:`peakfit` seems to increase the fitting time significantly. This might be an issue with :py:mod:`xarray.DataArray.curvefit`.
                - By passing a previous fit result, using the optional parameter ``fitresult``, we can just plot the fit result at multiple regions of the map.
                - In case of using double Lorentz fitting, the names of the parameters change! See: :func:`lorentz2`.

        .. seealso::

                It is good practice, to crop the data to the vicinity of the peak you want to fit to.

        """        
        # get the parameters used by the function: func
        # and also get the default values for the keyword arguments
        param_count = func.__code__.co_argcount
        param_names = func.__code__.co_varnames[:param_count]
        defaults = func.__defaults__
        # get the starting index for the keyword arguments
        kwargs_start_index = param_count - len(defaults)
        # make a dictionary with the keyword arguments (parameters) and their default values specified in the function: func
        kwargs_with_defaults = dict(zip(param_names[kwargs_start_index:], defaults))

        # loop over the keys in stval and fill missing values with defaults
        if stval is None:
                stval = kwargs_with_defaults
        else:
                # if only some values are missing, fill in the rest
                for key in kwargs_with_defaults.keys():
                        if key not in stval:
                                stval[key] = kwargs_with_defaults[key]
        
        # fit
        # the `xrobj` should have a 'ramanshift' coordinate
        # `nan_policy = omit` skips values with NaN. This is passed to scipy.optimize.curve_fit
        # `max_nfev` is passed to leastsq(). It determines the number of function calls, before quitting.
        if fitresult is None:
                fit = xrobj.curvefit('ramanshift', func, p0 = stval, bounds = bounds, errors = 'ignore', kwargs = {'maxfev': 1000})
        else:
                fit = fitresult

        # plot the resulting fit
        if toplot is True:
                #check if the xrobj is a single spectrum or map
                if 'width' in xrobj.dims:
                        # it's a map
                        if (width is not None) and (height is not None):
                                # get coordinates to plot, or take the middle spectrum
                                plotwidth = width
                                plotheight = height
                        else:
                                # Use the helper to compute central coordinates for plotting
                                plotwidth = _midpoint(xrobj.width.data)
                                plotheight = _midpoint(xrobj.height.data)
                        
                        paramnames = fit['curvefit_coefficients'].sel(width = plotwidth, height = plotheight, method = 'nearest').param.values
                        funcparams = fit['curvefit_coefficients'].sel(width = plotwidth, height = plotheight, method = 'nearest').data
                        # plot the data
                        xrobj.sel(width = plotwidth, height = plotheight, method = 'nearest').plot(color = 'black', marker = '.', lw = 0, label = 'data')                        
                else:
                        paramnames = fit['curvefit_coefficients'].param.values
                        funcparams = fit['curvefit_coefficients'].data
                        # plot the data
                        xrobj.plot(color = 'black', marker = '.', lw = 0, label = 'data')

                # print the starting and fitted values of the parameters
                print('Values of starting parameters: \n', stval, '\n')
                print('Values of fitted parameters:\n')
                for name, fitvalues in zip(paramnames, funcparams):
                        print(name, ':', f'{fitvalues:.2f}')

                shiftmin = min(xrobj.ramanshift.data)
                shiftmax = max(xrobj.ramanshift.data)
                shift = np.linspace(shiftmin, shiftmax, num = int((shiftmax - shiftmin)*100))
                
                funcvalues = func(shift, *funcparams)
                # set plotting variables based on the datapoints
                # this should be the ramanshift of the peak maximum if the fit was successful
                fitpeakpos = shift[np.argmax(funcvalues)]
                plotarea_x = 100
                plotarea_y = [0.8*np.min(funcvalues), 1.1*np.max(funcvalues)]

                plt.plot(shift, funcvalues, color = 'red', lw = 3, alpha = 0.5, label = 'fit')
                plt.xlim([fitpeakpos - plotarea_x, fitpeakpos + plotarea_x])
                plt.ylim(plotarea_y)
                plt.legend()
        
        # copy attributes to the fit dataset, update the 'comments'
        fit.attrs = xrobj.attrs.copy()
        # update the comments if it exists
        if hasattr(fit, 'comments'):
                fit.attrs['comments'] += 'peak fitting, using ' + str(func.__name__) + '\n'
                
        return fit


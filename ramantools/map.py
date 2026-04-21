"""The ``ramanmap`` container class.

Extracted from ``ramantools/ramantools.py`` during the Phase 1 refactor.
The public API (class name, attributes, method signatures, return shapes)
is unchanged; this file only moves the code, it does not modify it.
"""
import os  # Filename handling for the no-info-file fallback path
import re  # Regular expressions for parsing metadata
import copy  # Object copying utilities
import numpy as np  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import xarray as xr  # type: ignore

# Internal helpers (moved out of the original single-file layout).
from ._helpers import _midpoint, _NO_INFO_STR, _NO_INFO_NUM
from ._witec import _parse_witec_datafile_header, _graphname_to_name
# lorentz2 is referenced only as a guard target (``is lorentz2``); bgsubtract
# and peakfit are called directly by ``remove_bg`` / ``calibrate`` /
# ``normalize`` / ``peakmask``.
from ._fitting import lorentz2, bgsubtract, peakfit

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

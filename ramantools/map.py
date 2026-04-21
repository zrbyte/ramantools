"""The ``ramanmap`` container class.

Extracted from ``ramantools/ramantools.py`` during the Phase 1 refactor.
The public API (class name, attributes, method signatures, return shapes)
is unchanged; this file only moves the code, it does not modify it.
"""
# Defer annotation evaluation so ``X | None`` and forward references (e.g.
# ``-> ramanmap`` inside its own class body) work without runtime cost or
# Python-version gymnastics.
from __future__ import annotations

import os  # Filename handling for the no-info-file fallback path
import copy  # Object copying utilities
import numpy as np  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import xarray as xr  # type: ignore

# Shared helpers (dedup core extracted during Phase 1.2 of the refactor).
# ``_apply_default_common_metadata`` / ``_apply_parsed_header`` /
# ``_set_common_xarray_attrs`` fold the metadata-scaffolding that used to
# be copy-pasted between map and spec; ``_crr_xarray`` is the rolling-
# window CRR core; ``_as_float`` / ``_require_mode`` are the validator
# helpers.
from ._helpers import (
        _midpoint,
        _as_float,
        _require_mode,
        _crr_xarray,
        _apply_default_common_metadata,
        _apply_parsed_header,
        _set_common_xarray_attrs,
)
from ._witec import _parse_info_file
from ._io import _load_witec_datafile
# Named thresholds previously spelled as magic numbers in peakmask.
from ._constants import PEAKMASK_VICINITY_FACTOR
# bgsubtract + peakfit are used by remove_bg / peakmask.
# _compute_calibshift / _normalize_to_peak are the shared algorithm
# cores for calibrate / normalize. _reject_double_peak replaces the
# four inlined lorentz2 guards in calibrate + normalize.
from ._fitting import (
        bgsubtract,
        peakfit,
        _compute_calibshift,
        _normalize_to_peak,
        _reject_double_peak,
)

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

        def history(self) -> None:
                """Display the notes accumulated in the 'comments' attribute of the :class:`ramanmap.mapxr` `xarray` variable.
                """
                print('Data modification history:\n')
                print(self.mapxr.attrs['comments'])

        def print_metadata(self) -> None:
                """
                Prints the metadata of the :class:`ramanmap` instance, imported from the info file.

                :return: none
                """
                print('Comments of the `xarray` DataArray \n')
                print(self.mapxr.attrs['comments'])
                print('------------------')
                print(self.metadata)

        def plotspec(self, width: float, height: float, shift: float) -> None:
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

        def remove_bg(self, mode: str = 'const', fitmask = None, height: float | None = None, width: float | None = None, **kwargs):
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
                # Validate optional parameters. Centralized helpers replace
                # the four near-identical inline isinstance blocks the
                # pre-refactor code carried.
                _require_mode(mode, ['const', 'individual'])
                if fitmask is not None:
                        # fitmask stays inline because it accepts list/ndarray
                        # (not scalar) and needs a dtype=bool cast — no other
                        # caller in the package duplicates this idiom.
                        if not isinstance(fitmask, (np.ndarray, list)):
                                raise TypeError(f"fitmask must be a numpy array or list, got {type(fitmask)}")
                        fitmask = np.asarray(fitmask, dtype=bool)
                width = _as_float('width', width, optional=True)
                height = _as_float('height', height, optional=True)

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

        def calibrate(self, peakshift: float, calibfactor: float = 0, width: float | None = None, height: float | None = None, **kwargs) -> ramanmap:
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
                # Reject double-peak shape functions early — calibrate hardcodes
                # ``.sel(param='x0')`` which lorentz2 (x01/x02 names) breaks.
                _reject_double_peak(kwargs.get('func'), 'calibrate')
                # Validate parameters via the shared helpers.
                peakshift = _as_float('peakshift', peakshift)
                calibfactor = _as_float('calibfactor', calibfactor)
                width = _as_float('width', width, optional=True)
                height = _as_float('height', height, optional=True)

                # Pick the reference spectrum — user-supplied coords if both
                # width and height are given, otherwise the middle of the map.
                if (width is None) or (height is None):
                        spectofit = self.mapxr.sel(
                                width=self.size_x / 2,
                                height=self.size_y / 2,
                                method='nearest',
                        )
                else:
                        spectofit = self.mapxr.sel(width=width, height=height, method='nearest')

                # Delegate to the shared calibshift core (crops, fits, returns
                # the correction factor; skips the fit when calibfactor != 0).
                calibshift = _compute_calibshift(spectofit, peakshift, calibfactor, **kwargs)

                # Lightweight copy of the instance — only the ramanshift coord
                # changes, so a full deep copy would be wasteful.
                map_mod = copy.copy(self)
                map_mod.mapxr = self.mapxr.assign_coords(
                        ramanshift=self.mapxr['ramanshift'] + calibshift
                )
                # Append to the processing history (guard just in case attrs
                # was wiped somewhere upstream).
                if hasattr(map_mod.mapxr, 'comments'):
                        map_mod.mapxr.attrs['comments'] += (
                                'calibrated Raman shift by adding '
                                + f'{calibshift:.2f}'
                                + ' cm^-1 to the raw ramanshift \n'
                        )
                return map_mod

        def normalize(self, peakshift: float, width: float | None = None, height: float | None = None, mode: str = 'const', **kwargs) -> ramanmap:
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
                # Guard + validate via the shared helpers.
                _reject_double_peak(kwargs.get('func'), 'normalize')
                peakshift = _as_float('peakshift', peakshift)
                _require_mode(mode, ['const', 'individual'])
                width = _as_float('width', width, optional=True)
                height = _as_float('height', height, optional=True)

                # Build the reference-coords dict for the shared core. Only
                # both-or-neither here — the legacy behaviour ignored a
                # partial specification, and we preserve that.
                if (width is not None) and (height is not None):
                        ref_coords = {'width': width, 'height': height}
                else:
                        # None lets the core compute midpoints of the non-
                        # ramanshift dims itself (width + height for maps).
                        ref_coords = None

                # Delegate the crop / bg-check / fit / divide dance to the
                # shared core; we keep only the class-specific wrap-up here.
                normalized, peakampl, peakpos = _normalize_to_peak(
                        self.mapxr,
                        peakshift,
                        ref_coords=ref_coords,
                        mode=mode,
                        **kwargs,
                )
                # Map-specific comment line includes the ``in mode == XXX``
                # token (singlespec's version omits it).
                normalized.attrs['comments'] += (
                        'normalized to peak at: '
                        + f'{peakpos:.2f}'
                        + f' in mode == {mode}'
                        + ' by a factor of '
                        + f'{peakampl:.2f}'
                        + '\n'
                )

                # Lightweight copy of the instance and attach the new data.
                map_norm = copy.copy(self)
                map_norm.mapxr = normalized.copy()
                map_norm.normfactor = peakampl
                return map_norm


        def crr(self, cutoff: float = 2, window: int = 2, **kwargs) -> ramanmap:
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

                # Delegate to the shared rolling-window CRR core; the attrs
                # copy that xr.where strips is handled inside _crr_xarray.
                cleaned, n_spikes = _crr_xarray(self.mapxr, cutoff, window)
                map_crr = copy.copy(self)
                map_crr.mapxr = cleaned
                # Map-specific comment wording — "coordinates" (vs. "Ramanshift
                # coordinates" on singlespec.crr) preserved byte-for-byte.
                map_crr.mapxr.attrs['comments'] += (
                        'replaced cosmic ray values with local mean at '
                        + f'{n_spikes}'
                        + ' coordinates.\n'
                )
                return map_crr


        def peakmask(self, peakpos: float, cutoff: float = 0.1, width: float | None = None, height: float | None = None, **kwargs):
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
                # Validate parameters via the shared float-coercer. The
                # 0 <= cutoff <= 1 range check stays inline because no
                # other method carries the same range constraint.
                peakpos = _as_float('peakpos', peakpos)
                cutoff = _as_float('cutoff', cutoff)
                if not (0 <= cutoff <= 1):
                        raise ValueError(f"cutoff must be between 0 and 1, got {cutoff}")
                width = _as_float('width', width, optional=True)
                height = _as_float('height', height, optional=True)

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
                peakvicinity = slice(
                        peakx0 - PEAKMASK_VICINITY_FACTOR * peakwidth,
                        peakx0 + PEAKMASK_VICINITY_FACTOR * peakwidth,
                )

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

        def __init__(self, map_path: str, info_path: str | None = None) -> None:
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
                # Shared scaffolding (metadata / date / time / sample / laser /
                # itime / grating / objname / objmagn / positioner_x / _y) in
                # one call; map-specific extras follow below.
                _apply_default_common_metadata(self)
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


        def _apply_datafile_header(self):
                """Override sentinel metadata with values parsed from the data header.

                Called by :py:meth:`_load_map` after ``self.metadata_datafile``
                has been captured. Mutates attributes directly. Missing header
                fields leave the corresponding attribute at its sentinel value.
                """
                # Shared parse + setattr iteration. ``name_attr='mapname'``
                # routes the GraphName override to this class's name field;
                # the other five keys get copied verbatim.
                from ._witec import _parse_witec_datafile_header
                fields = _parse_witec_datafile_header(self.metadata_datafile)
                _apply_parsed_header(
                        self,
                        fields,
                        name_attr='mapname',
                        copy_keys=('pixel_x', 'pixel_y', 'size_x', 'size_y',
                                   'positioner_x', 'positioner_y'),
                )


        def _load_info(self, info_path, **kwargs):
                """
                Load the file containing the metadata.
                The metadata will be filled by searching the info file for various patterns, using regular expressions.
                """
                # Shared parser returns the raw text and a {attr: value} dict.
                # The ``is_map=True`` flag adds pixel_x / pixel_y / size_x /
                # size_y to the set of fields extracted.
                self.metadata, fields = _parse_info_file(info_path, is_map=True)
                # First line is the mapname; everything else copies 1:1 onto self.
                self.mapname = fields.pop('name')
                for attr, value in fields.items():
                        setattr(self, attr, value)
                return self.metadata

        def _load_map(self, map_path):
                """
                Load the Raman map data into a numpy array.
                """
                # One call owns: header-window scan, toskip detection,
                # metadata capture, FileName extraction, and np.loadtxt.
                _, self.metadata_datafile, self.wipfilename, m = _load_witec_datafile(map_path)
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
                # The raman shift is the first column in the exported table;
                # the remaining columns are the map intensities reshaped to
                # the pixel grid.
                self.ramanshift = m[:, 0]
                self.map = np.reshape(m[:, 1:], (m.shape[0], self.pixel_y, self.pixel_x))
                return self.map

        def _toxarray(self):
                """
                Load the raw numpy data, as well as the metadata into an xarray object.
                """
                width = np.linspace(0, self.size_x, num=self.pixel_x)
                height = np.linspace(0, self.size_y, num=self.pixel_y)
                # We need to flip the array along the height axis, so that
                # the data shows up in the same orientation as in Witec Project.
                self.mapxr = xr.DataArray(
                        np.flip(self.map, axis=1),
                        dims=['ramanshift', 'height', 'width'],
                        coords={
                                'ramanshift': self.ramanshift,
                                'width': width,
                                'height': height,
                        },
                )
                # Comment text notes the load provenance so a downstream
                # reader can tell whether metadata came from an info file or
                # purely from the data-file header.
                if self._info_loaded:
                        self.mapxr.attrs['comments'] = 'raw data loaded \n'
                else:
                        self.mapxr.attrs['comments'] = 'raw data loaded (no info file; metadata from data header) \n'
                # Shared attribute population (wipfile / units / long_name /
                # sample / laser / time / date / integration time / positioner
                # X + Y / objective name / magnification / grating, plus the
                # ramanshift coord metadata).
                _set_common_xarray_attrs(self.mapxr, self, self._info_loaded)
                # Map-specific extras: physical scan dimensions in the attrs
                # dict, plus width / height coord metadata (maps only).
                self.mapxr.attrs['map width'] = str(self.size_x) + ' um'
                self.mapxr.attrs['map height'] = str(self.size_y) + ' um'
                self.mapxr.coords['width'].attrs['units'] = 'um'  # type: ignore
                self.mapxr.coords['width'].attrs['long_name'] = 'width'
                self.mapxr.coords['height'].attrs['units'] = 'um'  # type: ignore
                self.mapxr.coords['height'].attrs['long_name'] = 'height'

"""The ``singlespec`` container class.

Extracted from ``ramantools/ramantools.py`` during the Phase 1 refactor.
Public API is unchanged; this file only moves the code.
"""
# Defer annotation evaluation — same rationale as in ``map.py``.
from __future__ import annotations

import os  # Filename handling for the no-info-file fallback path
import copy  # Object copying utilities
import numpy as np  # type: ignore
import xarray as xr  # type: ignore

# Shared helpers. Same set ``map.py`` imports from the internals, minus
# the map-only _midpoint (singlespec methods never pick spatial coords).
from ._helpers import (
        _NO_INFO_NUM,
        _as_float,
        _crr_xarray,
        _apply_default_common_metadata,
        _apply_parsed_header,
        _set_common_xarray_attrs,
)
from ._witec import _parse_info_file
from ._io import _load_witec_datafile
# bgsubtract is used by remove_bg; _compute_calibshift / _normalize_to_peak
# are the shared cores for calibrate / normalize; _reject_double_peak
# centralises the lorentz2 guard shared with map.
from ._fitting import (
        bgsubtract,
        _compute_calibshift,
        _normalize_to_peak,
        _reject_double_peak,
)

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

        def history(self) -> None:
                """Display the notes accumulated in the 'comments' attribute of the :class:`singlespec.ssxr` `xarray` variable.
                """
                print('Data modification history:\n')
                print(self.ssxr.attrs['comments'])

        def print_metadata(self) -> None:
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

        def calibrate(self, peakshift: float, calibfactor: float = 0, **kwargs) -> singlespec:
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
                # Guard + validate via the shared helpers.
                _reject_double_peak(kwargs.get('func'), 'calibrate')
                peakshift = _as_float('peakshift', peakshift)
                calibfactor = _as_float('calibfactor', calibfactor)

                # Delegate: spec has no spatial coords, so the whole ssxr is
                # already the 1-D "reference spectrum" the core expects.
                calibshift = _compute_calibshift(self.ssxr, peakshift, calibfactor, **kwargs)

                # Shallow copy — ramanshift is the only coord that changes.
                ss_mod = copy.copy(self)
                ss_mod.ssxr = self.ssxr.assign_coords(
                        ramanshift=self.ssxr['ramanshift'] + calibshift
                )
                # Note: no trailing space before ``\n`` on the singlespec
                # comment (map carries a trailing space — keep the divergence
                # byte-exact so the checksum test passes).
                if hasattr(ss_mod.ssxr, 'comments'):
                        ss_mod.ssxr.attrs['comments'] += (
                                'calibrated Raman shift by adding '
                                + f'{calibshift:.2f}'
                                + ' cm^-1 to the raw ramanshift\n'
                        )
                return ss_mod

        def normalize(self, peakshift: float, **kwargs) -> singlespec:
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
                # Guard + validate.
                _reject_double_peak(kwargs.get('func'), 'normalize')
                peakshift = _as_float('peakshift', peakshift)

                # Delegate the crop / bg-check / fit / divide to the shared
                # core. ``ref_coords=None`` + a 1-D DataArray means the core
                # computes no midpoints (no non-ramanshift dims); mode is
                # always 'const' for spectra (no per-pixel variant exists).
                normalized, peakampl, peakpos = _normalize_to_peak(
                        self.ssxr,
                        peakshift,
                        ref_coords=None,
                        mode='const',
                        **kwargs,
                )
                # Spec comment omits the ``in mode == XXX`` token that map's
                # version carries. Preserved byte-for-byte for the checksum.
                normalized.attrs['comments'] += (
                        'normalized to peak at: '
                        + f'{peakpos:.2f}'
                        + ' by a factor of '
                        + f'{peakampl:.2f}'
                        + '\n'
                )

                # Lightweight copy and attach normalized data.
                ss_norm = copy.copy(self)
                ss_norm.ssxr = normalized.copy()
                ss_norm.normfactor = peakampl
                return ss_norm

        def crr(self, cutoff: float = 2, window: int = 2, **kwargs) -> singlespec:
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

                # Delegate to the shared rolling-window CRR core.
                cleaned, n_spikes = _crr_xarray(self.ssxr, cutoff, window)
                ss_crr = copy.copy(self)
                ss_crr.ssxr = cleaned
                # Spec comment wording differs from map — "Ramanshift
                # coordinates" here vs. just "coordinates" on the map side.
                # Preserved byte-for-byte to keep the pipeline checksum happy.
                ss_crr.ssxr.attrs['comments'] += (
                        'replaced cosmic ray values with local mean at '
                        + f'{n_spikes}'
                        + ' Ramanshift coordinates.\n'
                )
                return ss_crr


        # internal functions ----------------------------------

        def __init__(self, spec_path: str, info_path: str | None = None) -> None:
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
                # Shared scaffolding (metadata / date / time / sample / laser /
                # itime / grating / objname / objmagn / positioner_x / _y).
                _apply_default_common_metadata(self)
                # Spectrum name derived from the data file's stem; the data
                # header's GraphName, if present, will overwrite this via
                # _apply_datafile_header.
                self.specname = os.path.splitext(os.path.basename(self.filename))[0]
                # Spec-only extra: the Z positioner coordinate that maps lack.
                self.positioner_z = _NO_INFO_NUM


        def _apply_datafile_header(self):
                """Override sentinel metadata with values parsed from the data header.

                Called by :py:meth:`_load_singlespec` after
                ``self.metadata_datafile`` has been captured. Overrides are
                limited to keys the parser actually finds; missing fields
                leave sentinels in place.
                """
                # Shared parse + setattr iteration. ``name_attr='specname'``
                # routes the GraphName override to this class's name field.
                # ``copy_keys`` deliberately excludes pixel_x / pixel_y / size_x
                # / size_y: spectra ignore those keys even if ``SizeX = SizeY
                # = 1`` appears in the header.
                from ._witec import _parse_witec_datafile_header
                fields = _parse_witec_datafile_header(self.metadata_datafile)
                _apply_parsed_header(
                        self,
                        fields,
                        name_attr='specname',
                        copy_keys=('positioner_x', 'positioner_y', 'positioner_z'),
                )


        def _load_info(self, info_path):
                """
                Load the file containing the metadata.
                The metadata will be filled by searching the info file for various patterns, using regular expressions.
                """
                # Shared parser returns raw text + a {attr: value} dict.
                # ``is_map=False`` swaps the four map-only pixel / scan keys
                # for the single spec-only ``positioner_z`` key.
                self.metadata, fields = _parse_info_file(info_path, is_map=False)
                self.specname = fields.pop('name')
                for attr, value in fields.items():
                        setattr(self, attr, value)
                return self.metadata

        def _load_singlespec(self, spec_path):
                """
                Load the Raman map data into a numpy array.
                """
                # One call owns header detection, metadata capture, FileName
                # extraction, and np.loadtxt — same shared loader map.py uses.
                _, self.metadata_datafile, self.wipfilename, ss = _load_witec_datafile(spec_path)
                # When no info file was supplied, the data-file header (if any)
                # is our only source for extra metadata such as PositionX/Y/Z.
                # A spectrum with no header still loads fine — the parser just
                # returns an empty dict and sentinels remain in place.
                if not self._info_loaded:
                        self._apply_datafile_header()
                # Spec is a plain two-column table: ramanshift in col 0,
                # counts in col 1. No reshape needed.
                self.ramanshift = ss[:, 0]
                self.counts = ss[:, 1]
                return self.ramanshift, self.counts

        def _toxarray(self):
                """
                Load the raw numpy data, as well as the metadata into an xarray object
                """
                self.ssxr = xr.DataArray(
                        self.counts,
                        dims=['ramanshift'],
                        coords={'ramanshift': self.ramanshift},
                )
                # Comment text flags whether metadata came from an info file
                # or from the data-file header / defaults.
                if self._info_loaded:
                        self.ssxr.attrs['comments'] = 'raw data loaded \n'
                else:
                        self.ssxr.attrs['comments'] = 'raw data loaded (no info file; metadata from data header) \n'
                # Shared attribute population — same block map.py uses. The
                # helper includes ramanshift coord units / long_name so they
                # don't need to be set twice.
                _set_common_xarray_attrs(self.ssxr, self, self._info_loaded)
                # Spec-only extra: Z positioner coordinate (maps don't carry it).
                self.ssxr.attrs['sample positioner Z'] = self.positioner_z

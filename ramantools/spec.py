"""The ``singlespec`` container class.

Extracted from ``ramantools/ramantools.py`` during the Phase 1 refactor.
Public API is unchanged; this file only moves the code.
"""
import os  # Filename handling for the no-info-file fallback path
import re  # Regular expressions for parsing metadata
import copy  # Object copying utilities
import numpy as np  # type: ignore
import xarray as xr  # type: ignore

from ._helpers import _NO_INFO_STR, _NO_INFO_NUM
from ._witec import _parse_witec_datafile_header, _graphname_to_name
from ._fitting import lorentz2, bgsubtract, peakfit

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

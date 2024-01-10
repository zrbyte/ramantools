import re, copy
import numpy as np
import matplotlib.pyplot as pl
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import xarray as xr

"""
Module ramantools
=============
Tools to analize Raman spectroscopy data, measured using the Witec 300rsa+ confocal Raman spectrometer.
"""

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
	:param info_path: Path to the info file, containing the metadata, exported from Witec
	:type info_path: str

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

		# Creating a figure with two subplots and a given size
		fig, [ax0, ax1] = pl.subplots(1, 2, figsize = (9, 4))
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
		pl.tight_layout()

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
		# create a copy of the instance
		map_mod = copy.deepcopy(self)

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
			# Make an individual fit to all spectra in the map
			print('not implemented yet. Sorry')
			coeff = 0
			covar = 0
			pass
		
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
		"""
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

		# create a copy of the instance
		map_mod = copy.deepcopy(self)

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
		"""

		# get the middle of the map
		if (width is not None) and (height is not None):
			mapwidth = width
			mapheight = height
		else:
			# if no width and height parameters are supplied, take the middle spectrum
			wmin = min(self.mapxr.width.data)
			wmax = max(self.mapxr.width.data)
			hmin = min(self.mapxr.height.data)
			hmax = max(self.mapxr.height.data)
			mapwidth = (wmax - wmin)/2 + wmin
			mapheight = (hmax - hmin)/2 + hmin

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

			map_norm = normalized

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
		
		# create a copy of the instance
		map_norm = copy.deepcopy(self)
		# replace the xarray variable with the normalized one
		map_norm.mapxr = normalized

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

		# shift the data by `window` pixels to the left and right
		left = self.mapxr.shift(ramanshift = -window)
		right = self.mapxr.shift(ramanshift = window)

		# the standard deviation of this will give us the noise level in the data
		# we can then take a `cutoff` times the std to locate the CRR peaks
		difference = left - right
		noise_std = difference.std()
		# calculate the average of the Raman intensity positions of the neighbor pixels from window away
		average = (left + right)/2

		# positions of the CRR peaks, where the Raman signal is higher than `cutoff*noise_std`
		crrpos = (self.mapxr > (left + cutoff*noise_std)) & (self.mapxr > (right + cutoff*noise_std))

		# make new Dataarray with the values with the averages at the CRR positions, where the crrpos mask is True and with the original values where it is False.
		map_crr_removed = xr.where(crrpos, average.where(crrpos), self.mapxr)
		# make a copy of the spec instance
		map_crr = copy.deepcopy(self)
		map_crr.mapxr.data = map_crr_removed.data

		# add comment to attributes
		map_crr.mapxr.attrs['comments'] += 'replaced cosmic ray values with average of neighbors at ' + f'{crrpos.sum().data}' + ' coordinates.\n'

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
		# get the middle of the map
		if (width is not None) and (height is not None):
			mapwidth = width
			mapheight = height
		else:
			# if no width and height parameters are supplied, take the middle spectrum
			wmin = min(self.mapxr.width.data)
			wmax = max(self.mapxr.width.data)
			hmin = min(self.mapxr.height.data)
			hmax = max(self.mapxr.height.data)
			mapwidth = (wmax - wmin)/2 + wmin
			mapheight = (hmax - hmin)/2 + hmin

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
		mapmasked = copy.deepcopy(self)
		mapmasked.mapxr = mapmasked.mapxr.where(peakmask)

		# update the comment attributes if it exists
		if hasattr(mapmasked.mapxr, 'comments'):
			mapmasked.mapxr.attrs['comments'] += 'cropped regions, where mean Raman int. of peak: ' + f'{peakx0:.2f}' + ' is less than ' + f'{cutoff:.2f}' + ' of selected peak\n'

		return mapmasked, peakmask

	# internal functions --------------------------

	def __init__(self, map_path, info_path):
		"""Constructor for :class:`ramanmap`
		"""		
		# filename
		self.filename = map_path
		# fitmask
		self.mask = None
		# normalization factor, factor by which the raman intensity values are divided during normalize
		self.normfactor = None
		# Load the metadata
		self._load_info(info_path)
		# Load the Raman map
		self._load_map(map_path)
		# load the data into an xarray container
		self._toxarray()


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

		# Check to see if metadata are present in the data file
		with open(map_path, 'r', encoding = 'latin1') as file:
			lines = [next(file).strip() for _ in range(2)]
		
		if 'Header' in lines[1]:
			# we have a header
			# load additional metadata from the data file itself, ie the first 19 lines we have skipped.
			with open(map_path, 'r', encoding = 'latin1') as file:
				lines = [next(file).strip() for _ in range(17)]
				self.metadata_datafile = '\n'.join(lines)
			# need to skip the header when loading
			toskip = 19

			# extract the WIP filename
			self.wipfilename = re.findall(r'FileName = (.*?)(?:\n|$)', self.metadata_datafile)[0]
		else:
			# there is no header
			toskip = 0
			self.wipfilename = map_path
		
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
		# add a comment field
		self.mapxr.attrs['comments'] = 'raw data loaded \n'
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
		self.mapxr.attrs['objective magnification'] = self.objmagn + 'x'
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
	:param info_path: Path to the info file, containing the metadata, exported from Witec
	:type info_path: str

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

		# create a copy of the instance
		singlesp_mod = copy.deepcopy(self)

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
		"""		

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

		# create a copy of the instance
		ss_mod = copy.deepcopy(self)

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
		"""

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

		# copy the singlespec instance
		ss_norm = copy.deepcopy(self)
		ss_norm.ssxr = normalized

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

		# shift the data by `window` pixels to the left and right
		left = self.ssxr.shift(ramanshift = -window)
		right = self.ssxr.shift(ramanshift = window)

		# the standard deviation of this will give us the noise level in the data
		# we can then take a `cutoff` times the std to locate the CRR peaks
		difference = left - right
		noise_std = difference.std()
		# calculate the average of the Raman intensity positions of the neighbor pixels from window away
		average = (left + right)/2

		# positions of the CRR peaks, where the Raman signal is higher than `cutoff*noise_std`
		crrpos = (self.ssxr > (left + cutoff*noise_std)) & (self.ssxr > (right + cutoff*noise_std))

		# Ramanshift coordinates of the CRR peaks
		crr_coords = self.ssxr.ramanshift.where(crrpos, drop = True)

		# make a copy of the spec instance
		ss_crr = copy.deepcopy(self)
		# replace the values with the averages at the CRR positions
		ss_crr.ssxr.loc[{'ramanshift': crr_coords}] = average.sel(ramanshift=crr_coords).data

		# add comment to attributes
		ss_crr.ssxr.attrs['comments'] += 'replaced cosmic ray values with average of neighbors at ' + f'{crrpos.sum().data}' + ' Ramanshift coordinates.\n'

		return ss_crr


	# internal functions ----------------------------------

	def __init__(self, spec_path, info_path):
		"""Constructor for :class:`singlespec`
		"""	
		# filename
		self.filename = spec_path
		# fit mask	
		self.mask = None
		# normalization factor, factor by which the raman intensity values are divided during normalize
		self.normfactor = None
		# Load the metadata
		self._load_info(info_path)
		# Load the Raman map
		self._load_singlespec(spec_path)
		# load the data into an xarray container
		self._toxarray()

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

		# Check to see if metadata are present in the data file
		with open(spec_path, 'r', encoding = 'latin1') as file:
			lines = [next(file).strip() for _ in range(2)]
		
		if 'Header' in lines[1]:
			# we have a header
			# load additional metadata from the data file itself, ie the first 19 lines we have skipped.
			with open(spec_path, 'r', encoding = 'latin1') as file:
				lines = [next(file).strip() for _ in range(17)]
				self.metadata_datafile = '\n'.join(lines)
			# need to skip the header when loading
			toskip = 17

			# extract the WIP filename
			self.wipfilename = re.findall(r'FileName = (.*?)(?:\n|$)', self.metadata_datafile)[0]
		else:
			# there is no header
			toskip = 0
			self.wipfilename = spec_path

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
		
		# add a comment field
		self.ssxr.attrs['comments'] = 'raw data loaded \n'
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
		self.ssxr.attrs['objective magnification'] = self.objmagn + 'x'
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
	:param y_data: Raman intesity values
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
		pl.plot(x_data, y_data, label = 'Raman spectrum')

		# Highlight the peaks
		if fitmask is None:
			pl.scatter(x_data[peak_indices], y_data[peak_indices], color = 'green', label = 'peaks')
		else:
			pass

		# Plot the fitted polynomial
		pl.plot(x_data, bg_values, color = 'k', ls = "dashed", label = 'fitted polynomial')

		# Highlight the background used for fitting
		pl.scatter(uncovered_x_data, uncovered_y_data, color = 'red', marker= 'o', alpha = 1, label = 'background used for fit') # type: ignore

		pl.xlabel('Raman shift (cm$^{-1}$)')
		pl.ylabel('Raman intensity (a.u.)')
		pl.title('Data plot with peaks, fitted line and background highlighted.')
		pl.legend()
	
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
				wmin = np.min(xrobj.width.data)
				wmax = np.max(xrobj.width.data)
				hmin = np.min(xrobj.height.data)
				hmax = np.max(xrobj.height.data)
				plotwidth = (wmax - wmin)/2 + wmin
				plotheight = (hmax - hmin)/2 + hmin
			
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

		pl.plot(shift, funcvalues, color = 'red', lw = 3, alpha = 0.5, label = 'fit')
		pl.xlim([fitpeakpos - plotarea_x, fitpeakpos + plotarea_x])
		pl.ylim(plotarea_y)
		pl.legend()
	
	# copy attributes to the fit dataset, update the 'comments'
	fit.attrs = xrobj.attrs.copy()
	# update the comments if it exists
	if hasattr(fit, 'comments'):
		fit.attrs['comments'] += 'peak fitting, using ' + str(func.__name__) + '\n'
		
	return fit


import re
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
	:var samplename: (type: str) name of the sample, as shown in the Witec software.
	:var mapname: (type: str) contains the name of the Raman map, as shown in the Witec software.

	For a compete list see example below.

	Examples::

		import ramantools as rt

		map_path = r'data path on you machine'
		info_path = r'metadata path on your machine'
		# use raw strings, starting with `r'` to escape special characters, such as backslash

		map = rt.ramanmap(map_path, info_path)
		# list of the variables stored in the `ramanmap` instance
		print(list(map.__dict__))
	
	"""	

	def __init__(self, map_path, info_path):
		"""Constructor for :class:`ramanmap`
		"""		
		# Load the metadata
		self._load_info(info_path)
		# Load the Raman map
		self._load_map(map_path)
		# load the data into an xarray container
		self._toxarray()


	def print_metadata(self):
		"""
		Prints the metadata of the :class:`ramanmap` instance, imported from the info file.

		:return: none
		"""
		print(self.metadata)

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
		self.samplename = re.findall(r'(?<=Sample Name:\t)-?.+', metadata)[0]
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
		m = np.loadtxt(map_path, skiprows=19, encoding='latin1')
		# The raman shift is the first column in the exported table.
		self.ramanshift = m[:, 0]
		self.map = np.reshape(m[:, 1:], (m.shape[0], self.pixel_y, self.pixel_x))

		# load additional metadata from the data file itself, ie the first 19 lines we have skipped.
		with open(map_path, 'r', encoding = 'latin1') as file:
			lines = [next(file).strip() for _ in range(17)]
			self.metadata_datafile = '\n'.join(lines)

		# extract the WIP filename
		self.wipfilename = re.findall(r'FileName = (.*?)(?:\n|$)', self.metadata_datafile)[0]

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
		self.mapxr.coords['ramanshift'].attrs['units'] = 'cm$^{-1}$'
		self.mapxr.coords['ramanshift'].attrs['long_name'] = 'Raman shift'
		self.mapxr.coords['width'].attrs['units'] = '$\mathrm{\mu m}$'
		self.mapxr.coords['width'].attrs['long_name'] = 'width'
		self.mapxr.coords['height'].attrs['units'] = '$\mathrm{\mu m}$'
		self.mapxr.coords['height'].attrs['long_name'] = 'height'

## ----------------------------------------------------

class singlespec:
	"""
	Container for Raman single spectra, imported from a text file.
	The text file needs to be exported as a "table" from Witec Project or Witec Control
	Additional info also needs to be exported, containing the metadata for the measurement.
	This is the text next to the map data in the Witec software.
	It takes two arguments, the first is the path to the file containing the spectroscopy data, the second is the path to the metadata.

	:return: object containing the data and metadata
	:rtype: :class:`singlespec` instance

	:param spec_path: Path to the text file, containing the Raman spectrum, exported from Witec
	:type spec_path: str
	:param info_path: Path to the info file, containing the metadata, exported from Witec
	:type info_path: str

	Most important variables of the :class:`singlespec` instance:

	:var ssxr: (type :py:mod:`xarray` DataArray) all data, coordinates and metadata
	:var counts: (type :py:mod:`numpy` array) Raman intensity values
	:var ramanshift: (type :py:mod:`numpy` array) Raman shift values for the datapoints stored in `map`
	:var samplename: (type: str) name of the sample, as shown in the Witec software.
	:var specname: (type: str) contains the name of the Raman single spectrum, as shown in the Witec software.

	For a compete list see example below.

	Examples::

		import ramantools as rt

		spec_path = r'data path on you machine'
		info_path = r'metadata path on your machine'
		# use raw strings, starting with `r'` to escape special characters, such as backslash

		single_spectrum = rt.singlespec(spec_path, info_path)
		# list of variables stored in the `singlespec` instance
		print(list(single_spectrum.__dict__))

	"""

	def __init__(self, spec_path, info_path):
		"""Constructor for :class:`singlespec`
		"""		
		# Load the metadata
		self._load_info(info_path)
		# Load the Raman map
		self._load_singlespec(spec_path)
		# load the data into an xarray container
		self._toxarray()


	def print_metadata(self):
		"""
		Prints the metadata of the :class:`singlespec` instance, imported from the info file.

		:return: none
		"""
		print(self.metadata)

	def remove_bg(self, **kwargs):
		"""Remove the background of Raman spectra.
		It takes the same optional arguments as :func:`bgsubtract`.

		:return: Returns an :py:mod:`xarray` instance, with the same data and metadata, but the background removed
		:rtype: :py:mod:`xarray`
		"""		
		data_nobg, bg_values, coeff = bgsubtract(self.ramanshift, self.counts, **kwargs)

		return data_nobg, bg_values, coeff

	# internal functions

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
		self.samplename = re.findall(r'(?<=Sample Name:\t)-?.+', metadata)[0]
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
		ss = np.loadtxt(spec_path, skiprows = 17, encoding = 'latin1')
		self.ramanshift = ss[:, 0]
		self.counts = ss[:, 1]

		# load additional metadata from the data file itself, ie the first 19 lines we have skipped.
		with open(spec_path, 'r', encoding = 'latin1') as file:
			lines = [next(file).strip() for _ in range(17)]
			self.metadata_datafile = '\n'.join(lines)

		self.wipfilename = re.findall(r'FileName = (.*?)(?:\n|$)', self.metadata_datafile)[0]

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
		self.ssxr.coords['ramanshift'].attrs['units'] = 'cm$^{-1}$'
		self.ssxr.coords['ramanshift'].attrs['long_name'] = 'Raman shift'

## internal functions ------------------------------------------------------------

# nothing here yet

## Tools -----------------------------------------------------------------

def lorentz(x, x0, area, width, offset):
	"""Single Lorentz function

	:return: values of a single Lorentz function
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

	The amplitude of the peak can be given by::
	(2*area)/(np.pi*width)
	
	"""
	return offset + (2/np.pi) * (area * width) / (4*(x - x0)**2 + width**2)

def lorentz2(x, x01, area1, width1, x02, area2, width2, offset):
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
	return offset + (2/np.pi) * (area1 * width1) / (4*(x - x01)**2 + width1**2) + (2/np.pi) * (area2 * width2) / (4*(x - x02)**2 + width2**2)

def polynomial_fit(order, x_data, y_data):
	"""Polinomial fit to `x_data`, `y_data`

	:param order: order of the polinomial to be fit
	:type order: int
	:param x_data: x coordinate of the data, this would be Raman shift
	:type x_data: :py:mod:`numpy` array
	:param y_data: y coordinate of the data, this would be Raman intensity
	:type y_data: :py:mod:`numpy` array
	:return: coefficients of the polinomial, as used by `np.polyval`
	:rtype: :py:mod:`numpy` array

	"""    
	# Define polynomial function of given order
	def poly_func(x, *coeffs):
		y = np.polyval(coeffs, x)
		return y

	# Initial guess for the coefficients is all ones
	guess = np.ones(order + 1)

	# Use curve_fit to find best fit parameters
	coeff, _ = curve_fit(poly_func, x_data, y_data, p0 = guess)

	return coeff

def plotspec(xrobject, width, height, shift):
	"""
	Plots a Raman map at a given Raman shift and displays alongside a selected spectrum at a specified width and height.
	First variable is a `ramanmap` object, followed by the specific width and height coordinates for the single spectrum.
	
	:return: none

	:param xrobject: This is a Raman map
	:type xrobject: :py:mod:`xarray` DataArray
	:param width: 'width' coordinate in um, from xrobject
	:type width: float
	:param height: 'height' coordinate in um, from xrobject
	:type height: float
	:param shift: 'ramanshift' coordinate in um, from xrobject
	:type shift: float

	"""
	spec = xrobject.sel(width = width, height = height, method = 'nearest')
	ramanintensity = xrobject.sel(ramanshift = shift, method = 'nearest')

	# Creating a figure with two subplots and a given size
	fig, [ax0, ax1] = pl.subplots(1, 2, figsize = (9, 4))
	# plotting the density plot of the 2D peak area ratio
	ramanintensity.plot(
		ax = ax0,
		cmap = 'plasma')
	# plotting the spectrum at the given coordinates on the right plot
	spec.plot(ax = ax1, color = 'lime')
	# plotting a dot on the density map, where the selected spectrum was measured
	ax0.scatter(
		spec.coords['width'].data,
		spec.coords['height'].data,
		color = 'lime', marker = 'x')
	ax0.set_aspect('equal', 'box')
	ax0.axes.title.set_size(10)
	ax1.axes.title.set_size(10)
	pl.tight_layout()

def bgsubtract(x_data, y_data,
	       polyorder = 1,
	       toplot = False,
	       hmin = 50,
		   hmax = 10000,
		   wmin = 4,
		   vmax = 60,
		   prom = 10,
		   exclusion_factor = 6,
		   peak_pos = None):
	"""Takes Raman shift and Raman intensity data and automatically finds peaks in the spectrum, using :py:mod:`scipy.find_peaks`.
	These peaks are then used to define the areas of the background signal.
	In the areas with the peaks removed, the background is fitted, using :py:mod:`scipy.curvefit`.
	The function returns the Raman intensity counts with the background removed, the background polinomial values themselves and the coefficients of the background fit results, as used by :py:mod:`numpy.polyval`

	:return: ``y_data_nobg``, ``bg_values``, ``coeff``
	:rtype: :py:mod:`numpy` array, :py:mod:`numpy` array, :py:mod:`numpy` array

	:param x_data: Raman shift values
	:type x_data: :py:mod:`numpy` array
	:param y_data: Raman intesity values
	:type y_data: :py:mod:`numpy` array
	:param polyorder: order of polynomial used to fit the background, defaults to 1
	:type polyorder: int, optional
	:param toplot: if `True` a plot of: the fit, the background used and positions of the peaks is shown, defaults to False
	:type toplot: bool, optional
	:param hmin: minimum height of the peaks passed to :py:mod:`scipy.signal.find_peaks`, defaults to 50
	:type hmin: int, optional
	:param hmax: maximum height of the peaks passed to :py:mod:`scipy.signal.find_peaks`, defaults to 10000
	:type hmax: int, optional
	:param wmin: minimum width of the peaks, passed to :py:mod:`scipy.signal.find_peaks`, defaults to 4
	:type wmin: int, optional
	:param vmax: maximum width of the peaks passed to :py:mod:`scipy.signal.find_peaks`, defaults to 60
	:type vmax: int, optional
	:param prom: prominence of the peaks, passed to :py:mod:`scipy.signal.find_peaks`, defaults to 10
	:type prom: int, optional
	:param exclusion_factor: this parameter multiplies the width of the peaks found by :py:mod:`scipy.signal.find_peaks`, or specified by ``wmin`` if the peak positions are passed by hand, using ``peak_pos``, defaults to 6
	:type exclusion_factor: int, optional
	:param peak_pos: _description_, defaults to None
	:type peak_pos: _type_, optional

	.. note::
		bla

	"""	
	if peak_pos is None:
		# Find the peaks with specified minimum height and width
		# re_height sets the width from the maximum at which value the width is evaluated
		peak_properties = find_peaks(y_data, height = (hmin, hmax), width = (wmin, vmax), prominence = prom, rel_height = 0.5)

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
	# make the mask False for the region below the notch filter cutoff (~80 cm^{-1})
	x_data_notch = x_data[x_data < 80]
	mask[:x_data_notch.shape[0]] = False
	uncovered_x_data = x_data[mask]
	uncovered_y_data = y_data[mask]

	# Fit polynomial to the remaining data
	coeff = polynomial_fit(polyorder, uncovered_x_data, uncovered_y_data)

	# Calculate the fitted polynomial values
	bg_values = np.polyval(coeff, x_data)

	# Line subtracted data
	y_data_nobg = y_data - bg_values

	if toplot == True:
		# Plot the data and peaks
		pl.plot(x_data, y_data, label = 'Raman spectrum')

		# Highlight the peaks
		pl.scatter(x_data[peak_indices], y_data[peak_indices], color = 'green', label = 'peaks')

		# Plot the fitted polynomial
		pl.plot(x_data, bg_values, color = 'k', ls = "dashed", label = 'fitted polynomial')

		# Highlight the background used for fitting
		pl.scatter(uncovered_x_data, uncovered_y_data, color = 'red', marker= 'o', alpha = 1, label = 'background used for fit')

		pl.xlabel('Raman shift (cm$^{-1}$)')
		pl.ylabel('Raman intensity (a.u.)')
		pl.title('Data plot with peaks, fitted line and background highlighted.')
		pl.legend()

	return y_data_nobg, bg_values, coeff
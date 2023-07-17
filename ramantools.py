import re
import numpy as np
import pylab as pl
from scipy.optimize import curve_fit
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
		Prints the imported metadata, loaded from the info file.
		
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
		Prints the metadata of the `singlespec` object.

		:return: none
		"""
		print(self.metadata)

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


## Tools -------------------------------------------------

def plotspec(xrobject, width, height, shift):
	"""
	`plotspec` plots a Raman map at a given Raman shift and displays alongside a selected spectrum at a specified width and height.
	First variable is a `ramanmap` object, followed by the specific width and height coordinates for the single spectrum.
	
	:return: none

	:param xrobject: This is a Raman map
	:type xrobject: `xarray DataArray`
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


Introduction
============

Tools to analize Raman spectroscopy data, measured using the Witec 300rsa+ confocal Raman spectrometer.
Source is available from `GitHub <https://github.com/zrbyte/ramantools/>`_.

Requires python 3.10 or later.

.. Operations made to the xarray instances, using the methods of the `ramanmap` and `singlespec` classes are appended to the 'comments' attribute.

Installation
=============

.. code-block:: 
	
	pip install ramantools


Examples
=============

Take a look at the ``ramantools demo.ipynb`` and ``ramantools tutorial.ipynb`` Jupyter notebooks in the `GitHub repository <https://github.com/zrbyte/ramantools/>`_.

Below is a simple example for loading a Raman map and doing some basic analysis.

.. code-block:: python
	
	import ramantools as rt

	map_path = r'data path on you machine'
	info_path = r'metadata path on your machine'
	# Use raw strings, starting with `r'` to escape special characters, such as backslash.

	# Load a map
	m = rt.ramanmap(map_path, info_path)

	# plot a density plot of the Raman intensity at 2750 cm^-1
	m.mapxr.sel(ramanshift = 2750, method = 'nearest').plot()
	
	# plot the middle spectrum (using "numpy-like" index based slicing) of the mapxr `xarray` instance
	m.mapxr[:, int(map.size_x/2), int(map.size_y/2)].plot()

	# remove background
	m_nobg, coeff, covar = m.remove_bg()

	# plot a spectrum from the `ramanmap` instance, with the background removed, at the specified coordinates, using the `sel()` method of `xarray`
	m_nobg.mapxr.sel(width = 31, height = 42, method = 'nearest').plot()

	# fit a Lorenzian to the peak at ~2750
	fit = rt.peakfit(m_nobg.mapxr, stval = {'x0': 2750, 'ampl': 100, 'width': 50, 'offset': 900})

	# plot the width of the peak on a density plot
	fit['curvefit_coefficients'].sel(param = 'width').plot(vmin = 40, vmax = 80)

	# save the map without the background, together with its history, mask and metadata,
	# and the fit results, for easy loading later
	m_nobg.save(data_path + 'map.nc')
	fit.to_netcdf(data_path + 'map_Lorentz fit.nc')

	# in a later session: restore the object and continue
	m_nobg = rt.load(data_path + 'map.nc')
	m_nobg.history()

Working without a notebook
==========================

Typing an object name shows a summary (``repr(m)``; ``m.summary()`` returns it as a dict),
``m.history(as_list=True)`` returns the processing steps with their parameters,
every plotting option accepts ``savepath=`` to write an image instead of displaying it,
and the ``ramantools`` command-line tool inspects, exports and plots files directly.
The ``AGENTS.md`` file in the repository collects all of this for scripts and AI agents.



# ramantools
Tools for analyzing Raman spectroscopy data, measured by a Witec confocal Raman microscope. It uses [xarray](https://docs.xarray.dev/en/stable/) DataArrays to effiently store and analyze spectroscopy maps and individual spectra, as well as their metadata. There are two containers for Raman spectroscopy data: [ramanmap](https://zrbyte.github.io/ramantools/ramantools.html#ramantools.ramantools.ramanmap) and [singlespec](https://zrbyte.github.io/ramantools/ramantools.html#ramantools.ramantools.singlespec), for use with spectroscopy maps and single spectra.

Documentation can be found here: [zrbyte.github.io/ramantools](https://zrbyte.github.io/ramantools/)

Installation:
`pip install ramantools`

WITec project files (`.wip` / `.wid`) are read directly with `rt.wipfile()` through the [WITecSDK](https://pypi.org/project/WITecSDK/) package, which is installed as a dependency.

Processed maps and spectra can be written to netCDF with `obj.save('file.nc')` and restored, history and metadata included, with `rt.load('file.nc')`. A command-line tool, `ramantools` (or `python -m ramantools`), inspects, exports and plots data without writing Python: `ramantools inspect project.wip`, `ramantools export project.wip --id 28 --out map.nc`, `ramantools plot map.nc --out map.png`.

## Using ramantools from scripts and AI agents

[`AGENTS.md`](AGENTS.md) is a compact usage guide for working without a notebook: the public API with return shapes and units, structured processing history (`obj.history(as_list=True)`), headless plotting (`savepath=` on every plot), save/load, the CLI, logging and a synthetic data generator (`testing/synthetic.py`) for reproducible examples. Text versions of the tutorial notebooks are in [`docs/notebooks/`](docs/notebooks/). A Claude Code skill wrapping the same workflow ships in `.claude/skills/ramantools/`.

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/zrbyte/ramantools/HEAD?labpath=ramantools%20demo.ipynb)

If you use this package in your publication, consider citing it:
Peter Nemes-I. (2023) “zrbyte/ramantools: v0.3.1”. Zenodo. doi: 10.5281/zenodo.10143138.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10143138.svg)](https://doi.org/10.5281/zenodo.10143138)

## Summary of Core API

### `ramanmap`
Container class for Raman maps exported from Witec software. Key attributes include:
- `mapxr`: `xarray.DataArray` storing intensity values, coordinates and metadata
- `map`: raw `numpy` array of intensities
- `ramanshift`: array of Raman shift values
- `mask`: boolean mask populated after background removal
- `samplename` and `mapname`: names extracted from metadata
- `repr(m)` / `m.summary()`: one-screen summary (grid, size, ramanshift range, instrument, processing steps)
- `m.history()` prints the processing log; `m.history(as_list=True)` returns it as a list of step dicts
- `m.save(path)` / `ramanmap.load(path)`: complete netCDF round trip

### `singlespec`
Container for individual Raman spectra. Important variables:
- `ssxr`: `xarray.DataArray` with spectral data and metadata
- `counts` and `ramanshift`: raw spectral data
- `mask`: background fit mask
- `samplename` and `specname`: metadata fields
- `repr(s)`, `s.summary()`, `s.history(as_list=True)`, `s.save(path)` / `singlespec.load(path)` as for maps

### `wipfile`
Container for a WITec project file (`.wip` / `.wid`), opened with the WITecSDK package. Lists the project's contents and extracts them:
- `list()`, `tree()`, `maps`, `spectra`, `images`, `masks`, `texts`: inventory by object ID and caption (`tree()` draws the project as an ASCII tree, grouped per scan like Witec Project); `entries`, `list(as_list=True)` and `tree(as_string=True)` return the same information instead of printing it
- `ramanmap(id)` and `singlespec(id)`: extract a map or spectrum as the containers above, with metadata
- `mask(id)` and `image(id)`: masks drawn in Witec Project and fit-result images as 2-D `xarray.DataArray` objects aligned with the map
- `info(id)`: the Witec "Information" text of the scan
- `project`: the raw WITecSDK `Project` object

### Utility functions
- `gaussian`, `lorentz` and `lorentz2` implement standard peak shapes
- `polynomial_fit` performs polynomial baseline fitting
- `bgsubtract` removes background using automatic peak detection
- `peakfit` fits peak functions to spectra or maps; the result carries `attrs['param_units']`
- `load` opens a `.nc` file written by `save()` and returns a `ramanmap` or `singlespec`

Each class also provides methods for history tracking, background subtraction, calibration, normalization and cosmic ray removal as described in the module docstrings. Every plotting option (`toplot=True` in `bgsubtract` / `peakfit` / `remove_bg`, and `plotspec`) also accepts `savepath=` to write the figure to a file and `ax=` to draw into existing axes, for use without a display.

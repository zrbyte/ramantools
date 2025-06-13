# ramantools
Tools for analyzing Raman spectroscopy data, measured by a Witec confocal Raman microscope. It uses [xarray](https://docs.xarray.dev/en/stable/) DataArrays to effiently store and analyze spectroscopy maps and individual spectra, as well as their metadata. There are two containers for Raman spectroscopy data: [ramanmap](https://zrbyte.github.io/ramantools/ramantools.html#ramantools.ramantools.ramanmap) and [singlespec](https://zrbyte.github.io/ramantools/ramantools.html#ramantools.ramantools.singlespec), for use with spectroscopy maps and single spectra.

Documentation can be found here: [zrbyte.github.io/ramantools](https://zrbyte.github.io/ramantools/)

Installation:
`pip install ramantools`

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

### `singlespec`
Container for individual Raman spectra. Important variables:
- `ssxr`: `xarray.DataArray` with spectral data and metadata
- `counts` and `ramanshift`: raw spectral data
- `mask`: background fit mask
- `samplename` and `specname`: metadata fields

### Utility functions
- `gaussian`, `lorentz` and `lorentz2` implement standard peak shapes
- `polynomial_fit` performs polynomial baseline fitting
- `bgsubtract` removes background using automatic peak detection
- `peakfit` fits peak functions to spectra or maps

Each class also provides methods for history tracking, background subtraction, calibration, normalization and cosmic ray removal as described in the module docstrings.

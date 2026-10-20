# ramantools for agents and scripts

This file is the usage guide for anyone, human or AI agent, who drives `ramantools` from code without a notebook in front of them. It lists the public API with return shapes and units, the headless ways to look at data, and the pitfalls. Development conventions for changing the package live in the repository's `CLAUDE.md`; the full API reference is at https://zrbyte.github.io/ramantools/.

The text versions of the three tutorial notebooks are in `docs/notebooks/` (`ramantools-tutorial.md`, `ramantools-wip-tutorial.md`, `ramantools-demo.md`). Read those instead of the `.ipynb` files, which carry megabytes of embedded output.

## Install and import

```bash
pip install ramantools          # pulls numpy, scipy, xarray, matplotlib, WITecSDK, h5netcdf
pip install -e .                # inside a clone
```

```python
import ramantools as rt
```

Everything public is on the `rt` namespace: `ramanmap`, `singlespec`, `wipfile`, `load`, `peakfit`, `bgsubtract`, `polynomial_fit`, `gaussian`, `lorentz`, `lorentz2`.

## Data model in one paragraph

A `ramanmap` holds a 3-D `xarray.DataArray` at `m.mapxr` with dims `(ramanshift, height, width)`. Raman shift is in 1/cm, height and width in micrometres, values are CCD counts (`attrs['units'] == 'au'`) until `normalize`, after which they are dimensionless. A `singlespec` holds a 1-D DataArray at `s.ssxr` with the single dim `ramanshift`. Both are immutable in practice: every processing method returns a new instance and leaves the input untouched. The raw arrays as loaded stay available as `m.map` / `m.ramanshift` and `s.counts` / `s.ramanshift`. The height axis is flipped relative to the text file so that plots match Witec Project.

## Loading

Text exports from Witec Project (a data table plus, optionally, the info text next to it):

```python
m = rt.ramanmap('map.txt', 'map_info.txt')        # info file optional if the table has a [Header]
s = rt.singlespec('spectrum.txt', 'spectrum_info.txt')
```

WITec project files, without exporting anything first:

```python
wip = rt.wipfile('project.wip')
wip.tree()                      # ASCII tree like Witec Project; IDs are what you pass below
wip.entries                     # list of {'id', 'kind', 'caption', 'shape'} dicts (JSON-ready)
wip.list(as_list=True)          # same, filtered to the kinds list() prints
wip.tree(as_string=True)        # the tree as text
wip.maps, wip.spectra, wip.masks, wip.images   # (id, caption) pairs by kind
m = wip.ramanmap(28)            # by integer ID or exact caption
s = wip.singlespec(200)
mask = wip.mask(254)            # boolean DataArray aligned with the map: m.mapxr.where(mask)
img = wip.image(260)            # numeric image (fit result, band sum) as DataArray
wip.info(28)                    # the Witec "Information" text
```

Files written by `save()` (see below):

```python
m = rt.load('map_nobg.nc')      # dispatches to ramanmap or singlespec
m = rt.ramanmap.load('map_nobg.nc')   # typed; raises TypeError on a spectrum file
```

## Inspecting an object

`repr(m)` (or just `m` in a REPL) prints a multi-line summary: name, sample, source, pixel grid and size in micrometres, ramanshift range and step, instrument settings, value units and the processing steps so far. `m.summary()` returns the same facts as a JSON-serialisable dict. Both are instant; they never read the intensity data.

Processing history exists in two forms. `m.history()` prints the free-text log that every step appends to `m.mapxr.attrs['comments']`. `m.history(as_list=True)` returns the structured log stored in `attrs['processing_steps']`: a list of `{'step': name, 'params': {...}}` dicts, one per step, with the actual parameters used (polynomial coefficients, applied calibration shift, normalisation factor, number of cosmic rays replaced, pixels kept by a peak mask). Step names are `load`, `wip_extract`, `remove_bg`, `calibrate`, `normalize`, `crr`, `peakmask`; `peakfit` writes its own record onto the fit result.

`m.print_metadata()` prints the raw Witec info text; `m.metadata` is that text as a string.

## The pipeline and what each step returns

Order matters: background must be removed before `normalize` (it raises otherwise). Calibration shifts the ramanshift coordinate, so do it before any fit whose positions you will report.

```python
m_nobg, coeff, covar = m.remove_bg()                  # tuple of three
m_cal = m_nobg.calibrate(peakshift=520.5)             # new ramanmap
m_norm = m_cal.normalize(peakshift=1582)              # new ramanmap; m_norm.normfactor
m_clean = m_norm.crr(cutoff=2, window=2)              # new ramanmap
m_masked, peakmask = m_clean.peakmask(1582, cutoff=0.1)   # tuple of two
```

Return shapes:

- `remove_bg(mode='const', fitmask=None, width=None, height=None, **bgsubtract kwargs)` returns `(new_instance, coeff, covar)`. In `'const'` mode `coeff` is a 1-D numpy array in `numpy.polyval` order (highest power first) and `covar` the 2-D covariance. In `mode='individual'` (maps only) they are DataArrays over `(height, width, coef)` and `(height, width, coef_i, coef_j)`. The boolean fit mask lands on `new_instance.mask`. Pass `polyorder=` for the degree and `peak_pos=[...]` to tell the peak finder where the peaks are. `toplot=True` is refused in individual mode.
- `calibrate(peakshift, calibfactor=0, width=None, height=None, **peakfit kwargs)` returns a new instance whose `ramanshift` coordinate is shifted so the fitted peak near `peakshift` lands at `peakshift`. The applied shift is in `history(as_list=True)[-1]['params']['shift_applied']`. Give `calibfactor` to apply a known shift without fitting.
- `normalize(peakshift, width=None, height=None, mode='const', **peakfit kwargs)` returns a new instance divided by the fitted amplitude of the peak near `peakshift`. `normfactor` is a float in `'const'` mode and a `(height, width)` DataArray in `'individual'` mode. `singlespec.normalize` has no `mode`.
- `crr(cutoff=2, window=2)` returns a new instance with spikes replaced by the local mean. A one-sample spike among `2*window+1` points can only reach a z-score of about `sqrt(2*window)` against a local std that includes the spike, so the defaults sit at the detection limit; use `window=3` or lower `cutoff` when spikes survive. The first and last `window` samples come back as NaN.
- `peakmask(peakpos, cutoff=0.1, width=None, height=None)` (maps only) returns `(masked_instance, boolean_mask)`; the mask is a `(height, width)` DataArray, True where the mean intensity of the peak exceeds `cutoff` times its value in the reference spectrum.

Reference spectrum: `remove_bg`, `calibrate`, `normalize` and `peakmask` on a map use the spectrum at the centre of the map unless `width=` and `height=` (micrometres, nearest pixel) are given.

## Fitting

```python
crop = m_nobg.mapxr.sel(ramanshift=slice(1450, 1700))
fit = rt.peakfit(crop, func=rt.lorentz, stval={'x0': 1582, 'ampl': 500, 'width': 20, 'offset': 0})
fit['curvefit_coefficients'].sel(param='x0')      # DataArray over (height, width) for a map, scalar for a spectrum
fit['curvefit_covariance']
fit.attrs['param_units']                          # {'x0': '1/cm', 'ampl': 'au', 'width': '1/cm', 'offset': 'au'}
```

`peakfit(xrobj, func=lorentz, fitresult=None, stval=None, bounds=None, toplot=False, width=None, height=None, ax=None, savepath=None)` wraps `xarray.DataArray.curvefit` and returns an `xarray.Dataset`. Crop to the peak first; fits over the whole spectrum rarely converge. `func` can be `gaussian`, `lorentz` or `lorentz2` (two Lorentzians; parameters are then `x01, ampl1, width1, x02, ampl2, width2, offset`). `width` is the FWHM. The width can come back negative because the Lorentzian is symmetric in its sign; use `abs()`. Starting values default to the shape function's keyword defaults, so always pass at least `x0`.

`rt.bgsubtract(x, y, polyorder=1, ...)` is the module-level background routine on plain arrays; it returns `(y_nobg, bg_values, coeff, params_used, mask, covar)`.

## Looking at data without a display

Every plotting path takes `savepath=` to write an image and close the figure, and `ax=` to draw into Axes you own. Nothing returns a Figure, so a notebook user never sees a plot twice.

```python
m.plotspec(width=6, height=5, shift=1582, savepath='map.png')     # map image at one shift + one spectrum
s.remove_bg(savepath='bg_fit.png')                                # background fit, peaks and fit points
rt.peakfit(crop, stval={'x0': 1582}, savepath='fit.png')          # data and fitted curve around the peak
```

For anything else use xarray's plotting on the DataArray with your own `fig, ax = plt.subplots()`, then `fig.savefig(...)`. In a script that imports matplotlib, call `matplotlib.use('Agg')` before importing `ramantools` if no display is available.

## Saving and resuming

```python
m_nobg.save('map_nobg.nc')          # netCDF via h5netcdf; data, attrs, history, mask, normfactor, raw array, metadata
m_nobg = rt.load('map_nobg.nc')     # identical object; the pipeline continues from here
m_nobg.save('small.nc', include_raw=False)   # roughly half the size; raw array rebuilt from processed data on load
fit.to_netcdf('fit.nc')             # peakfit results are plain xarray Datasets
```

`mapxr.to_netcdf()` still works but keeps only the DataArray.

## Command line

The same chores without Python, via `ramantools ...` or `python -m ramantools ...`:

```bash
ramantools inspect project.wip                  # repr + object tree
ramantools inspect project.wip --json           # inventory as JSON
ramantools info project.wip --id 28             # Witec Information text
ramantools export project.wip --id 28 --out map.nc
ramantools export map.txt --info map_info.txt --out map.nc
ramantools inspect map.nc [--json]              # container summary
ramantools history map.nc --json                # processing steps
ramantools plot map.nc --out map.png [--shift 1582 --width 6 --height 5]
ramantools plot spectrum.nc --out spec.png
```

Exit status is 0 on success, 1 on a library error (message on stderr), 2 on a usage error.

## Logging

The package logs through `logging.getLogger('ramantools')`. By default nothing is configured and the few status messages (the parameter report of `peakfit(toplot=True)`) are printed as before. Once a handler exists and the level admits INFO, those messages go through logging only, never twice. Every pipeline step also emits a DEBUG record with its parameters:

```python
import logging
logging.basicConfig(level=logging.DEBUG, format='%(name)s %(levelname)s %(message)s')
```

To silence the package: `logging.getLogger('ramantools').setLevel(logging.WARNING)`.

## Synthetic data for experiments and tests

`testing/synthetic.py` writes small Witec-format files with known content: a silicon line at 520.5, graphene G at 1582 and 2D at 2700 1/cm on a flat baseline, a flake-shaped map, injected cosmic rays and an optional miscalibration. Use it to build a reproducible example without any measurement file:

```bash
python testing/synthetic.py /tmp/raman-synthetic
```

```python
import sys; sys.path.insert(0, 'testing')
import synthetic
files = synthetic.write_sample_set('/tmp/raman-synthetic')
m = rt.ramanmap(files['map']['map_path'], files['map']['info_path'])
files['map']['truth']            # flake mask, cosmic ray positions, miscalibration
```

`pytest testing/test_synthetic.py` runs on any clone; the other test modules skip when the large private fixtures in `test-data/` are absent. `pytest testing/test_notebook_outputs.py` executes `testing/notebook_output_checks.ipynb` in a Jupyter kernel and checks, cell by cell, that nothing is displayed twice (repr echo, print-and-return-None methods, `toplot` / `savepath` / `ax` plotting, print-or-log status text). The notebook is generated by `testing/make_notebook_output_checks.py`; edit the generator, not the notebook.

## Pitfalls

- `remove_bg` and `peakmask` return tuples. Unpack them.
- `normalize` raises `ValueError` if `remove_bg` has not run on that object.
- Coordinates are physical (micrometres, 1/cm). Use `.sel(..., method='nearest')`, not integer indices, unless you mean `.isel`.
- `calibrate` and `normalize` refuse `func=lorentz2`; fit the double peak with `peakfit` directly.
- The `comments` attribute is free text and may change wording; parse `history(as_list=True)` instead.
- The first and last `window` samples after `crr` are NaN.
- Info files are latin-1; the loaders handle that, but open them with `encoding='latin1'` if you read them yourself.

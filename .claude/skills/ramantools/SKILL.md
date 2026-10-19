---
name: ramantools
description: Analyse Witec Raman spectroscopy data with the ramantools Python package. Use when the user mentions Raman maps or spectra, Witec exports, .wip project files, background subtraction, peak fitting, calibration, normalization or cosmic ray removal of Raman data, or asks to inspect, plot or export such data from the command line.
---

# ramantools

Python package for Witec confocal Raman data. Two containers, `ramanmap` (3-D xarray at `.mapxr`, dims `ramanshift, height, width`) and `singlespec` (1-D at `.ssxr`), a `.wip` project reader (`wipfile`), module-level fitting helpers, a netCDF round trip and a CLI. Read `AGENTS.md` in the repository root for the full API with return shapes and units before writing more than a few lines; the text tutorials are in `docs/notebooks/`.

## Workflow

1. Find out what you have. For a project file run `ramantools inspect project.wip` (object tree with IDs) or `wip = rt.wipfile(path); wip.entries`. For a text export or a saved `.nc` run `ramantools inspect FILE --json` or `repr(obj)` / `obj.summary()`.
2. Load. `rt.ramanmap(map_txt, info_txt)`, `rt.singlespec(spec_txt, info_txt)`, `wip.ramanmap(ID)`, `wip.singlespec(ID)`, or `rt.load('x.nc')`.
3. Process in this order, each call returns a new object: `remove_bg` (returns a 3-tuple), `calibrate`, `normalize` (needs remove_bg first), `crr`, optionally `peakmask` (2-tuple). Check `obj.history(as_list=True)` to see what ran and with which parameters.
4. Fit with `rt.peakfit(obj.mapxr.sel(ramanshift=slice(lo, hi)), stval={'x0': ...})`; crop to the peak first. Units of every parameter are in `fit.attrs['param_units']`.
5. Look at results headlessly with `savepath=`: `m.plotspec(w, h, shift, savepath='map.png')`, `s.remove_bg(savepath='bg.png')`, `rt.peakfit(..., savepath='fit.png')`, or the CLI `ramantools plot FILE --out img.png`. Then read the PNG.
6. Persist with `obj.save('name.nc')` so later steps or tool calls start from `rt.load('name.nc')` instead of re-running the pipeline.

## Rules

- Never mutate `mapxr` / `ssxr` in place; use the methods, which keep the history attributes.
- Use physical coordinates with `.sel(method='nearest')`; width/height are micrometres, ramanshift is 1/cm.
- Default `crr(cutoff=2, window=2)` misses isolated one-sample spikes; try `window=3`.
- `calibrate` / `normalize` take single-peak shape functions only (`lorentz`, `gaussian`).
- Fitted `width` can be negative; report `abs()`.
- For a reproducible example without measurement files, generate data with `python testing/synthetic.py OUTDIR` (known peaks, flake mask, cosmic rays, 2.5 1/cm miscalibration).
- To trace a run, `logging.basicConfig(level=logging.DEBUG)` before calling the pipeline; each step logs its parameters under the `ramantools` logger.

## Testing changes to the package

`pytest testing/test_synthetic.py` runs anywhere. The other modules need the private fixtures in `test-data/` and skip without them. Use the `datadigger` conda environment on the maintainer's machine.

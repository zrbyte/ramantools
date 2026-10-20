<!-- Generated from "ramantools wip tutorial.ipynb" with: jupyter nbconvert --to markdown --TemplateExporter.exclude_output=True. Edit the notebook, not this file. -->

# Loading WITec project files (`.wip`) with ramantools

This notebook shows how to work directly with a WITec project file, without exporting each map and spectrum as a text table first. The main [ramantools tutorial](ramantools%20tutorial.ipynb) covers everything that happens *after* loading (background subtraction, calibration, normalization, peak fitting); all of it applies unchanged to data loaded from a project file.

Reading `.wip` / `.wid` files uses the [WITecSDK](https://pypi.org/project/WITecSDK/) package, which is installed together with ramantools.

# Table of contents
- [Opening a project file](#Opening-a-project-file)
- [What is in the project?](#What-is-in-the-project?)
- [Extracting a map](#Extracting-a-map)
- [Metadata](#Metadata)
- [Running the processing pipeline](#Running-the-processing-pipeline)
- [Extracting single spectra](#Extracting-single-spectra)
- [Masks drawn in Witec Project](#Masks-drawn-in-Witec-Project)
- [Images: fit results and band sums](#Images:-fit-results-and-band-sums)
- [Going beyond the wrapper: the raw SDK object](#Going-beyond-the-wrapper:-the-raw-SDK-object)


```python
%load_ext autoreload
%autoreload 2
```


```python
import ramantools as rt
import numpy as np
import pylab as pl
```

## Opening a project file

The example file is `FLG42_532.wip` in the `test-data` folder (it is large and not part of the git repository, so point `wip_path` at a project file of your own if it is missing).


```python
wip_path = 'test-data/FLG42_532.wip'
wip = rt.wipfile(wip_path)
wip
```

Opening is instant, even for files of several hundred MB: the SDK only reads the directory structure of the file. The actual spectra are decoded when you extract an object.

## What is in the project?

`wip.list()` prints every object in the project, in the order Witec Project shows them. The first column is the object **ID**, which is how you refer to objects from now on. The `kind` column tells you what ramantools can make of the object:

- `map`: a 2D spectral map, extracted with `wip.ramanmap(ID)`
- `spectrum`: a single spectrum, extracted with `wip.singlespec(ID)`
- `mask`: a boolean mask drawn in Witec Project, extracted with `wip.mask(ID)`
- `image`: a numeric image (fit results, band sums, ...), extracted with `wip.image(ID)`
- `text`: the "Information" blocks with the measurement metadata, and user notes
- `bitmap`: video images (not wrapped, see the last section)
- `graph`: other graph types such as line scans or time series (not wrapped)


```python
wip.list()
```


```python
wip.tree()
```

To show only some kinds, pass them explicitly:


```python
wip.list(kinds = ('map', 'spectrum', 'mask'))
```

`wip.tree()` shows the same objects as a tree, grouped per scan the way Witec Project displays them. Under each scan comes the raw spectral data, with everything derived from it nested one level deeper: extracted spectra, background-subtracted copies, fit results, band-sum images and masks (user-named masks and images are attached to the scan they were drawn on). Captions are shortened to what they add beyond their parent, and the IDs are the same ones you pass to `wip.ramanmap()`, `wip.singlespec()`, `wip.mask()` and `wip.image()`.


```python
wip.tree()
```

The SDK-internal objects (coordinate transformations, interpretations, colour profiles) are hidden by default; `wip.tree(show_all = True)` includes them.

The same inventory is available as lists of `(ID, caption)` pairs, which is handy when you want to pick objects in code, e.g. all maps whose caption contains a sample name:


```python
wip.maps
```


```python
[(i, caption) for i, caption in wip.maps if caption.startswith('FLG_42')]
```

## Extracting a map

`wip.ramanmap(ID)` returns an ordinary `ramanmap` object. You can also pass the exact caption string instead of the ID. Let's take the full-size map of the FLG_42 flake, object 188.


```python
m = wip.ramanmap(188)
m.mapxr
```

This is the same `xarray` container the text-export loader produces: the `ramanshift`, `height` and `width` coordinates are in 1/cm and µm, the metadata is in the attributes, and the data is oriented the way Witec Project displays it.


```python
m.mapxr.sel(ramanshift = 1580, method = 'nearest').plot()
```

Derived maps that you created inside Witec Project (background subtracted copies, "Sub min", fit curves, ...) are extracted the same way. Their caption keeps the suffix, so you can tell them apart from the raw scan:


```python
m_small = wip.ramanmap(107)
m_small.mapname
```

## Metadata

The measurement metadata comes from the "Information" text that Witec stores next to each scan. It is the same text you would export as the info file next to a table export, and it is parsed with the same rules, so the familiar attributes are all there:


```python
m.mapname, m.samplename, m.laser, m.itime, m.grating, m.objname, m.objmagn, m.date, m.time
```


```python
m.pixel_x, m.pixel_y, m.size_x, m.size_y
```

The pixel counts and the physical size of the scan are read from the project's coordinate transformation rather than from the text, so they are also correct for cropped or derived maps.

`wip.info(ID)` returns the raw Information text, and `print_metadata()` works as usual:


```python
print(wip.info(188))
```

The processing history records where the data came from:


```python
m.history()
```

## Running the processing pipeline

Everything in the main tutorial applies: `remove_bg()`, `calibrate()`, `normalize()`, `crr()` and `rt.peakfit()` all work on the extracted map. Let's run the standard chain on the small background-subtracted map 107.


```python
m_bg, coeff, covar = m_small.remove_bg()
m_norm = m_bg.normalize(1582)
m_crr = m_norm.crr()
m_crr.history()
```


```python
m_crr.plotspec(5, 5, 1582)
```

## Extracting single spectra

Single spectra, including the ones you extracted from a map inside Witec Project, are returned as `singlespec` objects. Witec names the latter after their pixel coordinates, e.g. `(31/33)` is the spectrum at pixel x = 31, y = 33 of the map.


```python
s = wip.singlespec(200)
s.ssxr.plot()
```

An extracted spectrum has no Information block of its own in the project, so the instrument settings (laser, integration time, grating) are inherited from the parent scan. Its position comes from the spectrum's own coordinates:


```python
s.specname, s.laser, s.itime, s.positioner_x, s.positioner_y, s.positioner_z
```

We can check that the extracted spectrum really is pixel (31, 33) of the map. Because `mapxr` is flipped along `height` to match the Witec Project orientation, pixel row `y` sits at height index `pixel_y - 1 - y`:


```python
np.allclose(s.counts, m.mapxr.data[:, m.pixel_y - 1 - 33, 31])
```

## Masks drawn in Witec Project

Masks are returned as boolean `xarray` DataArrays on the same `(height, width)` grid as the map they belong to. `True` marks the inside of the mask.


```python
mask = wip.mask(254)
mask.plot()
```

Because the coordinates match, the mask can be applied to the map directly with `where()`, which replaces everything outside the mask with NaN:


```python
m_masked = m.mapxr.where(mask)
m_masked.sel(ramanshift = 1580, method = 'nearest').plot()
```

A typical use is averaging all spectra inside the mask:


```python
m_masked.mean(dim = ['width', 'height']).plot()
pl.xlim([1400, 3000])
```

## Images: fit results and band sums

Everything Witec Project computed from a map and stored as an image (peak fits, band sums, peak widths, ...) comes out with `wip.image(ID)` as a 2D DataArray with the Witec unit in `attrs['units']`. Object 208 is the Lorentz peak position of the G peak fitted in Witec Project.


```python
x0 = wip.image(208)
x0.attrs['units']
```

Witec writes a huge sentinel value into pixels where the fit failed, so restrict the range before plotting. Pixels outside the range become NaN and are left blank:


```python
x0.where((x0 > 1560) & (x0 < 1600)).plot(cmap = 'viridis')
```

The band-sum images Witec computed can be compared to the same quantity calculated from the extracted map, which is also a good check that the map is oriented correctly:


```python
gsum_witec = wip.image(126)
gsum_ours = m_small.mapxr.sel(ramanshift = slice(1500, 1660)).sum('ramanshift')

fig, ax = pl.subplots(1, 2, figsize = (9, 4))
gsum_witec.plot(ax = ax[0]); ax[0].set_title('G-Sum stored by Witec')
gsum_ours.plot(ax = ax[1]); ax[1].set_title('G band sum from the extracted map')
pl.tight_layout()
print('correlation:', np.corrcoef(gsum_witec.data.ravel(), gsum_ours.data.ravel())[0, 1])
```

`wip.mask()` also accepts a numeric image and thresholds it at `!= 0`, in case you want to turn an image into a mask quickly.

## Going beyond the wrapper: the raw SDK object

Video bitmaps, coordinate transformations, colour profiles and anything else ramantools does not wrap are reachable through the raw WITecSDK `Project` object in `wip.project`. For example, the user notes stored as text objects in this project:


```python
wip.texts
```


```python
wip.project.GetDataClassByID(53).PlainText
```

See the [WITecSDK documentation](https://pypi.org/project/WITecSDK/) for the full API of the `Project`, `TDGraph`, `TDImage`, `TDBitmap` and `TDText` classes.


```python

```

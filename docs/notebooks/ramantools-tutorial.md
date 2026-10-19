<!-- Generated from "ramantools tutorial.ipynb" with: jupyter nbconvert --to markdown --TemplateExporter.exclude_output=True. Edit the notebook, not this file. -->

One can start from this notebook, when working with Raman measurements. Just change the data path below to point to your exported Raman measurements in text files.
The Raman measurements need to be exported from Witec Project as *table* and make sure to have the Raman shift dimensions in cm$^{-1}$.

For in depth documentation of `ramantools`, check the documentation on [github.io](https://zrbyte.github.io/ramantools/).

To use this notebook on your PC, the data needs to be loaded as described below, in section [Loading on PC](#Loading-on-PC).

# Table of contents
- [Loading a Raman map](#Loading-a-Raman-map)
- [Working with ramantools](#Working-with-ramantools)
    - [Loading a WITec project file directly (`.wip`)](#Loading-a-WITec-project-file-directly-(.wip))
- [Working with data](#Working-with-data)
    - [Calibration](#Calibration)
    - [Subtracting background](#subtracting-background)
    - [Normalizing spectra](#normalizing-spectra)
    - [Curve fitting](#curve-fitting)


```python
%load_ext autoreload
%autoreload 2
```


```python
import ramantools as rt
import glob, sys
import pylab as pl
import numpy as np
import xarray as xr
```

# Loading a Raman map

For this we need to load the map, exported as *table* from *Witec Project* or *Witec Control*, as well as the text file next to the map. This text file is used to read the metadata of the Raman measurement.

## Loading on PC

<mark>You do not need to run this section when working in *binder*.</mark>

**Skip to [Loading test data from a URL](#Loading-test-data-from-a-URL)**.


```python
data_path = 'test-data/'
sys.path.append(data_path)
```

Need to add `r` in front of the file path to treat it as a *raw string*, meaning that `\` is treated as a character.

Using `glob` to read the contents to the data folder above. The use of `glob` helps with searching for files according to patterns. See below, where we identify the map called: *ABC_36* and load its contents, including the text file containing the metadata.


```python
# list all the files in the data folder
glob.glob(data_path + '*')
```


```python
map_path = glob.glob(data_path + 'map*')
ss_path = glob.glob(data_path + 'ss*')
```


```python
map_path
```


```python
ss_path
```

# Working with ramantools

Let's load the data from the text files. We can do this by passing the path of the map file and the info file to the `ramanmap` class. Let's call it `m`.
<a id='loading_map'></a>


```python
# m = rt.ramanmap(map_path, info_path)
m = rt.ramanmap(map_path[3])
```

`rt.ramanmap(map_path, info_path)` returns a `ramanmap` object, which among other data, contains as variables:
- the raw imported, Raman measurements are stored in: `m.map`
- all the metadata in: `m.metadata`. The `m.print_metadata()` function, prints the contents of the info file.
- The Raman shift of the measurement is stored in `m.ramanshift`

In addition to the above, the `ramanmap` object, called `m` in this example has the following variables, mostly containing the measurement metadata:


```python
list(m.__dict__)
```

The Raman shift is stored in the variable: `m.ramanshift`


```python
m.ramanshift
```

## The Raman map in `xarray`

The data is also loaded into an `xarray` container. This can be found in the `mapxr` variable of the `ramanmap` object.


```python
# let's just call this `xr` for ease of use.
xrm = m.mapxr
```

These are the metadata loaded also into the xarray. For example this makes plotting easier by including the relevant metadata as axis labels, etc.


```python
xrm.attrs
```


```python
print(xrm.attrs['comments'])
```


```python
xrm.attrs['map width'], xrm.attrs['map height']
```

We can see that the map is 77x97 $\mu m$ in size. Let's select a spectrum from somewhere in the middle and plot it.


```python
_ = xrm.sel(width = 50, height = 45, method = 'nearest').plot()
```

Let's zoom in on the G peak area, by slicing, ie. giving a range of values to plot, using the `slice()` function. For this we need to use another `sel()` function.
One can pass options to the `plot()` function, similarly to a normal matplotlib plot. In this case, I've specified the marker and the color of the plot.


```python
ss = xrm.sel(width = 50, height = 45, method = 'nearest')
ss.sel(ramanshift = slice(1530, 1620)).plot(marker = 'o', color = 'tab:orange')
```

Ok. Now let's plot the intensity of the G peak over the whole map.


```python
xrm.sel(ramanshift = 1580, method = 'nearest').plot()
```

`ramantools` has a built in function for maps, called `plotspec()`, which can be used to quickly plot the position of a single spectrum on the map. This takes the arguments for the coordinates at which we can plot it. Furthermore, it takes a Raman shift as the last argument, at which to plot the Raman intensity in the map.


```python
m.plotspec(100, 30, 1583)
```

For more complicated plots, you can just do whatever you need by hand.

## Saving the `xarray` DataArray to netcdf format


```python
xrm.to_netcdf(data_path + 'xrm.nc')
```


```python
xrm2 = xr.open_dataarray(data_path + 'xrm.nc')
```


```python
xrm2
```

## Loading single Raman spectra

For this we will use the test data on Github, in the folder: 'test data'


```python
single_s = rt.singlespec(ss_path[2])
```

For single spectra, we also get an `xarray` object, called `ssxr`. 

We can plot this similarly to the map. This has only one coordinate.


```python
single_s_plot = single_s.ssxr.plot()
```


```python
single_s.print_metadata()
```

## Loading a WITec project file directly (`.wip`)

Instead of exporting every map and spectrum as a text table plus an info file, the whole project file can be opened with `rt.wipfile()` and its maps, spectra, masks and images extracted from there. This uses the [WITecSDK](https://pypi.org/project/WITecSDK/) package, which is installed together with ramantools. It has its own walkthrough: see the [ramantools wip tutorial](ramantools%20wip%20tutorial.ipynb) notebook next to this one. In short:

```python
wip = rt.wipfile('FLG42_532.wip')
wip.list()                 # what's inside, with object IDs
m = wip.ramanmap(188)      # a map, as a regular ramanmap
s = wip.singlespec(200)    # a single spectrum
mask = wip.mask(254)       # a mask drawn in Witec Project, aligned with m.mapxr
```

Everything that follows in this notebook applies unchanged to data loaded this way.

# Working with data

## Calibration


```python
print(single_s.ssxr.attrs['comments'])
```

We can calibrate, by just specifying the expected position of the peak to be used for calibration. `singlespec.calibrate()` takes one argument: the expected peak position. In this case it's the Si first order peak at 520.5 cm$^{-1}$. The method will search for a peak in this vicinity and fit a Lorenztian to this and return a new `singlespec` instance, with the ramanshift calibrated.

The method also takes all possible optional arguments for `peakfit()`. In the example below we use the toplot keyword to show the result of the fitting to the Si peak.


```python
single_s_calib = single_s.calibrate(0, toplot = True)
```


```python
# comments are updated by the calibration factor
print(single_s_calib.ssxr.attrs['comments'])
```

Let's plot the result and the original data.


```python
single_s_calib.ssxr.plot(color = 'r', label = 'calibrated')
single_s.ssxr.plot(color = 'b', label = 'raw data')
pl.xlim([-30, 30])
pl.grid()
pl.legend()
```

We can also specify a calibration factor, using the optional keyword argument: `calibfactor`. In this case the `peakshift` argument is disregarded.

For maps it is similar.


```python
print(m.mapxr.attrs['comments'])
```


```python
m_calib = m.calibrate(520)
```


```python
print(m_calib.mapxr.attrs['comments'])
```

Let's check a spectrum from the map, to see the change.


```python
m.mapxr[:, 10, 4].plot(color = 'b', label = 'raw')
m_calib.mapxr[:, 10, 4].plot(color = 'r', label = 'calibrated')
pl.xlim([480, 560])
pl.grid()
pl.legend()
```

## Subtracting background

First thing we do is we can subtract the background from the spectra. One can easily do arthmetic with an `xarray` DataArray. Let's just subtract 800 from the spectra in the map.


```python
single_bgsub = single_s.ssxr - 850
```

As you can see this is just another `xarray` object, with the Raman intensity values being smaller by 800.


```python
single_bgsub
```

The attributes from the old `xarray` object are not carried over to the new one. One reason for this is that units are generally stored in attributes and most of the time arithmetic involves changes in units.
In `ramantools`, if a method or function generates a new `xarray` instance, we try to make sure that the attributes stay up to date and carry over any changes in units.

### Background subtraction by fitting to the baseline

#### Single spectra

Let's try this for a single spectrum. This can be done by the `remove_bg()` method of the `singlespec` class.
Usually this can be run without any arguments. In this case the method tries to find the peaks and disregard their surroundings. More details in the [documentation](https://zrbyte.github.io/ramantools/index.html).


```python
# This is our raw data.
single_s.ssxr.plot()
```


```python
# Let's remove the background, with the toplot = True option to see what `removebg()` does
new_single_s, coeff, covar = single_s.remove_bg(
    toplot = True,
    wmin = 10,
    hmin = 1000,
    hmax = 100000
)
```


```python
new_single_s.ssxr.plot()
```

Many optional arguments can be passed to `remove_bg()`, the list can be found [here](https://zrbyte.github.io/ramantools/ramantools.html#ramantools.bgsubtract).

We can see that `remove_bg()` has found the peaks marked by green dots and disregarded the area around them. The fitting parameters can be found in the 'comments' attribute of the `xarray` instance.


```python
print(new_single_s.ssxr.attrs['comments'])
```

This 'comments' field also acts as a log of what happened to the data, as we work with the `ramantools` methods and functions that act on `xarray` objects.

Let's try to change the polynomial order to 2 and give the peaks to be fit by hand:


```python
new_single_s, coeff, covar = single_s.remove_bg(
    toplot = True,
    polyorder = 2,
    peak_pos = [520, 950, 1583, 2710])
# The position of the peaks to be removed is given in a list
```

We can see that not all of the peaks, especially the 2D peak is not removed entirely, we can correct this be either increasing the `wmin` parameter, which is the minimum width of the peaks considered by the method, or by increasing the `exclusion_factor` parameter. This last parameter means: how much of the FWHM of the peak (defined by `wmin`) is removed.

Let's change `wmin` to 12.


```python
new_single_s, covar, covar = single_s.remove_bg(
    toplot = True,
    polyorder = 2,
    peak_pos = [520, 950, 1583, 2710],
    wmin = 12)
```

This is much better, giving us a nice second order fit.

Final tweaks: If we don't want the area around 0 cm$^{-1}$, we can just "add a peak there". Let's change back to linear fit.


```python
new_single_s, coeff, covar = single_s.remove_bg(
    toplot = True,
    polyorder = 1,
    peak_pos = [120, 520, 950, 1583, 2710],
    wmin = 12)
```

The fit mask used is also returned as a variable of the [`singlespec`](https://zrbyte.github.io/ramantools/ramantools.html#ramantools.singlespec) instance, called `single_s` in this example.


```python
new_single_s.mask
```


```python
# Let's plot the mask
new_single_s.ssxr.plot()
pl.plot(new_single_s.ssxr.ramanshift, new_single_s.mask*500, 'r')
```

The returned `xarray` instance has the same attributes as the original, with the "comments" field appended.


```python
new_single_s.ssxr.attrs
```


```python
print(new_single_s.ssxr.attrs['comments'])
```

The ammented "comments" acts as a log of the parameters used for fitting.

Let's plot the result:

#### Background fitting for maps

Let's try the simplest approach. Just calling the `remove_bg()` method of the `ramanmap` instance called `m`. This is the map we loaded at the beginning of the tutorial, here: [Loading a Raman map](#Loading-a-Raman-map).


```python
new_m, coeff, covar = m.remove_bg()
```


```python
print(new_m.mapxr.attrs['comments'])
```


```python
new_m.pixel_x, new_m.pixel_y
```


```python
# selecting a spectrum from the map to plot
new_m.mapxr[:, 10, 10].plot()
```

Looks good. Let's look at some more options.

Using it without options, fits a background to the middle spectrum of the map. This simplest approach, uses the `bgsubtract()` function, with no options, meaning automatic detection of peaks. The method `ramanmap.remove_bg()` accepts the same parameters as the `singlespec.remove_bg()`, but has additional options. For a list of parameters see the documentation of [`bgsubtract()`](https://zrbyte.github.io/ramantools/ramantools.html#ramantools.bgsubtract) and [remove_bg](https://zrbyte.github.io/ramantools/ramantools.html#ramantools.ramanmap.remove_bg).

If the automatic background subtraction doesn't produce the desired outcome, one can tweak the fitting procedure. A good idea would be to take the most "problematic" spectrum from the map and fit a background to that by hand, either using the function `bgsubtract()`, which acts on `numpy` arrays or `singlespec.remove_bg()`, which acts on `xarray` instances.

Let's try this by selecting one of the spectra from the map. We can then tune the fitting parameters to suit our needs. After a satisfactory fit parameter has been found, we can pass that to the `ramanmap.remove_bg()`. Most importantly, we can pass a `fitmask` supplied by all background fitting functions. Let's see an example of this.


```python
# selecting a spectrum
spec = m.mapxr.sel(width = 30, height = 40, method = 'nearest')
```


```python
# let's try some fit parameters with this
y_data_nobg, bg_values, coeff, params_used_at_run, mask, covar = rt.bgsubtract(spec.coords['ramanshift'].data, spec.data, toplot = True, peak_pos = [140, 520, 1000, 1583, 2730], wmin = 15)
```

The above `bgsubtract()` has returned among other parameters, the mask used to fit the data: `mask`. We can pass this to the `fitmask` optional parameter of `ramanmap.remove_bg()`. In this case the peak search will be skipped and a polynomial fit will be applied based on the data covered by the mask. Default polynomial fit is of order 1.


```python
mask
```


```python
new_m, coeff, covar = m.remove_bg(fitmask = mask, toplot = True)
```

We can see that the previously used `mask` is also applied to the map.


```python
m_nobg = m.remove_bg(height = 20, width = 50)
```


```python
m_nobg = m.remove_bg(polyorder = 1, fitmask = mask)
```

`remove_bg` individual mode

So far every spectrum in the map had the *same* background subtracted: the polynomial fitted to one reference spectrum. With `mode = 'individual'`, `remove_bg()` fits a separate polynomial to each pixel of the map. There are two ways to run it:

- **With a `fitmask`**: all pixels share the same masked-in region (the one we got from `bgsubtract()` above), and a single vectorized least-squares solve handles the whole map at once. This is fast.
- **Without a `fitmask`**: `bgsubtract()` runs on every pixel, including its automatic peak search, so each pixel gets its own mask. This is slower, but adapts to spectra whose peaks are at different positions.

Let's start with the fast version, reusing the `mask` from above.


```python
new_m_ind, coeff_ind, covar_ind = m.remove_bg(mode = 'individual', fitmask = mask)
```


```python
print(new_m_ind.mapxr.attrs['comments'])
```

In `'individual'` mode, `coeff` and `covar` are no longer plain `numpy` arrays. They are `xarray` DataArrays indexed by the `height` and `width` coordinates of the map, with an extra `coef` dimension holding the polynomial coefficients in `numpy.polyval` order (highest power first). For the default `polyorder = 1`, `coef = 0` is the slope and `coef = 1` is the offset.


```python
# the fitted background offset for every pixel of the map
coeff_ind.sel(coef = 1).plot(vmin = 880, vmax =910)
pl.title('background offset (counts)')
```

We can also pick a single pixel and compare: the raw spectrum, the `'const'` mode result (`new_m`, where the background of the middle spectrum was subtracted everywhere) and the `'individual'` mode result.


```python
w, h = 30, 40
m.mapxr.sel(width = w, height = h, method = 'nearest').plot(color = 'k', label = 'raw')
new_m.mapxr.sel(width = w, height = h, method = 'nearest').plot(color = 'b', label = "mode = 'const'")
new_m_ind.mapxr.sel(width = w, height = h, method = 'nearest').plot(color = 'r', label = "mode = 'individual'")
pl.legend()
```

Now the slow path: no `fitmask`, so every pixel runs its own peak search. All keyword arguments of `bgsubtract()` can be passed, as before. Note that `toplot = True` is rejected in this mode, since it would create one plot for every pixel.


```python
%%time
new_m_ind2, coeff_ind2, covar_ind2 = m.remove_bg(mode = 'individual', polyorder = 1, wmin = 15)
```


```python
print(new_m_ind2.mapxr.attrs['comments'])
```

In this case the fit mask differs from pixel to pixel, so `mask` is a 3D boolean DataArray over (`height`, `width`, `ramanshift`). Let's look at the mask found for one pixel, on top of its spectrum.


```python
pixel_mask = new_m_ind2.mask.sel(width = w, height = h, method = 'nearest')
new_m_ind2.mapxr.sel(width = w, height = h, method = 'nearest').plot()
pl.plot(pixel_mask.ramanshift, pixel_mask * 500, 'r')
```

## Normalizing spectra

### Single spectra


```python
# Let's remove the background first
new_single_s, _, _ = single_s.remove_bg()
```


```python
# Let's normalize to the G peak
new_single_s_norm = new_single_s.normalize(1580)
```

The normalized `xarray` is now unitless:


```python
new_single_s_norm.ssxr.attrs['long_name'], new_single_s.ssxr.attrs['units']
```


```python
print(new_single_s_norm.ssxr.attrs['comments'])
```


```python
new_single_s_norm.ssxr.plot()
```

`singlespec.normalize()` can take any keyword argument which is used by `peakfit()`. For example, we can plot the fit that is used for normalization:


```python
_ = new_single_s.normalize(1580, toplot = True)
```

### Maps


```python
new_m, _, _ = m.remove_bg()
```


```python
print(new_m.mapxr.attrs['comments'])
```


```python
new_m_norm = new_m.normalize(1580)
```


```python
new_m_norm.history()
```


```python
new_m_norm.plotspec(100, 31, 1580)
```


```python
new_m_norm = new_m.normalize(1580, width = 20, height = 40)
```


```python
new_m_norm.plotspec(20, 40, 1580)
```

Let's try to normalize to each individual spectrum in the map.


```python
new_m_norm = new_m.normalize(1582, mode = 'individual')
```


```python
new_m_norm.history()
```

Now, we can see that the comments are appended, with another note, since we normalized an already normalized map.


```python
new_m_norm.mapxr.sel(ramanshift = 1582, method = 'nearest').plot(vmin = 0.8, vmax = 1.1)
```

## Curve fitting

One can either use the basic functionality of `curvefit` from the `scipy` package, which works on `numpy` arrays. However, `xarray` also has a built in functionality for curve fitting, where the end result is also an xarray object. This is just a wrapper around `scipy.optimize.curvefit`, so usage is similar.

### Let's try the curvefit within `xarray`.


```python
ssfit = single_s.ssxr.curvefit(
    'ramanshift', rt.lorentz,
    p0 = {
        'x0': 1582,
        # 'ampl': 100,
        # 'width': 15,
        # 'offset': 0
    },
    # bounds = {'width': [0, 100]}
)
```


```python
# let's take a section of the map, to make fitting faster.
m_small = m.mapxr.sel(width = slice(30, 40), height = slice(20, 30))
```

Fitting to the map: `m_small`


```python
mfit = m_small.curvefit(
    'ramanshift', rt.lorentz,
    p0 = {
        'x0': 1581,
        'ampl': 100,
        'width': 15,
        'offset': 0
    },
    # bounds = {'width': [0, 100]}
)
```

Both of these fit results: `ssfit` and `mfit` are `xarray` Datasets, with the fit parameters of the fit function and the covariances.

`ramantools` has a built in `peakfit()` function which takes as the argument the `xarray` DataArray of a single spectrum or map. It has several options to tweak the fit, including starting values, bounds and an option to plot the resulting fit.

Let's try this for a single spectrum.

#### `peakfit()` within ramantools

With the default values for the start parameters, the `peakfit` method finds the G peak.
To fit a specific peak, the starting values can be adjusted, or the data can be sliced to the region around the peak in question.

Let's try to fit the 2D peak, by adjusting the starting values of the parameters.


```python
ssfit = rt.peakfit(single_s.ssxr, toplot = True, stval = {'x0': 2700, 'ampl': 100, 'width': 15, 'offset': 900})
```

Perfect. Let's see the values of the fit parameters:


```python
ssfit['curvefit_coefficients']
```


```python
print(ssfit.attrs['comments'])
```

We can use these parameters as the starting value of the next fit. We will fit the 2D peak of the entire map: `m.mapxr`. For this we need to build a dictionary of the Lorentz parameters and pass it to the `peakfit()` method.


```python
# Let's subtract the background first
m_nobg, coeff, covar = m.remove_bg()
```


```python
# Creating the dictionary
p = {'x0': 2724, 'ampl': 313, 'width': 49, 'offset': 0}
# passing the starting values contained in `p` to the `peakfit()` method. 
# Let's also add some boundaries for the parameters
b = {'x0': [2500, 2900], 'ampl': [0, 900], 'width': [20, 100], 'offset': [-10, 50]}
mapfit = rt.peakfit(m_nobg.mapxr, stval = p, bounds = None, toplot = True)
```

Let's look at the result, by plotting the width parameter of the fit Dataset.


```python
mapfit['curvefit_coefficients'].sel(param = 'width').plot(vmin = 40, vmax = 70)
```


```python
print(mapfit.attrs['comments'])
```

By passing a fit result (in this case `mapfit`) to the `peakfit()` function, we can skip the fitting and use it to plot the fit at other values of the `width` and `height` coordinates.


```python
_ = rt.peakfit(m_nobg.mapxr, fitresult = mapfit, toplot = True, width = 32, height = 42)
```


```python
# plot the spectra and fits along height = 42, for widths from 10 to 40, every 5 microns
for w in np.arange(10, 40, 5):
    _ = rt.peakfit(m_nobg.mapxr, fitresult = mapfit, toplot = True, width = w, height = 42)
pl.legend(loc = 'upper right')
```

#### Drawmask
As used in Witec software


```python
# create a boolean mask based on the intensity of the G peak.
gpeak = m.mapxr.sel(ramanshift = slice(1550, 1610))
```

Let's create a mean filter from this.


```python
gmean = gpeak.mean(dim = 'ramanshift')
```


```python
gmean.plot()
```


```python
gmask = gmean > 900
```


```python
gmask
```


```python
gmask.plot()
```


```python
gfilter_map = m_nobg.mapxr.where(gmask)
```


```python
gfilter_map.sel(ramanshift = 1583, method = 'nearest').plot()
```

Let's calculate the area of the flake, by using this mask. We just need to count the number of `True` values in the mask.


```python
# this is the area of a pixel
pixel_area = (m_nobg.size_x / m_nobg.pixel_x) * (m_nobg.size_y / m_nobg.pixel_y)
# to get the flake area we just need to sum the mask. This works because: True + True = 2 :)
flake_area = gmask.sum() * pixel_area
print('flake area is:', flake_area.data, 'um^2')
```


```python
p = {'x0': 2724, 'ampl': 313, 'width': 49, 'offset': 900}
mapfit_gmask = rt.peakfit(gfilter_map, stval = p, toplot = True)
```


```python
twodpeak_width = mapfit_gmask['curvefit_coefficients'].sel(param = 'width')
twodpeak_width.plot(vmin = 45, vmax = 70)
```

Let's calculate the area with wide 2D peaks.


```python
# make another mask, based on the peak width
wide_twod_mask = twodpeak_width > 55
wide_twod = twodpeak_width.where(wide_twod_mask)
```

Let's highlight with red, the areas with a peak width > 55


```python
twodpeak_width.plot(cmap = 'viridis', vmin = 45, vmax = 70)
wide_twod.plot(cmap = 'Reds', alpha = 0.8)
ax = pl.gca()
ax.set_aspect(1)
```


```python
wide_twod_area = wide_twod_mask.sum() * pixel_area
print('the area with a peak width > 55 is:', wide_twod_area.data, 'um^2')
print('This is ', str(100*wide_twod_area.data / flake_area.data)[:4], '% of the total flake area.')
```


```python

```

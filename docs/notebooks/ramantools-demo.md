<!-- Generated from "ramantools demo.ipynb" with: jupyter nbconvert --to markdown --TemplateExporter.exclude_output=True. Edit the notebook, not this file. -->

# This is a quick demo of `ramantools`. Documentation can be found [here](https://zrbyte.github.io/ramantools/).


```python
import ramantools as rt
import matplotlib.pyplot as pl
```

Loading test data from a URL


```python
# For use only in this tutorial.
import urllib

# link to datafile
datalink = 'https://public.ek-cer.hu/~nemes/testdata/FLG36.MK_FLG_ABC_36--Scan%20Piezo--013--Spec.Data%201.txt'
# link to infofile
infolink = 'https://public.ek-cer.hu/~nemes/testdata/FLG36.MK_FLG_ABC_36--Scan%20Piezo--013--Information.txt'

f = urllib.request.urlopen(datalink)
myfile = f.read().decode('latin1').replace('\n', ' ')
with open('data.txt', mode = 'w', encoding = 'latin1') as f:
    f.write(myfile)

f = urllib.request.urlopen(infolink)
myfile = f.read().decode('latin1').replace('\n', ' ')
with open('info.txt', mode = 'w', encoding = 'latin1') as f:
    f.write(myfile)
```


```python
map_path = 'data.txt'
info_path = 'info.txt'
```


```python
# Load the raman map
m = rt.ramanmap(map_path, info_path)
```


```python
# remove the downloaded files
import os
os.remove('data.txt')
os.remove('info.txt')
```


```python
# plot a density plot of the Raman intensity at 2750 cm^-1
m.mapxr.sel(ramanshift = 2750, method = 'nearest').plot()
```


```python
# plot the middle spectrum (using "numpy-like" index based slicing) of the mapxr `xarray` instance
m.mapxr[:, int(m.size_x/2), int(m.size_y/2)].plot()
```

Let's calibrate the Raman shift to the Si peak at 520.5 1/cm.


```python
m = m.calibrate(520.5)
```


```python
# remove background
map_nobg, coeff, covar = m.remove_bg()
```


```python
# plot a spectrum from the `ramanmap` instance, with the background removed,
# at the specified coordinates, using the `sel()` method of `xarray`
map_nobg.mapxr.sel(width = 31, height = 42, method = 'nearest').plot()
```

Let's normalize the spectra to the G peak amplitude.


```python
# normalize each spectrum to the G peak amplitude
map_norm = map_nobg.normalize(1582, mode = 'individual')

# plot one of the spectra
map_norm.mapxr.sel(width = 31, height = 42, method = 'nearest').plot()
```


```python
# fit a Lorenzian to the peak at ~2750
fit = rt.peakfit(map_norm.mapxr, toplot = True, stval = {'x0': 2750, 'ampl': 0.5, 'width': 50, 'offset': 0})
```


```python
# plot the width of the peak on a density plot
fit['curvefit_coefficients'].sel(param = 'width').plot(vmin = 40, vmax = 80)
```

Let's create a boolean mask based on the intensity of the G peak. After which we will use this for more efficient fitting and to plot the flake area and the area of the regions with a wide 2D peak.


```python
gpeak = m.mapxr.sel(ramanshift = slice(1550, 1610))
```


```python
# Taking the mean value of the G peak in the above range (1550 to 1610).
# This is similar to the "sum" filter in the Witec software
gmean = gpeak.mean(dim = 'ramanshift')
```


```python
# Let's plot this
gmean.plot()
```


```python
# Let's now make a boolean mask `gmask` in the areas where the mean of the G peak intenity is larger than 900
gmask = gmean > 900
```


```python
# This dropps all data, where the G peak is not there.
flake_only = map_nobg.mapxr.where(gmask)
```

Let's calculate the area of the flake, by using this mask. We just need to count the number of `True` values in the mask.


```python
# this is the area of a pixel
pixel_area = (map_nobg.size_x / map_nobg.pixel_x) * (map_nobg.size_y / map_nobg.pixel_y)
# to get the flake area we just need to sum the mask. This works because: True + True = 2 :)
flake_area = gmask.sum() * pixel_area
print('flake area is:', flake_area.data, 'um^2')
```


```python
# Let's make another fit. This time fitting only to the flake
p = {'x0': 2724, 'ampl': 313, 'width': 49, 'offset': 0}
mapfit_gmask = rt.peakfit(flake_only, stval = p)
```


```python
# Let's plot the 2D peak width
twodpeak_width = mapfit_gmask['curvefit_coefficients'].sel(param = 'width')
twodpeak_width.plot(vmin = 40, vmax = 65)
```


```python
# make another mask, based on the peak width
wide_twod_mask = twodpeak_width > 55
wide_twod = twodpeak_width.where(wide_twod_mask)
```

Let's highlight with red, the areas with a peak width > 55


```python
twodpeak_width.plot(cmap = 'viridis', vmin = 40, vmax = 65)
wide_twod.plot(cmap = 'Reds', alpha = 1)
ax = pl.gca()
ax.set_aspect(1)
```

Let's now calculate the percentage of the flake that is covered by areas, where the peak is wider than 55 cm$^{-1}$


```python
wide_twod_area = wide_twod_mask.sum() * pixel_area
print('the area with a peak width > 55 is:', wide_twod_area.data, 'um^2')
print('This is ', str(100*wide_twod_area.data / flake_area.data)[:4], '% of the total flake area.')
```

Let's plot a histogram of the 2D peak width values only in the area where it is larger than 55 cm^-1


```python
flake_flatten = wide_twod.stack(pos = ['width', 'height'])
# drop the NaN values
flake_flatten = flake_flatten.dropna(dim = 'pos')
_ = flake_flatten.plot.hist(bins = 60)
pl.xlabel('peak width (cm$^{-1}$)')
```

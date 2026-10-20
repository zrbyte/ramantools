"""netCDF round-trip for :class:`ramanmap` and :class:`singlespec`.

Before 0.6.0 the only way to persist a processed object was
``m.mapxr.to_netcdf(...)``, which keeps the DataArray but silently drops
everything else the container carries: the background fit mask, the
normalisation factor, the raw (unprocessed) array and the Python-side
metadata attributes. This module writes all of that into one netCDF
file (HDF5 under the hood, via the ``h5netcdf`` engine) and rebuilds an
identical container from it, so an analysis can be split across
processes, scripts or agent tool calls without re-running the pipeline.

File layout (one :class:`xarray.Dataset`):

* ``data`` — the processed DataArray (``mapxr`` / ``ssxr``) with all its
  attrs, including ``comments`` and ``processing_steps``.
* ``mask`` — the background fit mask, when present. A 1-D numpy mask is
  stored along ``ramanshift``; a per-pixel 3-D mask (from
  ``remove_bg(mode='individual')`` without a fitmask) keeps its own dims.
* ``normfactor`` — scalar or per-pixel normalisation factor, when present.
* ``raw`` — the untouched array as loaded from disk (``map`` / ``counts``)
  with its own ``raw_ramanshift`` coordinate, unless ``include_raw=False``.
* Dataset attributes ``ramantools_version``, ``ramantools_class``,
  ``ramantools_container`` (JSON of the plain metadata attributes such
  as ``samplename`` / ``laser`` / ``pixel_x``) and a few bookkeeping keys.

Both classes expose this as ``obj.save(path)`` / ``cls.load(path)``; the
package-level :func:`load` dispatches on the stored class name so a
caller that does not know what a file holds can still open it.
"""
from __future__ import annotations

import json  # JSON encode / decode of the container attribute dict
import numpy as np  # type: ignore
import xarray as xr  # type: ignore

from .version import __version__
from ._helpers import _jsonable
from ._log import logger

# netCDF engine. h5netcdf is a hard dependency (setup.py) because the
# scipy fallback writes netCDF3, which cannot hold 64-bit integers,
# booleans or compressed variables.
_ENGINE = 'h5netcdf'

# Plain Python attributes of each container that are not derivable from
# the DataArray and therefore have to be written out explicitly. The
# order is irrelevant; the tuple is what ``_save_container`` iterates.
_COMMON_ATTRS = (
        'filename', 'samplename', 'date', 'time', 'laser', 'itime', 'grating',
        'objname', 'objmagn', 'positioner_x', 'positioner_y',
        'metadata', 'metadata_datafile', 'wipfilename', '_info_loaded',
)
_MAP_ATTRS = _COMMON_ATTRS + ('mapname', 'pixel_x', 'pixel_y', 'size_x', 'size_y')
_SPEC_ATTRS = _COMMON_ATTRS + ('specname', 'positioner_z')

# Compression for the two bulk variables. Level 4 gzip is a good
# trade-off: raw CCD counts (integers stored as float64) shrink several
# fold, and both reading and writing stay fast.
_BULK_ENCODING = {'zlib': True, 'complevel': 4}


def _save_container(obj, da: xr.DataArray, path, *, kind: str, include_raw: bool = True) -> None:
        """Write ``obj`` (whose DataArray is ``da``) to ``path`` as netCDF.

        ``kind`` is ``'ramanmap'`` or ``'singlespec'``; it selects the
        attribute list and tells :func:`_load_container` which class to
        rebuild. ``include_raw=False`` leaves out the raw array, roughly
        halving the file for a map; on load the raw attributes are then
        reconstructed from the processed data.
        """
        # The processed DataArray becomes the ``data`` variable; its own
        # attrs (comments, processing_steps, instrument metadata) stay on
        # the variable, separate from the Dataset-level bookkeeping below.
        ds = da.to_dataset(name='data')
        encoding = {'data': dict(_BULK_ENCODING)}

        # --- optional per-object variables ---------------------------------
        mask = getattr(obj, 'mask', None)
        if mask is not None:
                if isinstance(mask, xr.DataArray):
                        ds['mask'] = mask
                        ds.attrs['ramantools_mask_kind'] = 'dataarray'
                else:
                        arr = np.asarray(mask, dtype=bool)
                        # Normally the mask is one flag per ramanshift sample;
                        # fall back to a private dim if the lengths disagree
                        # (e.g. the user cropped the DataArray after remove_bg)
                        # rather than refusing to save.
                        dim = 'ramanshift' if arr.ndim == 1 and arr.size == ds.sizes.get('ramanshift') else 'mask_point'
                        ds['mask'] = (dim, arr)
                        ds.attrs['ramantools_mask_kind'] = 'ndarray'

        normfactor = getattr(obj, 'normfactor', None)
        if normfactor is not None:
                if isinstance(normfactor, xr.DataArray):
                        ds['normfactor'] = normfactor
                        ds.attrs['ramantools_normfactor_kind'] = 'dataarray'
                else:
                        # 0-d variable; stored as float so numpy scalars and
                        # Python floats come back identically.
                        ds['normfactor'] = xr.DataArray(float(normfactor))
                        ds.attrs['ramantools_normfactor_kind'] = 'scalar'

        # --- raw array -----------------------------------------------------
        if include_raw:
                raw_rs = np.asarray(obj.ramanshift)
                if kind == 'ramanmap':
                        raw = np.asarray(obj.map)
                        ds['raw'] = (('raw_ramanshift', 'raw_row', 'raw_col'), raw)
                else:
                        raw = np.asarray(obj.counts)
                        ds['raw'] = (('raw_ramanshift',), raw)
                ds.coords['raw_ramanshift'] = raw_rs
                encoding['raw'] = dict(_BULK_ENCODING)
        ds.attrs['ramantools_include_raw'] = int(bool(include_raw))

        # --- container attributes -----------------------------------------
        names = _MAP_ATTRS if kind == 'ramanmap' else _SPEC_ATTRS
        container = {name: _jsonable(getattr(obj, name, None)) for name in names}
        ds.attrs['ramantools_container'] = json.dumps(container)
        ds.attrs['ramantools_class'] = kind
        ds.attrs['ramantools_version'] = __version__
        # DataArray ``.name`` is not an attr, so remember it separately.
        ds.attrs['ramantools_dataarray_name'] = str(da.name) if da.name is not None else ''

        logger.debug('save: writing %s to %s (include_raw=%s)', kind, path, include_raw)
        ds.to_netcdf(path, engine=_ENGINE, encoding=encoding)


def _load_container(path, *, expect: str | None = None):
        """Rebuild a container from a file written by :func:`_save_container`.

        ``expect`` optionally pins the class (``'ramanmap'`` /
        ``'singlespec'``); a mismatch raises ``TypeError`` so
        ``ramanmap.load`` cannot silently hand back a spectrum.
        """
        # Deferred imports: map.py / spec.py import this module, so a
        # top-level import here would be circular.
        from .map import ramanmap
        from .spec import singlespec

        # load_dataset (not open_dataset) reads everything into memory and
        # closes the file, so the returned object has no handle on disk.
        ds = xr.load_dataset(path, engine=_ENGINE)
        kind = ds.attrs.get('ramantools_class')
        if kind not in ('ramanmap', 'singlespec'):
                raise ValueError(
                        f"{path!s} was not written by ramantools.save(): "
                        "missing or unknown 'ramantools_class' attribute"
                )
        if expect is not None and kind != expect:
                raise TypeError(f"{path!s} holds a {kind}, not a {expect}; use rt.load() or {kind}.load()")

        cls = ramanmap if kind == 'ramanmap' else singlespec
        # Allocate without running __init__ (which would try to read a
        # Witec text file).
        obj = cls.__new__(cls)
        for name, value in json.loads(ds.attrs['ramantools_container']).items():
                setattr(obj, name, value)

        # Processed DataArray. ``.copy()`` detaches it from the Dataset so
        # later in-place edits (``mapxr[:] -= bg``) do not touch siblings.
        da = ds['data'].copy()
        name = ds.attrs.get('ramantools_dataarray_name', '')
        da.name = name if name else None
        if kind == 'ramanmap':
                obj.mapxr = da
        else:
                obj.ssxr = da

        # Optional variables; absent ones are restored as None, matching a
        # freshly loaded object.
        if 'mask' in ds:
                if ds.attrs.get('ramantools_mask_kind') == 'dataarray':
                        obj.mask = ds['mask'].copy()
                else:
                        obj.mask = ds['mask'].values.astype(bool)
        else:
                obj.mask = None
        if 'normfactor' in ds:
                if ds.attrs.get('ramantools_normfactor_kind') == 'dataarray':
                        obj.normfactor = ds['normfactor'].copy()
                else:
                        obj.normfactor = float(ds['normfactor'].values)
        else:
                obj.normfactor = None

        # Raw array: either the stored copy, or a best-effort reconstruction
        # from the processed data when the file was saved without it.
        if 'raw' in ds:
                raw = ds['raw'].values
                obj.ramanshift = ds['raw_ramanshift'].values
        else:
                logger.debug('load: %s has no raw array; reconstructing from processed data', path)
                obj.ramanshift = da.coords['ramanshift'].values.copy()
                # Undo the height flip _toxarray applies so ``map`` has the
                # same orientation it would have straight after loading.
                raw = np.flip(da.values, axis=1) if kind == 'ramanmap' else da.values.copy()
        if kind == 'ramanmap':
                obj.map = raw
        else:
                obj.counts = raw
        return obj


def load(path):
        """Open a file written by ``ramanmap.save`` / ``singlespec.save``.

        Returns whichever container the file holds. Use this when the
        caller does not know (or care) whether a ``.nc`` file is a map or
        a spectrum; otherwise ``ramanmap.load`` / ``singlespec.load`` give
        a typed result.

        :param path: path of the ``.nc`` file.
        :return: :class:`ramanmap` or :class:`singlespec` instance.

        :Example:

        .. code-block:: python

                import ramantools as rt
                m_nobg, _, _ = m.remove_bg()
                m_nobg.save('flake_nobg.nc')
                # later, in another process:
                m_nobg = rt.load('flake_nobg.nc')
                m_nobg.history()
        """
        return _load_container(path)

"""Shared internal helpers for the ramantools package.

Houses small, reusable primitives extracted during the Phase 1 refactor so
that ``ramanmap`` and ``singlespec`` can share implementation instead of
copy-pasting. Organized in four groups:

* **_midpoint / sentinels**: coordinate-midpoint utility and the
  no-info-file sentinel values.
* **Parameter validation** (``_as_float``, ``_require_mode``): centralize
  the isinstance/value-check idiom that every public method used to
  duplicate.
* **Metadata helpers** (``_apply_default_common_metadata``,
  ``_apply_parsed_header``, ``_set_common_xarray_attrs``): fill the
  sentinel attributes, apply fields parsed from a data-file header, and
  set the ~15 attrs that are identical between map and spec xarray
  objects.
* **Algorithm helpers** (``_crr_xarray``): the rolling-window
  cosmic-ray-removal core shared by ``ramanmap.crr`` and
  ``singlespec.crr``.

None of these are part of the public API — they live under single-underscore
names and are not re-exported from the package ``__init__``.
"""
import numpy as np  # type: ignore
import xarray as xr  # type: ignore


# ---------------------------------------------------------------------------
# Midpoint + sentinels
# ---------------------------------------------------------------------------

def _midpoint(values: np.ndarray) -> float:
        """Return the midpoint of an array of values.

        This function centralizes the midpoint calculation used throughout the
        module to determine the center of width and height coordinates. Having
        a single helper avoids repeated formulae and clarifies intent.
        """
        vmin = np.min(values)
        vmax = np.max(values)
        return (vmax - vmin) / 2 + vmin


# Sentinels used when no info file is available. Strings use "N/A" so printed
# metadata stays self-describing; numeric fields use NaN so downstream math
# that touches them propagates NaN rather than a silent zero.
_NO_INFO_STR = 'N/A'
_NO_INFO_NUM = float('nan')


# ---------------------------------------------------------------------------
# Parameter validation
# ---------------------------------------------------------------------------

def _as_float(name: str, value, *, optional: bool = False):
        """Coerce ``value`` to ``float`` with the same TypeError format the
        original inline validators used.

        Passing ``optional=True`` short-circuits ``None`` to ``None`` so the
        width / height kwargs (which default to ``None``) can flow through
        unchanged. For required parameters, ``None`` falls through to the
        isinstance check and raises ``got <class 'NoneType'>`` — the exact
        message the pre-refactor code produced, so downstream error-matching
        code keeps working.
        """
        # Short-circuit for optional kwargs. Not merging into the isinstance
        # check below because we want the *required* path to produce the
        # legacy "got NoneType" message for BC.
        if optional and value is None:
                return None
        if not isinstance(value, (int, float, np.number)):
                raise TypeError(f"{name} must be a number, got {type(value)}")
        return float(value)


def _require_mode(value: str, allowed) -> None:
        """Raise ``ValueError`` if ``value`` is not in ``allowed``.

        Message format matches the pre-refactor inline checks exactly
        (``"mode must be 'const' or 'individual', got 'xyz'"``) so any
        upstream test that pattern-matches the error text still passes.
        """
        if value not in allowed:
                # "or"-join the quoted allowed values to reproduce the
                # original's wording regardless of how many options there are.
                quoted = " or ".join(f"'{a}'" for a in allowed)
                raise ValueError(f"mode must be {quoted}, got '{value}'")


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------

def _apply_default_common_metadata(obj) -> None:
        """Populate the shared no-info-file sentinel attributes on ``obj``.

        Both ``ramanmap._load_defaults`` and ``singlespec._load_defaults``
        begin by setting the same 11 attributes to the sentinel values; each
        then adds its own class-specific extras (pixel/scan dims for maps,
        positioner_z for spectra). Centralizing the common subset keeps the
        class methods tiny and reduces the odds of the two drifting.
        """
        # No raw metadata text yet — print_metadata then shows just the
        # processing comments, which is what we want.
        obj.metadata = ''
        obj.date = _NO_INFO_STR
        obj.time = _NO_INFO_STR
        obj.samplename = _NO_INFO_STR
        obj.laser = _NO_INFO_NUM
        obj.itime = _NO_INFO_NUM
        obj.grating = _NO_INFO_STR
        obj.objname = _NO_INFO_STR
        obj.objmagn = _NO_INFO_STR
        obj.positioner_x = _NO_INFO_NUM
        obj.positioner_y = _NO_INFO_NUM


def _apply_parsed_header(obj, fields: dict, name_attr: str, copy_keys) -> None:
        """Apply parsed Witec-header fields to ``obj`` as attributes.

        ``fields`` is the dict returned by
        :func:`ramantools._witec._parse_witec_datafile_header`. Only keys
        actually present in the dict are applied, so missing header fields
        leave the existing sentinel values in place.

        * ``name_attr`` receives ``_graphname_to_name(fields['graphname'])``
          when ``'graphname'`` is present. Different classes pick a different
          attribute name (``mapname`` vs ``specname``), hence the kwarg.
        * ``copy_keys`` is the list of keys whose header value is copied
          verbatim (``pixel_x``, ``positioner_x``, etc.). ``'wipfilename'``
          is handled here automatically because it's the only key used by
          both classes and always wants straight copy.
        """
        # Deferred import: _witec imports nothing from this module but we
        # still only need _graphname_to_name on the successful path.
        from ._witec import _graphname_to_name

        if 'graphname' in fields:
                setattr(obj, name_attr, _graphname_to_name(fields['graphname']))
        # wipfilename is shared by both classes, so fold it into copy_keys
        # transparently rather than forcing each caller to list it.
        all_copy_keys = ('wipfilename',) + tuple(copy_keys)
        for key in all_copy_keys:
                if key in fields:
                        setattr(obj, key, fields[key])


def _set_common_xarray_attrs(da: xr.DataArray, src, info_loaded: bool) -> None:
        """Set the attributes that are identical between ``mapxr`` and ``ssxr``.

        Both classes' ``_toxarray`` methods used to carry the same block of
        ~15 attribute assignments. This helper writes them once; each class
        then adds its own extras (map adds ``map width`` / ``map height`` /
        ``width`` / ``height`` coord metadata; spec adds ``sample positioner Z``).

        ``src`` is the host instance (``ramanmap`` or ``singlespec``) — we
        pull its previously-populated Python attributes to assemble the
        xarray ``.attrs`` dict.

        ``info_loaded`` controls whether ``objective magnification`` gets
        the conventional ``'x'`` suffix. When no info file was supplied
        ``objmagn`` may be the ``'N/A'`` sentinel; appending ``'x'`` would
        produce an awkward ``'N/Ax'`` so the suffix is omitted in that path.
        """
        # Name is needed by hvplot to label the colour-axis / y-axis.
        da.name = 'Raman intensity'
        da.attrs['wipfile name'] = src.wipfilename
        da.attrs['units'] = 'au'
        da.attrs['long_name'] = 'Raman intensity'
        da.attrs['sample name'] = src.samplename
        da.attrs['laser excitation'] = str(src.laser) + ' nm'
        da.attrs['time of measurement'] = src.time
        da.attrs['date of measurement'] = src.date
        da.attrs['integration time'] = str(src.itime) + ' s'
        da.attrs['sample positioner X'] = src.positioner_x
        da.attrs['sample positioner Y'] = src.positioner_y
        da.attrs['objective name'] = src.objname
        if info_loaded:
                da.attrs['objective magnification'] = src.objmagn + 'x'
        else:
                da.attrs['objective magnification'] = src.objmagn
        da.attrs['grating'] = src.grating
        # Ramanshift coord metadata is shared; width/height attrs stay
        # class-specific since only maps have those dimensions.
        da.coords['ramanshift'].attrs['units'] = r'1/cm'
        da.coords['ramanshift'].attrs['long_name'] = 'Raman shift'


# ---------------------------------------------------------------------------
# Cosmic-ray removal core
# ---------------------------------------------------------------------------

def _crr_xarray(da: xr.DataArray, cutoff: float, window: int):
        """Rolling-window cosmic-ray removal shared by map and spec.

        Works for both 1-D spectra (``singlespec.ssxr``) and 3-D maps
        (``ramanmap.mapxr``) because xarray rolling/where operations
        broadcast over the non-``ramanshift`` dims transparently.

        Algorithm (same as before the refactor):
          1. Compute a centered rolling mean and standard deviation of
             size ``2*window + 1`` along ``ramanshift``.
          2. Flag any sample whose excursion above the local mean exceeds
             ``cutoff * local_std`` — these are the cosmic-ray spikes.
          3. Replace each flagged sample with its local-mean value.

        ``xr.where`` does not preserve ``.attrs`` by default, so the
        cleaned DataArray inherits ``da.attrs`` here before the caller
        appends its class-specific comment line.

        Returns ``(cleaned_da, n_spikes)`` so callers can drop the spike
        count into their ``comments`` annotation.
        """
        rolling = da.rolling(ramanshift=2 * window + 1, center=True)
        local_mean = rolling.mean()
        local_std = rolling.std()
        # Boolean mask: True at positions that significantly exceed the
        # local mean, i.e. the cosmic-ray spikes we want to replace.
        crrpos = (da - local_mean) > (cutoff * local_std)
        cleaned = xr.where(crrpos, local_mean, da)
        # Restore attrs that xr.where strips, keeping the chain consistent
        # so the caller can append a comment line without a KeyError.
        cleaned.attrs = da.attrs.copy()
        return cleaned, int(crrpos.sum().data)

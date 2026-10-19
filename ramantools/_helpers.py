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


# ---------------------------------------------------------------------------
# Structured processing history
# ---------------------------------------------------------------------------
#
# ``attrs['comments']`` is a free-text log meant for humans. Agents and
# scripts need the same information as data: which step ran, with which
# parameters, in which order. ``attrs['processing_steps']`` carries that
# as a JSON string (a plain string so it survives ``to_netcdf`` — netCDF
# attributes cannot hold nested Python objects). Every pipeline method
# appends one record through ``_record_step`` right next to its
# ``comments`` line; ``_history_steps`` decodes the list back.

import json  # JSON encode / decode of the step records

# Attribute key under which the JSON-encoded step list is stored.
_STEPS_ATTR = 'processing_steps'


def _jsonable(value):
        """Coerce ``value`` into something :func:`json.dumps` accepts.

        Pipeline parameters are mostly floats, ints, strings and bools,
        but a few are numpy scalars / arrays (``calibshift`` comes back
        as a 0-d numpy array, ``peak_pos`` may be a list of numpy
        floats) or callables (the ``func`` passed to ``peakfit``). This
        maps each of those onto a JSON-friendly equivalent instead of
        raising ``TypeError`` deep inside a processing method.
        """
        # numpy scalars / 0-d arrays → Python scalar
        if isinstance(value, np.generic):
                return value.item()
        if isinstance(value, np.ndarray):
                if value.ndim == 0:
                        return value.item()
                # Small arrays (e.g. polynomial coefficients) are worth
                # keeping; large ones (a per-pixel array) are summarised
                # by shape so the attribute stays small.
                if value.size <= 32:
                        return [ _jsonable(v) for v in value.tolist() ]
                return {'shape': list(value.shape), 'dtype': str(value.dtype)}
        if isinstance(value, xr.DataArray):
                # Per-pixel results (individual-mode normfactor etc.):
                # record summary statistics, not the whole array.
                finite = value.data[np.isfinite(value.data)] if np.issubdtype(value.dtype, np.floating) else value.data
                summary = {'shape': list(value.shape), 'dims': list(value.dims)}
                if finite.size:
                        summary.update(mean=float(np.mean(finite)), min=float(np.min(finite)), max=float(np.max(finite)))
                return summary
        if isinstance(value, dict):
                return {str(k): _jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
                return [_jsonable(v) for v in value]
        if callable(value):
                # Shape functions: their name is what a reader wants.
                return getattr(value, '__name__', repr(value))
        if isinstance(value, (str, int, float, bool)) or value is None:
                return value
        # Anything else (Path objects, …) falls back to its string form.
        return str(value)


def _history_steps(da: xr.DataArray) -> list:
        """Decode ``attrs['processing_steps']`` into a list of dicts.

        Returns an empty list when the attribute is absent (objects
        created before 0.6.0, or DataArrays assembled by hand). A corrupt
        attribute is also treated as empty rather than raising, because
        a broken history must never prevent the data from being used.
        """
        raw = da.attrs.get(_STEPS_ATTR, '')
        if not raw:
                return []
        try:
                steps = json.loads(raw)
        except (TypeError, ValueError):
                return []
        return steps if isinstance(steps, list) else []


def _record_step(da: xr.DataArray, step: str, **params) -> None:
        """Append one ``{'step': step, 'params': {...}}`` record to ``da``.

        Mutates ``da.attrs`` in place; callers invoke it on the *new*
        DataArray they are about to hand back, immediately after
        appending the human-readable ``comments`` line, so the two logs
        stay in lockstep. Parameters pass through :func:`_jsonable`.
        """
        steps = _history_steps(da)
        steps.append({'step': step, 'params': _jsonable(params)})
        da.attrs[_STEPS_ATTR] = json.dumps(steps)


def _step_names(da: xr.DataArray) -> list:
        """Just the ordered step names — what ``__repr__`` shows."""
        return [s.get('step', '?') for s in _history_steps(da)]


def _comments_as_steps(da: xr.DataArray) -> list:
        """Fallback for ``history(as_list=True)`` on pre-0.6.0 objects.

        Splits the free-text ``comments`` into one ``{'step': 'comment',
        'text': line}`` record per non-empty line, so callers always get
        a list even when the structured attribute is missing.
        """
        text = da.attrs.get('comments', '')
        return [{'step': 'comment', 'text': line.strip()} for line in text.splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _plot_target(ax, savepath):
        """Resolve the ``ax`` / ``savepath`` plotting kwargs to an Axes.

        Shared by every method that accepts the opt-in ``ax=`` and
        ``savepath=`` keywords added in 0.6.0:

        * ``ax`` given → draw there; the caller owns the figure.
        * ``savepath`` given (no ``ax``) → a fresh figure is created so
          the saved image contains only this plot, and
          :func:`_finish_plot` closes it afterwards. This is the
          headless path agents use.
        * neither → ``plt.gca()``, i.e. the historical behaviour of the
          bare ``plt.plot`` calls (draw on the current axes, let the
          inline backend display it at the end of the cell).

        Returns ``(ax, owns_figure)`` where ``owns_figure`` tells
        :func:`_finish_plot` whether closing is our responsibility.
        """
        import matplotlib.pyplot as plt  # local import keeps module import light
        if ax is not None:
                return ax, False
        if savepath is not None:
                _, ax = plt.subplots()
                return ax, True
        return plt.gca(), False


def _finish_plot(ax, savepath, owns_figure: bool) -> None:
        """Save and, if we created the figure, close it.

        Closing matters in two ways: it frees memory in long batch runs,
        and in Jupyter it prevents the figure from *also* being rendered
        inline after it has been written to disk.
        """
        import matplotlib.pyplot as plt  # local import keeps module import light
        if savepath is None:
                return
        fig = ax.get_figure()
        fig.savefig(savepath, bbox_inches='tight')
        if owns_figure:
                plt.close(fig)


# ---------------------------------------------------------------------------
# Container summary / repr
# ---------------------------------------------------------------------------

def _fmt_num(value, digits: int = 3) -> str:
        """Format a metadata number compactly; sentinels print as ``N/A``."""
        try:
                f = float(value)
        except (TypeError, ValueError):
                return str(value)
        if np.isnan(f):
                return 'N/A'
        # Integers print without a decimal part (``1 s`` not ``1.000 s``).
        if f == int(f) and abs(f) < 1e6:
                return str(int(f))
        return f'{f:.{digits}f}'


def _container_summary(obj, da: xr.DataArray) -> dict:
        """Collect the facts ``__repr__`` and ``summary()`` report, as a dict.

        Shared by :class:`ramanmap` and :class:`singlespec`; the map adds
        its pixel grid / physical size, the spectrum its Z positioner.
        Every value is a plain Python type so the dict can be dumped as
        JSON by the CLI. Only cheap operations are performed here — the
        intensity data itself is never touched, so ``repr(m)`` on a
        100 MB map is instant.
        """
        rs = da.coords['ramanshift'].data
        is_map = 'width' in da.dims
        out = {
                'class': 'ramanmap' if is_map else 'singlespec',
                'name': getattr(obj, 'mapname', None) if is_map else getattr(obj, 'specname', None),
                'sample': getattr(obj, 'samplename', None),
                'source': str(getattr(obj, 'filename', '')),
                'wipfile': getattr(obj, 'wipfilename', ''),
                'date': getattr(obj, 'date', None),
                'time': getattr(obj, 'time', None),
                'n_ramanshift': int(rs.size),
                'ramanshift_min': float(rs.min()) if rs.size else None,
                'ramanshift_max': float(rs.max()) if rs.size else None,
                # Mean sample spacing — spectrometers are not perfectly
                # linear in 1/cm, so a single step value is approximate.
                'ramanshift_step': float(np.mean(np.diff(rs))) if rs.size > 1 else None,
                'laser_nm': _jsonable(getattr(obj, 'laser', None)),
                'integration_time_s': _jsonable(getattr(obj, 'itime', None)),
                'grating': getattr(obj, 'grating', None),
                'objective': getattr(obj, 'objname', None),
                'magnification': getattr(obj, 'objmagn', None),
                'positioner_x': _jsonable(getattr(obj, 'positioner_x', None)),
                'positioner_y': _jsonable(getattr(obj, 'positioner_y', None)),
                'units': da.attrs.get('units'),
                'long_name': da.attrs.get('long_name'),
                'steps': _step_names(da),
                'has_mask': getattr(obj, 'mask', None) is not None,
                'normfactor': _jsonable(getattr(obj, 'normfactor', None)),
        }
        if is_map:
                out['pixel_x'] = int(da.sizes['width'])
                out['pixel_y'] = int(da.sizes['height'])
                out['size_x_um'] = _jsonable(getattr(obj, 'size_x', None))
                out['size_y_um'] = _jsonable(getattr(obj, 'size_y', None))
        else:
                out['positioner_z'] = _jsonable(getattr(obj, 'positioner_z', None))
        return out


def _format_summary(s: dict) -> str:
        """Render a :func:`_container_summary` dict as the multi-line repr.

        One fact per line, aligned labels, no data statistics. The
        ``steps`` line is the processing history at a glance; ``load``
        is always first, so an object that shows only ``load`` is raw.
        """
        lines = [f"<ramantools.{s['class']} {s['name']!r}>"]
        lines.append(f"  sample:      {s['sample']}")
        lines.append(f"  source:      {s['source']}")
        if s['class'] == 'ramanmap':
                lines.append(
                        f"  map:         {s['pixel_x']} x {s['pixel_y']} pixels (width x height), "
                        f"{_fmt_num(s['size_x_um'])} x {_fmt_num(s['size_y_um'])} um"
                )
        else:
                lines.append(
                        f"  position:    X {_fmt_num(s['positioner_x'])}, Y {_fmt_num(s['positioner_y'])}, "
                        f"Z {_fmt_num(s['positioner_z'])} um"
                )
        if s['n_ramanshift']:
                lines.append(
                        f"  ramanshift:  {s['n_ramanshift']} points, "
                        f"{s['ramanshift_min']:.1f} .. {s['ramanshift_max']:.1f} 1/cm "
                        f"(step {s['ramanshift_step']:.2f})" if s['ramanshift_step'] is not None else
                        f"  ramanshift:  {s['n_ramanshift']} points"
                )
        lines.append(
                f"  instrument:  laser {_fmt_num(s['laser_nm'])} nm, integration {_fmt_num(s['integration_time_s'])} s, "
                f"grating {s['grating']}, objective {s['magnification']}x"
        )
        lines.append(f"  values:      {s['long_name']} [{s['units']}]")
        steps = ' -> '.join(s['steps']) if s['steps'] else '(no structured history)'
        lines.append(f"  steps:       {steps}")
        return '\n'.join(lines)

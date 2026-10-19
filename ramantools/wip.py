"""Loader for WITec project files (``.wip`` / ``.wid``) built on the WITec SDK.

The text-export path (:class:`ramantools.ramanmap` / :class:`ramantools.singlespec`
constructed from a tab-delimited table plus an info file) stays exactly as it
was. This module adds a *second* way in: open the project file itself with
`WITecSDK <https://pypi.org/project/WITecSDK/>`_, list what it contains, and
extract individual objects as the familiar ``ramanmap`` / ``singlespec``
instances (or as plain :mod:`xarray` DataArrays for images and masks).

Typical usage::

        import ramantools as rt
        wip = rt.wipfile('FLG42_532.wip')
        wip.list()                      # what's inside
        m = wip.ramanmap(188)           # a map, by object ID (or by caption)
        s = wip.singlespec(200)         # a single spectrum
        mask = wip.mask(254)            # boolean DataArray matching m.mapxr's coords
        m.mapxr.where(mask)             # apply it

Design notes
------------
* ``WITecSDK`` is an *optional* dependency, imported lazily inside
  :class:`wipfile` so that users of the text-export path never need it.
  Install with ``pip install ramantools[wip]`` or ``pip install WITecSDK``.
* Spectral data are decoded straight from the SDK's byte buffer with
  :func:`numpy.frombuffer`. The SDK's own ``TDGraph.Graph`` property goes
  through nested Python tuples, which is slower and loses the dtype.
* The buffer layout is ``[y][x][ramanshift]`` when the graph's
  ``DataFieldInverted`` flag is set (every graph in the files seen so far)
  and ``[x][y][ramanshift]`` otherwise, mirroring the SDK's
  ``reshape_graph``. Either way the result is handed to
  :py:meth:`ramanmap._from_arrays` as ``(ramanshift, pixel_y, pixel_x)``, the
  same order the text exporter writes, so the usual height flip in
  ``_toxarray`` produces the Witec Project orientation. Verified against the
  project's own derived "G-Sum" image (correlation 0.997 in the
  ``[y][x]`` orientation, ~0 in the alternatives).
* Physical sizes and stage positions come from the object's
  ``TDSpaceTransformation`` (pixel count times scale; affine world origin),
  not from regexes on the Information text. The SDK's ``Position`` property
  is broken in SDK 2.0.9 (``Mat3n * Mat3n`` is not defined), so the affine
  transform is evaluated here with numpy.
* Instrument metadata (laser, integration time, grating, date, objective,
  …) comes from the project's embedded "Information" text object, which has
  the same layout as the info file the text-export path parses. It is
  matched to a graph by caption prefix (``"<scan>--Information"``), and for
  objects without a caption prefix (e.g. user-named masks) by the shared
  space-transformation ID. The text is parsed with the same regex table as
  the info file (:func:`ramantools._witec._parse_info_text`) in lenient mode.
"""
from __future__ import annotations

import re  # Caption-prefix matching for Information text lookup
import numpy as np  # type: ignore
import xarray as xr  # type: ignore

from ._witec import _parse_info_text, _graphname_to_name
from ._helpers import _NO_INFO_NUM
from .map import ramanmap
from .spec import singlespec

# SDK ``DataClass.DataType`` code → numpy dtype. Mirrors the SDK's own
# ``_DataTypeToFormat`` struct table (q/l/h/b/I/H/B/?/f/d); WITec files are
# written on x86 Windows, hence explicit little-endian.
_SDK_DTYPE = {
        1: '<i8', 2: '<i4', 3: '<i2', 4: '<i1',
        5: '<u4', 6: '<u2', 7: '<u1', 8: '?',
        9: '<f4', 10: '<f8',
}


def _require_witecsdk():
        """Import and return the SDK's ``Project`` class, or raise a clear error.

        Kept in a function so the import cost (and the hard dependency) is
        only paid when a :class:`wipfile` is actually constructed.
        """
        try:
                from WITecSDK import Project  # type: ignore
        except ImportError as exc:  # pragma: no cover - depends on environment
                raise ImportError(
                        "ramantools.wipfile needs the optional 'WITecSDK' package. "
                        "Install it with: pip install WITecSDK  (or pip install ramantools[wip])"
                ) from exc
        return Project


def _buffer_to_array(dataclass) -> np.ndarray:
        """Decode an SDK ``DataClass`` byte buffer into a flat numpy array."""
        dtype = _SDK_DTYPE.get(dataclass.DataType)
        if dtype is None:
                raise NotImplementedError(f"WITec data type code {dataclass.DataType} is not supported")
        return np.frombuffer(dataclass.Data, dtype=dtype)


def _affine_position(transform, model_point) -> np.ndarray:
        """Evaluate ``Rotation @ Scale @ (p - ModelOrigin) + WorldOrigin``.

        The SDK stores the 3x3 ``Scale`` / ``Rotation`` matrices as flat
        9-tuples; interpreting them as **column-major** (``reshape(3, 3).T``)
        reproduces the stage positions stored for spectra extracted from a
        map to ~1e-9 µm, the row-major reading does not.
        """
        vp = transform.ViewPort3D
        rot = np.array(vp.Rotation, dtype=float).reshape(3, 3).T
        scale = np.array(vp.Scale, dtype=float).reshape(3, 3).T
        p = np.asarray(model_point, dtype=float) - np.array(vp.ModelOrigin, dtype=float)
        return rot @ scale @ p + np.array(vp.WorldOrigin, dtype=float)


class wipfile:
        """
        Container for a WITec project file (``.wip`` or ``.wid``), opened with the WITec SDK.

        The project is parsed lazily by the SDK, so opening even a multi-hundred-MB
        file is instant; data buffers are only decoded when an object is extracted.

        :param path: path to the ``.wip`` / ``.wid`` file.
        :type path: str

        :raises ImportError: if the optional ``WITecSDK`` package is not installed.

        .. note::
                Objects are addressed by their integer **ID** (first column of
                :py:meth:`list`) or by their exact caption string. IDs are stable
                within a file; captions are what Witec Project shows in its tree.

        :Example:

        ::

                import ramantools as rt
                wip = rt.wipfile('FLG42_532.wip')
                wip.list()
                m = wip.ramanmap(188)
                s = wip.singlespec(200)
                mask = wip.mask(254)
        """

        def __init__(self, path: str) -> None:
                Project = _require_witecsdk()
                # Keep the path under the same attribute name the other
                # classes use, so print_metadata-style introspection is uniform.
                self.filename = path
                # The raw SDK project — exposed for anything this wrapper
                # does not cover (bitmaps, transformations, …).
                self.project = Project.OpenFile(path)
                # Build the inventory once; everything else looks objects up here.
                self._entries = self._build_inventory()

        # ------------------------------------------------------------------
        # Inventory
        # ------------------------------------------------------------------

        def _build_inventory(self) -> list:
                """Classify every data object in the project into a plain record.

                Each record is ``{'id', 'kind', 'caption', 'shape', 'obj'}``.
                ``kind`` is one of ``'map'``, ``'spectrum'``, ``'graph'`` (other
                TDGraph types such as line scans / series), ``'mask'``,
                ``'image'``, ``'bitmap'``, ``'text'``, ``'other'``. Hidden and
                intermediate objects are included so IDs match Witec Project.
                """
                from WITecSDK import TDGraph, TDImage, TDBitmap, TDText  # type: ignore
                entries = []
                for obj in self.project.Data:
                        rec = {'id': obj.ID, 'caption': obj.Caption, 'obj': obj, 'shape': ''}
                        if isinstance(obj, TDGraph):
                                # A graph with a single spatial pixel is a spectrum;
                                # a 2-D grid of spectra is a map. Anything else
                                # (line scans, time series, histograms) is 'graph'.
                                if obj.SizeX == 1 and obj.SizeY == 1:
                                        rec['kind'] = 'spectrum'
                                        rec['shape'] = f'({obj.SizeGraph},)'
                                elif obj.DataType == '2DImage':
                                        rec['kind'] = 'map'
                                        rec['shape'] = f'({obj.SizeGraph}, {obj.SizeY}, {obj.SizeX})'
                                else:
                                        rec['kind'] = 'graph'
                                        rec['shape'] = f'({obj.SizeGraph}, {obj.SizeY}, {obj.SizeX})'
                        elif isinstance(obj, TDImage):
                                # The SDK flags boolean "Mask" images via the
                                # DataCategories metadata; everything else is a
                                # numeric image (fit results, band sums, …).
                                rec['kind'] = 'mask' if obj.DataType == 'Mask' else 'image'
                                rec['shape'] = f'({obj.SizeY}, {obj.SizeX})'
                        elif isinstance(obj, TDBitmap):
                                rec['kind'] = 'bitmap'
                        elif isinstance(obj, TDText):
                                rec['kind'] = 'text'
                        else:
                                rec['kind'] = 'other'
                        entries.append(rec)
                return entries

        def _of_kind(self, *kinds) -> list:
                """``(ID, caption)`` pairs of the inventory entries of the given kinds."""
                return [(e['id'], e['caption']) for e in self._entries if e['kind'] in kinds]

        @property
        def maps(self) -> list:
                """``(ID, caption)`` of every 2-D spectral map in the project."""
                return self._of_kind('map')

        @property
        def spectra(self) -> list:
                """``(ID, caption)`` of every single spectrum in the project."""
                return self._of_kind('spectrum')

        @property
        def images(self) -> list:
                """``(ID, caption)`` of every numeric image (fit results, band sums, …)."""
                return self._of_kind('image')

        @property
        def masks(self) -> list:
                """``(ID, caption)`` of every boolean mask image."""
                return self._of_kind('mask')

        @property
        def texts(self) -> list:
                """``(ID, caption)`` of every text object (Information blocks, notes)."""
                return self._of_kind('text')

        def list(self, kinds=None) -> None:
                """Print an inventory of the project: ID, kind, shape and caption.

                :param kinds: optional iterable of kinds to show, e.g.
                        ``('map', 'spectrum')``. Defaults to everything except
                        the SDK-internal ``'other'`` objects (transformations,
                        interpretations, colour profiles).
                """
                if kinds is None:
                        kinds = ('map', 'spectrum', 'graph', 'image', 'mask', 'bitmap', 'text')
                print(f"{'ID':>5}  {'kind':<9} {'shape':<20} caption")
                for e in self._entries:
                        if e['kind'] in kinds:
                                print(f"{e['id']:>5}  {e['kind']:<9} {e['shape']:<20} {e['caption']}")

        # ------------------------------------------------------------------
        # ASCII tree view
        # ------------------------------------------------------------------

        def _tree_groups(self, show_all: bool = False) -> list:
                """Group inventory entries the way Witec Project's tree does.

                Returns a list of ``(group_name, raw_entry_or_None, members)``
                tuples ordered by first appearance (lowest ID). ``group_name``
                is the scan caption prefix (``'FLG_42--Scan Piezo--010'``);
                ``raw_entry`` is the un-derived data object of that scan
                (caption exactly ``'<prefix>--Spec.Data N'``) if there is one;
                ``members`` are all other entries of the group in ID order.

                Membership is decided by caption prefix first; objects whose
                caption carries no scan prefix (user-named masks, images) are
                attached to the scan that shares their space-transformation
                ID. Anything still unassigned lands in a ``'(unlinked)'``
                group. ``show_all=False`` hides the SDK-internal ``'other'``
                objects (transformations, interpretations, colour profiles).
                """
                # Prefix → group; a prefix counts as a scan name only if some
                # object in the project actually uses it as a "--Spec.Data",
                # "--Information" or "--Transformation" stem, or it is a
                # bitmap/video image; otherwise the caption is a free name.
                known_prefixes = set()
                for e in self._entries:
                        cap = e['caption']
                        for token in ('--Spec.Data', '--Information', '--Transformation'):
                                if token in cap:
                                        known_prefixes.add(cap.split(token)[0].strip())
                        if e['kind'] == 'bitmap':
                                known_prefixes.add(cap.strip())

                # Map each space-transformation ID to the scan prefix of a
                # graph that uses it, so prefix-less objects can be attached.
                st_to_prefix = {}
                for e in self._entries:
                        if e['kind'] in ('map', 'spectrum', 'graph'):
                                st_id = self._space_transform_id(e['obj'])
                                prefix = self._caption_prefix(e['caption'])
                                if st_id and prefix in known_prefixes:
                                        st_to_prefix.setdefault(st_id, prefix)

                groups: dict = {}
                order: list = []
                for e in self._entries:
                        if e['kind'] == 'other' and not show_all:
                                continue
                        cap = e['caption']
                        prefix = None
                        # Longest known prefix that the caption starts with wins
                        # (handles names like 'FLG_42--Scan Piezo--010--Spec.Data 1').
                        candidates = [k for k in known_prefixes if cap == k or cap.startswith(k + '--') or cap.startswith(k + ' ')]
                        if candidates:
                                prefix = max(candidates, key=len)
                        else:
                                st_id = self._space_transform_id(e['obj'])
                                prefix = st_to_prefix.get(st_id, '(unlinked)')
                        if prefix not in groups:
                                groups[prefix] = {'raw': None, 'members': []}
                                order.append(prefix)
                        # The raw data object is captioned '<prefix>--Spec.Data N',
                        # optionally with a bare suffix such as '-anal0' but
                        # without the ' (...)' / ': ...' decorations Witec adds
                        # to derived objects.
                        if e['kind'] in ('map', 'spectrum', 'graph') and re.fullmatch(re.escape(prefix) + r'--Spec\.Data\s*\d+[^ (:]*', cap):
                                if groups[prefix]['raw'] is None:
                                        groups[prefix]['raw'] = e
                                        continue
                        groups[prefix]['members'].append(e)
                # '(unlinked)' goes last regardless of ID.
                if '(unlinked)' in order:
                        order.remove('(unlinked)'); order.append('(unlinked)')
                return [(name, groups[name]['raw'], groups[name]['members']) for name in order]

        def _tree_lines(self, show_all: bool = False) -> list:
                """Build the ASCII tree as a list of lines (see :py:meth:`tree`)."""
                groups = self._tree_groups(show_all=show_all)
                lines = [f"{self.filename}  (WITec project v{self.project.Version}, {len(self.project.Data)} objects)"]

                def kind_of(e) -> str:
                        # SDK-internal objects show their class name instead of a kind.
                        return type(e['obj']).__name__ if e['kind'] == 'other' else e['kind']

                # Pad the kind column to the longest label shown so IDs line up.
                shown = [raw for _, raw, _ in groups if raw is not None] + [m for _, _, ms in groups for m in ms]
                kind_width = max((len(kind_of(e)) for e in shown), default=4) + 2

                def label(e, strip: str = '') -> str:
                        """One tree row: ``[kind]  ID  caption  shape``."""
                        cap = e['caption']
                        if strip and cap.startswith(strip):
                                # Show only what the caption adds beyond its parent.
                                cap = cap[len(strip):].lstrip('-').strip() or cap
                        shape = f"  {e['shape']}" if e['shape'] else ''
                        return f"{('[' + kind_of(e) + ']').ljust(kind_width)} {e['id']:>4}  {cap}{shape}"

                for gi, (name, raw, members) in enumerate(groups):
                        last_group = gi == len(groups) - 1
                        lines.append(f"{'└── ' if last_group else '├── '}{name}")
                        gpad = '    ' if last_group else '│   '
                        # Children of the group: the raw object (if any) comes
                        # first and carries the derived objects beneath it.
                        if raw is not None:
                                raw_cap = raw['caption']
                                derived = [m for m in members if m['caption'].startswith(raw_cap) or m['kind'] in ('mask', 'image')]
                                plain = [m for m in members if m not in derived]
                                children = [('raw', raw)] + [('plain', m) for m in plain]
                        else:
                                derived, plain = [], members
                                children = [('plain', m) for m in plain]
                        for ci, (role, e) in enumerate(children):
                                last_child = ci == len(children) - 1
                                lines.append(f"{gpad}{'└── ' if last_child else '├── '}{label(e, strip=name)}")
                                if role == 'raw' and derived:
                                        cpad = gpad + ('    ' if last_child else '│   ')
                                        for di, d in enumerate(derived):
                                                last_d = di == len(derived) - 1
                                                lines.append(f"{cpad}{'└── ' if last_d else '├── '}{label(d, strip=raw['caption'])}")
                return lines

        def tree(self, show_all: bool = False) -> None:
                """Print the project as an ASCII tree, grouped per scan like Witec Project's own tree.

                Each top-level node is a scan (or video image); beneath it come
                the raw spectral data object, with everything derived from it
                (extracted spectra, background-subtracted copies, fit results,
                band-sum images, masks) nested one level deeper, followed by
                the scan's Information text. Objects that Witec Project
                does not display (coordinate transformations, interpretations,
                colour profiles) are hidden unless ``show_all=True``.

                Each row reads ``[kind]  ID  caption  shape``; the ID is what
                :py:meth:`ramanmap`, :py:meth:`singlespec`, :py:meth:`mask`
                and :py:meth:`image` take. Captions are shortened to what
                they add beyond their parent node.

                :param show_all: also list the SDK-internal objects, with
                        their class name in place of the kind.
                """
                print('\n'.join(self._tree_lines(show_all=show_all)))

        def __repr__(self) -> str:
                return (
                        f"wipfile('{self.filename}': {len(self.maps)} maps, "
                        f"{len(self.spectra)} spectra, {len(self.images)} images, "
                        f"{len(self.masks)} masks)"
                )

        # ------------------------------------------------------------------
        # Lookup helpers
        # ------------------------------------------------------------------

        def _entry(self, key) -> dict:
                """Resolve an ID (int) or exact caption (str) to an inventory record."""
                if isinstance(key, (int, np.integer)):
                        for e in self._entries:
                                if e['id'] == int(key):
                                        return e
                        raise KeyError(f"no object with ID {key} in {self.filename}")
                if isinstance(key, str):
                        matches = [e for e in self._entries if e['caption'] == key]
                        if not matches:
                                raise KeyError(f"no object with caption {key!r} in {self.filename}")
                        if len(matches) > 1:
                                ids = [e['id'] for e in matches]
                                raise KeyError(f"caption {key!r} is ambiguous (IDs {ids}); use the ID instead")
                        return matches[0]
                raise TypeError("key must be an integer object ID or a caption string")

        def _entry_of_kind(self, key, *kinds) -> dict:
                """Like :py:meth:`_entry` but also checks the object's kind."""
                e = self._entry(key)
                if e['kind'] not in kinds:
                        raise TypeError(
                                f"object {e['id']} ({e['caption']!r}) is a {e['kind']}, "
                                f"expected one of {kinds}"
                        )
                return e

        @staticmethod
        def _caption_prefix(caption: str) -> str:
                """``'FLG_42--Scan Piezo--010--Spec.Data 1 (Sub min)'`` → ``'FLG_42--Scan Piezo--010'``.

                Witec names every object derived from a scan with the scan's
                name followed by ``--Spec.Data N``; the scan's Information
                text is captioned ``'<scan>--Information'``.
                """
                return re.split(r'--Spec\.Data', caption)[0].strip()

        @staticmethod
        def _space_transform_id(obj) -> int:
                """The space-transformation ID an object is positioned with (0 if none)."""
                if hasattr(obj, '_TDGraph'):
                        return obj._TDGraph.SpaceTransformationID
                if hasattr(obj, '_TDImage'):
                        return obj._TDImage.PositionTransformationID
                return 0

        def _info_entry(self, obj):
                """Find the Information text object belonging to ``obj``, or ``None``.

                Strategy: (1) caption prefix → ``'<prefix>--Information'``;
                (2) otherwise borrow the prefix of a map / spectrum that shares
                the same space-transformation ID (covers user-named masks and
                fit-result images, which keep the scan's transformation).
                """
                texts = {e['caption']: e for e in self._entries if e['kind'] == 'text'}
                prefix = self._caption_prefix(obj.Caption)
                if prefix + '--Information' in texts:
                        return texts[prefix + '--Information']
                st_id = self._space_transform_id(obj)
                if st_id:
                        for e in self._entries:
                                if e['kind'] in ('map', 'spectrum', 'graph') and self._space_transform_id(e['obj']) == st_id:
                                        prefix = self._caption_prefix(e['caption'])
                                        if prefix + '--Information' in texts:
                                                return texts[prefix + '--Information']
                return None

        def info(self, key) -> str | None:
                """Return the Witec "Information" text for an object (map, spectrum, image, mask).

                This is the same text the Witec software exports as the info
                file next to a table export. Returns ``None`` if the project
                holds no Information block for that object.

                :param key: object ID or caption.
                """
                e = self._info_entry(self._entry(key)['obj'])
                return None if e is None else e['obj'].PlainText

        # ------------------------------------------------------------------
        # Geometry / metadata assembly
        # ------------------------------------------------------------------

        def _geometry(self, obj) -> dict:
                """Pixel counts, physical size (µm) and stage position of an object.

                Returns ``pixel_x``, ``pixel_y``, ``size_x``, ``size_y`` and
                ``positioner_x/y/z``. Sizes are ``pixels × |scale|`` from the
                space transformation. The positioner values are the stage
                coordinates (µm) of the transformation's model origin, i.e.
                the scan centre for maps and the measurement point for single
                spectra. Falls back to pixel units / NaN when the object has
                no space transformation.
                """
                nx, ny = obj.SizeX, obj.SizeY
                geo = {'pixel_x': nx, 'pixel_y': ny}
                st_id = self._space_transform_id(obj)
                if st_id:
                        transform = self.project.GetDataClassByID(st_id)
                        scale = np.array(transform.ViewPort3D.Scale, dtype=float).reshape(3, 3).T
                        geo['size_x'] = float(nx * abs(scale[0, 0]))
                        geo['size_y'] = float(ny * abs(scale[1, 1]))
                        pos = _affine_position(transform, transform.ViewPort3D.ModelOrigin)
                        geo['positioner_x'] = float(pos[0])
                        geo['positioner_y'] = float(pos[1])
                        geo['positioner_z'] = float(pos[2])
                else:
                        geo['size_x'] = float(nx)
                        geo['size_y'] = float(ny)
                        geo['positioner_x'] = _NO_INFO_NUM
                        geo['positioner_y'] = _NO_INFO_NUM
                        geo['positioner_z'] = _NO_INFO_NUM
                return geo

        def _fields_for(self, obj, *, is_map: bool):
                """Assemble ``(fields, metadata_text)`` for ``_from_arrays``.

                Instrument metadata comes from the Information text (lenient
                regex parse, missing keys keep sentinels); geometry from the
                SDK overrides whatever the text said so derived objects
                (crops, extracted spectra) get *their own* dimensions.
                """
                info = self._info_entry(obj)
                if info is None:
                        text = ''
                        fields = {}
                else:
                        text = info['obj'].PlainText
                        fields = _parse_info_text(text, is_map=is_map, strict=False)
                # Name follows the text-export convention: caption minus the
                # trailing "--Spec.Data N" (derived objects keep their suffix).
                fields['name'] = _graphname_to_name(obj.Caption)
                geo = self._geometry(obj)
                if not is_map:
                        # singlespec has no pixel / size attributes.
                        for k in ('pixel_x', 'pixel_y', 'size_x', 'size_y'):
                                geo.pop(k)
                else:
                        geo.pop('positioner_z')
                fields.update(geo)
                return fields, text, info is not None

        @staticmethod
        def _graph_arrays(obj):
                """``(ramanshift, data)`` for a TDGraph, ``data`` shaped ``(N, SizeY, SizeX)``.

                Casts to float64 so the result behaves exactly like a map
                loaded with :func:`numpy.loadtxt` from a text export (raw
                WITec maps are uint16, which would silently truncate in
                background subtraction).
                """
                n, nx, ny = obj.SizeGraph, obj.SizeX, obj.SizeY
                flat = _buffer_to_array(obj._TDGraph.GraphData)
                if flat.size != n * nx * ny:
                        raise ValueError(
                                f"object {obj.ID}: buffer holds {flat.size} values, "
                                f"expected {n}*{nx}*{ny}"
                        )
                if obj._TDGraph.DataFieldInverted:
                        # [y][x][k]  →  (k, y, x)
                        data = flat.reshape(ny, nx, n).transpose(2, 0, 1)
                else:
                        # [x][y][k]  →  (k, y, x)
                        data = flat.reshape(nx, ny, n).transpose(2, 1, 0)
                _, xaxis = obj.GraphXAxis
                ramanshift = np.asarray(xaxis, dtype=float)
                return ramanshift, np.ascontiguousarray(data, dtype=float)

        def _xy_coords(self, geo: dict):
                """Width / height coordinate vectors with the same convention as ``ramanmap._toxarray``."""
                width = np.linspace(0, geo['size_x'], num=geo['pixel_x'])
                height = np.linspace(0, geo['size_y'], num=geo['pixel_y'])
                return width, height

        # ------------------------------------------------------------------
        # Extraction
        # ------------------------------------------------------------------

        def ramanmap(self, key) -> ramanmap:
                """Extract a 2-D spectral map as a :class:`ramantools.ramanmap` instance.

                The returned object is indistinguishable from one loaded from
                a text export plus info file: same attributes, same
                ``mapxr`` layout, coordinates and attrs, so the whole
                processing pipeline (``remove_bg`` → ``calibrate`` →
                ``normalize`` → ``crr``) applies unchanged.

                :param key: object ID (int) or exact caption (str) of a map.
                :raises TypeError: if the object is not a 2-D map.
                """
                e = self._entry_of_kind(key, 'map')
                obj = e['obj']
                ramanshift, data = self._graph_arrays(obj)
                fields, text, has_info = self._fields_for(obj, is_map=True)
                m = ramanmap._from_arrays(
                        ramanshift, data,
                        filename=self.filename, fields=fields,
                        metadata=text, info_loaded=has_info,
                )
                # Record provenance in the processing history.
                m.mapxr.attrs['comments'] += f'source: WITec project object ID {obj.ID}: "{obj.Caption}" \n'
                return m

        def singlespec(self, key) -> singlespec:
                """Extract a single spectrum as a :class:`ramantools.singlespec` instance.

                Spectra extracted inside Witec Project from a map (captions
                like ``"... (31/33)"``) carry no Information text of their
                own; the parent scan's text is used for the instrument
                metadata, while the stage position comes from the spectrum's
                own space transformation.

                :param key: object ID (int) or exact caption (str) of a spectrum.
                :raises TypeError: if the object is not a single spectrum.
                """
                e = self._entry_of_kind(key, 'spectrum')
                obj = e['obj']
                ramanshift, data = self._graph_arrays(obj)
                fields, text, has_info = self._fields_for(obj, is_map=False)
                s = singlespec._from_arrays(
                        ramanshift, data[:, 0, 0],
                        filename=self.filename, fields=fields,
                        metadata=text, info_loaded=has_info,
                )
                s.ssxr.attrs['comments'] += f'source: WITec project object ID {obj.ID}: "{obj.Caption}" \n'
                return s

        def _image_dataarray(self, obj, dtype=None) -> xr.DataArray:
                """Shared body of :py:meth:`image` / :py:meth:`mask`.

                The buffer is row-major ``[y][x]`` (as the SDK's
                ``reshape_image`` assumes); the height axis is flipped exactly
                like ``ramanmap._toxarray`` does, so the result overlays the
                map extracted from the same scan.
                """
                flat = _buffer_to_array(obj._TDImage.ImageData)
                if flat.size != obj.SizeX * obj.SizeY:
                        raise ValueError(
                                f"object {obj.ID}: buffer holds {flat.size} values, "
                                f"expected {obj.SizeX}*{obj.SizeY}"
                        )
                img = flat.reshape(obj.SizeY, obj.SizeX)
                if dtype is not None:
                        img = img.astype(dtype)
                geo = self._geometry(obj)
                width, height = self._xy_coords(geo)
                da = xr.DataArray(
                        np.flip(img, axis=0),
                        dims=['height', 'width'],
                        coords={'width': width, 'height': height},
                        name=obj.Caption,
                )
                da.attrs['long_name'] = obj.Caption
                da.attrs['wipfile name'] = self.filename
                da.attrs['source'] = f'WITec project object ID {obj.ID}'
                da.attrs['map width'] = str(geo['size_x']) + ' um'
                da.attrs['map height'] = str(geo['size_y']) + ' um'
                da.coords['width'].attrs['units'] = 'um'  # type: ignore
                da.coords['width'].attrs['long_name'] = 'width'
                da.coords['height'].attrs['units'] = 'um'  # type: ignore
                da.coords['height'].attrs['long_name'] = 'height'
                return da

        def image(self, key) -> xr.DataArray:
                """Extract a numeric image (fit result, band sum, …) as a 2-D :class:`xarray.DataArray`.

                Dims are ``('height', 'width')`` with the same coordinates a
                :py:meth:`ramanmap` of the same scan gets, so the image can be
                plotted or combined with the map directly. ``attrs['units']``
                carries the Witec unit string (e.g. ``'rel. 1/cm'``).

                :param key: object ID (int) or exact caption (str) of an image
                        (a mask is accepted too and returned as bool).
                """
                e = self._entry_of_kind(key, 'image', 'mask')
                obj = e['obj']
                da = self._image_dataarray(obj, dtype=bool if e['kind'] == 'mask' else float)
                da.attrs['units'] = obj.ImageZUnit
                return da

        def mask(self, key) -> xr.DataArray:
                """Extract a Witec mask as a boolean 2-D :class:`xarray.DataArray`.

                ``True`` marks pixels *inside* the mask, as drawn in Witec
                Project. Coordinates match :py:meth:`ramanmap` of the scan
                the mask belongs to, so ``m.mapxr.where(wip.mask(id))``
                blanks everything outside it. Any numeric image is accepted
                as well and thresholded at ``!= 0``.

                :param key: object ID (int) or exact caption (str).
                """
                e = self._entry_of_kind(key, 'mask', 'image')
                da = self._image_dataarray(e['obj'])
                da = da != 0
                da.name = e['caption']
                da.attrs = {
                        'long_name': e['caption'],
                        'units': 'bool',
                        'wipfile name': self.filename,
                        'source': f"WITec project object ID {e['id']}",
                }
                return da

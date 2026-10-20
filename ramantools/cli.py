"""Command-line interface: ``ramantools <command> ...`` or ``python -m ramantools``.

Covers the chores that an agent or a shell script does before and after
the Python-side analysis, without writing Python:

* ``inspect``  — what is in a file. For a WITec project this is the
  object tree (or the JSON inventory); for a text export or a saved
  ``.nc`` file it is the container summary.
* ``info``     — the Witec "Information" text of a project object.
* ``export``   — extract a map or spectrum from a project (or load a
  text export) and write it to a ``.nc`` file that :func:`ramantools.load`
  reads back, so later steps never touch the 260 MB project again.
* ``history``  — processing history of a saved ``.nc`` file.
* ``plot``     — render a map (image at one Raman shift + one spectrum)
  or a spectrum to an image file, for headless inspection.

Every command accepts ``--json`` where structured output makes sense,
and :func:`main` returns the exit status instead of calling ``sys.exit``
so the CLI can be exercised in-process by the test suite.

Imports of the heavy modules happen inside the command functions, so
``ramantools --help`` stays fast and the CLI module can be imported
without pulling in matplotlib or the WITec SDK.
"""
from __future__ import annotations

import argparse  # command-line parsing
import json  # --json output
import os  # file-extension dispatch
import sys  # stderr for error messages


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _kind_of_path(path: str) -> str:
        """Classify an input path by extension: ``'wip'``, ``'nc'`` or ``'txt'``."""
        ext = os.path.splitext(path)[1].lower()
        if ext in ('.wip', '.wid'):
                return 'wip'
        if ext in ('.nc', '.h5', '.hdf5', '.netcdf'):
                return 'nc'
        return 'txt'


def _parse_key(value: str):
        """``--id`` accepts an integer object ID or an exact caption string."""
        try:
                return int(value)
        except ValueError:
                return value


def _load_any(path: str, info: str | None, key, as_map: bool | None):
        """Open ``path`` as a ramanmap or singlespec, whatever its format.

        * ``.wip`` → needs ``key`` (``--id``); the object's kind decides
          the class unless ``as_map`` forces it.
        * ``.nc`` → :func:`ramantools.load`.
        * text export → ``as_map`` picks the class; when ``None`` the
          file is treated as a map if it has more than two columns.
        """
        import ramantools as rt
        kind = _kind_of_path(path)
        if kind == 'wip':
                if key is None:
                        raise SystemExit("a WITec project needs --id <object ID or caption>")
                wip = rt.wipfile(path)
                entry = wip._entry(key)
                if entry['kind'] == 'map' or (as_map and entry['kind'] == 'graph'):
                        return wip.ramanmap(key)
                if entry['kind'] == 'spectrum':
                        return wip.singlespec(key)
                raise SystemExit(f"object {entry['id']} is a {entry['kind']}; only maps and spectra can be loaded")
        if kind == 'nc':
                return rt.load(path)
        # Text export: peek at the first data row to tell map from spectrum.
        if as_map is None:
                with open(path, 'r', encoding='latin1') as fh:
                        for line in fh:
                                parts = line.strip().split('\t')
                                if len(parts) >= 2 and parts[0].replace('.', '', 1).replace('-', '', 1).replace('E', '', 1).replace('+', '', 1).isdigit():
                                        as_map = len(parts) > 2
                                        break
                if as_map is None:
                        as_map = False
        return rt.ramanmap(path, info) if as_map else rt.singlespec(path, info)


def _emit(obj, as_json: bool) -> None:
        """Print ``obj`` as JSON or as plain text (its repr)."""
        if as_json:
                print(json.dumps(obj, indent=2, default=str))
        else:
                print(obj)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_inspect(args) -> int:
        """``inspect FILE``: tree / inventory of a project, or a container summary.

        A project without ``--id`` is listed as a whole; with ``--id`` the
        selected object is loaded and summarised exactly like a text
        export or a ``.nc`` file.
        """
        import ramantools as rt
        kind = _kind_of_path(args.path)
        if kind == 'wip' and not args.id:
                wip = rt.wipfile(args.path)
                if args.json:
                        _emit(wip.list(as_list=True) if not args.all else wip.entries, True)
                else:
                        print(repr(wip))
                        print(wip.tree(show_all=args.all, as_string=True))
                return 0
        obj = _load_any(args.path, args.info, _parse_key(args.id) if args.id else None, args.map if args.map else None)
        if args.json:
                _emit(obj.summary(), True)
        else:
                print(repr(obj))
        return 0


def cmd_info(args) -> int:
        """``info FILE --id ID``: the Witec Information text of a project object."""
        import ramantools as rt
        if _kind_of_path(args.path) != 'wip':
                obj = _load_any(args.path, args.info, None, args.map if args.map else None)
                print(obj.metadata if obj.metadata else obj.metadata_datafile)
                return 0
        wip = rt.wipfile(args.path)
        text = wip.info(_parse_key(args.id))
        if text is None:
                print(f"no Information text for object {args.id}", file=sys.stderr)
                return 1
        print(text)
        return 0


def cmd_export(args) -> int:
        """``export FILE --out OUT.nc [--id ID] [--info INFO]``: save as netCDF."""
        obj = _load_any(args.path, args.info, _parse_key(args.id) if args.id else None, args.map if args.map else None)
        obj.save(args.out, include_raw=not args.no_raw)
        if args.json:
                _emit({'written': args.out, **obj.summary()}, True)
        else:
                print(f"wrote {args.out}")
                print(repr(obj))
        return 0


def cmd_history(args) -> int:
        """``history FILE.nc``: processing steps of a saved container."""
        obj = _load_any(args.path, args.info, _parse_key(args.id) if args.id else None, args.map if args.map else None)
        if args.json:
                _emit(obj.history(as_list=True), True)
        else:
                obj.history()
        return 0


def cmd_plot(args) -> int:
        """``plot FILE --out IMG``: render a map or spectrum to an image file."""
        import matplotlib
        # Force a non-interactive backend before pyplot is imported anywhere;
        # the CLI never shows a window.
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import ramantools as rt
        obj = _load_any(args.path, args.info, _parse_key(args.id) if args.id else None, args.map if args.map else None)
        if isinstance(obj, rt.ramanmap):
                # Defaults: centre pixel and the strongest Raman shift of the
                # mean spectrum, so a bare ``plot`` shows something sensible.
                width = args.width if args.width is not None else float(obj.size_x) / 2
                height = args.height if args.height is not None else float(obj.size_y) / 2
                if args.shift is not None:
                        shift = args.shift
                else:
                        mean_spec = obj.mapxr.mean(dim=('width', 'height'))
                        shift = float(mean_spec.ramanshift[int(mean_spec.argmax(dim='ramanshift'))])
                obj.plotspec(width, height, shift, savepath=args.out)
        else:
                fig, ax = plt.subplots(figsize=(7, 4))
                obj.ssxr.plot(ax=ax)
                ax.set_title(obj.specname, fontsize=10)
                fig.tight_layout()
                fig.savefig(args.out, bbox_inches='tight')
                plt.close(fig)
        print(f"wrote {args.out}")
        return 0


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
        """Assemble the argparse tree. Kept separate so docs / tests can introspect it."""
        parser = argparse.ArgumentParser(
                prog='ramantools',
                description='Inspect, export and plot Witec Raman data from the command line.',
        )
        from .version import __version__
        parser.add_argument('--version', action='version', version=f'ramantools {__version__}')
        sub = parser.add_subparsers(dest='command', required=True)

        def add_input_args(p, with_id: bool = True):
                """Arguments shared by every command that opens a data file."""
                p.add_argument('path', help='.wip/.wid project, Witec text export, or .nc file written by save()')
                if with_id:
                        p.add_argument('--id', help='object ID or exact caption inside a .wip project')
                p.add_argument('--info', help='info (metadata) text file accompanying a text export')
                p.add_argument('--map', action='store_true', help='treat a text export as a map (default: guess from the columns)')
                p.add_argument('--json', action='store_true', help='machine-readable JSON output')

        p = sub.add_parser('inspect', help='tree / inventory of a project (or, with --id, one object), or summary of a map / spectrum')
        add_input_args(p)
        p.add_argument('--all', action='store_true', help='include SDK-internal objects of a project')
        p.set_defaults(func=cmd_inspect)

        p = sub.add_parser('info', help='Witec Information text of an object')
        add_input_args(p)
        p.set_defaults(func=cmd_info)

        p = sub.add_parser('export', help='write a map / spectrum to a .nc file readable by rt.load()')
        add_input_args(p)
        p.add_argument('--out', required=True, help='output .nc path')
        p.add_argument('--no-raw', action='store_true', help='omit the raw (unprocessed) array')
        p.set_defaults(func=cmd_export)

        p = sub.add_parser('history', help='processing history of a map / spectrum')
        add_input_args(p)
        p.set_defaults(func=cmd_history)

        p = sub.add_parser('plot', help='render a map (image + one spectrum) or a spectrum to an image file')
        add_input_args(p)
        p.add_argument('--out', required=True, help='output image path (.png, .pdf, .svg …)')
        p.add_argument('--shift', type=float, help='Raman shift (1/cm) of the map image; default: strongest band')
        p.add_argument('--width', type=float, help='width coordinate (um) of the spectrum to show; default: centre')
        p.add_argument('--height', type=float, help='height coordinate (um) of the spectrum to show; default: centre')
        p.set_defaults(func=cmd_plot)
        return parser


def main(argv=None) -> int:
        """Run the CLI; returns the exit status (0 on success)."""
        parser = build_parser()
        args = parser.parse_args(argv)
        try:
                return args.func(args)
        except SystemExit as exc:
                # ``raise SystemExit("message")`` inside a command: print and
                # map to status 2 (usage error) instead of propagating.
                if isinstance(exc.code, str):
                        print(f"ramantools: {exc.code}", file=sys.stderr)
                        return 2
                raise
        except (KeyError, TypeError, ValueError, IndexError, FileNotFoundError) as exc:
                # Library errors carry a readable message already. IndexError
                # is what the info-file parser raises when a required field
                # is missing (e.g. a spectrum info file passed for a map).
                print(f"ramantools: {type(exc).__name__}: {exc}", file=sys.stderr)
                return 1


if __name__ == '__main__':  # pragma: no cover - ``python -m ramantools`` uses __main__.py
        sys.exit(main())

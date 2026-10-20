"""Generate ``testing/notebook_output_checks.ipynb``.

The notebook exercises every code path of ramantools whose behaviour is
*visible* in a Jupyter cell rather than in a return value: the
``__repr__`` echo, methods that print and return ``None`` (``history``,
``wipfile.list`` …), the three plotting routes (``toplot=True``,
``savepath=``, ``ax=``) and the ``_say`` print-or-log switch. Each code
cell carries an ``expect`` dict in its metadata describing exactly what
the cell is allowed to emit (how many images, whether an
``execute_result`` echo, stdout / stderr). ``test_notebook_outputs.py``
executes the notebook with nbclient and checks every cell against that
contract, so a regression such as a figure being displayed twice, a
printing method starting to return a value, or a status line showing up
both on stdout and through logging is caught mechanically.

The notebook is generated rather than hand-edited so the expectations
stay next to the code they describe and the JSON stays diff-friendly
(no outputs are stored). Regenerate after editing this file::

    python testing/make_notebook_output_checks.py

``test_notebook_outputs.py`` also fails when the committed notebook is out
of sync with this generator (an optional first argument selects another
output path, which that check uses).

Expectation keys (all optional):

``images``          exact number of outputs carrying ``image/png``
``result``          whether the cell may end with an ``execute_result``
``stdout`` / ``stderr``  whether anything may appear on that stream
``stdout_once`` / ``stderr_once``  substring that must appear exactly once
``result_includes`` / ``result_excludes``  substrings of the echo's text/plain
``when_has_output`` only check the cell when it produced any output at
                    all (used for the optional ``.wip`` section, which is
                    silent when the sample file is absent)
"""

import sys  # optional output path on the command line
from pathlib import Path  # output location next to this script
from textwrap import dedent  # keep cell sources readable in this file

import nbformat  # notebook JSON model
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = Path(__file__).resolve().parent
# Default target is the committed notebook next to this script; the sync
# test passes a temporary path instead and diffs the two.
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / 'notebook_output_checks.ipynb'

# Cells are collected in order; ``md`` / ``code`` append to this list.
cells = []


def md(text: str) -> None:
        """Append a markdown cell (dedented, surrounding blank lines stripped)."""
        cells.append(new_markdown_cell(dedent(text).strip()))


def code(source: str, **expect) -> None:
        """Append a code cell; keyword arguments become its ``expect`` metadata."""
        cell = new_code_cell(dedent(source).strip())
        if expect:
                cell.metadata['expect'] = expect
        cells.append(cell)


# ---------------------------------------------------------------------------
# Introduction and setup
# ---------------------------------------------------------------------------

md("""
# Notebook output checks for ramantools

Every cell below is a case where Jupyter could show something twice, show
something it should not, or swallow something it should show:

- the multi-line `repr` echo of `ramanmap` / `singlespec` / `wipfile`
- methods that **print and return `None`** (`history()`, `print_metadata()`,
  `wipfile.list()`, `wipfile.tree()`) versus their opt-in structured returns
- the three plotting routes: `toplot=True` (draw on the current axes),
  `savepath=` (own figure, saved and closed, nothing displayed) and `ax=`
  (draw into a user-owned figure)
- status text that must appear exactly once, on stdout by default or through
  `logging` when the user configured it, never both

Each code cell carries an `expect` block in its metadata stating what the
cell is allowed to emit. `testing/test_notebook_outputs.py` executes this
notebook and checks every cell against it. When reading the notebook by eye:
**one** image per plotting cell, no `None` echoes, no doubled text.

The data are synthetic (`testing/synthetic.py`), so the notebook runs on a
fresh clone. The `.wip` section runs only when `test-data/FLG42_532.wip` is
present.
""")

code("""
%matplotlib inline
import logging
import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import ramantools as rt

# testing/ holds the synthetic data generator. Make it importable whether the
# kernel was started in the repository root or inside testing/.
for candidate in (Path.cwd() / 'testing', Path.cwd(), Path.cwd().parent / 'testing'):
    if (candidate / 'synthetic.py').exists():
        sys.path.insert(0, str(candidate))
        break
import synthetic

# Fresh scratch directory for the Witec-format files and every image / netCDF
# written below. Printed so the files can be inspected after a manual run.
WORKDIR = Path(tempfile.mkdtemp(prefix='ramantools_nb_'))
files = synthetic.write_sample_set(WORKDIR)
m = rt.ramanmap(files['map']['map_path'], files['map']['info_path'])
s = rt.singlespec(files['spectrum']['spec_path'], files['spectrum']['info_path'])
print('scratch directory:', WORKDIR)
""", images=0, result=False, stdout=True, stderr=False)

# ---------------------------------------------------------------------------
# repr / print-and-return-None contracts
# ---------------------------------------------------------------------------

md("""
## 1. `repr` echo

A bare object name must produce exactly one `execute_result` with the
multi-line summary and nothing on stdout. `print(obj)` must produce the same
text once on stdout and no echo.
""")

code("m", images=0, result=True, stdout=False, result_includes="<ramantools.ramanmap")
code("s", images=0, result=True, stdout=False, result_includes="<ramantools.singlespec")
code("print(m)", images=0, result=False, stdout=True, stdout_once="<ramantools.ramanmap")
code("""
# Containers inside another container: the list repr calls __repr__ of each
# element. Must not raise and must not print.
[m, s]
""", images=0, result=True, stdout=False, result_includes="<ramantools.singlespec")

md("""
## 2. Printing methods versus structured returns

`history()`, `print_metadata()` and `summary()` are the cases that would
double up if a printing method ever started returning its text: the cell
would show the text **and** echo the returned string.
""")

code("m.history()", images=0, result=False, stdout=True)
code("m.history(as_list=True)", images=0, result=True, stdout=False)
code("s.summary()", images=0, result=True, stdout=False)
code("m.print_metadata()", images=0, result=False, stdout=True)
code("""
# The returned value of a printing method must be None, so a cell that ends
# in it shows nothing beyond the printed text.
assert m.history() is None
assert m.print_metadata() is None
""", images=0, result=False, stdout=True)

# ---------------------------------------------------------------------------
# Plotting: toplot / savepath / ax
# ---------------------------------------------------------------------------

md("""
## 3. Plotting routes

The inline backend renders every open figure once at the end of a cell. A
function that **returns** a `Figure` as part of its result would be rendered
again by the `execute_result` echo. `savepath=` must create, save and close
its own figure so nothing is displayed; `ax=` must draw into the caller's
figure and leave it open.

### 3.1 `singlespec.remove_bg`
""")

code("""
# Historical route: draw on plt.gca(), one image, no echo because of the assignment.
nobg_s, coeff, covar = s.remove_bg(toplot=True)
""", images=1, result=False)

code("""
# Same call as the last expression: the tuple is echoed, but it must not
# contain a Figure (which the inline backend would render a second time).
s.remove_bg(toplot=True)
""", images=1, result=True, result_excludes="Figure")

code("""
# Headless route: the figure is written to disk and closed, nothing is shown.
out = WORKDIR / 'spec_bg.png'
_ = s.remove_bg(savepath=out)
assert out.exists() and out.stat().st_size > 0, 'savepath did not write a file'
assert plt.get_fignums() == [], f'savepath left figures open: {plt.get_fignums()}'
""", images=0, result=False, stdout=False)

code("""
# toplot=True together with savepath=: savepath wins, still nothing displayed.
out = WORKDIR / 'spec_bg_both.png'
_ = s.remove_bg(toplot=True, savepath=out)
assert out.exists()
assert plt.get_fignums() == []
""", images=0, result=False, stdout=False)

code("""
# ax= route: the user owns the figure, it is displayed once at the end of the cell.
fig, ax = plt.subplots(figsize=(6, 3))
_ = s.remove_bg(ax=ax)
ax.set_title('drawn into a user-owned Axes');
""", images=1, result=False)

code("""
# ax= together with savepath=: the user's figure is saved AND stays open, so
# it is displayed exactly once; a second, private figure must not appear.
out = WORKDIR / 'spec_bg_ax.png'
fig, ax = plt.subplots(figsize=(6, 3))
_ = s.remove_bg(ax=ax, savepath=out)
assert out.exists()
assert plt.get_fignums() == [fig.number], plt.get_fignums()
""", images=1, result=False, stdout=False)

md("### 3.2 `ramanmap.remove_bg` and `plotspec`")

code("""
# mode='const' plots the single reference spectrum: one image.
nobg_m, _, _ = m.remove_bg(toplot=True)
""", images=1, result=False)

code("""
# mode='individual' would make one plot per pixel; every plotting keyword is
# rejected up front, before any figure is created.
for kw in ({'toplot': True}, {'savepath': WORKDIR / 'never.png'}, {'ax': plt.gca()}):
    try:
        m.remove_bg(mode='individual', **kw)
    except ValueError as err:
        print('rejected', list(kw), '->', str(err)[:60])
    else:
        raise AssertionError(f'{kw} was accepted in individual mode')
plt.close('all')  # discard the empty Axes created by plt.gca() above
assert not (WORKDIR / 'never.png').exists()
""", images=0, result=False, stdout=True)

code("""
# plotspec: map image plus spectrum in one figure, displayed once.
m.plotspec(6, 5, 1582)
""", images=1, result=False, stdout=False)

code("""
# plotspec with savepath: nothing displayed, figure closed.
out = WORKDIR / 'plotspec.png'
m.plotspec(6, 5, 1582, savepath=out)
assert out.exists()
assert plt.get_fignums() == []
""", images=0, result=False, stdout=False)

code("""
# plotspec into user-owned axes, also saved: one image (the user's figure).
out = WORKDIR / 'plotspec_axes.png'
fig, axes = plt.subplots(1, 2, figsize=(9, 4))
m.plotspec(6, 5, 1582, savepath=out, axes=axes)
assert out.exists()
assert plt.get_fignums() == [fig.number]
""", images=1, result=False, stdout=False)

md("""
### 3.3 `peakfit` and `bgsubtract`

`peakfit(toplot=True)` also prints the starting and fitted parameters through
`_say`, so these cells check that the text appears exactly once as well.
""")

code("""
crop_s = nobg_s.ssxr.sel(ramanshift=slice(1450, 1700))
fit = rt.peakfit(crop_s, stval={'x0': 1582}, toplot=True)
""", images=1, result=False, stdout_once="Values of fitted parameters")

code("""
# Map fit as the last expression: one image, the Dataset echo must not carry a Figure.
crop_m = nobg_m.mapxr.sel(ramanshift=slice(1450, 1700))
rt.peakfit(crop_m, stval={'x0': 1582}, toplot=True, width=6, height=5)
""", images=1, result=True, stdout_once="Values of fitted parameters", result_excludes="Figure")

code("""
# savepath: no image in the notebook, the parameter printout still appears once.
out = WORKDIR / 'fit.png'
_ = rt.peakfit(crop_s, stval={'x0': 1582}, savepath=out)
assert out.exists()
assert plt.get_fignums() == []
""", images=0, result=False, stdout_once="Values of fitted parameters")

code("""
# Two ramantools plots in one user-owned figure: one image.
fig, (ax_bg, ax_fit) = plt.subplots(1, 2, figsize=(10, 3.5))
_ = s.remove_bg(ax=ax_bg)
_ = rt.peakfit(crop_s, stval={'x0': 1582}, ax=ax_fit)
assert plt.get_fignums() == [fig.number]
""", images=1, result=False)

code("""
# The fit Dataset itself echoes once; its attrs carry the parameter units.
fit.attrs['param_units']
""", images=0, result=True, stdout=False, result_includes="x0")

code("""
# Module-level bgsubtract on raw arrays, historical toplot route.
_ = rt.bgsubtract(s.ramanshift, s.counts, toplot=True)
""", images=1, result=False)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

md("""
## 4. Status text: print or log, never both

`_say` prints to stdout unless a real logging handler is configured *and*
admits the record; then the text goes through logging only (the default
`basicConfig` handler writes to stderr). An explicit level on the
`ramantools` logger silences it. Each cell below must show the
`peakfit` parameter printout on exactly one stream, or on none.
""")

code("""
# Default: no logging configured, text on stdout.
_ = rt.peakfit(crop_s, stval={'x0': 1582}, savepath=WORKDIR / 'log1.png')
""", images=0, result=False, stdout_once="Values of fitted parameters", stderr=False)

code("""
# basicConfig at INFO: delivered through the root handler (stderr) only.
logging.basicConfig(level=logging.INFO, force=True)
_ = rt.peakfit(crop_s, stval={'x0': 1582}, savepath=WORKDIR / 'log2.png')
""", images=0, result=False, stdout=False, stderr_once="Values of fitted parameters")

code("""
# basicConfig at its WARNING default drops INFO records: fall back to stdout
# so the text does not silently disappear.
logging.basicConfig(level=logging.WARNING, force=True)
_ = rt.peakfit(crop_s, stval={'x0': 1582}, savepath=WORKDIR / 'log3.png')
""", images=0, result=False, stdout_once="Values of fitted parameters", stderr=False)

code("""
# Explicit opt-out on the package logger: nothing anywhere.
logging.basicConfig(level=logging.INFO, force=True)
logging.getLogger('ramantools').setLevel(logging.WARNING)
_ = rt.peakfit(crop_s, stval={'x0': 1582}, savepath=WORKDIR / 'log4.png')
""", images=0, result=False, stdout=False, stderr=False)

code("""
# DEBUG trace of a pipeline: every step emits one record, none twice.
class ListHandler(logging.Handler):
    # Minimal sink that keeps the formatted messages for assertions.
    def __init__(self):
        super().__init__()
        self.records = []
    def emit(self, record):
        self.records.append(record.getMessage())

rt_logger = logging.getLogger('ramantools')
rt_logger.setLevel(logging.DEBUG)
sink = ListHandler()
rt_logger.addHandler(sink)
try:
    _ = nobg_s.calibrate(peakshift=520.5).normalize(peakshift=1582).crr(cutoff=2, window=3)
finally:
    rt_logger.removeHandler(sink)
assert len(sink.records) == len(set(sink.records)), 'a record was emitted twice'
assert len(sink.records) >= 3, sink.records
print(len(sink.records), 'debug records:')
print(*sink.records, sep='\\n')
""", images=0, result=False, stdout=True)

code("""
# Restore the default state: no handlers, no explicit level -> stdout again.
rt_logger.setLevel(logging.NOTSET)
for handler in list(logging.getLogger().handlers):
    logging.getLogger().removeHandler(handler)
_ = rt.peakfit(crop_s, stval={'x0': 1582}, savepath=WORKDIR / 'log5.png')
""", images=0, result=False, stdout_once="Values of fitted parameters", stderr=False)

# ---------------------------------------------------------------------------
# save / load
# ---------------------------------------------------------------------------

md("""
## 5. `save` / `load`

`save()` returns `None` (no echo); `rt.load()` as the last expression echoes
the restored object's repr once, identical to the original's summary.
""")

code("""
nc = WORKDIR / 'spec.nc'
nobg_s.save(nc)
""", images=0, result=False, stdout=False)

code("rt.load(nc)", images=0, result=True, stdout=False, result_includes="<ramantools.singlespec")

code("""
loaded = rt.load(nc)
assert loaded.summary() == nobg_s.summary(), 'summary changed across the round trip'
assert loaded.history(as_list=True) == nobg_s.history(as_list=True)
loaded.history()
""", images=0, result=False, stdout=True)

# ---------------------------------------------------------------------------
# wipfile (optional)
# ---------------------------------------------------------------------------

md("""
## 6. `wipfile` (only with `test-data/FLG42_532.wip`)

Same contracts as above: the object echoes once, `list()` / `tree()` print
and return `None`, `entries` / `as_list` / `as_string` return without
printing. Each cell is silent when the sample file is absent and is then
skipped by the test.
""")

code("""
WIP_PATH = next((p for p in (Path('test-data/FLG42_532.wip'), Path('../test-data/FLG42_532.wip')) if p.exists()), None)
wip = rt.wipfile(WIP_PATH) if WIP_PATH else None
wip
""", when_has_output=True, images=0, result=True, stdout=False)

code("""
if wip:
    # prints the inventory and returns None (no echo)
    assert wip.list() is None
""", when_has_output=True, images=0, result=False, stdout=True)

code("""
if wip:
    assert wip.tree() is None
""", when_has_output=True, images=0, result=False, stdout=True)

code("""
wip.entries[:3] if wip else None
""", when_has_output=True, images=0, result=True, stdout=False)

code("""
wip.tree(as_string=True).splitlines()[:5] if wip else None
""", when_has_output=True, images=0, result=True, stdout=False)

# ---------------------------------------------------------------------------
# CLI from a notebook
# ---------------------------------------------------------------------------

md("""
## 7. Calling the CLI in-process

`ramantools.cli.main()` returns an exit status, so a cell ending in a bare
`main([...])` would print the report **and** echo `0`. Assign the status
instead.
""")

code("""
from ramantools.cli import main
status = main(['inspect', str(files['spectrum']['spec_path']), '--info', str(files['spectrum']['info_path'])])
assert status == 0
""", images=0, result=False, stdout_once="<ramantools.singlespec")

md("""
## Reading the results

Scroll through once: every plotting cell shows a single image, no cell shows
`None`, and no text block appears twice. The mechanical version of that check
is

    pytest testing/test_notebook_outputs.py

which executes this notebook and compares each cell's outputs with its
`expect` metadata.
""")

# ---------------------------------------------------------------------------
# Write the notebook
# ---------------------------------------------------------------------------

nb = new_notebook(cells=cells)
# Kernel metadata so Jupyter and nbclient pick the default Python kernel.
nb.metadata['kernelspec'] = {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}
nb.metadata['language_info'] = {'name': 'python'}
nbformat.write(nb, OUT)
print(f'wrote {OUT} ({len(cells)} cells)')

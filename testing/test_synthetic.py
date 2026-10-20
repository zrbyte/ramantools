"""Tests that run on generated data, so they work on a fresh clone.

Everything here uses ``testing/synthetic.py`` to write small Witec-format
files into a temporary directory; no file from ``test-data/`` is needed.
Beyond covering the 0.6.0 additions (repr / summary, structured history,
save / load, ``savepath`` plotting, logging, the CLI) these tests check
the algorithms against *known* ground truth, which the real-data tests
cannot do: the calibration offset, the cosmic-ray positions and the
flake mask are all chosen by the generator.
"""

import json  # CLI --json output parsing
import logging  # logging-capture checks
import sys  # repo root on sys.path
from pathlib import Path  # path handling
import numpy as np  # numeric assertions
import pytest  # fixtures / raises
import xarray as xr  # type checks on returned DataArrays

# Headless backend for the whole module: plots are only ever written to
# files here, and CI has no display.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))
import ramantools as rt  # noqa: E402
import synthetic  # noqa: E402 — the generator lives next to this file


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def files(tmp_path_factory):
    """Write the sample set once per module; returns the generator's dict."""
    return synthetic.write_sample_set(tmp_path_factory.mktemp('synthetic'))


@pytest.fixture(scope='module')
def spec(files):
    """Loaded synthetic spectrum (with info file)."""
    f = files['spectrum']
    return rt.singlespec(f['spec_path'], f['info_path'])


@pytest.fixture(scope='module')
def rmap(files):
    """Loaded synthetic map (with info file)."""
    f = files['map']
    return rt.ramanmap(f['map_path'], f['info_path'])


@pytest.fixture(scope='module')
def spec_nobg(spec):
    """Background-subtracted spectrum via automatic peak detection."""
    nobg, _, _ = spec.remove_bg()
    return nobg


@pytest.fixture(scope='module')
def map_nobg(rmap):
    """Background-subtracted map (const mode, automatic peak detection)."""
    nobg, _, _ = rmap.remove_bg()
    return nobg


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

class TestLoading:
    def test_spectrum_metadata(self, spec, files):
        """Info-file fields land on the instance and the data matches the generator."""
        f = files['spectrum']
        assert spec.specname == f['name']
        assert spec.samplename == f['sample']
        assert spec.laser == pytest.approx(synthetic.LASER_NM)
        assert spec.positioner_z == 0.0
        np.testing.assert_allclose(spec.counts, f['counts'])
        np.testing.assert_allclose(spec.ramanshift, f['ramanshift'], rtol=1e-5)

    def test_map_shape_and_geometry(self, rmap, files):
        """Pixel grid, physical size and cube layout survive the text round trip."""
        f = files['map']
        assert rmap.map.shape == f['cube'].shape
        assert (rmap.pixel_x, rmap.pixel_y) == (f['cube'].shape[2], f['cube'].shape[1])
        assert rmap.size_x == f['size_x'] and rmap.size_y == f['size_y']
        np.testing.assert_allclose(rmap.map, f['cube'])
        # xarray layout is (ramanshift, height, width) with the height flipped.
        assert rmap.mapxr.dims == ('ramanshift', 'height', 'width')
        np.testing.assert_allclose(rmap.mapxr.values, np.flip(f['cube'], axis=1))

    def test_header_only_load(self, tmp_path):
        """Loading without an info file takes the geometry from the [Header] block."""
        f = synthetic.write_map(tmp_path, pixel_x=6, pixel_y=5, size_x=6.0, size_y=5.0, cosmic_rays=0)
        m = rt.ramanmap(f['map_path'])
        assert (m.pixel_x, m.pixel_y, m.size_x, m.size_y) == (6, 5, 6.0, 5.0)
        assert m.mapname == f['name']
        assert m.samplename == 'N/A'  # sentinel: no info file
        s = rt.singlespec(synthetic.write_spectrum(tmp_path, cosmic_rays=0)['spec_path'])
        assert s.positioner_x == 100.0


# ---------------------------------------------------------------------------
# repr / summary / structured history
# ---------------------------------------------------------------------------

class TestIntrospection:
    def test_repr_is_summary(self, rmap, spec):
        """repr shows name, grid, ramanshift range and the step list, not an address."""
        r = repr(rmap)
        assert r.startswith("<ramantools.ramanmap 'SYN_FLAKE--Scan Piezo--001'>")
        assert '12 x 10 pixels' in r and '1600 points' in r and 'steps:       load' in r
        assert ' at 0x' not in r  # no bare object address
        r = repr(spec)
        assert r.startswith("<ramantools.singlespec") and 'position:' in r

    def test_repr_without_info_has_no_dangling_units(self, files):
        """Header-only loading leaves instrument fields unknown; the repr must
        print a bare ``N/A`` for them, not ``N/A nm`` / ``N/Ax``."""
        r = repr(rt.ramanmap(files['map']['map_path']))
        assert 'objective N/A' in r and 'laser N/A' in r
        assert 'N/Ax' not in r and 'N/A nm' not in r and 'N/A s' not in r
        # The Witec text header carries the stage position, so a header-only
        # spectrum still knows it; exercise the all-unknown branch of the
        # formatter directly instead.
        from ramantools._helpers import _format_summary
        summ = rt.singlespec(files['spectrum']['spec_path']).summary()
        summ.update(positioner_x=float('nan'), positioner_y=float('nan'), positioner_z=float('nan'))
        assert 'position:    N/A' in _format_summary(summ)

    def test_summary_is_json_serialisable(self, rmap, spec):
        """summary() is a plain dict that json.dumps accepts without a default."""
        for obj in (rmap, spec):
            s = obj.summary()
            json.dumps(s)
            assert s['n_ramanshift'] == 1600
            assert s['steps'] == ['load']
        assert rmap.summary()['pixel_x'] == 12
        assert spec.summary()['positioner_z'] == 0.0

    def test_history_records_every_step(self, spec_nobg):
        """Each pipeline method appends one structured record with its parameters."""
        s = spec_nobg.calibrate(peakshift=520.5).normalize(peakshift=1582).crr(cutoff=3, window=2)
        steps = s.history(as_list=True)
        assert [st['step'] for st in steps] == ['load', 'remove_bg', 'calibrate', 'normalize', 'crr']
        assert steps[1]['params']['polyorder'] == 1
        assert steps[2]['params']['peakshift'] == 520.5
        assert steps[3]['params']['func'] == 'lorentz'
        assert steps[4]['params'] == {'cutoff': 3, 'window': 2, 'n_replaced': steps[4]['params']['n_replaced']}
        # The free-text log is still maintained in parallel.
        assert 'background subtracted' in s.ssxr.attrs['comments']
        # Default call still prints and returns None (backward compatible).
        assert s.history() is None

    def test_history_falls_back_to_comments(self, spec):
        """Objects without the structured attr still give a list (one entry per comment line)."""
        da = spec.ssxr.copy()
        del da.attrs['processing_steps']
        import copy
        s = copy.copy(spec)
        s.ssxr = da
        steps = s.history(as_list=True)
        assert steps and steps[0]['step'] == 'comment' and 'raw data loaded' in steps[0]['text']

    def test_peakfit_units_and_history(self, spec_nobg):
        """peakfit attaches param_units and its own history record to the result Dataset."""
        fit = rt.peakfit(spec_nobg.ssxr.sel(ramanshift=slice(1450, 1700)), stval={'x0': 1582})
        assert fit.attrs['param_units'] == {'x0': '1/cm', 'ampl': 'au', 'width': '1/cm', 'offset': 'au'}
        steps = json.loads(fit.attrs['processing_steps'])
        assert steps[-1]['step'] == 'peakfit' and steps[-1]['params']['func'] == 'lorentz'
        fit2 = rt.peakfit(spec_nobg.ssxr.sel(ramanshift=slice(2600, 2800)), func=rt.lorentz2,
                          stval={'x01': 2690, 'x02': 2715})
        assert fit2.attrs['param_units']['x02'] == '1/cm' and fit2.attrs['param_units']['ampl1'] == 'au'


# ---------------------------------------------------------------------------
# Algorithms against ground truth
# ---------------------------------------------------------------------------

class TestAlgorithms:
    def test_remove_bg_flattens_baseline(self, spec_nobg):
        """After background removal the featureless region sits at ~0 counts."""
        quiet = spec_nobg.ssxr.sel(ramanshift=slice(3200, 3900))
        assert abs(float(quiet.median())) < 5
        assert spec_nobg.mask is not None and spec_nobg.mask.dtype == bool

    def test_calibrate_recovers_miscalibration(self, spec_nobg, files):
        """The generator shifted all peaks by +2.5 cm⁻¹; calibrate must undo it."""
        truth = files['spectrum']['truth']
        cal = spec_nobg.calibrate(peakshift=synthetic.SI_PEAK['x0'])
        shift_applied = cal.history(as_list=True)[-1]['params']['shift_applied']
        assert shift_applied == pytest.approx(-truth['miscalibration'], abs=0.3)
        # After calibration the G band sits where the model put it.
        fit = rt.peakfit(cal.ssxr.sel(ramanshift=slice(1450, 1700)), stval={'x0': 1582})
        assert float(fit['curvefit_coefficients'].sel(param='x0')) == pytest.approx(synthetic.G_PEAK['x0'], abs=0.5)
        # |width|: the Lorentzian is symmetric in the sign of the width parameter.
        # No abs(): peakfit reports the FWHM as a positive number (see TestPeakfitWidthSign).
        assert float(fit['curvefit_coefficients'].sel(param='width')) == pytest.approx(synthetic.G_PEAK['width'], rel=0.1)

    def test_normalize_scales_peak_to_one(self, spec_nobg):
        """Normalising to the G band makes its fitted amplitude ≈ 1."""
        norm = spec_nobg.normalize(peakshift=synthetic.G_PEAK['x0'] + 2.5)
        assert norm.normfactor == pytest.approx(synthetic.G_PEAK['ampl'], rel=0.05)
        assert float(norm.ssxr.sel(ramanshift=slice(1550, 1620)).max()) == pytest.approx(1.0, abs=0.05)
        assert norm.ssxr.attrs['long_name'] == 'normalized Raman intensity'

    def test_normalize_requires_remove_bg(self, spec):
        """The guard still refuses raw data (structured history has no remove_bg)."""
        with pytest.raises(ValueError, match='background was not removed'):
            spec.normalize(peakshift=1582)

    def test_crr_hits_the_injected_spikes(self, spec, files):
        """crr replaces the cosmic-ray samples the generator injected.

        A single-sample spike among ``2*window+1`` points can reach a
        z-score of at most about ``sqrt(2*window)`` against a local std
        that includes the spike itself, so the default ``cutoff=2,
        window=2`` sits right at the detection limit; ``window=3`` gives
        the rolling statistics enough room (see roadmap item 2.1).
        """
        idx = files['spectrum']['truth']['cosmic_ray_indices']
        clean = spec.crr(cutoff=2, window=3)
        before = spec.ssxr.values[idx]
        after = clean.ssxr.values[idx]
        assert np.all(before - after > 0.5 * synthetic.COSMIC_HEIGHT)
        assert clean.history(as_list=True)[-1]['params']['n_replaced'] >= len(idx)

    def test_map_crr_hits_the_injected_spikes(self, rmap, files):
        """Same check on the map, using the (row, col, index) triples (height is flipped)."""
        clean = rmap.crr(cutoff=2, window=3)
        ny = rmap.pixel_y
        for j, i, k in files['map']['truth']['cosmic_rays']:
            before = rmap.mapxr.values[k, ny - 1 - j, i]
            after = clean.mapxr.values[k, ny - 1 - j, i]
            assert before - after > 0.5 * synthetic.COSMIC_HEIGHT

    def test_individual_bg_removes_gradient(self, rmap):
        """Per-pixel background removal flattens the left-to-right baseline drift."""
        fitmask = np.ones_like(rmap.ramanshift, dtype=bool)
        # Exclude the three peaks from the polynomial fit.
        rs = rmap.ramanshift
        for x0, w in ((520.5, 19), (1582, 22), (2700, 60)):
            fitmask &= ~((rs > x0 + 2.5 - 6 * w) & (rs < x0 + 2.5 + 6 * w))
        nobg, coeff, covar = rmap.remove_bg(mode='individual', fitmask=fitmask)
        assert isinstance(coeff, xr.DataArray) and coeff.dims == ('height', 'width', 'coef')
        quiet = nobg.mapxr.sel(ramanshift=slice(3200, 3900)).median(dim='ramanshift')
        # Lorentzian tails bias a straight-line background by a few counts
        # everywhere, so the residual is small but not zero …
        assert float(np.abs(quiet).max()) < 12
        # … while the 5 % (≈ 44 counts) left-to-right ramp must be gone.
        assert float(quiet.std()) < 3
        # Fitted intercepts follow the generator's 5 % baseline ramp.
        intercept = coeff.sel(coef=1)
        assert float(intercept.isel(width=-1).mean()) > float(intercept.isel(width=0).mean())

    def test_peakmask_finds_the_flake(self, map_nobg, files):
        """peakmask on the G band selects (roughly) the pixels inside the flake."""
        masked, mask = map_nobg.peakmask(synthetic.G_PEAK['x0'] + 2.5, cutoff=0.3)
        truth = np.flip(files['map']['truth']['flake'], axis=0) > 0.5  # height flipped like mapxr
        agreement = (mask.values == truth).mean()
        assert agreement > 0.9
        rec = masked.history(as_list=True)[-1]
        assert rec['step'] == 'peakmask' and rec['params']['pixels_kept'] == int(mask.sum())

    def test_normalize_individual(self, map_nobg):
        """Per-pixel normalisation returns a (height, width) normfactor."""
        norm = map_nobg.normalize(peakshift=synthetic.SI_PEAK['x0'] + 2.5, mode='individual')
        assert isinstance(norm.normfactor, xr.DataArray) and norm.normfactor.dims == ('height', 'width')
        rec = norm.history(as_list=True)[-1]['params']
        assert rec['mode'] == 'individual' and 'mean' in rec['normfactor']


# ---------------------------------------------------------------------------
# save / load
# ---------------------------------------------------------------------------

class TestSaveLoad:
    def test_spectrum_round_trip(self, spec_nobg, tmp_path):
        """Everything on the container comes back: data, attrs, mask, normfactor, raw, metadata."""
        s = spec_nobg.calibrate(peakshift=520.5).normalize(peakshift=1582)
        path = tmp_path / 's.nc'
        s.save(path)
        back = rt.singlespec.load(path)
        xr.testing.assert_identical(s.ssxr, back.ssxr)
        np.testing.assert_array_equal(s.mask, back.mask)
        assert back.normfactor == pytest.approx(s.normfactor)
        np.testing.assert_array_equal(s.counts, back.counts)
        np.testing.assert_array_equal(s.ramanshift, back.ramanshift)
        for attr in ('specname', 'samplename', 'laser', 'itime', 'grating', 'positioner_z', 'metadata', '_info_loaded'):
            assert getattr(back, attr) == getattr(s, attr)
        assert back.history(as_list=True) == s.history(as_list=True)
        # Pipeline continues on the loaded object.
        back.crr()

    def test_map_round_trip_and_dispatch(self, map_nobg, tmp_path):
        """ramanmap round trip, rt.load() dispatch, typed-load guard and include_raw=False."""
        m = map_nobg.normalize(peakshift=1582 + 2.5, mode='individual')
        path = tmp_path / 'm.nc'
        m.save(path)
        back = rt.load(path)
        assert isinstance(back, rt.ramanmap)
        xr.testing.assert_identical(m.mapxr, back.mapxr)
        xr.testing.assert_allclose(m.normfactor, back.normfactor)
        np.testing.assert_array_equal(m.map, back.map)
        assert (back.pixel_x, back.pixel_y, back.size_x, back.size_y) == (m.pixel_x, m.pixel_y, m.size_x, m.size_y)
        with pytest.raises(TypeError, match='holds a ramanmap'):
            rt.singlespec.load(path)
        # Without the raw array the file is smaller and ``map`` is rebuilt
        # from the processed data.
        small = tmp_path / 'm_small.nc'
        m.save(small, include_raw=False)
        assert small.stat().st_size < path.stat().st_size
        back2 = rt.ramanmap.load(small)
        np.testing.assert_array_equal(back2.map, np.flip(m.mapxr.values, axis=1))

    def test_load_rejects_foreign_file(self, tmp_path, spec):
        """A plain xarray netCDF without our attributes is refused with a clear message."""
        path = tmp_path / 'plain.nc'
        spec.ssxr.to_netcdf(path, engine='h5netcdf')
        with pytest.raises(ValueError, match='not written by ramantools'):
            rt.load(path)


# ---------------------------------------------------------------------------
# Plotting without a display
# ---------------------------------------------------------------------------

class TestHeadlessPlots:
    def test_savepath_writes_and_closes(self, spec, map_nobg, tmp_path):
        """savepath produces a file and leaves no open figure behind (no double display)."""
        plt.close('all')
        spec.remove_bg(savepath=tmp_path / 'bg.png')
        rt.peakfit(map_nobg.mapxr.sel(ramanshift=slice(1450, 1700)), stval={'x0': 1582}, savepath=tmp_path / 'fit.png')
        map_nobg.plotspec(6, 5, 1582, savepath=tmp_path / 'map.png')
        for name in ('bg.png', 'fit.png', 'map.png'):
            assert (tmp_path / name).stat().st_size > 1000
        assert plt.get_fignums() == []

    def test_ax_draws_into_user_axes(self, spec):
        """ax= draws on the caller's axes and the figure stays open for them."""
        fig, ax = plt.subplots()
        spec.remove_bg(ax=ax)
        assert len(ax.lines) >= 2  # spectrum + fitted polynomial
        plt.close(fig)

    def test_individual_mode_rejects_plot_kwargs(self, rmap, tmp_path):
        """savepath / ax would mean one plot per pixel in individual mode."""
        fitmask = np.ones_like(rmap.ramanshift, dtype=bool)
        with pytest.raises(ValueError, match='toplot=True'):
            rmap.remove_bg(mode='individual', fitmask=fitmask, savepath=tmp_path / 'x.png')


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

class TestLogging:
    def test_prints_by_default(self, spec_nobg, capsys, tmp_path):
        """With no logging configured, peakfit's toplot report goes to stdout as before."""
        rt.peakfit(spec_nobg.ssxr.sel(ramanshift=slice(1450, 1700)), stval={'x0': 1582}, savepath=tmp_path / 'f.png')
        out = capsys.readouterr().out
        assert 'Values of fitted parameters' in out and 'x0 :' in out

    def test_logs_instead_of_printing_when_configured(self, spec_nobg, capsys, caplog, tmp_path):
        """With a handler attached, the same text is logged once and not printed."""
        logger = logging.getLogger('ramantools')
        handler = logging.StreamHandler()
        logger.addHandler(handler)
        try:
            with caplog.at_level(logging.DEBUG, logger='ramantools'):
                rt.peakfit(spec_nobg.ssxr.sel(ramanshift=slice(1450, 1700)), stval={'x0': 1582}, savepath=tmp_path / 'f.png')
                spec_nobg.crr()
        finally:
            logger.removeHandler(handler)
        assert 'Values of fitted parameters' not in capsys.readouterr().out
        messages = [r.getMessage() for r in caplog.records]
        assert any('Values of fitted parameters' in m for m in messages)
        assert any(m.startswith('crr: cutoff=') for m in messages)


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------

class TestCLI:
    def test_inspect_text_export(self, files, capsys):
        """inspect prints the repr, or the summary as JSON."""
        f = files['map']
        from ramantools.cli import main
        assert main(['inspect', str(f['map_path']), '--info', str(f['info_path'])]) == 0
        assert '<ramantools.ramanmap' in capsys.readouterr().out
        assert main(['inspect', str(f['map_path']), '--info', str(f['info_path']), '--json']) == 0
        data = json.loads(capsys.readouterr().out)
        assert data['class'] == 'ramanmap' and data['pixel_x'] == 12

    def test_export_history_plot(self, files, tmp_path, capsys):
        """export writes a .nc that history and plot can consume."""
        from ramantools.cli import main
        f = files['spectrum']
        out = tmp_path / 'spec.nc'
        assert main(['export', str(f['spec_path']), '--info', str(f['info_path']), '--out', str(out)]) == 0
        assert out.exists()
        assert 'wrote' in capsys.readouterr().out  # consume export's output
        assert main(['history', str(out), '--json']) == 0
        steps = json.loads(capsys.readouterr().out)
        assert steps[0]['step'] == 'load'
        png = tmp_path / 'spec.png'
        assert main(['plot', str(out), '--out', str(png)]) == 0
        assert png.stat().st_size > 1000
        # Map plot picks the strongest band by default.
        m = files['map']
        mpng = tmp_path / 'map.png'
        assert main(['plot', str(m['map_path']), '--info', str(m['info_path']), '--out', str(mpng)]) == 0
        assert mpng.stat().st_size > 1000

    def test_errors_are_reported_not_raised(self, tmp_path, capsys):
        """Missing files give a non-zero status and a message on stderr."""
        from ramantools.cli import main
        assert main(['inspect', str(tmp_path / 'missing.txt')]) == 1
        assert 'ramantools:' in capsys.readouterr().err

    def test_mismatched_info_file_is_reported(self, files, tmp_path, capsys):
        """A spectrum info file given for a map makes the Witec parser raise
        IndexError (missing 'Points per Line'); the CLI must report it on
        stderr with status 1 instead of dumping a traceback."""
        from ramantools.cli import main
        m, s = files['map'], files['spectrum']
        status = main(['export', str(m['map_path']), '--info', str(s['info_path']), '--out', str(tmp_path / 'x.nc')])
        assert status == 1
        err = capsys.readouterr().err
        assert err.startswith('ramantools: IndexError') and 'Points per Line' in err
        assert not (tmp_path / 'x.nc').exists()

    def test_inspect_accepts_id(self):
        """``inspect project.wip --id N`` must parse; it was the only command
        without --id, so the documented workflow failed with a usage error."""
        from ramantools.cli import build_parser
        args = build_parser().parse_args(['inspect', 'project.wip', '--id', '28', '--json'])
        assert args.id == '28' and args.json

    def test_module_entry_point(self, files):
        """python -m ramantools --version works end to end."""
        import subprocess
        res = subprocess.run([sys.executable, '-m', 'ramantools', '--version'], capture_output=True, text=True, cwd=ROOT)
        assert res.returncode == 0 and rt.__version__ in res.stdout


# ---------------------------------------------------------------------------
# Width sign normalisation in peakfit
# ---------------------------------------------------------------------------

class TestPeakfitWidthSign:
    """``peakfit`` must report widths as positive numbers.

    The synthetic G peak is the case that exposed the problem: with the
    default starting width of 14 the Lorentz fit converges to width = -22,
    a perfectly good fit in the mirrored parametrisation. These tests pin
    that the reported width is +22, that the fix is an exact
    reparametrisation (covariance consistent with a raw ``curvefit``), and
    that it covers maps, ``lorentz2`` and user-supplied ``fitresult``.
    """

    # Full starting-value dict, identical to what peakfit fills in from the
    # lorentz defaults, so the raw curvefit below runs the very same problem.
    STVAL = {'x0': 1582, 'ampl': 1, 'width': 14, 'offset': 0}

    def test_spectrum_width_positive_and_exact(self, spec_nobg):
        """Reported width is +FWHM; coefficients and covariance equal the raw
        curvefit result up to the sign flip of the width row/column."""
        crop = spec_nobg.ssxr.sel(ramanshift=slice(1450, 1700))
        fit = rt.peakfit(crop, stval=dict(self.STVAL))
        raw = crop.curvefit('ramanshift', rt.lorentz, p0=dict(self.STVAL), errors='ignore', kwargs={'maxfev': 1000})
        w = float(fit['curvefit_coefficients'].sel(param='width'))
        assert w > 0 and w == pytest.approx(synthetic.G_PEAK['width'], rel=0.1)
        # Exactness: every coefficient agrees with the raw fit in magnitude
        # bit for bit (the fix multiplies by +-1.0, nothing else).
        assert np.array_equal(np.abs(fit['curvefit_coefficients'].values), np.abs(raw['curvefit_coefficients'].values))
        # The raw fit may come out with either sign; the reported covariance
        # must be the raw one with the width row and column multiplied by
        # that sign, which leaves the diagonal untouched.
        s = np.sign(float(raw['curvefit_coefficients'].sel(param='width')))
        expected = raw['curvefit_covariance'].values.copy()
        i = list(raw.param.values).index('width')
        expected[i, :] *= s
        expected[:, i] *= s
        assert np.array_equal(fit['curvefit_covariance'].values, expected)
        assert np.array_equal(np.diag(fit['curvefit_covariance'].values), np.diag(raw['curvefit_covariance'].values))

    def test_map_widths_positive(self, map_nobg):
        """Every converged pixel of a map fit reports a positive width."""
        crop = map_nobg.mapxr.sel(ramanshift=slice(1450, 1700))
        fit = rt.peakfit(crop, stval={'x0': 1582})
        widths = fit['curvefit_coefficients'].sel(param='width').values
        finite = widths[np.isfinite(widths)]
        assert finite.size > 0 and np.all(finite > 0)

    def test_lorentz2_negative_start_reported_positive(self):
        """Double Lorentz: widths started negative stay negative in the raw
        optimiser result and must come out positive from peakfit, mixed
        signs included.

        A constructed two-peak spectrum is used because the synthetic 2D
        band is a single Lorentzian, on which a seven-parameter lorentz2
        fit does not converge from any reasonable start.
        """
        x = np.linspace(2600, 2800, 200)
        rng = np.random.default_rng(0)
        y = rt.lorentz2(x, 2690, 400, 40, 2740, 300, 40, 5) + rng.normal(0, 2, x.size)
        da = xr.DataArray(y, coords={'ramanshift': x}, dims='ramanshift')
        for w1, w2 in ((-30, -30), (30, -30)):
            stval = {'x01': 2685, 'x02': 2745, 'ampl1': 300, 'ampl2': 300, 'width1': w1, 'width2': w2, 'offset': 0}
            raw = da.curvefit('ramanshift', rt.lorentz2, p0=dict(stval), errors='ignore', kwargs={'maxfev': 1000})
            fit = rt.peakfit(da, func=rt.lorentz2, stval=dict(stval))
            raw_w = raw['curvefit_coefficients'].sel(param=['width1', 'width2']).values
            fit_w = fit['curvefit_coefficients'].sel(param=['width1', 'width2']).values
            # The optimiser keeps the sign it started with (the function is
            # even in width), which is exactly the case being normalised.
            assert np.all(np.sign(raw_w) == np.sign([w1, w2])), raw_w
            assert np.array_equal(fit_w, np.abs(raw_w)) and np.all(fit_w > 0)
            assert fit_w == pytest.approx([40, 40], rel=0.05)
            # Variances are invariant under the sign flip.
            assert np.array_equal(np.diag(fit['curvefit_covariance'].values), np.diag(raw['curvefit_covariance'].values))

    def test_fitresult_input_is_normalised_not_mutated(self, spec_nobg):
        """A caller-supplied ``fitresult`` with a negative width is returned
        positive, and the caller's Dataset is left as it was."""
        crop = spec_nobg.ssxr.sel(ramanshift=slice(1450, 1700))
        fit = rt.peakfit(crop, stval={'x0': 1582})
        negative = fit.copy(deep=True)
        negative['curvefit_coefficients'].loc[{'param': 'width'}] = -abs(float(fit['curvefit_coefficients'].sel(param='width')))
        out = rt.peakfit(crop, fitresult=negative)
        assert float(out['curvefit_coefficients'].sel(param='width')) > 0
        assert float(negative['curvefit_coefficients'].sel(param='width')) < 0  # input untouched

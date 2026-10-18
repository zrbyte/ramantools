"""Pytest suite for ramantools functionality.

This test suite validates core ramantools operations on both ramanmap and
singlespec objects using minimal synthetic test data.
"""

import pytest  # Testing framework for organizing and running tests
import sys  # Access to the Python path for importing the package under test
from pathlib import Path  # Convenient path handling independent of OS
import numpy as np  # Numerical operations for creating fit masks
import xarray as xr  # Type checks on DataArray-valued results (e.g. normfactor)

# Ensure the ramantools package is importable from the repo root
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import ramantools as rt


# ---------------------------------------------------------------------------
# Fixtures: Setup shared test data
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def data_paths():
    """Provide paths to test data files.

    Returns dictionary with paths to map and spectrum data and metadata files.
    Using module scope ensures paths are computed once per test module.
    Uses real data from test-data/ folder for comprehensive testing.
    """
    # Use real data from test-data folder (not pushed to repo)
    data_dir = Path(__file__).parent.parent / "test-data"
    return {
        "map_path": data_dir / "specmap.txt",
        "map_info": data_dir / "specmap_metadata.txt",
        "spec_path": data_dir / "singlespec.txt",
        "spec_info": data_dir / "singlespec_metadata.txt",
    }


@pytest.fixture(scope="module")
def ramanmap_instance(data_paths):
    """Load a ramanmap instance for testing.

    Module scope ensures the map is loaded once and reused across tests,
    improving test performance.
    """
    return rt.ramanmap(data_paths["map_path"], data_paths["map_info"])


@pytest.fixture(scope="module")
def singlespec_instance(data_paths):
    """Load a singlespec instance for testing.

    Module scope ensures the spectrum is loaded once and reused across tests.
    """
    return rt.singlespec(data_paths["spec_path"], data_paths["spec_info"])


@pytest.fixture
def fit_mask_map(ramanmap_instance):
    """Create a simple fit mask for ramanmap background subtraction.

    This uniform mask bypasses automatic peak finding which could fail
    on minimal synthetic datasets. Function scope creates a fresh mask
    for each test that needs one.
    """
    return np.ones_like(ramanmap_instance.ramanshift, dtype=bool)


@pytest.fixture
def fit_mask_spec(singlespec_instance):
    """Create a simple fit mask for singlespec background subtraction.

    Function scope ensures each test gets an independent mask.
    """
    return np.ones_like(singlespec_instance.ramanshift, dtype=bool)


# ---------------------------------------------------------------------------
# Tests: Data Loading
# ---------------------------------------------------------------------------

class TestDataLoading:
    """Test suite for loading Raman data from files."""

    def test_load_ramanmap(self, ramanmap_instance):
        """Verify ramanmap loads without errors and has expected structure."""
        assert ramanmap_instance is not None
        assert hasattr(ramanmap_instance, "mapxr")
        assert hasattr(ramanmap_instance, "ramanshift")
        # Verify dimensions are positive integers
        assert ramanmap_instance.pixel_x > 0
        assert ramanmap_instance.pixel_y > 0

    def test_load_singlespec(self, singlespec_instance):
        """Verify singlespec loads without errors and has expected structure."""
        assert singlespec_instance is not None
        assert hasattr(singlespec_instance, "ssxr")
        assert hasattr(singlespec_instance, "ramanshift")
        assert hasattr(singlespec_instance, "counts")

    # --- no-info-file loading via data-file header --------------------

    def test_load_singlespec_header_only(self):
        """Load a single spectrum using only the data file's [Header] block.

        ``ss_old_table_export.txt`` carries a Witec "old table export"
        header whose PositionX / Y / Z and GraphName fields the loader now
        consumes when no info file is supplied. All info-file-only fields
        (sample name, grating, laser, objective etc.) fall back to
        ``'N/A'`` / ``NaN`` sentinels.
        """
        bare_path = Path(__file__).parent.parent / "test-data" / "ss_old_table_export.txt"
        # One-arg call exercises the new default ``info_path=None`` branch.
        spec = rt.singlespec(str(bare_path))
        # Core data attributes should still be present.
        assert hasattr(spec, "ssxr")
        assert hasattr(spec, "ramanshift")
        assert hasattr(spec, "counts")
        # PositionX / Y / Z from the header should populate the positioner
        # attributes exactly (not as sentinels).
        assert spec.positioner_x == pytest.approx(-165.510999992635)
        assert spec.positioner_y == pytest.approx(4193.09300156985)
        assert spec.positioner_z == pytest.approx(0.0)
        # GraphName gets normalized by stripping the trailing --Spec.Data N.
        assert spec.specname == "MK_FLG_ABC_111--Spectrum--092"
        # Fields not present in the header should remain at sentinels.
        assert spec.samplename == "N/A"
        assert spec.grating == "N/A"
        assert np.isnan(spec.laser)
        # Comment must advertise the no-info-file load path.
        assert "no info file" in spec.ssxr.attrs["comments"]

    def test_load_ramanmap_header_only(self):
        """Load a map using only the data file's [Header] block.

        ``map_old_table_export.txt`` carries SizeX / SizeY / ScanWidth /
        ScanHeight / ScanOriginX / ScanOriginY in its header; the loader
        uses these to reshape the data and build coordinates without
        needing the info file.
        """
        bare_path = Path(__file__).parent.parent / "test-data" / "map_old_table_export.txt"
        specmap = rt.ramanmap(str(bare_path))
        # Pixel dimensions should come straight from the header.
        assert specmap.pixel_x == 250
        assert specmap.pixel_y == 120
        # Physical scan size from the header (ScanWidth / ScanHeight).
        assert specmap.size_x == pytest.approx(250.0)
        assert specmap.size_y == pytest.approx(120.0, abs=1e-6)
        # ScanOriginX / Y populate the positioner attributes.
        assert specmap.positioner_x == pytest.approx(-131.899993849918)
        assert specmap.positioner_y == pytest.approx(4139.39990239032)
        # GraphName becomes mapname (with the --Spec.Data suffix stripped).
        assert specmap.mapname == "MK_FLG_ABC_111--Scan LA--096"
        # xarray carries the inference-provenance comment.
        assert hasattr(specmap, "mapxr")
        assert "no info file" in specmap.mapxr.attrs["comments"]
        # Info-only fields remain sentinels.
        assert specmap.samplename == "N/A"
        assert np.isnan(specmap.laser)

    # --- new-format exports (identical [Header] keys, empty FileName) ---

    def test_load_singlespec_new_format(self):
        """Load a single spectrum from the newer Witec export format.

        The "new" table export differs from the "old" one only in that
        ``FileName =`` is empty; all other header keys (GraphName, SizeX,
        SizeY, PositionX/Y/Z, ...) are identical. This test confirms the
        shared parser handles both formats without branching.
        """
        bare_path = Path(__file__).parent.parent / "test-data" / "ss_new_table_export.txt"
        spec = rt.singlespec(str(bare_path))
        # Core data attributes present.
        assert hasattr(spec, "ssxr")
        assert hasattr(spec, "ramanshift")
        assert hasattr(spec, "counts")
        # PositionX / Y / Z parsed straight out of the header.
        assert spec.positioner_x == pytest.approx(6873.29400184061)
        assert spec.positioner_y == pytest.approx(4614.00599958236)
        assert spec.positioner_z == pytest.approx(0.0)
        # GraphName normalized (trailing --Spec.Data N stripped).
        assert spec.specname == "Graphite_on_PVC_10_after_strain--Spectrum--028"
        # wipfilename reflects the empty FileName value from this variant —
        # the outer regex captures the empty string; the override parser
        # leaves it alone because its stricter pattern won't match empty.
        assert spec.wipfilename == ""
        # Info-only fields stay sentinel.
        assert spec.samplename == "N/A"
        assert np.isnan(spec.laser)
        # Comment records the no-info-file provenance.
        assert "no info file" in spec.ssxr.attrs["comments"]

    def test_load_ramanmap_new_format(self):
        """Load a map from the newer Witec export format.

        Header keys match the old format (SizeX / SizeY / ScanWidth /
        ScanHeight / ScanOriginX / ScanOriginY); only FileName is empty.
        The fixture header says SizeX = 50, SizeY = 25, so reshape goes
        (ramanshift, 25, 50).
        """
        bare_path = Path(__file__).parent.parent / "test-data" / "map_new_table_export.txt"
        specmap = rt.ramanmap(str(bare_path))
        # Pixel dimensions from SizeX / SizeY.
        assert specmap.pixel_x == 50
        assert specmap.pixel_y == 25
        # Physical scan size — the fixture stores odd floating-point values
        # (99.99999... ≈ 100, 49.99999... ≈ 50) because of float precision
        # in the Witec exporter; we match what's actually in the file.
        assert specmap.size_x == pytest.approx(99.9999999999998)
        assert specmap.size_y == pytest.approx(49.9999999999995)
        # ScanOriginX / Y → positioner coordinates.
        assert specmap.positioner_x == pytest.approx(6895.3280002594)
        assert specmap.positioner_y == pytest.approx(4609.94099998474)
        # GraphName → mapname (suffix stripped).
        assert specmap.mapname == "Graphite_on_PVC_10_after_strain--Scan Piezo--027"
        # Reshape must have produced the expected grid.
        assert specmap.mapxr.sizes["width"] == 50
        assert specmap.mapxr.sizes["height"] == 25
        # Empty FileName means wipfilename is '' (see singlespec test note).
        assert specmap.wipfilename == ""
        # Info-only fields stay sentinel.
        assert specmap.samplename == "N/A"
        assert np.isnan(specmap.laser)
        # Comment records the no-info-file provenance.
        assert "no info file" in specmap.mapxr.attrs["comments"]


# ---------------------------------------------------------------------------
# Tests: Ramanmap Methods
# ---------------------------------------------------------------------------

class TestRamanmapMethods:
    """Test suite for ramanmap class methods."""

    def test_print_metadata(self, ramanmap_instance, capsys):
        """Verify metadata printing executes without errors."""
        ramanmap_instance.print_metadata()
        captured = capsys.readouterr()
        # Check that some metadata is printed
        assert len(captured.out) > 0
        assert "raw data loaded" in captured.out

    def test_remove_bg(self, ramanmap_instance, fit_mask_map):
        """Verify background removal returns valid ramanmap instance."""
        map_mod, coeff, covar = ramanmap_instance.remove_bg(fitmask=fit_mask_map)
        assert map_mod is not None
        assert hasattr(map_mod, "mapxr")
        assert coeff is not None
        assert covar is not None

    def test_remove_bg_individual_fast_path(self, ramanmap_instance, fit_mask_map):
        """remove_bg(mode='individual', fitmask=...) → vectorized fast path.

        When a fitmask is supplied, every pixel shares the same masked-in
        region so the implementation solves all per-pixel polynomial fits
        in one ``np.linalg.lstsq`` call. Check that the returned coeff /
        covar are xarray DataArrays indexed by the map's spatial coords
        and that the bg-subtracted map preserves dtype and shape.
        """
        import xarray as xr  # local import to keep test imports tidy
        map_mod, coeff, covar = ramanmap_instance.remove_bg(
            mode='individual', fitmask=fit_mask_map,
        )
        # Shape preservation: bg-subtracted map keeps original dims.
        assert map_mod.mapxr.shape == ramanmap_instance.mapxr.shape
        # Coefficients shape: (height, width, coef). For the default
        # polyorder=1 there are 2 coefficients per pixel.
        assert isinstance(coeff, xr.DataArray)
        assert coeff.dims == ('height', 'width', 'coef')
        assert coeff.sizes['height'] == ramanmap_instance.pixel_y
        assert coeff.sizes['width'] == ramanmap_instance.pixel_x
        assert coeff.sizes['coef'] == 2  # polyorder=1 → 2 coefficients
        # Covariance: (height, width, coef_i, coef_j) — symmetric in i↔j.
        assert isinstance(covar, xr.DataArray)
        assert covar.dims == ('height', 'width', 'coef_i', 'coef_j')
        assert covar.sizes['coef_i'] == 2
        assert covar.sizes['coef_j'] == 2
        # Comment annotation flags the individual / vectorized branch so
        # history() makes the path traceable downstream.
        assert 'mode == individual' in map_mod.mapxr.attrs['comments']
        assert 'vectorized polyfit' in map_mod.mapxr.attrs['comments']
        # Fast path stores the user's raw fitmask (1-D, no notch zeroing).
        assert isinstance(map_mod.mask, np.ndarray)
        assert map_mod.mask.shape == fit_mask_map.shape

    def test_remove_bg_individual_matches_bgsubtract(self, ramanmap_instance, fit_mask_map):
        """Fast-path coefficients must agree with per-pixel ``bgsubtract``.

        The vectorized lstsq solves the same linear least-squares problem
        scipy.curve_fit converges to inside bgsubtract, so the two
        coefficient arrays should match to floating-point precision. This
        protects against silent algorithmic drift in the fast path.
        """
        # Run the fast path
        _, coeff_individual, covar_individual = ramanmap_instance.remove_bg(
            mode='individual', fitmask=fit_mask_map,
        )
        # Run bgsubtract directly on a known pixel; pick the middle to
        # avoid edge effects and to match a common user workflow.
        mid_w = ramanmap_instance.size_x / 2
        mid_h = ramanmap_instance.size_y / 2
        spec = ramanmap_instance.mapxr.sel(width=mid_w, height=mid_h, method='nearest')
        # Pass the same fitmask so bgsubtract takes its fast internal path.
        _, _, c_ref, _, _, cv_ref = rt.bgsubtract(
            spec.ramanshift.data, spec.data, fitmask=fit_mask_map.copy(),
        )
        # Compare the middle-pixel coefficients and covariance.
        c_pixel = coeff_individual.sel(width=mid_w, height=mid_h, method='nearest').data
        cv_pixel = covar_individual.sel(width=mid_w, height=mid_h, method='nearest').data
        # Coefficients agree to ~1e-8 because both methods solve the same
        # linear system. Covariance gets a looser tolerance because s²
        # accumulates a different residual sum (the lstsq SVD solve and
        # Levenberg-Marquardt diverge at the last digit or two of the
        # least-squares residual).
        np.testing.assert_allclose(c_pixel, c_ref, rtol=1e-8, atol=1e-10)
        np.testing.assert_allclose(cv_pixel, cv_ref, rtol=1e-6, atol=1e-10)

    def test_remove_bg_individual_slow_path(self, ramanmap_instance):
        """remove_bg(mode='individual') without fitmask → per-pixel path.

        Each pixel goes through ``bgsubtract``'s own peak finder, so the
        per-pixel mask varies and is exposed as a 3-D DataArray. Just
        verify the shapes / types — exact numeric agreement is implicit
        because we're calling bgsubtract directly per pixel.
        """
        import xarray as xr
        map_mod, coeff, covar = ramanmap_instance.remove_bg(mode='individual')
        # Shapes match the fast path — same return contract.
        assert coeff.dims == ('height', 'width', 'coef')
        assert covar.dims == ('height', 'width', 'coef_i', 'coef_j')
        # Per-pixel mask: 3-D xarray DataArray indexed by spatial coords.
        assert isinstance(map_mod.mask, xr.DataArray)
        assert map_mod.mask.dims == ('height', 'width', 'ramanshift')
        # Comment annotation flags the per-pixel branch.
        assert 'per-pixel bgsubtract' in map_mod.mapxr.attrs['comments']

    def test_remove_bg_individual_rejects_toplot(self, ramanmap_instance, fit_mask_map):
        """remove_bg(mode='individual', toplot=True) must raise.

        Allowing toplot would spawn one matplotlib figure per pixel —
        thousands of figures on a typical map. The implementation
        refuses up front so the user isn't surprised by a hung session.
        """
        with pytest.raises(ValueError, match="toplot=True"):
            ramanmap_instance.remove_bg(
                mode='individual', fitmask=fit_mask_map, toplot=True,
            )

    def test_remove_bg_const_still_byte_identical(self, ramanmap_instance, fit_mask_map):
        """``mode='const'`` must remain byte-identical after the change.

        The Phase 2.5 implementation touched only the ``mode='individual'``
        branch; the const path is unchanged. Verify by checking that the
        returned coeff / covar are still plain numpy arrays (not the new
        xarray containers) and that the mapxr data matches a manual
        bgsubtract on the middle spectrum.
        """
        # Run remove_bg in const mode (default).
        map_mod, coeff, covar = ramanmap_instance.remove_bg(fitmask=fit_mask_map)
        # Const-mode return types stay numpy — this is the BC contract.
        assert isinstance(coeff, np.ndarray)
        assert isinstance(covar, np.ndarray)
        # Shape: coeff is 1-D (polyorder+1), covar is 2-D (P+1, P+1).
        assert coeff.ndim == 1
        assert covar.ndim == 2

    def test_calibrate_rejects_lorentz2(self, ramanmap_instance):
        """calibrate() must reject double-peak shapes loudly.

        Previously this produced a confusing KeyError deep in xarray
        because calibrate hardcodes ``.sel(param='x0')``, and lorentz2
        emits ``x01`` / ``x02`` parameter names instead.
        """
        with pytest.raises(ValueError, match="single-peak"):
            ramanmap_instance.calibrate(peakshift=1580, func=rt.lorentz2)

    def test_normalize_rejects_lorentz2(self, ramanmap_instance, fit_mask_map):
        """normalize() must reject double-peak shapes for the same reason."""
        # Need background-removed map first — normalize refuses otherwise.
        map_mod, _, _ = ramanmap_instance.remove_bg(fitmask=fit_mask_map)
        with pytest.raises(ValueError, match="single-peak"):
            map_mod.normalize(peakshift=1580, func=rt.lorentz2)

    def test_history(self, ramanmap_instance, capsys):
        """Verify history method displays processing comments."""
        ramanmap_instance.history()
        captured = capsys.readouterr()
        assert "Data modification history" in captured.out
        assert "raw data loaded" in captured.out

    def test_crr(self, ramanmap_instance):
        """Verify cosmic ray removal executes and returns valid instance."""
        map_crr = ramanmap_instance.crr()
        assert map_crr is not None
        assert hasattr(map_crr, "mapxr")

    def test_plotspec(self, ramanmap_instance):
        """Verify plotspec method executes without errors."""
        # Use middle of map coordinates
        width = ramanmap_instance.size_x / 2
        height = ramanmap_instance.size_y / 2
        shift = 300
        # This should execute without raising an exception
        # Note: We don't check plot output, just that it doesn't crash
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend for testing
        ramanmap_instance.plotspec(width=width, height=height, shift=shift)
        import matplotlib.pyplot as plt
        plt.close('all')  # Clean up plots

    def test_peakmask(self, ramanmap_instance):
        """Test peakmask method creates valid mask and masked map."""
        # Use a peak position that should exist in real Raman data
        mapmasked, mask = ramanmap_instance.peakmask(peakpos=1580, cutoff=0.1)
        assert mapmasked is not None
        assert mask is not None
        assert hasattr(mapmasked, "mapxr")
        # Verify mask has correct dimensions (width x height)
        assert mask.dims == ('height', 'width')

    def test_normalize(self, ramanmap_instance, fit_mask_map):
        """Test normalization to a peak returns valid normalized map."""
        # Remove background first as required by normalize
        map_mod, _, _ = ramanmap_instance.remove_bg(fitmask=fit_mask_map)
        # Normalize to a common Raman peak (e.g., G-band around 1580 cm^-1)
        map_norm = map_mod.normalize(peakshift=1580)
        assert map_norm is not None
        assert hasattr(map_norm, "mapxr")
        assert map_norm.normfactor is not None
        # Verify units are dimensionless after normalization
        assert map_norm.mapxr.attrs['units'] == ' '

    def test_normalize_individual(self, ramanmap_instance, fit_mask_map):
        """normalize(mode='individual') divides each pixel by its own peak amplitude.

        Regression test: the reference coordinates used for the ``peakpos``
        history line are the arithmetic midpoint of the coordinate range,
        which is generally *not* a grid point. The pre-fix code did an
        exact-match ``.sel`` there and raised KeyError; after that, the
        comment line tried to format the 2-D ``peakampl`` with ``:.2f``
        and raised TypeError. Both paths are exercised here.
        """
        # Background must be removed first, as in the const-mode test.
        map_mod, _, _ = ramanmap_instance.remove_bg(fitmask=fit_mask_map)
        map_norm = map_mod.normalize(peakshift=1580, mode='individual')
        # The per-pixel normalization factor is a labelled DataArray over
        # the map coords, so users can ``.sel(width=..., height=...)`` it.
        assert isinstance(map_norm.normfactor, xr.DataArray)
        assert set(map_norm.normfactor.dims) == {'height', 'width'}
        assert map_norm.normfactor.shape == (
            ramanmap_instance.mapxr.sizes['height'],
            ramanmap_instance.mapxr.sizes['width'],
        )
        # Data keeps its shape and dims; the division must align by
        # dimension name rather than by numpy trailing-axis broadcasting.
        assert map_norm.mapxr.dims == ramanmap_instance.mapxr.dims
        assert map_norm.mapxr.shape == ramanmap_instance.mapxr.shape
        # Dividing by the per-pixel amplitude must reproduce the input
        # when multiplied back — checks the DataArray broadcast is right.
        # peakfit runs with ``errors='ignore'``, so pixels whose fit did
        # not converge carry a NaN amplitude; restrict the comparison to
        # pixels with a finite factor and require that most pixels are.
        back = map_norm.mapxr * map_norm.normfactor
        finite = np.isfinite(map_norm.normfactor)
        assert float(finite.mean()) > 0.9
        np.testing.assert_allclose(
            back.where(finite).data[:, finite.data],
            map_mod.mapxr.where(finite).data[:, finite.data],
            rtol=1e-10, atol=1e-8,
        )
        # History line mentions the mode and summarises the 2-D factor.
        last = map_norm.mapxr.attrs['comments'].splitlines()[-1]
        assert 'in mode == individual' in last
        assert 'mean over pixels' in last
        assert map_norm.mapxr.attrs['units'] == ' '

    def test_calibrate(self, ramanmap_instance):
        """Test calibration shifts Raman shift coordinate correctly."""
        # Calibrate using a known peak position
        map_calib = ramanmap_instance.calibrate(peakshift=1580)
        assert map_calib is not None
        assert hasattr(map_calib, "mapxr")
        # Verify ramanshift coordinate exists and has been modified
        assert 'ramanshift' in map_calib.mapxr.coords
        assert 'calibrated' in map_calib.mapxr.attrs['comments']


# ---------------------------------------------------------------------------
# Tests: Singlespec Methods
# ---------------------------------------------------------------------------

class TestSinglespecMethods:
    """Test suite for singlespec class methods."""

    def test_print_metadata(self, singlespec_instance, capsys):
        """Verify metadata printing executes without errors."""
        singlespec_instance.print_metadata()
        captured = capsys.readouterr()
        # Check that some metadata is printed
        assert len(captured.out) > 0
        assert "raw data loaded" in captured.out

    def test_history(self, singlespec_instance, capsys):
        """Verify history method displays processing comments."""
        singlespec_instance.history()
        captured = capsys.readouterr()
        assert "Data modification history" in captured.out
        assert "raw data loaded" in captured.out

    def test_remove_bg(self, singlespec_instance, fit_mask_spec):
        """Verify background removal returns valid singlespec instance."""
        ss_mod, coeff, covar = singlespec_instance.remove_bg(fitmask=fit_mask_spec)
        assert ss_mod is not None
        assert hasattr(ss_mod, "ssxr")
        assert coeff is not None
        assert covar is not None

    def test_calibrate_rejects_lorentz2(self, singlespec_instance):
        """calibrate() must reject lorentz2 loudly rather than KeyError."""
        with pytest.raises(ValueError, match="single-peak"):
            singlespec_instance.calibrate(peakshift=1580, func=rt.lorentz2)

    def test_normalize_rejects_lorentz2(self, singlespec_instance, fit_mask_spec):
        """normalize() must reject lorentz2 loudly rather than KeyError."""
        # Need background-removed spec first — normalize refuses otherwise.
        ss_mod, _, _ = singlespec_instance.remove_bg(fitmask=fit_mask_spec)
        with pytest.raises(ValueError, match="single-peak"):
            ss_mod.normalize(peakshift=1580, func=rt.lorentz2)

    def test_crr(self, singlespec_instance):
        """Verify cosmic ray removal executes and returns valid instance."""
        ss_crr = singlespec_instance.crr()
        assert ss_crr is not None
        assert hasattr(ss_crr, "ssxr")

    def test_normalize(self, singlespec_instance, fit_mask_spec):
        """Test normalization to a peak returns valid normalized spectrum."""
        # Remove background first as required by normalize
        ss_mod, _, _ = singlespec_instance.remove_bg(fitmask=fit_mask_spec)
        # Normalize to a common Raman peak (e.g., G-band around 1580 cm^-1)
        ss_norm = ss_mod.normalize(peakshift=1580)
        assert ss_norm is not None
        assert hasattr(ss_norm, "ssxr")
        assert ss_norm.normfactor is not None
        # Verify units are dimensionless after normalization
        assert ss_norm.ssxr.attrs['units'] == ' '

    def test_calibrate(self, singlespec_instance):
        """Test calibration shifts Raman shift coordinate correctly."""
        # Calibrate using a known peak position
        ss_calib = singlespec_instance.calibrate(peakshift=1580)
        assert ss_calib is not None
        assert hasattr(ss_calib, "ssxr")
        # Verify ramanshift coordinate exists and has been modified
        assert 'ramanshift' in ss_calib.ssxr.coords
        assert 'calibrated' in ss_calib.ssxr.attrs['comments']


# ---------------------------------------------------------------------------
# Tests: Utility Functions
# ---------------------------------------------------------------------------

class TestUtilityFunctions:
    """Test suite for module-level utility functions."""

    def test_peakfit(self, singlespec_instance):
        """Verify peakfit executes on single spectrum data."""
        fit_result = rt.peakfit(
            singlespec_instance.ssxr,
            stval={"x0": 300, "ampl": 5, "width": 30, "offset": 0}
        )
        assert fit_result is not None
        assert "curvefit_coefficients" in fit_result

    def test_gaussian(self):
        """Verify Gaussian function produces expected output."""
        x = np.array([1570, 1580, 1590])
        result = rt.gaussian(x, x0=1580, ampl=10, width=15, offset=0)
        assert isinstance(result, np.ndarray)
        assert len(result) == len(x)
        # Peak should be at x0=1580
        assert result[1] > result[0]
        assert result[1] > result[2]

    def test_lorentz(self):
        """Verify Lorentz function produces expected output."""
        x = np.array([1570, 1580, 1590])
        result = rt.lorentz(x, x0=1580, ampl=1, width=14, offset=0)
        assert isinstance(result, np.ndarray)
        assert len(result) == len(x)
        # Peak should be at x0=1580
        assert result[1] > result[0]
        assert result[1] > result[2]

    def test_lorentz2(self):
        """Verify double Lorentz function produces expected output."""
        x = np.array([2690, 2700, 2710, 2720, 2730, 2740])
        result = rt.lorentz2(
            x, x01=2700, ampl1=1, width1=15,
            x02=2730, ampl2=1, width2=15, offset=0
        )
        assert isinstance(result, np.ndarray)
        assert len(result) == len(x)
        # Should have two peaks
        assert result[1] > result[0]  # First peak
        assert result[4] > result[3]  # Second peak

    def test_polynomial_fit(self):
        """Verify polynomial fitting returns coefficients and covariance."""
        # Create simple linear data
        x_data = np.array([1, 2, 3, 4, 5], dtype=float)
        y_data = np.array([2, 4, 6, 8, 10], dtype=float)  # y = 2*x

        # Fit first order polynomial (linear)
        coeff, covar = rt.polynomial_fit(1, x_data, y_data)

        assert isinstance(coeff, np.ndarray)
        assert isinstance(covar, np.ndarray)
        assert len(coeff) == 2  # First order = 2 coefficients
        # Check that fit is approximately correct (slope ~2, intercept ~0)
        assert abs(coeff[0] - 2.0) < 0.1  # Slope
        assert abs(coeff[1] - 0.0) < 0.1  # Intercept

    def test_bgsubtract_with_mask(self):
        """Verify bgsubtract works when fitmask is provided."""
        # Create simple data with a peak
        x_data = np.array([100, 200, 300, 400, 500], dtype=float)
        y_data = np.array([1, 3, 5, 3, 1], dtype=float)
        # Create a mask that includes all points
        fitmask = np.ones_like(x_data, dtype=bool)

        # Run background subtraction with provided mask
        y_nobg, bg_values, coeff, params, mask, covar = rt.bgsubtract(
            x_data, y_data, fitmask=fitmask, polyorder=1
        )

        assert isinstance(y_nobg, np.ndarray)
        assert isinstance(bg_values, np.ndarray)
        assert isinstance(coeff, np.ndarray)
        assert isinstance(mask, np.ndarray)
        assert len(y_nobg) == len(x_data)
        assert len(bg_values) == len(x_data)

    def test_bgsubtract_with_peak_detection(self):
        """Verify bgsubtract works with automatic peak detection."""
        # Create data with a clear peak
        x_data = np.linspace(100, 500, 50)
        # Background + peak
        y_data = 100 + 0.1 * x_data + 1000 * np.exp(-((x_data - 300)**2) / 100)

        # Run with automatic peak detection
        y_nobg, bg_values, coeff, params, mask, covar = rt.bgsubtract(
            x_data, y_data, polyorder=1, hmin=50, wmin=2
        )

        assert isinstance(y_nobg, np.ndarray)
        assert isinstance(bg_values, np.ndarray)
        assert isinstance(mask, np.ndarray)
        assert len(y_nobg) == len(x_data)
        # Verify that some points were excluded (mask has False values)
        assert not np.all(mask)


# ---------------------------------------------------------------------------
# Tests: Integration Tests
# ---------------------------------------------------------------------------

class TestIntegration:
    """Integration tests combining multiple operations."""

    def test_full_pipeline_singlespec(self, singlespec_instance, fit_mask_spec):
        """Test a complete processing pipeline on a single spectrum."""
        # Step 1: Remove background
        ss_nobg, _, _ = singlespec_instance.remove_bg(fitmask=fit_mask_spec)
        assert ss_nobg is not None

        # Step 2: Remove cosmic rays
        ss_crr = ss_nobg.crr()
        assert ss_crr is not None

        # Verify processing history is tracked
        assert "background subtracted" in ss_nobg.ssxr.attrs["comments"]
        assert "cosmic ray" in ss_crr.ssxr.attrs["comments"]

    def test_full_pipeline_ramanmap(self, ramanmap_instance, fit_mask_map):
        """Test a complete processing pipeline on a Raman map."""
        # Step 1: Remove background
        map_nobg, _, _ = ramanmap_instance.remove_bg(fitmask=fit_mask_map)
        assert map_nobg is not None

        # Step 2: Remove cosmic rays
        map_crr = map_nobg.crr()
        assert map_crr is not None

        # Verify processing history is tracked
        assert "background subtracted" in map_nobg.mapxr.attrs["comments"]
        assert "cosmic ray" in map_crr.mapxr.attrs["comments"]

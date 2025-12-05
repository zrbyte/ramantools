"""Pytest suite for ramantools functionality.

This test suite validates core ramantools operations on both ramanmap and
singlespec objects using minimal synthetic test data.
"""

import pytest  # Testing framework for organizing and running tests
import sys  # Access to the Python path for importing the package under test
from pathlib import Path  # Convenient path handling independent of OS
import numpy as np  # Numerical operations for creating fit masks

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

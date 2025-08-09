"""Test suite executing core ramantools functionality.

This script loads a sample single spectrum and a minimal Raman map from the
current directory, applies several operations from the `ramantools` module, and
logs any failures. A success message is written to `test.log` if all calls
execute without raising errors.
"""

import logging  # Built-in logging module for reporting test results
import sys  # Access to the Python path for importing the package under test
from pathlib import Path  # Convenient path handling independent of OS
import numpy as np  # Numerical operations for creating fit masks

# Ensure the pre-built package in ``build/lib`` is discovered before the source
# files. This avoids indentation issues in the development version while still
# allowing the tests to execute the library's functionality.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'build/lib'))
import ramantools as rt  # Import the module under test

# ---------------------------------------------------------------------------
# Set up logging to capture test outcomes in a dedicated file
# ---------------------------------------------------------------------------
LOG_PATH = Path(__file__).with_suffix(".log")  # Log file resides next to script
logging.basicConfig(
    filename=LOG_PATH,
    filemode="w",  # Start each test run with a fresh log file
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Keep track of any failing steps so we can summarise at the end
failures: list[str] = []


def log_step(name: str, func) -> None:
    """Run ``func`` and log its success or failure.

    Parameters
    ----------
    name:
        Human readable description of the tested action.
    func:
        A zero-argument callable executing the action to test.
    """
    try:
        func()  # Execute the provided action
        logger.info("%s passed", name)  # Record success in the log
    except Exception as exc:  # Catch any exception to keep subsequent tests running
        failures.append(name)  # Remember which step failed
        logger.error("%s failed: %s", name, exc)  # Provide the exception message


# Resolve data files relative to this script's directory
DATA_DIR = Path(__file__).parent
MAP_PATH = DATA_DIR / "map.txt"
MAP_INFO = DATA_DIR / "map_info.txt"
SPEC_PATH = DATA_DIR / "spectrum.txt"
SPEC_INFO = DATA_DIR / "spectrum_info.txt"

# ---------------------------------------------------------------------------
# Loading of test datasets
# ---------------------------------------------------------------------------
log_step(
    "load ramanmap",
    lambda: rt.ramanmap(MAP_PATH, MAP_INFO),  # Attempt to construct the map object
)
log_step(
    "load singlespec",
    lambda: rt.singlespec(SPEC_PATH, SPEC_INFO),  # Attempt to construct the spectrum object
)

# After construction we re-use the objects for subsequent operations
m = rt.ramanmap(MAP_PATH, MAP_INFO)
s = rt.singlespec(SPEC_PATH, SPEC_INFO)

# Create a simple fit mask consisting of True values for all spectral points.
# This bypasses automatic peak finding in background subtraction which could
# otherwise fail on the minimal synthetic datasets used here.
mask_map = np.ones_like(m.ramanshift, dtype=bool)
mask_spec = np.ones_like(s.ramanshift, dtype=bool)

# ---------------------------------------------------------------------------
# Raman map methods
# ---------------------------------------------------------------------------
log_step("ramanmap.print_metadata", m.print_metadata)
log_step(
    "ramanmap.remove_bg",
    lambda: m.remove_bg(fitmask=mask_map),  # Remove background using uniform mask
)
log_step(
    "ramanmap.normalize",
    lambda: m.normalize(peakshift=300),  # Normalize around the known peak
)
log_step(
    "ramanmap.calibrate",
    lambda: m.calibrate(peakshift=300),  # Calibrate spectrum using same peak
)
log_step("ramanmap.crr", m.crr)  # Cosmic ray removal using rolling window

# ---------------------------------------------------------------------------
# Single spectrum methods
# ---------------------------------------------------------------------------
log_step("singlespec.print_metadata", s.print_metadata)
log_step(
    "singlespec.remove_bg",
    lambda: s.remove_bg(fitmask=mask_spec),  # Background removal for single spectrum
)
log_step(
    "singlespec.normalize",
    lambda: s.normalize(peakshift=300),  # Normalize single spectrum
)
log_step(
    "singlespec.calibrate",
    lambda: s.calibrate(peakshift=300),  # Calibrate single spectrum
)
log_step("singlespec.crr", s.crr)

# ---------------------------------------------------------------------------
# Module level function `peakfit` using the single spectrum data
# ---------------------------------------------------------------------------
log_step(
    "peakfit",
    lambda: rt.peakfit(
        s.ssxr,
        stval={"x0": 300, "ampl": 5, "width": 30, "offset": 0},
    ),
)

# ---------------------------------------------------------------------------
# Summarise overall result in the log file
# ---------------------------------------------------------------------------
if failures:
    logger.error("Testing completed with failures: %s", ", ".join(failures))
else:
    logger.info("ALL CLEAR")  # Everything executed without raising an exception

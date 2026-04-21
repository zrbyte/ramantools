"""Pipeline checksum safety net for the Phase 1 refactor.

Runs the full processing pipeline (remove_bg → calibrate → normalize →
crr) on both ``ramanmap`` and ``singlespec`` fixtures and compares every
numeric output against a pickled baseline produced by
``testing/generate_baseline.py``. Any accidental behavior drift during
the refactor trips these assertions with tight tolerances.

When baseline files or real-data fixtures are missing, the tests *skip*
rather than fail — so CI without the private corpus still runs cleanly.

To (deliberately) update the baseline, run::

    python testing/generate_baseline.py

and commit the regenerated ``.npz`` files alongside the code change.
"""

import sys  # For inserting the repo root onto sys.path
import hashlib  # SHA256 fingerprinting, matches generate_baseline.py
from pathlib import Path  # Cross-platform path handling
import numpy as np  # Array comparison and I/O
import pytest  # Skip / fixture primitives

# Make the in-tree ramantools importable without `pip install -e .`.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import ramantools as rt  # noqa: E402 — sys.path must be set first


# Real-data fixtures live outside the repo (gitignored); the baseline
# .npz files *are* committed. If either side is missing the checksum
# test has nothing to compare against and must skip.
DATA_DIR = ROOT / "test-data"
BASELINE_DIR = Path(__file__).parent / "baseline"
MAP_BASELINE = BASELINE_DIR / "pipeline_map.npz"
SPEC_BASELINE = BASELINE_DIR / "pipeline_spec.npz"

# Tight tolerances: pipeline numerics are fully deterministic for fixed
# numpy / scipy versions. Only the ``normalize`` / ``calibrate`` paths
# involve curve_fit, and those converge bit-exactly from identical
# starting guesses on identical inputs. Loosen these ONLY if a
# deliberate algorithm change lands; at that point regenerate the
# baseline instead of relaxing the tolerance.
RTOL = 1e-10
ATOL = 1e-12


# --- skip helpers -----------------------------------------------------------

def _skip_if_data_missing() -> None:
    """Skip if the real-data fixtures are not present in this checkout."""
    if not DATA_DIR.exists():
        pytest.skip(
            f"real-data fixtures at {DATA_DIR} not present — "
            f"skipping pipeline checksum tests"
        )


def _skip_if_baseline_missing(path: Path) -> None:
    """Skip if the .npz baseline is absent (first-time setup on a new clone)."""
    if not path.exists():
        pytest.skip(
            f"baseline file {path.name} missing under {BASELINE_DIR} — "
            f"run `python testing/generate_baseline.py` first"
        )


def _assert_fingerprint_matches(actual: np.ndarray, expected: np.lib.npyio.NpzFile,
                                prefix: str) -> None:
    """Assert that ``actual`` matches a packed fingerprint in ``expected``.

    Mirrors the encoding in ``generate_baseline._fingerprint`` /
    ``_pack_fingerprint``. The SHA256 is the primary guard; the stats
    are re-asserted with tight ``np.isclose`` tolerances so a mismatch
    produces a readable diagnostic (the hash alone tells you it broke
    but not *how*).
    """
    # Recompute the same fingerprint structure on the actual array.
    flat = np.ascontiguousarray(actual)
    actual_hash = hashlib.sha256(flat.tobytes()).hexdigest()
    if np.issubdtype(flat.dtype, np.floating):
        finite = flat[~np.isnan(flat)]
        nan_count = int(np.isnan(flat).sum())
    else:
        finite = flat
        nan_count = 0

    # Shape / dtype first — a mismatch here explains any hash failure.
    assert tuple(expected[f"{prefix}__shape"]) == flat.shape, (
        f"{prefix}: shape drifted {flat.shape} vs baseline {tuple(expected[f'{prefix}__shape'])}"
    )
    assert str(expected[f"{prefix}__dtype"]) == str(flat.dtype), (
        f"{prefix}: dtype drifted {flat.dtype} vs baseline {expected[f'{prefix}__dtype']}"
    )
    assert int(expected[f"{prefix}__nan_count"]) == nan_count, (
        f"{prefix}: NaN count drifted {nan_count} vs baseline {int(expected[f'{prefix}__nan_count'])}"
    )

    # Summary statistics — small numerical wiggle is permitted here because
    # float64 sums aren't strictly associative. The hash is the strict check.
    for key in ("mean", "std", "min", "max", "sum"):
        baseline_val = float(expected[f"{prefix}__{key}"])
        actual_val = float(getattr(finite, key)()) if finite.size else 0.0
        assert np.isclose(actual_val, baseline_val, rtol=1e-10, atol=1e-12), (
            f"{prefix}: {key} drifted — actual={actual_val}, baseline={baseline_val}"
        )

    # Finally the hash. A mismatch here with stats all matching means
    # some tiny per-element change canceled out in the aggregates —
    # that is exactly the kind of subtle drift this safety net exists
    # to catch during refactoring.
    expected_hash = str(expected[f"{prefix}__sha256"])
    assert actual_hash == expected_hash, (
        f"{prefix}: SHA256 mismatch despite matching stats — subtle per-element drift. "
        f"actual={actual_hash} baseline={expected_hash}"
    )


# --- tests ------------------------------------------------------------------

def test_pipeline_checksum_ramanmap():
    """Full ramanmap pipeline must bit-stably match the committed baseline.

    Any drift in remove_bg / calibrate / normalize / crr — whether from a
    refactor, a dependency upgrade, or accidental parameter change — will
    surface here. The same pipeline is executed by
    ``testing/generate_baseline.py`` when regenerating, so the two paths
    cannot diverge unintentionally.
    """
    _skip_if_data_missing()
    _skip_if_baseline_missing(MAP_BASELINE)

    # Replay the exact same pipeline used at baseline generation time.
    specmap = rt.ramanmap(DATA_DIR / "specmap.txt", DATA_DIR / "specmap_metadata.txt")
    fit_mask = np.ones_like(specmap.ramanshift, dtype=bool)
    nobg, coeff, _ = specmap.remove_bg(fitmask=fit_mask)
    calibrated = nobg.calibrate(peakshift=1580)
    normalized = calibrated.normalize(peakshift=1580)
    cleaned = normalized.crr(cutoff=2, window=2)

    expected = np.load(MAP_BASELINE, allow_pickle=False)
    # Fingerprint check handles NaNs explicitly via the nan_count field
    # (the crr edge-NaN band is fixed-width and identical on both sides).
    _assert_fingerprint_matches(cleaned.mapxr.values, expected, prefix="values")
    np.testing.assert_allclose(
        cleaned.mapxr.ramanshift.values,
        expected["ramanshift"],
        rtol=RTOL, atol=ATOL,
        err_msg="ramanmap ramanshift coord drifted after calibrate",
    )
    np.testing.assert_allclose(
        coeff,
        expected["bg_coeff"],
        rtol=RTOL, atol=ATOL,
        err_msg="ramanmap bg polynomial coefficients drifted",
    )
    np.testing.assert_allclose(
        np.asarray(normalized.normfactor),
        expected["normfactor"],
        rtol=RTOL, atol=ATOL,
        err_msg="ramanmap normfactor drifted",
    )


def test_pipeline_checksum_singlespec():
    """Full singlespec pipeline must bit-stably match the committed baseline.

    Mirror of :func:`test_pipeline_checksum_ramanmap` against the 1-D fixture.
    """
    _skip_if_data_missing()
    _skip_if_baseline_missing(SPEC_BASELINE)

    spec = rt.singlespec(DATA_DIR / "singlespec.txt", DATA_DIR / "singlespec_metadata.txt")
    fit_mask = np.ones_like(spec.ramanshift, dtype=bool)
    nobg, coeff, _ = spec.remove_bg(fitmask=fit_mask)
    calibrated = nobg.calibrate(peakshift=1580)
    normalized = calibrated.normalize(peakshift=1580)
    cleaned = normalized.crr(cutoff=2, window=2)

    expected = np.load(SPEC_BASELINE, allow_pickle=False)
    # Same fingerprint approach as for the map — keeps the committed
    # baseline tiny while still catching any numeric drift.
    _assert_fingerprint_matches(cleaned.ssxr.values, expected, prefix="values")
    np.testing.assert_allclose(
        cleaned.ssxr.ramanshift.values,
        expected["ramanshift"],
        rtol=RTOL, atol=ATOL,
        err_msg="singlespec ramanshift coord drifted after calibrate",
    )
    np.testing.assert_allclose(
        coeff,
        expected["bg_coeff"],
        rtol=RTOL, atol=ATOL,
        err_msg="singlespec bg polynomial coefficients drifted",
    )
    np.testing.assert_allclose(
        np.asarray(normalized.normfactor),
        expected["normfactor"],
        rtol=RTOL, atol=ATOL,
        err_msg="singlespec normfactor drifted",
    )

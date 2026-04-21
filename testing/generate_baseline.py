"""Regenerate the pipeline-checksum baselines for ``test_pipeline_baseline.py``.

Usage::

    python testing/generate_baseline.py

Produces ``testing/baseline/pipeline_map.npz`` and
``testing/baseline/pipeline_spec.npz`` from the real-data fixtures under
``test-data/``. The fixtures themselves are gitignored; the resulting
baseline files are small and get committed to the repo so CI / future
refactors have something deterministic to diff against.

When to regenerate
------------------
* **Never** run just because a test fails — investigate first. Drift is
  exactly what the checksum is designed to catch.
* Run only after a **deliberate, vetted** numeric change (e.g. algorithm
  fix or intended update); commit the regenerated ``.npz`` files in the
  same PR as the code change so reviewers can audit both together.
"""

import sys  # To insert the repo root onto sys.path
import hashlib  # SHA256 fingerprinting of large arrays
from pathlib import Path  # Filesystem path handling
import numpy as np  # Array I/O (savez / load)

# Make the in-tree ramantools importable without `pip install -e .`.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import ramantools as rt  # noqa: E402 — import after sys.path tweak is intentional

# Absolute paths to data fixtures and output location.
DATA_DIR = ROOT / "test-data"
BASELINE_DIR = Path(__file__).parent / "baseline"


def _fingerprint(arr: np.ndarray) -> dict:
    """Compact, drift-detecting fingerprint for a numeric array.

    Full-array baselines would balloon past 60 MB for a typical Raman
    map; committing that to git defeats the purpose of a lightweight
    safety net. Instead we capture:

    * ``sha256`` — exact-match check over the raw bytes. Any drift,
      anywhere in the array, flips the hash. This is the primary guard.
    * ``mean / std / min / max / sum / nan_count`` — human-readable
      summary statistics. Cheap, diagnostic if the hash ever mismatches
      (tells the developer *how* the array changed, not just that it did).
    * ``shape / dtype`` — guards against silent shape or precision drift.

    The hash is computed on a C-contiguous copy so platform-specific
    memory layout cannot perturb it; the dtype field locks precision.
    """
    # Force contiguous + deterministic byte layout before hashing so two
    # arrays with identical *values* but different strides still hash equal.
    flat = np.ascontiguousarray(arr)
    # np.isnan only applies to floats; guard with issubdtype to also
    # support integer arrays where nan_count is trivially 0.
    if np.issubdtype(flat.dtype, np.floating):
        # np.nansum/nanmean ignore NaNs to produce stable summary stats
        # even when crr leaves edge NaNs in the output.
        finite_arr = flat[~np.isnan(flat)]
        nan_count = int(np.isnan(flat).sum())
    else:
        finite_arr = flat
        nan_count = 0
    return {
        "sha256": hashlib.sha256(flat.tobytes()).hexdigest(),
        "shape": np.asarray(flat.shape, dtype=np.int64),
        "dtype": str(flat.dtype),
        "mean": float(finite_arr.mean()) if finite_arr.size else 0.0,
        "std": float(finite_arr.std()) if finite_arr.size else 0.0,
        "min": float(finite_arr.min()) if finite_arr.size else 0.0,
        "max": float(finite_arr.max()) if finite_arr.size else 0.0,
        "sum": float(finite_arr.sum()) if finite_arr.size else 0.0,
        "nan_count": nan_count,
    }


def _pack_fingerprint(prefix: str, fp: dict) -> dict:
    """Flatten a fingerprint dict under a keyed prefix for np.savez.

    ``np.savez`` only accepts array-like values, so string fields are
    stored as 0-d object / unicode arrays and scalars are boxed.
    """
    out = {}
    for k, v in fp.items():
        # savez wants array-likes; wrap scalars/strings in np.array.
        out[f"{prefix}__{k}"] = np.asarray(v)
    return out


def build_map_baseline() -> dict:
    """Run the full ramanmap pipeline once and return checksum arrays.

    Pipeline order matches CLAUDE.md:
    load → remove_bg → calibrate → normalize → crr.

    The ``fitmask=ones`` (no peaks excluded) mirrors what the existing
    test suite already uses for remove_bg in ``test_remove_bg`` /
    ``test_normalize`` so numeric behavior is identical to those paths.
    """
    # Load map with info file (same as the module-scoped fixture).
    specmap = rt.ramanmap(DATA_DIR / "specmap.txt", DATA_DIR / "specmap_metadata.txt")
    # Uniform fitmask bypasses automatic peak detection — deterministic
    # and independent of scipy.signal.find_peaks version quirks.
    fit_mask = np.ones_like(specmap.ramanshift, dtype=bool)
    nobg, coeff, _ = specmap.remove_bg(fitmask=fit_mask)
    # Calibrate using the G-band as the reference peak.
    calibrated = nobg.calibrate(peakshift=1580)
    # Normalize to the same peak (requires bg already removed).
    normalized = calibrated.normalize(peakshift=1580)
    # Finally remove cosmic rays with the default parameters.
    cleaned = normalized.crr(cutoff=2, window=2)
    # Pack: fingerprint of the 3-D values (via hash+stats so the .npz
    # stays small), plus the full small-ish ramanshift coord, bg coeffs
    # and scalar normfactor (all tiny, safe to store verbatim).
    out = {
        "ramanshift": cleaned.mapxr.ramanshift.values,
        "bg_coeff": np.asarray(coeff),
        "normfactor": np.asarray(normalized.normfactor),
    }
    out.update(_pack_fingerprint("values", _fingerprint(cleaned.mapxr.values)))
    return out


def build_spec_baseline() -> dict:
    """Run the full singlespec pipeline once and return checksum arrays.

    Identical staging to :func:`build_map_baseline` but against the 1-D
    singlespec fixture.
    """
    spec = rt.singlespec(DATA_DIR / "singlespec.txt", DATA_DIR / "singlespec_metadata.txt")
    # Same fitmask strategy as the map baseline for consistency.
    fit_mask = np.ones_like(spec.ramanshift, dtype=bool)
    nobg, coeff, _ = spec.remove_bg(fitmask=fit_mask)
    calibrated = nobg.calibrate(peakshift=1580)
    normalized = calibrated.normalize(peakshift=1580)
    cleaned = normalized.crr(cutoff=2, window=2)
    # Spectrum values are small (~1600 points) but we still use the
    # fingerprint format for consistency with the map baseline path.
    out = {
        "ramanshift": cleaned.ssxr.ramanshift.values,
        "bg_coeff": np.asarray(coeff),
        "normfactor": np.asarray(normalized.normfactor),
    }
    out.update(_pack_fingerprint("values", _fingerprint(cleaned.ssxr.values)))
    return out


def main() -> None:
    """Regenerate both baselines and write them to ``testing/baseline/``."""
    # Guard: real-data fixtures are gitignored; fail loudly if absent so
    # a contributor without them knows exactly what's wrong.
    if not DATA_DIR.exists():
        raise SystemExit(
            f"Cannot regenerate baselines: {DATA_DIR} is missing. "
            f"The real Raman fixtures live outside the repo (gitignored); "
            f"obtain them from the project owner."
        )
    # mkdir is idempotent — creates the dir on first run, no-op otherwise.
    BASELINE_DIR.mkdir(exist_ok=True)
    # Save each baseline to its own .npz file so the two pipelines can
    # be audited independently.
    np.savez(BASELINE_DIR / "pipeline_map.npz", **build_map_baseline())
    np.savez(BASELINE_DIR / "pipeline_spec.npz", **build_spec_baseline())
    print(f"Wrote baselines to {BASELINE_DIR}")


if __name__ == "__main__":
    main()

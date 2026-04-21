"""Backward-compatibility safety net for the Phase 1 refactor.

Asserts every public name in ``ramantools.__all__`` resolves via BOTH:
  1. ``import ramantools`` as ``ramantools.<name>``
  2. ``from ramantools.ramantools import <name>``

Also locks the call signatures of public functions / constructors so the
refactor cannot silently rename / reorder parameters that downstream
notebooks may pass positionally.

These tests must pass BEFORE the refactor lands (baseline) and AFTER it
lands (confirms BC preserved). Adding the tests now — before touching
production code — is the whole point of "safety net first".
"""

import sys  # For inserting the repo root onto sys.path
from pathlib import Path  # Path handling independent of OS
import inspect  # To introspect function signatures without calling them

# Ensure the in-tree ramantools is importable without relying on the
# user having run `pip install -e .`.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import ramantools  # Package-level import path
from ramantools import ramantools as rt_module  # Legacy deep import path


# Names expected on the public package surface. Sourced explicitly rather
# than from __all__ so this test is the source of truth for "what is
# public"; drift here reflects a deliberate change to the public API.
EXPECTED_PUBLIC = [
    "ramanmap",
    "singlespec",
    "gaussian",
    "lorentz",
    "lorentz2",
    "polynomial_fit",
    "bgsubtract",
    "peakfit",
]


def test_all_covers_expected_public():
    """``__all__`` must list every public name we depend on.

    Failing here means either:
      - __all__ was narrowed (API removal) — update EXPECTED_PUBLIC only
        if the removal is intentional and documented.
      - A new public name was added — add it to EXPECTED_PUBLIC so future
        refactors cannot drop it without tripping this check.
    """
    # Using >= (superset) instead of == to allow __all__ to grow with
    # additions; only a regression (removal) is caught.
    assert set(ramantools.__all__) >= set(EXPECTED_PUBLIC)


def test_public_names_on_package():
    """Every expected public name is accessible as ``ramantools.<name>``."""
    for name in EXPECTED_PUBLIC:
        assert hasattr(ramantools, name), f"ramantools.{name} missing from package surface"


def test_public_names_on_submodule():
    """Every expected public name is accessible as ``ramantools.ramantools.<name>``.

    This is the legacy deep-import path the original single-file module
    exposed. The refactor (module split) must preserve it by keeping
    ``ramantools/ramantools.py`` as a re-export shim.
    """
    for name in EXPECTED_PUBLIC:
        assert hasattr(rt_module, name), (
            f"ramantools.ramantools.{name} missing — the submodule shim must re-export it"
        )


def test_public_names_identical_across_paths():
    """Both import paths must resolve to the *same* object (``is`` identity).

    Notebooks often mix ``import ramantools`` and ``from ramantools.ramantools
    import X`` styles; distinct objects would cause surprising ``isinstance``
    mismatches after the refactor. ``is`` catches the case where the shim
    accidentally creates a wrapper instead of re-exporting.
    """
    for name in EXPECTED_PUBLIC:
        assert getattr(ramantools, name) is getattr(rt_module, name), (
            f"{name} is a different object via the two import paths"
        )


def test_version_string_exposed():
    """``ramantools.__version__`` remains a non-empty string on the package root."""
    assert isinstance(ramantools.__version__, str)
    assert ramantools.__version__  # not empty


# --- class-surface checks ---------------------------------------------------

# Every method documented as public on ``ramanmap``. The refactor may
# deduplicate these internally (e.g. via a shared helper or mixin), but
# the bound-method names on the class must remain.
RAMANMAP_METHODS = [
    "history",
    "print_metadata",
    "plotspec",
    "remove_bg",
    "calibrate",
    "normalize",
    "crr",
    "peakmask",
]

# Every method documented as public on ``singlespec``. Shorter than the
# map because spectra have no plotspec / peakmask.
SINGLESPEC_METHODS = [
    "history",
    "print_metadata",
    "remove_bg",
    "calibrate",
    "normalize",
    "crr",
]


def test_ramanmap_methods_present():
    """All documented public methods on ``ramanmap`` stay callable."""
    for name in RAMANMAP_METHODS:
        method = getattr(ramantools.ramanmap, name, None)
        assert callable(method), f"ramanmap.{name} missing or not callable"


def test_singlespec_methods_present():
    """All documented public methods on ``singlespec`` stay callable."""
    for name in SINGLESPEC_METHODS:
        method = getattr(ramantools.singlespec, name, None)
        assert callable(method), f"singlespec.{name} missing or not callable"


# --- signature locks --------------------------------------------------------

def test_shape_function_signatures():
    """Shape functions (gaussian, lorentz, lorentz2) keep parameter names/order.

    Downstream users pass these positionally to ``peakfit`` / ``scipy.optimize``
    internals via ``curvefit``. Reordering would silently produce wrong fits.
    """
    # gaussian(x, x0=1580, ampl=10, width=15, offset=0)
    assert list(inspect.signature(ramantools.gaussian).parameters) == [
        "x", "x0", "ampl", "width", "offset",
    ]
    # lorentz(x, x0=1580, ampl=1, width=14, offset=0)
    assert list(inspect.signature(ramantools.lorentz).parameters) == [
        "x", "x0", "ampl", "width", "offset",
    ]
    # lorentz2(x, x01=2700, ampl1=1, width1=15, x02=2730, ampl2=1, width2=15, offset=0)
    assert list(inspect.signature(ramantools.lorentz2).parameters) == [
        "x", "x01", "ampl1", "width1", "x02", "ampl2", "width2", "offset",
    ]


def test_utility_function_signatures():
    """Core utility functions keep their leading positional parameters."""
    # polynomial_fit(order, x_data, y_data) — full signature locked.
    assert list(inspect.signature(ramantools.polynomial_fit).parameters) == [
        "order", "x_data", "y_data",
    ]
    # bgsubtract(x_data, y_data, ...) — lock leading two; keyword options
    # are free to grow in later phases.
    bg_params = list(inspect.signature(ramantools.bgsubtract).parameters)
    assert bg_params[:2] == ["x_data", "y_data"]
    # peakfit(xrobj, func=lorentz, ...) — leading positional is the xarray.
    pf_params = list(inspect.signature(ramantools.peakfit).parameters)
    assert pf_params[:1] == ["xrobj"]


def test_class_init_signatures():
    """Constructors ``ramanmap(map_path, info_path=None)`` and
    ``singlespec(spec_path, info_path=None)`` both require the data path
    positionally and allow omitting the info file (default ``None``).

    The ``info_path=None`` default was introduced earlier to support
    header-only loading; the Phase 1 refactor must preserve it.
    """
    params = list(inspect.signature(ramantools.ramanmap).parameters.values())
    assert params[0].name == "map_path"
    assert params[1].name == "info_path"
    assert params[1].default is None

    params = list(inspect.signature(ramantools.singlespec).parameters.values())
    assert params[0].name == "spec_path"
    assert params[1].name == "info_path"
    assert params[1].default is None

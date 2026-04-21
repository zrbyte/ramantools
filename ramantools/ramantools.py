"""Re-export shim — module split layout as of Phase 1 refactor.

The original ``ramantools.ramantools`` module was a single ~1955-line
file holding every class and utility. Phase 1 split it into focused
submodules (``_helpers``, ``_witec``, ``_fitting``, ``map``, ``spec``).
This shim preserves the legacy ``from ramantools.ramantools import X``
import path by re-exporting every public (and previously module-level
private) name the old file exposed.

Downstream behaviour is unchanged: the names here resolve to the exact
same objects as ``ramantools.X`` (the package ``__init__`` imports from
this shim).
"""

# Public API — these are the names listed in ``ramantools.__all__``.
from ._fitting import (
        gaussian,
        lorentz,
        lorentz2,
        polynomial_fit,
        bgsubtract,
        peakfit,
)
from .map import ramanmap
from .spec import singlespec

# Module-level internals preserved for compatibility — legacy notebooks
# occasionally poked at these (e.g. ``_midpoint`` in custom extensions).
# Keeping the re-exports costs nothing and prevents subtle breakage.
from ._helpers import _midpoint, _NO_INFO_STR, _NO_INFO_NUM
from ._witec import _parse_witec_datafile_header, _graphname_to_name

__all__ = [
        "ramanmap",
        "singlespec",
        "gaussian",
        "lorentz",
        "lorentz2",
        "polynomial_fit",
        "bgsubtract",
        "peakfit",
]

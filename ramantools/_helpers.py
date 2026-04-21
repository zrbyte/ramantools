"""Shared internal helpers for the ramantools package.

Houses the ``_midpoint`` utility used to compute the center of coordinate
arrays and the ``_NO_INFO_*`` sentinels used by the no-info-file load
path. Moved out of ``ramantools/ramantools.py`` during the Phase 1
refactor so both the map and spec modules can import them without a
circular-import risk.
"""
import numpy as np  # type: ignore

def _midpoint(values: np.ndarray) -> float:
        """Return the midpoint of an array of values.

        This function centralizes the midpoint calculation used throughout the
        module to determine the center of width and height coordinates. Having
        a single helper avoids repeated formulae and clarifies intent.
        """
        vmin = np.min(values)
        vmax = np.max(values)
        return (vmax - vmin) / 2 + vmin


# Sentinels used when no info file is available. Strings use "N/A" so printed
# metadata stays self-describing; numeric fields use NaN so downstream math
# that touches them propagates NaN rather than a silent zero.
_NO_INFO_STR = 'N/A'
_NO_INFO_NUM = float('nan')


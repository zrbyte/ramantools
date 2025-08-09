from .version import __version__  # Re-export package version
from .ramantools import (
        ramanmap,
        singlespec,
        gaussian,
        lorentz,
        lorentz2,
        polynomial_fit,
        bgsubtract,
        peakfit,
)

# Explicitly define the public interface for clarity
__all__ = [
        "__version__",
        "ramanmap",
        "singlespec",
        "gaussian",
        "lorentz",
        "lorentz2",
        "polynomial_fit",
        "bgsubtract",
        "peakfit",
]

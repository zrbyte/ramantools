"""Named constants used across the ramantools pipeline.

Pulling these out of the algorithm code makes intent obvious at the call
site and gives a single place to change each threshold. The names are
chosen to carry units explicitly (``_CM`` suffix for inverse-centimetre
values) so future readers don't have to reverse-engineer them.

All constants here match the pre-refactor magic numbers byte-for-byte;
the Phase 1 refactor does not change numeric behaviour.
"""

# Notch-filter cutoff in cm^-1. Frequencies below this are dominated by
# the Witec notch filter (Rayleigh rejection) and are forced False in
# the bgsubtract fitmask so the polynomial background fit doesn't chase
# the notch dip. Empirically ~80 cm^-1; 95 leaves a small safety margin.
NOTCH_CUTOFF_CM = 95

# Half-width in cm^-1 of the window cropped around the reference peak
# during ``normalize()``. The crop is ``[peakshift - X, peakshift + X]``
# so the peakfit has enough baseline on both sides.
NORMALIZE_CROP_REGION_CM = 100

# Background "is remove_bg done?" sanity threshold used by ``normalize()``.
# If the average of the cropped endpoint intensities exceeds this number
# in the reference spectrum, normalize assumes the background was NOT
# subtracted (because a clean baseline should be near zero) and refuses
# to proceed.
NORMALIZE_BG_THRESHOLD = 500

# Half-width in cm^-1 of the window cropped around the reference peak
# during ``calibrate()``. Same logic as NORMALIZE_CROP_REGION_CM but
# named separately so the two can diverge if tuning warrants it later.
CALIBRATE_FITRANGE_CM = 100

# Multiplicative factor applied to the fitted peak width when
# ``peakmask()`` defines the "vicinity of the peak" crop. Larger than
# bgsubtract's default ``exclusion_factor`` (6) because peakmask uses
# the crop for a mean-value measurement, not just fit exclusion — it
# wants to include the full peak tails.
PEAKMASK_VICINITY_FACTOR = 5

# ``maxfev`` passed to scipy.optimize.curve_fit via xarray.curvefit
# inside ``peakfit()``. Caps the number of objective-function calls the
# Levenberg-Marquardt solver can make before giving up.
PEAKFIT_MAXFEV = 1000

"""MIRI imaging flux-calibration uncertainties from Gordon et al. (2025).

Reference
---------
Gordon, K. D., et al. 2025, AJ, 169, 6
"Calibration Factors for MIRI Imaging and Coronagraphy"
https://doi.org/10.3847/1538-3881/ad8cd4

What the JWST pipeline already does (verified in jwst 2.0.1)
-------------------------------------------------------------
The JWST ``photom`` step (``jwst.photom.PhotomStep``) applies the following
to every MIRI imaging exposure before Eureka Stage 3 ever sees the data:

1. **Calibration factor (CF, Table 8 column A)** — converts DN/s pixel⁻¹ to
   MJy sr⁻¹.  The reference-file row is selected by matching both FILTER and
   SUBARRAY, so the **subarray-dependent throughput correction (DSA, Table 7)**
   is automatically embedded in the per-subarray CF value stored in CRDS.
2. **Time-dependent response loss (Table 8 columns B and τ)** — the exponential
   decay is divided out of the conversion factor when
   ``apply_time_correction=True`` (the default).

As a result, the calibrated pixel values and their ERR/VAR arrays are already
in physical units and already corrected for subarray throughput and temporal
response loss.

What is NOT done by the pipeline
---------------------------------
The pipeline treats the calibration factor as exact — it multiplies the flux
and errors by CF but adds **no uncertainty on CF itself**.  Gordon et al. (2025)
explicitly state (Sec. 3.5, and the Table 8 notes) that σ(CF) and σ(repeat)
must be combined in quadrature with the per-pixel statistical uncertainties
reported by the pipeline.

This module provides σ(CF) and σ(repeat) per filter so that Eureka can inflate
``aperr`` after aperture photometry extraction.

Notes on σ(repeat)
-------------------
σ(repeat) (Table 8, final column) quantifies the repeatability of
observations of point sources taken with a standard four-point dither pattern.
JWST time-series photometry observations (TSO) use a fixed pointing with no
dither, so σ(repeat) does not directly characterise TSO noise and should not
be applied by default.  It is provided here for users who want a conservative
noise floor (``phot_calib_repeat = True`` in the Stage 3 ECF).
"""

import math
import warnings

__all__ = ["GORDON2025_SIGMA_CF", "GORDON2025_SIGMA_REPEAT",
           "get_calib_unc_frac"]

# ---------------------------------------------------------------------------
# Table 8 of Gordon et al. (2025) — MIRI imaging filters only.
# Values are fractional (not percent): e.g. 0.0037 = 0.37 %.
# ---------------------------------------------------------------------------

#: Standard deviation of the calibration factor, σ(CF), as a fraction of CF.
#: Source: Gordon et al. (2025), AJ, 169, 6, Table 8, column "σ(CF) (%)".
GORDON2025_SIGMA_CF = {
    'F560W':  0.0037,
    'F770W':  0.0037,
    'F1000W': 0.0032,
    'F1130W': 0.0048,
    'F1280W': 0.0044,
    'FND':    0.0095,
    'F1500W': 0.0048,
    'F1800W': 0.0059,
    'F2100W': 0.0066,
    'F2550W': 0.0098,
    # F2550WR is the rerun filter; treat identically to F2550W.
    'F2550WR': 0.0098,
}

#: Repeatability, σ(repeat), as a fraction.
#: Source: Gordon et al. (2025), AJ, 169, 6, Table 8, column "σ(repeat) (%)".
#: FND has no measured repeatability (marked "L" in the paper); set to None.
GORDON2025_SIGMA_REPEAT = {
    'F560W':   0.0027,
    'F770W':   0.0025,
    'F1000W':  0.0008,
    'F1130W':  0.0016,
    'F1280W':  0.0020,
    'FND':     None,
    'F1500W':  0.0045,
    'F1800W':  0.0060,
    'F2100W':  0.0059,
    'F2550W':  0.0120,
    'F2550WR': 0.0120,
}


def get_calib_unc_frac(filter_name, include_repeat=False):
    """Return the fractional flux-calibration uncertainty for a MIRI filter.

    The returned value ``frac`` is used to inflate aperture-photometry errors
    in quadrature::

        aperr_total = sqrt(aperr_pipeline² + (frac × aplev)²)

    Parameters
    ----------
    filter_name : str
        MIRI imaging filter name as stored in the FITS primary header keyword
        ``FILTER`` (e.g. ``'F1500W'``, ``'F2550WR'``).  Case-insensitive.
    include_repeat : bool, optional
        If ``True``, combine σ(CF) and σ(repeat) in quadrature.  Defaults to
        ``False`` because σ(repeat) was measured for 4-point-dithered
        calibration-star observations and does not directly apply to fixed-
        pointing TSO data (see module docstring).

    Returns
    -------
    frac : float
        Fractional uncertainty (dimensionless).  Multiply by 100 to get
        percent.

    Raises
    ------
    ValueError
        If the filter name is not in the Gordon et al. (2025) Table 8 data.
    """
    key = filter_name.upper()

    if key not in GORDON2025_SIGMA_CF:
        raise ValueError(
            f"MIRI filter '{filter_name}' not found in Gordon et al. (2025) "
            f"Table 8.  Known filters: {sorted(GORDON2025_SIGMA_CF)}."
        )

    sigma_cf = GORDON2025_SIGMA_CF[key]

    if not include_repeat:
        return sigma_cf

    sigma_rep = GORDON2025_SIGMA_REPEAT[key]

    if sigma_rep is None:
        warnings.warn(
            f"No σ(repeat) measurement for filter '{filter_name}' in Gordon "
            "et al. (2025) Table 8 (marked 'L' in the paper).  Only σ(CF) "
            "will be used.",
            UserWarning,
            stacklevel=2,
        )
        return sigma_cf

    return math.sqrt(sigma_cf**2 + sigma_rep**2)

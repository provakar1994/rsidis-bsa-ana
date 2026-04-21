"""
Configurable event-selection cuts.

All cut thresholds are read from CutsConfig (loaded from YAML).
Nothing is hardcoded in the analysis code.

Branch names
------------
The constants below are the physical branch names in the skimmed ROOT T-tree
(using '_' separators, not '.').  They are defined here as named constants so
they're easy to audit or update if the tree layout changes.

Verified against: skimmed_coin_replay_production_{run}_-1.root (pass0p1)

IHWP correction
---------------
The In-Hole Wave Plate (IHWP) is inserted between helicity states to cancel
systematic effects.  When IHWP is IN, the physical beam helicity is flipped
relative to the DAQ-recorded helicity.

    effective_hel = raw_hel × (+1 if IHWP == "OUT" else −1)

This effective helicity is used for the helicity-split histograms.

Computed quantities
-------------------
zhad = E_π / ν = sqrt(p_π² + m_π²) / ν
Pt   = p_π × sin(θ_pq)
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from rsidis_ssa.config_loader import CutsConfig


# ---------------------------------------------------------------------------
# Branch name constants
# ---------------------------------------------------------------------------

# HMS electron PID
BRANCH_HSDELTA  = "H_gtr_dp"            # HMS focal-plane delta [%]
BRANCH_HCER_NPE = "H_cer_npeSum"        # HMS Cherenkov NPE sum
BRANCH_HETOTTRACKNORM  = "H_cal_etottracknorm" # HMS calorimeter E/p (track-normalized)

# SHMS pion PID
BRANCH_PSDELTA   = "P_gtr_dp"           # SHMS focal-plane delta [%]
BRANCH_PAERO_NPE = "P_aero_npeSum"      # SHMS aerogel Cherenkov NPE sum
BRANCH_PETOTTRACKNORM  = "P_cal_etottracknorm" # SHMS calorimeter E/p (track-normalized)
BRANCH_PHGC_NPE  = "P_hgcer_npeSum"     # SHMS heavy-gas Cherenkov NPE sum

# Coincidence time
BRANCH_CTIME = "CTime_ePiCoinTime_ROC2" # e−π coincidence time [ns]

# Per-event beam helicity (DAQ value, before IHWP correction)
BRANCH_HELICITY = "T_helicity_hel"      # +1 or −1 (or ±0.5 in some replay versions)

# Kinematics needed for computed quantities
BRANCH_PPi      = "P_gtr_p"                  # SHMS pion momentum [GeV/c]
BRANCH_NU       = "H_kin_primary_nu"          # Virtual-photon energy ν [GeV]
BRANCH_THETA_PQ = "P_kin_secondary_th_xq"    # θ_pq [rad]

# Pion mass [GeV/c²]
M_PI = 0.13957018

# IHWP sign convention
_IHWP_SIGN: dict[str, int] = {"OUT": +1, "IN": -1}


# ---------------------------------------------------------------------------
# IHWP correction
# ---------------------------------------------------------------------------

def ihwp_sign(ihwp: str) -> int:
    """
    Return the helicity sign factor for this run's IHWP state.

    +1 when IHWP is OUT (no flip), −1 when IHWP is IN (helicity flipped).

    Raises
    ------
    ValueError
        If *ihwp* is not 'IN' or 'OUT' (case-insensitive).
    """
    key = ihwp.strip().upper()
    if key not in _IHWP_SIGN:
        raise ValueError(
            f"Unknown IHWP state: {ihwp!r}.  Expected 'IN' or 'OUT'."
        )
    return _IHWP_SIGN[key]


def effective_helicity(arrays: dict[str, np.ndarray], ihwp: str) -> np.ndarray:
    """
    Per-event effective beam helicity after IHWP correction.

    Parameters
    ----------
    arrays : dict
        Must contain BRANCH_HELICITY.
    ihwp : str
        IHWP state for this run ('IN' or 'OUT').

    Returns
    -------
    np.ndarray
        Element-wise product of the raw helicity and the IHWP sign factor.
    """
    return arrays[BRANCH_HELICITY] * ihwp_sign(ihwp)


# ---------------------------------------------------------------------------
# Cut masks
# ---------------------------------------------------------------------------

def pid_mask(arrays: dict[str, np.ndarray], cuts_cfg: CutsConfig) -> np.ndarray:
    """
    Boolean mask: pass HMS + SHMS PID and acceptance cuts.

    Applied identically to real and random coincidence-time windows.

    Parameters
    ----------
    arrays : dict
        Must contain: BRANCH_HSDELTA, BRANCH_HCER_NPE, BRANCH_HSSHSUM,
        BRANCH_PSDELTA, BRANCH_PAERO_NPE, BRANCH_PHGC_NPE.
    cuts_cfg : CutsConfig
        Thresholds loaded from the YAML config.

    Returns
    -------
    np.ndarray of bool, shape (n_events,)
    """
    return (
        (arrays[BRANCH_HSDELTA]   >= cuts_cfg.hsdelta_lo)
        & (arrays[BRANCH_HSDELTA]   <= cuts_cfg.hsdelta_hi)
        & (arrays[BRANCH_HCER_NPE]  >= cuts_cfg.hcer_npe_min)
        & (arrays[BRANCH_HETOTTRACKNORM]   >= cuts_cfg.hsshsum_min)
        & (arrays[BRANCH_PSDELTA]   >= cuts_cfg.psdelta_lo)
        & (arrays[BRANCH_PSDELTA]   <= cuts_cfg.psdelta_hi)
        & (arrays[BRANCH_PAERO_NPE] >= cuts_cfg.paero_npe_min)
        & (arrays[BRANCH_PHGC_NPE]  >= cuts_cfg.phgc_npe_min)
        & (arrays[BRANCH_PETOTTRACKNORM]   <= cuts_cfg.psshsum_max)
    )


def real_ctime_mask(
    arrays: dict[str, np.ndarray],
    cuts_cfg: CutsConfig,
    ctmean: Optional[float],
    ctsigma: Optional[float],
) -> np.ndarray:
    """
    Boolean mask for the real coincidence-time peak.

    When ``cuts_cfg.ctime_real_center == 'auto'``:

    * ``center`` = *ctmean*  (per-run value from the CSV)
    * ``half_window`` = *nsigma* × *ctsigma*
    * If *ctsigma* is NaN/None, ``half_window = ctime_real_window_fallback``

    When ``ctime_real_center`` is a fixed float:

    * ``center`` = that float
    * ``half_window = ctime_real_window_fallback``

    Parameters
    ----------
    arrays : dict
        Must contain BRANCH_CTIME.
    cuts_cfg : CutsConfig
    ctmean : float or None
        Per-run coincidence-time mean [ns] (from CSV).  Used only in 'auto' mode.
    ctsigma : float or None
        Per-run coincidence-time sigma [ns] (from CSV).  May be NaN.

    Returns
    -------
    np.ndarray of bool
    """
    nsigma = cuts_cfg.ctime_real_nsigma
    if cuts_cfg.ctime_real_center == "auto":
        center = float(ctmean)  # type: ignore[arg-type]
        if nsigma is None or ctsigma is None or pd.isna(ctsigma):
            half_win = cuts_cfg.ctime_real_window_fallback
        else:
            half_win = nsigma * float(ctsigma)
    else:
        center   = float(cuts_cfg.ctime_real_center)
        half_win = cuts_cfg.ctime_real_window_fallback

    return np.abs(arrays[BRANCH_CTIME] - center) <= half_win


def random_ctime_mask(
    arrays: dict[str, np.ndarray],
    cuts_cfg: CutsConfig,
    ctmean: float,
    real_half_win: float,
    beam_bunch_ns: float,
) -> np.ndarray:
    """
    Boolean mask for the random coincidence-time sideband.

    Selects events inside the union of discrete windows around individual
    beam-bunch peaks.  The first ``ctime_random_n_skip`` peaks on each side
    of the real peak are excluded; the next ``ctime_random_n_peaks_lo``
    (lo side) and ``ctime_random_n_peaks_hi`` (hi side) peaks are used.

    Each random window uses the same half-width as the real peak so that the
    per-window statistics are directly comparable.

    Peak centers:
        lo side: ctmean − (n_skip + k) × beam_bunch_ns,  k = 1 … n_peaks_lo
        hi side: ctmean + (n_skip + k) × beam_bunch_ns,  k = 1 … n_peaks_hi

    Parameters
    ----------
    arrays : dict
        Must contain BRANCH_CTIME.
    cuts_cfg : CutsConfig
    ctmean : float
        Per-run coincidence-time mean of the real peak [ns].
    real_half_win : float
        Half-width of the real peak window [ns].  The same window is applied
        to every random peak.
    beam_bunch_ns : float
        Beam bunch spacing [ns].  Loaded from run_constants.yaml.

    Returns
    -------
    np.ndarray of bool
    """
    ctime  = arrays[BRANCH_CTIME]
    mask   = np.zeros(len(ctime), dtype=bool)
    n_skip = cuts_cfg.ctime_random_n_skip
    bbn    = beam_bunch_ns
    for k in range(1, cuts_cfg.ctime_random_n_peaks_lo + 1):
        center = ctmean - (n_skip + k) * bbn
        mask  |= np.abs(ctime - center) <= real_half_win
    for k in range(1, cuts_cfg.ctime_random_n_peaks_hi + 1):
        center = ctmean + (n_skip + k) * bbn
        mask  |= np.abs(ctime - center) <= real_half_win
    return mask


def build_real_mask(
    arrays: dict[str, np.ndarray],
    cuts_cfg: CutsConfig,
    ctmean: Optional[float],
    ctsigma: Optional[float],
) -> np.ndarray:
    """Combined PID + real coincidence-time mask."""
    return pid_mask(arrays, cuts_cfg) & real_ctime_mask(arrays, cuts_cfg, ctmean, ctsigma)


def build_random_mask(
    arrays: dict[str, np.ndarray],
    cuts_cfg: CutsConfig,
    ctmean: float,
    real_half_win: float,
    beam_bunch_ns: float,
) -> np.ndarray:
    """Combined PID + random coincidence-time sideband mask."""
    return pid_mask(arrays, cuts_cfg) & random_ctime_mask(
        arrays, cuts_cfg, ctmean, real_half_win, beam_bunch_ns
    )


# ---------------------------------------------------------------------------
# Computed physics quantities
# ---------------------------------------------------------------------------

def compute_zhad(arrays: dict[str, np.ndarray]) -> np.ndarray:
    """
    Hadronic energy fraction: z_had = E_π / ν = sqrt(p_π² + m_π²) / ν.

    Parameters
    ----------
    arrays : dict
        Must contain BRANCH_PPi and BRANCH_NU.

    Returns
    -------
    np.ndarray of float
    """
    return np.sqrt(arrays[BRANCH_PPi]**2 + M_PI**2) / arrays[BRANCH_NU]


def compute_Pt(arrays: dict[str, np.ndarray]) -> np.ndarray:
    """
    Pion transverse momentum: P_T = p_π × sin(θ_pq).

    Parameters
    ----------
    arrays : dict
        Must contain BRANCH_PPi and BRANCH_THETA_PQ.

    Returns
    -------
    np.ndarray of float
    """
    return arrays[BRANCH_PPi] * np.sin(arrays[BRANCH_THETA_PQ])

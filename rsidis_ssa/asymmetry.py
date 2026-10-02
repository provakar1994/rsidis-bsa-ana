"""
Beam single-spin asymmetry (A_LU) computation.

Entry point
-----------
    result = compute_asymmetry(h_plus, h_minus, beam_polarization, histogram_name)

Physics
-------
For a longitudinally polarized electron beam on an unpolarized target the
dominant φ_pq modulation is:

    A_LU(φ_pq) = A_LU^sinφ × sin(φ_pq)

The raw asymmetry per bin k is formed from the fully-subtracted weighted yields:

    A_raw(φ_k) = (W_+(φ_k) − W_-(φ_k)) / (W_+(φ_k) + W_-(φ_k))

Statistical uncertainty (exact error propagation through the Weight storage):

    σ²_A(φ_k) = 4 (W_+² Var(W_-) + W_-² Var(W_+)) / (W_+ + W_-)⁴

The physics asymmetry corrects for beam polarization:

    A_phys(φ_k) = A_raw(φ_k) / P_beam
    σ_A_phys(φ_k) = σ_A_raw(φ_k) / P_beam

The sin(φ) amplitude is extracted by analytic weighted least-squares
(single-parameter fit; no external solver required):

    A_fit = Σ_k  A_phys(φ_k) × sin(φ_k) / σ²_k
            ────────────────────────────────────
            Σ_k  sin²(φ_k) / σ²_k

    σ_fit² = 1 / Σ_k (sin²(φ_k) / σ²_k)

Bins where W_+ + W_- ≤ 0 are excluded from the fit (marked NaN in the output
arrays but never raise exceptions).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import boost_histogram as bh


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class AsymmetryResult:
    """
    Per-bin asymmetry and fitted sin(φ) amplitude for one histogram.

    Attributes
    ----------
    histogram_name : str
        Name of the base φ_pq histogram (e.g. ``"phipq"``).
    phi_centers : np.ndarray, shape (N,)
        Bin centres in radians.
    A_raw : np.ndarray, shape (N,)
        Raw beam-helicity asymmetry per bin.  NaN for empty bins.
    A_raw_err : np.ndarray, shape (N,)
        Statistical uncertainty on A_raw.  NaN for empty bins.
    A_phys : np.ndarray, shape (N,)
        Physics asymmetry A_raw / beam_polarization.
    A_phys_err : np.ndarray, shape (N,)
        Statistical uncertainty on A_phys.
    amplitude : float
        Fitted A_LU^sinφ amplitude (physics asymmetry, corrected for polarization).
    amplitude_err : float
        1σ uncertainty on the fitted amplitude.
    chi2_ndf : float
        χ²/ndf of the fit.  NaN when fewer than 2 valid bins remain.
    n_bins_used : int
        Number of bins included in the fit (W_+ + W_- > 0 AND σ > 0).
    beam_polarization : float
        |P_e| used for the polarization correction.
    N_plus, N_minus : np.ndarray or None
        Optional unweighted event counts per phi bin for positive and negative
        effective helicity.  These are diagnostic counts only; A_phys and its
        uncertainty are computed from the weighted/subtracted histograms.
    """
    histogram_name:    str
    phi_centers:       np.ndarray
    A_raw:             np.ndarray
    A_raw_err:         np.ndarray
    A_phys:            np.ndarray
    A_phys_err:        np.ndarray
    amplitude:         float
    amplitude_err:     float
    chi2_ndf:          float
    n_bins_used:       int
    beam_polarization: float
    N_plus:            np.ndarray | None = None
    N_minus:           np.ndarray | None = None


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def compute_raw_asymmetry(
    h_plus:  bh.Histogram,
    h_minus: bh.Histogram,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Per-bin raw beam-helicity asymmetry from two subtracted histograms.

    Parameters
    ----------
    h_plus, h_minus : bh.Histogram
        Fully-subtracted yield histograms for positive and negative effective
        helicity respectively.  Must share the same axis.

    Returns
    -------
    phi_centers : np.ndarray
        Bin centres [rad].
    A_raw : np.ndarray
        Raw asymmetry per bin (NaN where W_+ + W_- ≤ 0).
    A_raw_err : np.ndarray
        Statistical uncertainty per bin (NaN where W_+ + W_- ≤ 0).
    """
    phi_centers = h_plus.axes[0].centers.copy()

    W_p  = h_plus.values()
    W_m  = h_minus.values()
    VW_p = np.maximum(h_plus.variances(),  0.0)
    VW_m = np.maximum(h_minus.variances(), 0.0)

    denom = W_p + W_m
    valid = denom > 0.0

    A_raw     = np.full(len(phi_centers), np.nan)
    A_raw_err = np.full(len(phi_centers), np.nan)

    A_raw[valid] = (W_p[valid] - W_m[valid]) / denom[valid]

    # σ²_A = 4(W_+² Var(W_-) + W_-² Var(W_+)) / (W_+ + W_-)⁴
    d4         = denom[valid] ** 4
    numer_var  = 4.0 * (W_p[valid]**2 * VW_m[valid] + W_m[valid]**2 * VW_p[valid])
    A_raw_err[valid] = np.sqrt(np.maximum(numer_var / d4, 0.0))

    return phi_centers, A_raw, A_raw_err


def fit_sinphi(
    phi_centers: np.ndarray,
    A_phys:      np.ndarray,
    A_phys_err:  np.ndarray,
) -> tuple[float, float, float, int]:
    """
    Analytic weighted least-squares fit of A × sin(φ) to the physics asymmetry.

    Parameters
    ----------
    phi_centers : np.ndarray
        Bin centres [rad].
    A_phys : np.ndarray
        Physics asymmetry per bin (may contain NaN for empty bins).
    A_phys_err : np.ndarray
        Statistical uncertainty per bin (may contain NaN; zero entries skipped).

    Returns
    -------
    amplitude : float
        Best-fit A_LU^sinφ.  NaN if fewer than 2 valid bins.
    amplitude_err : float
        1σ uncertainty.  NaN if fewer than 2 valid bins.
    chi2_ndf : float
        χ²/ndf.  NaN if fewer than 2 valid bins.
    n_used : int
        Number of bins included in the fit.
    """
    valid = (
        np.isfinite(A_phys)
        & np.isfinite(A_phys_err)
        & (A_phys_err > 0.0)
    )
    n_used = int(valid.sum())

    if n_used < 2:
        return float("nan"), float("nan"), float("nan"), n_used

    sp  = np.sin(phi_centers[valid])
    A   = A_phys[valid]
    sig = A_phys_err[valid]
    w   = 1.0 / sig**2            # statistical weights

    # Analytic WLS for single-parameter model f(φ) = amp × sin(φ):
    #   amp = (Σ w_k A_k sin_k) / (Σ w_k sin²_k)
    wss       = float(np.sum(w * sp**2))
    amplitude     = float(np.sum(w * A * sp) / wss)
    amplitude_err = float(1.0 / np.sqrt(wss))

    residuals = A - amplitude * sp
    chi2      = float(np.sum((residuals / sig)**2))
    chi2_ndf  = chi2 / (n_used - 1)

    return amplitude, amplitude_err, chi2_ndf, n_used


def compute_asymmetry(
    h_plus:            bh.Histogram,
    h_minus:           bh.Histogram,
    beam_polarization: float,
    histogram_name:    str,
    N_plus:            np.ndarray | None = None,
    N_minus:           np.ndarray | None = None,
) -> AsymmetryResult:
    """
    Compute the full beam SSA from a pair of helicity-split histograms.

    Parameters
    ----------
    h_plus, h_minus : bh.Histogram
        Fully-subtracted yield histograms for positive and negative effective
        helicity.
    beam_polarization : float
        |P_e| in (0, 1].
    histogram_name : str
        Label for the result (e.g. ``"phipq"``).
    N_plus, N_minus : np.ndarray or None
        Optional unweighted per-bin counts to carry through to CSV output.

    Returns
    -------
    AsymmetryResult
    """
    if not (0.0 < beam_polarization <= 1.0):
        raise ValueError(
            f"beam_polarization must be in (0, 1], got {beam_polarization}"
        )

    phi_centers, A_raw, A_raw_err = compute_raw_asymmetry(h_plus, h_minus)

    A_phys     = A_raw     / beam_polarization
    A_phys_err = A_raw_err / beam_polarization

    amplitude, amplitude_err, chi2_ndf, n_used = fit_sinphi(
        phi_centers, A_phys, A_phys_err
    )

    return AsymmetryResult(
        histogram_name    = histogram_name,
        phi_centers       = phi_centers,
        A_raw             = A_raw,
        A_raw_err         = A_raw_err,
        A_phys            = A_phys,
        A_phys_err        = A_phys_err,
        amplitude         = amplitude,
        amplitude_err     = amplitude_err,
        chi2_ndf          = chi2_ndf,
        n_bins_used       = n_used,
        beam_polarization = beam_polarization,
        N_plus            = None if N_plus is None else np.asarray(N_plus).copy(),
        N_minus           = None if N_minus is None else np.asarray(N_minus).copy(),
    )


# ---------------------------------------------------------------------------
# Convenience: find matching helicity-split pairs from a histogram name list
# ---------------------------------------------------------------------------

def find_asymmetry_pairs(
    histogram_names: list[str],
) -> list[tuple[str, str, str]]:
    """
    Identify (base_name, hplus_name, hminus_name) triples from a list of
    histogram names.

    Convention: a histogram named ``foo_hplus`` (``foo_hminus``) is the
    positive (negative) helicity-split version of the base histogram ``foo``.
    Both ``foo_hplus`` and ``foo_hminus`` must be present; unpaired names are
    ignored.

    Returns
    -------
    list of (base_name, hplus_name, hminus_name) sorted by base_name.
    """
    name_set = set(histogram_names)
    pairs:       list[tuple[str, str, str]] = []
    seen_bases:  set[str]                   = set()

    for name in histogram_names:
        if not name.endswith("_hplus"):
            continue
        base        = name[: -len("_hplus")]
        hminus_name = base + "_hminus"
        if hminus_name in name_set and base not in seen_bases:
            pairs.append((base, name, hminus_name))
            seen_bases.add(base)

    return sorted(pairs, key=lambda t: t[0])

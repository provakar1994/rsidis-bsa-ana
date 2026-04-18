"""
Background subtraction routines.

Three independent operations, each returning a *new* histogram (inputs
are never mutated).  All use boost_histogram's arithmetic, which correctly
propagates the Weight storage (sum_w, sum_w²) through every operation.

Random-sideband subtraction
---------------------------
Fill real and random registries with identical per-event weights, then
subtract with the window-size scale factor:

    h_signal = h_real + h_random * (−scale)

where ``scale = real_window_half_width / random_window_half_width``.

For example, with a real window of ±nsigma×ctsigma ≈ ±2 ns and a random
sideband of ±6 ns, scale ≈ 2/6 ≈ 0.333.  The scale ensures equal
luminosity-exposure normalisation between the two windows before subtracting.

e⁺ charge-symmetric background subtraction
-------------------------------------------
1:1 subtraction (no scale factor).  The e⁺ runs share the same kinematic
setting and are already normalised per unit charge by the weight table.

    h_result = h_signal + h_eplus * (−1)

Dummy-target subtraction  (cryo targets only)
---------------------------------------------
1:1 subtraction.  Like e⁺, the dummy histograms are already luminosity-
normalised before being passed here.

    h_result = h_data + h_dummy * (−1)

Diagnostics
-----------
``SubtractionResult`` bundles before/background/after histograms for one
named step.  Its ``pulls`` property gives a bin-by-bin significance of the
subtracted background:

    pull[i] = background_scaled[i] / sqrt(variance_before[i])

Large |pull| values flag bins where the background is a significant fraction
of the statistical uncertainty — useful for validating the subtraction.

``subtraction_summary(results)`` returns a human-readable table of the
fraction of (normalised) yield removed at each step.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import boost_histogram as bh


# ---------------------------------------------------------------------------
# Pure subtraction functions
# ---------------------------------------------------------------------------

def subtract_randoms(
    h_real: bh.Histogram,
    h_random: bh.Histogram,
    scale: float,
) -> bh.Histogram:
    """
    Subtract the scaled random-coincidence sideband from the real-peak histogram.

    Parameters
    ----------
    h_real : bh.Histogram
        Histogram filled in the real coincidence-time window.
    h_random : bh.Histogram
        Histogram filled in the random sideband window.
    scale : float
        Ratio of real window half-width to random window half-width.
        E.g. if real = ±2 ns and random = ±6 ns, scale = 2/6 ≈ 0.333.

    Returns
    -------
    bh.Histogram
        h_real − scale × h_random.  Variance is propagated correctly:
        Var(result) = Var(h_real) + scale² × Var(h_random).
    """
    if scale <= 0:
        raise ValueError(f"scale must be > 0, got {scale}")
    return h_real + h_random * (-scale)


def subtract_eplus(
    h_signal: bh.Histogram,
    h_eplus: bh.Histogram,
) -> bh.Histogram:
    """
    Subtract the e⁺ charge-symmetric background (1:1, no scale factor).

    Both histograms must be normalised to the same luminosity (per unit charge)
    before calling this function — the weight table handles that.

    Parameters
    ----------
    h_signal : bh.Histogram
        Histogram from the e⁻ (signal) runs after random subtraction.
    h_eplus : bh.Histogram
        Histogram from the e⁺ (background) runs after random subtraction.

    Returns
    -------
    bh.Histogram
    """
    return h_signal + h_eplus * (-1.0)


def subtract_dummy(
    h_data: bh.Histogram,
    h_dummy: bh.Histogram,
    scale: float = 1.0,
) -> bh.Histogram:
    """
    Subtract the dummy-target contribution (cryo targets only).

    The dummy histogram must already be normalised per unit luminosity.
    *scale* is the inverse of the dummy-to-cryo-wall thickness ratio:

        scale = 1 / (dummy_thickness / cryo_wall_thickness)

    For LH2 the ratio is 7.2323, so scale ≈ 0.1382.
    The default scale=1.0 is kept for backward compatibility and tests.

    Parameters
    ----------
    h_data : bh.Histogram
        Histogram from the cryo-target runs.
    h_dummy : bh.Histogram
        Histogram from the dummy-target runs.
    scale : float
        Fraction of h_dummy to subtract.  Must be > 0.

    Returns
    -------
    bh.Histogram
    """
    if scale <= 0:
        raise ValueError(f"scale must be > 0, got {scale}")
    return h_data + h_dummy * (-scale)


# ---------------------------------------------------------------------------
# Diagnostic dataclass
# ---------------------------------------------------------------------------

@dataclass
class SubtractionResult:
    """
    Bundles the histograms and metadata for one background-subtraction step.

    Attributes
    ----------
    label : str
        Human-readable name for this step, e.g. "random", "eplus", "dummy".
    histogram_name : str
        Name of the histogram this result belongs to (e.g. "phipq").
    h_before : bh.Histogram
        Histogram before this subtraction.
    h_background : bh.Histogram
        The background histogram (unscaled, as filled).
    h_after : bh.Histogram
        Histogram after subtraction.
    scale : float
        Scale factor applied to h_background before subtracting (1.0 for
        e⁺ and dummy subtractions; the window ratio for random subtraction).
    """
    label:          str
    histogram_name: str
    h_before:       bh.Histogram
    h_background:   bh.Histogram
    h_after:        bh.Histogram
    scale:          float = 1.0

    @property
    def fraction_subtracted(self) -> float:
        """
        Fraction of the total (sum-of-weights) yield removed by this step.

        Returns 0.0 if h_before is empty.  May be negative if the background
        exceeds the signal in some bins — a warning sign worth investigating.
        """
        before = float(h_before_total := self.h_before.values().sum())
        if before == 0.0:
            return 0.0
        after = float(self.h_after.values().sum())
        return (before - after) / before

    @property
    def pulls(self) -> np.ndarray:
        """
        Bin-by-bin pull: background_scaled[i] / sqrt(variance_before[i]).

        Measures how many sigma (relative to the statistical uncertainty of
        h_before) the subtracted background is in each bin.

        Bins where variance_before == 0 return pull = 0.
        """
        background_scaled = self.h_background.values() * self.scale
        var_before = self.h_before.variances()
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(
                var_before > 0,
                background_scaled / np.sqrt(var_before),
                0.0,
            )

    @property
    def max_abs_pull(self) -> float:
        """Maximum absolute pull across all bins."""
        p = self.pulls
        return float(np.max(np.abs(p))) if p.size > 0 else 0.0


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def subtraction_summary(results: list[SubtractionResult]) -> str:
    """
    Return a human-readable table of fraction-subtracted per step.

    Parameters
    ----------
    results : list[SubtractionResult]
        One entry per subtraction step, in order of application.

    Returns
    -------
    str
        Multi-line table suitable for printing or writing to a log file.

    Example output::

        Background subtraction summary
        ─────────────────────────────────────────────────────────────────
          histogram      step      scale    subtracted(%)   max|pull|
        ─────────────────────────────────────────────────────────────────
          phipq          random    0.333       8.4 %          1.2
          phipq          eplus     1.000      12.1 %          2.4
        ─────────────────────────────────────────────────────────────────
    """
    header = (
        f"  {'histogram':<16}  {'step':<8}  {'scale':>7}  "
        f"{'subtracted(%)':>14}  {'max|pull|':>9}"
    )
    sep = "  " + "-" * (len(header) - 2)
    lines = [
        "Background subtraction summary",
        sep,
        header,
        sep,
    ]
    for r in results:
        frac_pct = r.fraction_subtracted * 100.0
        lines.append(
            f"  {r.histogram_name:<16}  {r.label:<8}  {r.scale:>7.3f}  "
            f"{frac_pct:>13.1f} %  {r.max_abs_pull:>9.2f}"
        )
    lines.append(sep)
    return "\n".join(lines)

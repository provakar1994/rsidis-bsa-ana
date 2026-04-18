"""
Histogram registry and filling.

Design
------
``build_histogram_registry(histo_cfgs)``
    Creates one boost_histogram per HistogramConfig entry (from the YAML).
    All histograms use the Weight storage so errors can be propagated.
    Adding a new histogram requires only a new YAML block — no code changes.

``fill_run(arrays, cut_mask, weight, registry, histo_cfgs, ihwp)``
    Fills all histograms for one run's events.  Handles:
    - Direct branches (read straight from the ROOT tree)
    - Computed quantities (zhad, Pt — delegated to cuts.compute_*)
    - Helicity-split histograms (helicity_cut: positive / negative)

Usage
-----
    registry = build_histogram_registry(cfg.histograms)
    fill_run(arrays, real_mask, run_weight.weight, registry, cfg.histograms, row["IHWP"])
    # ... repeat for every run, then subtract randoms, e+, dummy as needed
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import boost_histogram as bh

from rsidis_ssa.config_loader import HistogramConfig
from rsidis_ssa.cuts import (
    compute_zhad,
    compute_Pt,
    effective_helicity,
)


# ---------------------------------------------------------------------------
# Registry builder
# ---------------------------------------------------------------------------

def build_histogram_registry(
    histo_cfgs: list[HistogramConfig],
) -> dict[str, bh.Histogram]:
    """
    Create one boost_histogram per config entry, keyed by histogram name.

    All histograms use ``bh.storage.Weight()`` so they accumulate both the
    sum of weights and the sum of weights-squared (for error propagation).

    Parameters
    ----------
    histo_cfgs : list[HistogramConfig]
        Histogram definitions loaded from the YAML config.

    Returns
    -------
    dict[str, bh.Histogram]
        Keys are histogram names; values are freshly-zeroed histograms.
    """
    return {
        hcfg.name: bh.Histogram(
            bh.axis.Regular(hcfg.bins, hcfg.xmin, hcfg.xmax),
            storage=bh.storage.Weight(),
        )
        for hcfg in histo_cfgs
    }


# ---------------------------------------------------------------------------
# Value resolver
# ---------------------------------------------------------------------------

def _resolve_values(
    arrays: dict[str, np.ndarray],
    hcfg: HistogramConfig,
) -> np.ndarray:
    """
    Return the per-event values to fill for *hcfg*.

    Direct branches are looked up in *arrays* by name.
    Computed quantities are evaluated on the fly.

    Raises
    ------
    ValueError
        If an unrecognised computed-quantity name is encountered.
    """
    if not hcfg.is_computed:
        return arrays[hcfg.branch]

    cname = hcfg.computed_name
    if cname == "zhad":
        return compute_zhad(arrays)
    if cname == "Pt":
        return compute_Pt(arrays)
    raise ValueError(
        f"Unknown computed quantity: {cname!r}.  "
        "Register it in histograms._resolve_values() and cuts.py."
    )


# ---------------------------------------------------------------------------
# Filling
# ---------------------------------------------------------------------------

def fill_run(
    arrays: dict[str, np.ndarray],
    cut_mask: np.ndarray,
    weight: float,
    registry: dict[str, bh.Histogram],
    histo_cfgs: list[HistogramConfig],
    ihwp: str,
) -> None:
    """
    Fill all histograms in *registry* for one run's events.

    Parameters
    ----------
    arrays : dict[str, np.ndarray]
        All branches read from the ROOT file for this run.
        Keys are branch names (e.g. 'H_gtr_dp', 'P_gtr_p').
    cut_mask : np.ndarray of bool, shape (n_events,)
        Pre-computed combined PID + coincidence-time mask (real or random).
        Caller is responsible for building the right mask via
        ``cuts.build_real_mask`` or ``cuts.build_random_mask``.
    weight : float
        Per-event normalization weight for this run (from RunWeight.weight).
        Applied uniformly to every selected event.
    registry : dict[str, bh.Histogram]
        Target histograms (mutated in place).  Must contain an entry for
        every name in *histo_cfgs*.
    histo_cfgs : list[HistogramConfig]
        Config entries describing what to fill (same set as those used
        to build *registry*).
    ihwp : str
        IHWP state for this run ('IN' or 'OUT').
        Used only for histograms that have a ``helicity_cut``.
    """
    # Compute effective helicity once (only if any histogram needs it)
    needs_helicity = any(h.helicity_cut is not None for h in histo_cfgs)
    eff_hel: Optional[np.ndarray] = (
        effective_helicity(arrays, ihwp) if needs_helicity else None
    )

    for hcfg in histo_cfgs:
        mask = cut_mask

        if hcfg.helicity_cut == "positive":
            mask = mask & (eff_hel > 0.5)   # type: ignore[operator]
        elif hcfg.helicity_cut == "negative":
            mask = mask & (eff_hel < -0.5)  # type: ignore[operator]

        values = _resolve_values(arrays, hcfg)[mask]
        registry[hcfg.name].fill(values, weight=weight)

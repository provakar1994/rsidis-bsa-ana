"""
Per-run normalization weight calculation.

The per-event weight for run r is:

    w_r = (ps_factor_r × boil_corr_r)
          / (Q_r × h_esing_Eff_r × p_hadron_Eff_r × comp_livetime_r)

where:
  Q             — efficiency-corrected beam charge [mC], using the
                  configured normalization charge column
  h_esing_Eff   — HMS electron singles efficiency  (fraction, ~0.9997)
  p_hadron_Eff  — SHMS hadron tracking efficiency  (fraction, ~0.93)
  comp_livetime — computer live-time               (fraction, ~1.0; NOT percent)
  boil_corr     — target-density boiling correction (fraction ≥ 1, cryo only)
                  = 1.0 for non-cryo targets

boil_corr is in the **numerator** because a target that was less dense due
to beam heating (boil_corr > 1) has lower luminosity, so each recorded event
represents proportionally more cross-section.

Derivation:
  Y_nominal = Y_measured × boil_corr
  w = Y_nominal / (Q × h × p × livetime × N_events)
    = boil_corr / (Q × h × p × livetime)          [ps_factor=1 in this dataset]

Prescale (ps5/ps6) is extracted and recorded for auditing; in this dataset
the coin trigger is always unprescaled (ps_factor = 1.0).

Cross-check against the CSV:
  normyield ≈ ransubcoin / (Q × h_esing × p_hadron × comp_livetime)  [~1%]
The CSV normyield does NOT include boil_corr — it stores the uncorrected
normalized yield, consistent with the derivation above.

Combining runs
--------------
Fill histograms with per-event weight w_r for each run r.  The combined
histogram is the sum of all runs' weighted events — no further re-weighting.

Run exclusions
--------------
build_weight_table() never raises.  Any run with a missing or invalid
normalization value is silently excluded and reported in the returned
WeightTableResult.excluded list.  Pass log_path to write a structured
log file for auditing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Module logger — callers can configure handlers; we add a FileHandler when
# log_path is supplied to build_weight_table().
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class NormalizationError(ValueError):
    """
    Raised by compute_run_weight() when a normalization value is invalid.
    Carries the run number and the name of the offending column so that
    build_weight_table() can build a structured exclusion record.
    """
    def __init__(self, run: int, column: str, message: str) -> None:
        super().__init__(message)
        self.run = run
        self.column = column


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RunWeight:
    """
    Per-event normalization weight for one run, with a breakdown of
    each factor so any anomaly can be traced back to its source.

    For multi-run histogram filling use ``eff_scale`` (not ``weight``):
      - fill each run's events with weight = eff_scale
      - after all runs, multiply histograms by 1/Q_tot
    This gives the correct combined yield and error propagation.

    ``weight`` (= eff_scale / charge) is retained for single-run
    cross-checks (e.g. normyield ≈ ransubcoin × weight).
    """
    run:          int
    weight:       float   # eff_scale / charge — single-run cross-check only
    charge:       float   # configured charge-column value [mC]
    h_esing_eff:  float   # HMS electron singles efficiency
    p_hadron_eff: float   # SHMS hadron tracking efficiency
    ps_factor:    float   # prescale factor (1.0 = unprescaled)
    livetime:     float   # comp_livetime (fraction 0–1)
    boil_corr:    float   # boiling correction (1.0 for non-cryo)
    eff_scale:    float   # ps_factor * boil_corr / (h_esing_eff * p_hadron_eff * livetime)
                          # charge-independent; use for multi-run filling

    @property
    def norm_factor(self) -> float:
        """1 / weight — the effective luminosity denominator."""
        return 1.0 / self.weight


@dataclass(frozen=True)
class RunExclusion:
    """
    Records a run that was excluded from the weight table and why.
    """
    run:    int
    column: str   # which normalization column triggered the exclusion
    reason: str   # human-readable explanation


@dataclass
class WeightTableResult:
    """
    Returned by build_weight_table().

    Attributes
    ----------
    weights : dict[int, RunWeight]
        Per-event weight for every successfully validated run.
    excluded : list[RunExclusion]
        Every run that was dropped, with the column and reason.
    Q_tot : float
        Sum of the configured charge-column values [mC] over all valid runs.
        Used by the pipeline to divide histograms after filling with eff_scale.
    """
    weights:  dict[int, RunWeight]     = field(default_factory=dict)
    excluded: list[RunExclusion]       = field(default_factory=list)
    Q_tot:    float                    = 0.0

    @property
    def n_valid(self) -> int:
        return len(self.weights)

    @property
    def n_excluded(self) -> int:
        return len(self.excluded)

    def summary(self) -> str:
        lines = [
            f"Runs accepted : {self.n_valid}",
            f"Runs excluded : {self.n_excluded}",
            f"Q_tot         : {self.Q_tot:.3f} mC",
        ]
        if self.excluded:
            lines.append("")
            lines.append(f"  {'run':>8}  {'column':<20}  reason")
            lines.append("  " + "-" * 70)
            for ex in self.excluded:
                lines.append(f"  {ex.run:>8}  {ex.column:<20}  {ex.reason}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Prescale
# ---------------------------------------------------------------------------

def get_prescale_factor(row: pd.Series) -> float:
    """
    Return the active coincidence prescale factor for one run row.

    ps6 takes precedence; ps5 is the fallback.  Exactly one must be
    positive; the other should be NaN (was -999, replaced by load_runlist).

    Raises NormalizationError if neither slot is active or if both are active.
    """
    ps6 = row["ps6"]
    ps5 = row["ps5"]
    run = int(row["run"])

    ps6_active = pd.notna(ps6) and ps6 > 0
    ps5_active = pd.notna(ps5) and ps5 > 0

    if ps6_active and ps5_active:
        raise NormalizationError(
            run=run,
            column="ps5/ps6",
            message=(
                f"Run {run}: both prescale slots are active "
                f"(ps5={ps5}, ps6={ps6}).  Expected exactly one positive value."
            ),
        )

    if ps6_active:
        return float(ps6)
    if ps5_active:
        return float(ps5)

    raise NormalizationError(
        run=run,
        column="ps5/ps6",
        message=(
            f"Run {run}: no active prescale slot "
            f"(ps5={ps5}, ps6={ps6}).  Check the runlist."
        ),
    )


# ---------------------------------------------------------------------------
# Single-run weight
# ---------------------------------------------------------------------------

def compute_run_weight(
    row: pd.Series,
    charge_column: str = "BCM2_Q",
    apply_boil_corr: bool = False,
) -> RunWeight:
    """
    Compute the per-event normalization weight for one runlist row.

    Parameters
    ----------
    row : pd.Series
        A single row from the DataFrame returned by load_runlist().
    charge_column : str
        Name of the runlist column to use for beam charge normalization.
    apply_boil_corr : bool
        Pass True for LH2 / LD2 targets.

    Returns
    -------
    RunWeight

    Raises
    ------
    NormalizationError
        On any invalid or missing normalization value, carrying the
        run number and the offending column name.
    """
    run = int(row["run"])

    def _check(val, col, condition_ok, msg_suffix):
        if not condition_ok:
            raise NormalizationError(run=run, column=col,
                                     message=f"Run {run}: {col} {msg_suffix}")

    # ---- charge ----
    charge = row[charge_column]
    _check(charge, charge_column, pd.notna(charge),   "is NaN")
    _check(charge, charge_column, charge > 0,         f"= {charge:.4f} must be > 0")

    # ---- HMS electron singles efficiency ----
    h_eff = row["h_esing_Eff"]
    _check(h_eff, "h_esing_Eff", pd.notna(h_eff),     "is NaN")
    _check(h_eff, "h_esing_Eff", 0.0 < h_eff <= 1.0,  f"= {h_eff:.6f} outside (0, 1]")

    # ---- SHMS hadron tracking efficiency ----
    p_eff = row["p_hadron_Eff"]
    _check(p_eff, "p_hadron_Eff", pd.notna(p_eff),    "is NaN")
    _check(p_eff, "p_hadron_Eff", 0.0 < p_eff <= 1.0, f"= {p_eff:.6f} outside (0, 1]")

    # ---- computer live-time (fraction, NOT percent) ----
    livetime = row["comp_livetime"]
    _check(livetime, "comp_livetime", pd.notna(livetime), "is NaN")
    _check(livetime, "comp_livetime", 0.0 < livetime <= 1.0,
           f"= {livetime:.6f} outside (0, 1] — run may be flagged as bad")

    # ---- prescale ----
    ps_factor = get_prescale_factor(row)

    # ---- boiling correction ----
    if apply_boil_corr:
        boil = row["boil_corr"]
        _check(boil, "boil_corr", pd.notna(boil),
               "is NaN — exclude this run or assign a value manually")
        _check(boil, "boil_corr", boil > 0, f"= {boil:.6f} must be > 0")
    else:
        boil = 1.0

    # ---- assemble weight and eff_scale ----
    # eff_scale = ps_factor * boil / (h_eff × p_eff × livetime)  — charge-independent
    # weight    = eff_scale / charge                              — kept for single-run QA
    # boil is in the NUMERATOR: less-dense target → each event ∝ more cross-section.
    eff_scale   = ps_factor * boil / (h_eff * p_eff * livetime)
    norm_factor = charge / eff_scale                               # charge × h × p × lt / (ps × boil)
    weight      = 1.0 / norm_factor

    return RunWeight(
        run=run,
        weight=weight,
        charge=charge,
        h_esing_eff=h_eff,
        p_hadron_eff=p_eff,
        ps_factor=ps_factor,
        livetime=livetime,
        boil_corr=boil,
        eff_scale=eff_scale,
    )


# ---------------------------------------------------------------------------
# Multi-run weight table
# ---------------------------------------------------------------------------

def build_weight_table(
    df_runs: pd.DataFrame,
    charge_column: str = "BCM2_Q",
    apply_boil_corr: bool = False,
    log_path: str | Path | None = None,
) -> WeightTableResult:
    """
    Compute per-event weights for every run in *df_runs*.

    Runs with any invalid or missing normalization value are excluded
    (not raised) and recorded in the returned WeightTableResult.excluded.
    Every exclusion is also emitted as a WARNING through the module logger
    (``rsidis_ssa.normalization``).

    Parameters
    ----------
    df_runs : pd.DataFrame
        Subset of the runlist (e.g. the output of get_signal_runs()).
    charge_column : str
        Name of the runlist column to use for beam charge normalization.
    apply_boil_corr : bool
        Pass True for LH2 / LD2 targets.
    log_path : path-like or None
        If given, write a human-readable exclusion log to this file.
        The file is created (or overwritten) regardless of whether any
        runs were excluded — an empty exclusion section signals a clean run.

    Returns
    -------
    WeightTableResult
    """
    result = WeightTableResult()

    for _, row in df_runs.iterrows():
        try:
            rw = compute_run_weight(
                row,
                charge_column=charge_column,
                apply_boil_corr=apply_boil_corr,
            )
            result.weights[rw.run] = rw
        except NormalizationError as exc:
            excl = RunExclusion(run=exc.run, column=exc.column, reason=str(exc))
            result.excluded.append(excl)
            logger.warning("Excluded run %d (%s): %s", exc.run, exc.column, exc)

    result.Q_tot = sum(rw.charge for rw in result.weights.values())

    if result.excluded:
        logger.warning(
            "%d run(s) excluded from weight table due to missing/invalid "
            "normalization values — check the exclusion report.",
            result.n_excluded,
        )

    if log_path is not None:
        _write_log(
            result,
            Path(log_path),
            apply_boil_corr=apply_boil_corr,
            charge_column=charge_column,
        )

    return result


# ---------------------------------------------------------------------------
# Log writer
# ---------------------------------------------------------------------------

def _write_log(
    result: WeightTableResult,
    log_path: Path,
    apply_boil_corr: bool,
    charge_column: str,
) -> None:
    """Write a structured plain-text log of the weight-table build."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with log_path.open("w") as fh:
        fh.write("=" * 72 + "\n")
        fh.write("  RSIDIS SSA — normalization weight table build log\n")
        fh.write(f"  Generated : {timestamp}\n")
        fh.write(f"  apply_boil_corr = {apply_boil_corr}\n")
        fh.write("=" * 72 + "\n\n")

        fh.write(f"Runs accepted : {result.n_valid}\n")
        fh.write(f"Runs excluded : {result.n_excluded}\n")
        fh.write(f"Q_tot         : {result.Q_tot:.3f} mC\n\n")

        if result.n_valid:
            fh.write("--- Accepted runs ---\n")
            fh.write(f"  {'run':>8}  {'weight':>14}  {'eff_scale':>14}  {charge_column:>10}  "
                     f"{'h_eff':>8}  {'p_eff':>8}  {'livetime':>10}  "
                     f"{'boil_corr':>10}  {'ps':>4}\n")
            fh.write("  " + "-" * 95 + "\n")
            for rw in sorted(result.weights.values(), key=lambda r: r.run):
                fh.write(
                    f"  {rw.run:>8}  {rw.weight:>14.6e}  {rw.eff_scale:>14.6e}  "
                    f"{rw.charge:>10.3f}  "
                    f"{rw.h_esing_eff:>8.5f}  {rw.p_hadron_eff:>8.5f}  "
                    f"{rw.livetime:>10.6f}  {rw.boil_corr:>10.6f}  "
                    f"{rw.ps_factor:>4.1f}\n"
                )

        if result.n_excluded:
            fh.write("\n--- Excluded runs ---\n")
            fh.write(f"  {'run':>8}  {'column':<20}  reason\n")
            fh.write("  " + "-" * 72 + "\n")
            for ex in result.excluded:
                fh.write(f"  {ex.run:>8}  {ex.column:<20}  {ex.reason}\n")
        else:
            fh.write("\n--- No runs were excluded ---\n")

        fh.write("\n" + "=" * 72 + "\n")

    logger.info("Weight table log written to %s", log_path)


# ---------------------------------------------------------------------------
# Audit / QA helpers
# ---------------------------------------------------------------------------

def weight_table_to_df(weights: dict[int, RunWeight]) -> pd.DataFrame:
    """
    Convert a weight dict to a DataFrame for display and QA plots.
    Accepts either a plain dict or a WeightTableResult.weights dict.
    Sorted by run number.
    """
    rows = [
        {
            "run":           rw.run,
            "weight":        rw.weight,
            "norm_factor":   rw.norm_factor,
            "eff_scale":     rw.eff_scale,
            "BCM2_Q":        rw.charge,
            "h_esing_Eff":   rw.h_esing_eff,
            "p_hadron_Eff":  rw.p_hadron_eff,
            "ps_factor":     rw.ps_factor,
            "comp_livetime": rw.livetime,
            "boil_corr":     rw.boil_corr,
        }
        for rw in sorted(weights.values(), key=lambda r: r.run)
    ]
    return pd.DataFrame(rows)


def check_normyield_consistency(
    weights: dict[int, RunWeight],
    df_runlist: pd.DataFrame,
    charge_column: str = "BCM2_Q",
    rtol: float = 0.02,
) -> pd.DataFrame:
    """
    Cross-check computed weights against the pre-computed normyield column.

    For each run:
        normyield_expected ≈ ransubcoin / norm_factor_no_boil
    where norm_factor_no_boil = Q × h_eff × p_eff × livetime / ps_factor,
    using the configured charge column.

    The CSV normyield does not include boil_corr (confirmed from data),
    so this check is valid for both cryo and non-cryo runs.

    Returns a DataFrame with columns:
        run, normyield_csv, normyield_computed, residual_pct, flagged
    Any run with |residual_pct| > rtol*100 is flagged.
    """
    run_map = df_runlist.set_index("run")
    results = []

    for run_no, rw in sorted(weights.items(), key=lambda kv: kv[0]):
        row = run_map.loc[run_no]
        ny_csv = row["normyield"]
        ransubcoin = row["ransubcoin"]

        if pd.isna(ny_csv) or pd.isna(ransubcoin):
            continue

        norm_no_boil = rw.charge * rw.h_esing_eff * rw.p_hadron_eff * rw.livetime / rw.ps_factor
        ny_computed = ransubcoin / norm_no_boil
        residual = 100.0 * (ny_computed - ny_csv) / ny_csv

        results.append({
            "run":                run_no,
            "normyield_csv":      ny_csv,
            "normyield_computed": ny_computed,
            "residual_pct":       residual,
            "flagged":            abs(residual) > rtol * 100,
        })

    return pd.DataFrame(results)

"""
Analysis pipeline: load → fill → subtract.

``run_pipeline(cfg, config_dir)`` is the single entry point.  It:

  1. Loads the runlist CSV and selects runs for the requested setting.
  2. Builds normalization weight tables for each run type
     (signal / e⁺ / dummy).
  3. Loops over every run:
       - Opens the ROOT file and reads the minimal set of required branches.
       - Applies PID + coincidence-time cuts.
       - Fills two registries per run type:
           real   — events in the real coincidence-time peak, weight = w_run
           random — events in the discrete random-peak windows,
                    weight = w_run × win_scale
         where win_scale = 1 / n_random_peaks (constant) normalises the
         random sum to one equivalent real-window area.
  4. Returns a PipelineResult with all filled registries and metadata.

Missing ROOT files are logged and skipped; the run is recorded in
PipelineResult.file_skips so the caller can audit which runs contributed.
Branch or tree errors are fatal (they indicate a config or file mismatch).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import boost_histogram as bh

from rsidis_ssa.config_loader import AnalysisConfig, load_config
from rsidis_ssa.cuts import (
    BRANCH_CTIME,
    BRANCH_HELICITY,
    build_random_mask,
    build_real_mask,
    effective_helicity,
    pid_mask as build_pid_mask,
)
from rsidis_ssa.histograms import build_histogram_registry, fill_run
from rsidis_ssa.normalization import RunWeight, WeightTableResult, build_weight_table
from rsidis_ssa.reader import read_branches, required_branches
from rsidis_ssa.runlist import (
    get_dummy_runs,
    get_eplus_runs,
    get_signal_runs,
    load_runlist,
    select_runs,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class FilledRegistries:
    """
    Real and random histogram registries for one run type.

    Attributes
    ----------
    real : dict[str, bh.Histogram]
        Filled with events in the real coincidence-time window,
        weight = per-event normalization weight.
    random : dict[str, bh.Histogram]
        Filled with events in the random sideband,
        weight = per-event normalization weight × window_scale.
        window_scale absorbs the width difference between windows so that
        ``real − random`` is a valid subtraction without an additional factor.
    """
    real:   dict[str, bh.Histogram]
    random: dict[str, bh.Histogram]


@dataclass
class PipelineResult:
    """
    Complete output of ``run_pipeline()``.

    Attributes
    ----------
    signal : FilledRegistries
        Histograms from signal (e⁻) runs.
    eplus : FilledRegistries or None
        Histograms from e⁺ background runs.  None when
        ``cfg.do_eplus_subtraction`` is False.
    dummy : FilledRegistries or None
        Histograms from dummy-target runs.  None when
        ``cfg.do_dummy_subtraction`` is False.
    dummy_scale : float
        Scale applied to the dummy histogram at subtraction time:
        ``1 / (dummy_thickness / cryo_wall_thickness)``.
        1.0 when dummy subtraction is not active.
    ctime_hists : dict[str, bh.Histogram]
        Coincidence-time distributions (PID-cut only, unweighted) per run
        type.  Keys are "signal", "eplus", "dummy" as applicable.
        Axis: 1100 bins, 0–110 ns (0.1 ns/bin).
    ctime_run_windows : dict[str, list[tuple[float, float]]]
        Per-run real coincidence-time window parameters used during filling.
        Each entry is ``(ctmean, half_win)`` in ns.  Keys match ctime_hists.
    ctime_random_run_windows : dict[str, list[list[tuple[float, float]]]]
        Per-run discrete random-peak window parameters.
        Each outer-list element corresponds to one run; the inner list holds
        ``(center, half_win)`` for each discrete random peak used.
        Keys match ctime_hists.
    beam_bunch_ns : float
        Beam bunch spacing [ns] for this run period, from run_constants.yaml.
    weight_tables : dict[str, WeightTableResult]
        Keys are "signal", "eplus", "dummy" as applicable.
    file_skips : dict[str, list[int]]
        Runs skipped because their ROOT file was not found.
        Keys match weight_tables.
    run_dfs : dict[str, pd.DataFrame]
        Per-run-type runlist subsets (full CSV columns).  Used to access the
        pre-computed CSV normyield for cross-checking.
    normyield_per_run : dict[str, pd.DataFrame]
        Per-run normalized yield comparison table for each run type.
        Columns: run, n_real, n_random, normyield_workflow, normyield_csv,
        residual_pct, flagged, n_hplus, n_hminus, IHWP.
    raw_helicity : dict[str, np.ndarray]
        Concatenated raw T_helicity_hel values (real-window events, all runs)
        per run type.  Keys are "signal", "eplus", "dummy" as applicable.
    cfg : AnalysisConfig
        The config used for this run.
    """
    signal:            FilledRegistries
    eplus:             Optional[FilledRegistries]
    dummy:             Optional[FilledRegistries]
    dummy_scale:       float
    ctime_hists:       dict[str, bh.Histogram]
    ctime_run_windows:        dict[str, list[tuple[float, float]]]
    ctime_random_run_windows: dict[str, list[list[tuple[float, float]]]]
    beam_bunch_ns:            float
    weight_tables:            dict[str, WeightTableResult]
    file_skips:        dict[str, list[int]]
    run_dfs:           dict[str, pd.DataFrame]
    normyield_per_run: dict[str, pd.DataFrame]
    raw_helicity:      dict[str, np.ndarray]
    cfg:               AnalysisConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _apply_run_filter(
    df: pd.DataFrame,
    include: list[int] | None,
    exclude: list[int],
    label: str,
) -> pd.DataFrame:
    """
    Apply runs_include / runs_exclude filters to a runlist DataFrame.

    *include* = None means keep all runs.  *exclude* always wins over *include*.
    Logs a summary of how many runs were dropped and why.
    """
    n_before = len(df)
    if include is not None:
        df = df[df["run"].isin(include)]
    if exclude:
        dropped_excl = df["run"][df["run"].isin(exclude)].tolist()
        df = df[~df["run"].isin(exclude)]
        if dropped_excl:
            logger.info("[%s] runs_exclude removed %d run(s): %s",
                        label, len(dropped_excl), dropped_excl)
    n_after = len(df)
    if n_before != n_after:
        logger.info("[%s] run filter: %d → %d runs", label, n_before, n_after)
    return df.reset_index(drop=True)


def _real_half_win(row: pd.Series, cfg: AnalysisConfig) -> float:
    """Per-run real coincidence-time half-window in ns.

    Returns ``ctime_real_window_fallback`` when any of:
    - ctime_real_center is a fixed float (not 'auto')
    - ctime_real_nsigma is null
    - ctsigma from the CSV is NaN/missing
    """
    cuts = cfg.cuts
    if cuts.ctime_real_center == "auto":
        nsigma = cuts.ctime_real_nsigma
        if nsigma is None:
            return cuts.ctime_real_window_fallback
        ctsigma = row.get("ctsigma", float("nan"))
        if pd.isna(ctsigma):
            return cuts.ctime_real_window_fallback
        return nsigma * float(ctsigma)
    return cuts.ctime_real_window_fallback


_CTIME_AXIS = bh.axis.Regular(1100, 0.0, 110.0)  # 0.1 ns/bin, covers both peaks


def _fill_run_type(
    df_runs: pd.DataFrame,
    wt_result: WeightTableResult,
    cfg: AnalysisConfig,
    label: str,
    beam_bunch: float,
    config_dir: Optional[Path] = None,
    ctime_override: Optional[tuple[float, float]] = None,
) -> tuple[FilledRegistries, list[int], bh.Histogram, list[tuple[float, float]], list[list[tuple[float, float]]], pd.DataFrame]:
    """
    Fill real + random registries for all valid runs in *df_runs*.

    Also accumulates an unweighted coincidence-time histogram (PID cuts only,
    no ctime cut), records the per-run real-window parameters, and builds a
    per-run normalized yield comparison table.

    When *ctime_override* is provided the per-run CSV ctmean/ctsigma values
    are ignored and the override pair is used for every run.  This ensures
    that e⁺ and dummy runs use the signal-derived ctime windows rather than
    their own (often poorly fitted) CSV values.

    Returns
    -------
    FilledRegistries
        Real and random histogram registries.
    list[int]
        Run numbers whose ROOT file was not found.
    bh.Histogram
        Unweighted ctime distribution (PID-masked, all runs summed).
    list[tuple[float, float]]
        Per-run ``(ctmean, real_half_win)`` in ns for valid runs.
    list[list[tuple[float, float]]]
        Per-run list of ``(center, half_win)`` for each discrete random peak.
    pd.DataFrame
        Per-run normyield table with columns: run, n_real, n_random,
        normyield_workflow, normyield_csv, residual_pct, flagged.
        normyield_workflow = (n_real − n_random × win_scale) × weight_r,
        computed from this workflow's own event selection.
    """
    eff_histo_cfgs = cfg.effective_histograms()
    branches = required_branches(eff_histo_cfgs, cfg.cuts)
    branches.add(BRANCH_HELICITY)   # always needed for per-run helicity diagnostics
    real_reg   = build_histogram_registry(eff_histo_cfgs)
    random_reg = build_histogram_registry(eff_histo_cfgs)
    ctime_hist = bh.Histogram(_CTIME_AXIS, storage=bh.storage.Double())
    raw_helicity_values: list[np.ndarray] = []   # raw T_helicity_hel for all runs
    run_windows:        list[tuple[float, float]]       = []
    run_random_windows: list[list[tuple[float, float]]] = []
    file_skips:         list[int] = []
    run_normyields:     list[dict] = []

    # win_scale is constant: 1 / n_random_peaks (area-ratio normalisation).
    n_rand_peaks = cfg.cuts.ctime_random_n_peaks_lo + cfg.cuts.ctime_random_n_peaks_hi
    win_scale    = 1.0 / n_rand_peaks
    logger.debug("[%s] n_rand_peaks=%d  win_scale=%.6f", label, n_rand_peaks, win_scale)

    for run_no, rw in wt_result.weights.items():
        row = df_runs.set_index("run").loc[run_no]
        root_path = cfg.rootfiles.get_path(run_no, config_dir)

        try:
            arrays = read_branches(root_path, cfg.rootfiles.treename, branches)
        except FileNotFoundError:
            logger.warning("[%s] ROOT file not found, skipping run %d: %s",
                           label, run_no, root_path)
            file_skips.append(run_no)
            continue

        ihwp = str(row["IHWP"])
        if ctime_override is not None:
            ctmean, ctsigma = ctime_override
        else:
            ctmean  = row.get("ctmean",  None)
            ctsigma = row.get("ctsigma", None)

        # Real half-window varies per run (auto mode).
        # win_scale = 1 / n_rand_peaks is constant (computed before loop).
        half_win = _real_half_win(row, cfg)

        real_mask   = build_real_mask(arrays, cfg.cuts, ctmean, ctsigma)
        random_mask = build_random_mask(arrays, cfg.cuts, float(ctmean), half_win, beam_bunch)

        # Per-run helicity breakdown (real events only, after all PID+ctime cuts)
        eff_hel  = effective_helicity(arrays, ihwp)
        n_hplus  = int((real_mask & (eff_hel > 0.5)).sum())
        n_hminus = int((real_mask & (eff_hel < -0.5)).sum())
        raw_helicity_values.append(arrays[BRANCH_HELICITY][real_mask])

        # Absorb win_scale into the random fill weight so that
        # ``real − random`` is a valid subtraction without an extra factor.
        # eff_corrected_counts: fill with eff_scale; divide by Q_tot after loop.
        # eff_corrected_charge: fill with 1 (raw counts); divide by Q_eff_tot after loop.
        # charge_only:          fill with 1 (raw counts); divide by Q_tot after loop.
        #                       All efficiency factors are ignored — debugging only.
        scheme = cfg.normalization.weight_scheme
        fill_weight   = rw.eff_scale if scheme == "eff_corrected_counts" else 1.0
        random_weight = fill_weight * win_scale
        # (charge_only also uses fill_weight=1.0; it differs only in the post-loop denominator)

        fill_run(arrays, real_mask,   fill_weight,   real_reg,   eff_histo_cfgs, ihwp)
        fill_run(arrays, random_mask, random_weight, random_reg, eff_histo_cfgs, ihwp)

        # Ctime diagnostic: fill with PID-only mask, unweighted
        p_mask = build_pid_mask(arrays, cfg.cuts)
        ctime_hist.fill(arrays[BRANCH_CTIME][p_mask])

        # Record per-run window parameters (ctmean may be None for non-auto mode)
        real_center = (float(ctmean) if ctmean is not None and not np.isnan(float(ctmean))
                       else float(cfg.cuts.ctime_real_center)
                       if cfg.cuts.ctime_real_center != "auto"
                       else np.nan)
        run_windows.append((real_center, half_win))

        # Record discrete random-peak window parameters for this run
        run_peaks: list[tuple[float, float]] = []
        n_skip = cfg.cuts.ctime_random_n_skip
        for k in range(1, cfg.cuts.ctime_random_n_peaks_lo + 1):
            run_peaks.append((real_center - (n_skip + k) * beam_bunch, half_win))
        for k in range(1, cfg.cuts.ctime_random_n_peaks_hi + 1):
            run_peaks.append((real_center + (n_skip + k) * beam_bunch, half_win))
        run_random_windows.append(run_peaks)

        # Per-run normyield from this workflow's own event selection.
        # Equivalent to the integral of the unweighted random-subtracted histogram
        # of any kinematic variable, multiplied by weight_r.
        n_real        = int(real_mask.sum())
        n_random      = int(random_mask.sum())
        n_rand_scaled = n_random * win_scale   # scaled to real-window width
        n_rsc         = n_real - n_rand_scaled # random-subtracted count
        run_normyields.append({
            "run":                run_no,
            "n_real":             n_real,
            "n_random":           n_random,
            "win_scale":          win_scale,
            "n_rand_scaled":      n_rand_scaled,
            "n_rsc":              n_rsc,
            "normyield_workflow": n_rsc * rw.weight,
            "n_hplus":            n_hplus,
            "n_hminus":           n_hminus,
            "IHWP":               ihwp,
            # Normalization component breakdown (from RunWeight)
            "charge":             rw.charge,
            "charge_hp":          rw.charge_hp,
            "charge_hm":          rw.charge_hm,
            "h_esing_eff":        rw.h_esing_eff,
            "p_hadron_eff":       rw.p_hadron_eff,
            "ps_factor":          rw.ps_factor,
            "livetime":           rw.livetime,
            "boil_corr":          rw.boil_corr,
        })

        logger.debug(
            "[%s] run %d  eff_scale=%.4e  win_scale=%.4f  "
            "real_events=%d  random_events=%d  ny_wf=%.4e",
            label, run_no, rw.eff_scale, win_scale,
            n_real, n_random, n_rsc * rw.weight,
        )

    logger.info("[%s] filled %d runs (%d files missing)",
                label, len(wt_result.weights) - len(file_skips), len(file_skips))

    # Divide all histograms by the appropriate denominator to get the combined yield.
    # eff_corrected_counts:  fill_w=eff_scale, divide by Q_tot    (raw charge sum)
    # eff_corrected_charge:  fill_w=1,         divide by Q_eff_tot (efficiency-corrected charge)
    # charge_only:           fill_w=1,         divide by Q_tot    (no efficiency correction)
    # When use_helicity_gated_charge=True, _hplus/_hminus histograms are divided
    # by Q_hp_tot / Q_hm_tot instead of the common q_denom.
    scheme = cfg.normalization.weight_scheme
    if scheme in ("eff_corrected_counts", "charge_only"):
        q_denom = wt_result.Q_tot
        q_denom_label = f"Q_tot = {q_denom:.3f} mC"
    else:
        q_denom = wt_result.Q_eff_tot
        q_denom_label = f"Q_eff_tot = {q_denom:.3f} mC"
    if q_denom <= 0:
        raise ValueError(
            f"[{label}] {q_denom_label} — no valid charge accumulated. "
            "Check that at least one run passed normalization validation."
        )

    if cfg.normalization.use_helicity_gated_charge:
        q_hp = wt_result.Q_hp_tot
        q_hm = wt_result.Q_hm_tot
        for reg in (real_reg, random_reg):
            for key, h in reg.items():
                if key.endswith("_hplus"):
                    h *= 1.0 / q_hp
                elif key.endswith("_hminus"):
                    h *= 1.0 / q_hm
                else:
                    h *= 1.0 / q_denom
        logger.debug(
            "[%s] helicity-gated normalization: h+/Q_hp=%.3f mC, "
            "h-/Q_hm=%.3f mC, inclusive/%s  (scheme: %s)",
            label, q_hp, q_hm, q_denom_label, scheme,
        )
    else:
        scale = 1.0 / q_denom
        for h in real_reg.values():
            h *= scale
        for h in random_reg.values():
            h *= scale
        logger.debug("[%s] divided histograms by %s  (scheme: %s)",
                     label, q_denom_label, scheme)

    # Build normyield comparison DataFrame and merge in CSV normyield + raw counts
    ny_df = pd.DataFrame(run_normyields)
    if not ny_df.empty:
        # Join normyield and raw event-count columns from the CSV runlist
        csv_cols = ["normyield", "coin", "randoms", "ransubcoin"]
        csv_cols_present = [c for c in csv_cols if c in df_runs.columns]
        csv_join = df_runs.set_index("run")[csv_cols_present]
        csv_join = csv_join.rename(columns={"normyield": "normyield_csv",
                                            "coin":      "csv_coin",
                                            "randoms":   "csv_randoms",
                                            "ransubcoin":"csv_ransubcoin"})
        ny_df = ny_df.set_index("run").join(csv_join, how="left").reset_index()

        valid = ny_df["normyield_csv"].notna() & (ny_df["normyield_csv"] != 0)
        ny_df["residual_pct"] = np.where(
            valid,
            100.0 * (ny_df["normyield_workflow"] - ny_df["normyield_csv"]) / ny_df["normyield_csv"],
            np.nan,
        )
        ny_df["flagged"] = ny_df["residual_pct"].abs() > 2.0
    else:
        for col in ("win_scale", "n_rand_scaled", "n_rsc",
                    "normyield_csv", "residual_pct", "flagged",
                    "csv_coin", "csv_randoms", "csv_ransubcoin"):
            ny_df[col] = pd.Series(dtype=float)

    _log_normyield_table(ny_df, label)

    raw_hel_all = np.concatenate(raw_helicity_values) if raw_helicity_values else np.array([], dtype=float)
    return FilledRegistries(real=real_reg, random=random_reg), file_skips, ctime_hist, run_windows, run_random_windows, ny_df, raw_hel_all


def _log_normyield_table(ny_df: pd.DataFrame, label: str) -> None:
    """Print a per-run count + yield comparison table to stdout."""
    if ny_df.empty:
        return

    has_csv = "csv_coin" in ny_df.columns and ny_df["csv_coin"].notna().any()

    # Header
    # n_rand_sc = n_random * win_scale  (scaled to real-window width)
    # This is what should match CSV 'randoms' column.
    if has_csv:
        hdr = (f"{'run':>7}  "
               f"{'n_real':>8}  {'coin':>8}  {'r/c':>6}  "
               f"{'n_rand_sc':>9}  {'randoms':>9}  {'r/c':>6}  {'wscale':>7}  "
               f"{'n_rsc':>8}  {'ransubcoin':>10}  {'r/c':>6}  "
               f"{'ny_wf':>12}  {'ny_csv':>12}  {'Δ%':>7}  flag")
    else:
        hdr = (f"{'run':>7}  "
               f"{'n_real':>8}  {'n_rand_sc':>9}  {'wscale':>7}  {'n_rsc':>8}  "
               f"{'ny_wf':>12}")
    sep = "-" * len(hdr)

    lines = [f"\n[{label}] per-run yield/count comparison", sep, hdr, sep]

    def _ratio(a, b):
        try:
            return f"{float(a)/float(b):.4f}" if float(b) != 0 else "   nan"
        except (TypeError, ValueError):
            return "   nan"

    for _, row in ny_df.sort_values("run").iterrows():
        run           = int(row["run"])
        n_real        = int(row["n_real"])
        n_rand_sc     = float(row["n_rand_scaled"])
        win_scale     = float(row["win_scale"])
        n_rsc         = float(row["n_rsc"])
        ny_wf         = float(row["normyield_workflow"])

        if has_csv:
            coin    = row.get("csv_coin",       float("nan"))
            randoms = row.get("csv_randoms",    float("nan"))
            ransub  = row.get("csv_ransubcoin", float("nan"))
            ny_csv  = row.get("normyield_csv",  float("nan"))
            res_pct = row.get("residual_pct",   float("nan"))
            flagged = bool(row.get("flagged", False))

            lines.append(
                f"{run:>7}  "
                f"{n_real:>8d}  {_fmt(coin):>8}  {_ratio(n_real, coin):>6}  "
                f"{n_rand_sc:>9.2f}  {_fmt(randoms):>9}  {_ratio(n_rand_sc, randoms):>6}  {win_scale:>7.4f}  "
                f"{n_rsc:>8.1f}  {_fmt(ransub):>10}  {_ratio(n_rsc, ransub):>6}  "
                f"{ny_wf:>12.4e}  {_fmt_e(ny_csv):>12}  {_fmt_pct(res_pct):>7}  "
                f"{'***' if flagged else ''}"
            )
        else:
            lines.append(
                f"{run:>7}  {n_real:>8d}  {n_rand_sc:>9.2f}  {win_scale:>7.4f}  "
                f"{n_rsc:>8.1f}  {ny_wf:>12.4e}"
            )

    lines.append(sep)

    # Summary
    if has_csv:
        res = ny_df["residual_pct"].dropna()
        if len(res):
            lines.append(
                f"  median Δ(ny) = {res.median():+.2f}%   "
                f"max|Δ(ny)| = {res.abs().max():.2f}%   "
                f"flagged = {int(ny_df['flagged'].fillna(False).sum())} / {len(ny_df)}"
            )
        real_arr  = ny_df["n_real"].values.astype(float)
        nrsc_arr  = ny_df["n_rsc"].values.astype(float)
        coin_arr  = ny_df["csv_coin"].values.astype(float)
        ransub_arr= ny_df["csv_ransubcoin"].values.astype(float)
        ok_c  = (coin_arr   > 0) & np.isfinite(coin_arr)
        ok_rs = (ransub_arr > 0) & np.isfinite(ransub_arr)
        if ok_c.any():
            lines.append(f"  median n_real/coin       = {np.median(real_arr[ok_c]/coin_arr[ok_c]):.4f}")
        if ok_rs.any():
            lines.append(f"  median n_rsc/ransubcoin  = {np.median(nrsc_arr[ok_rs]/ransub_arr[ok_rs]):.4f}")

    # Normalization component breakdown
    norm_cols = ["charge", "h_esing_eff", "p_hadron_eff", "ps_factor", "livetime", "boil_corr"]
    if all(c in ny_df.columns for c in norm_cols):
        nhdr = (f"{'run':>7}  {'charge':>9}  {'h_e_eff':>8}  {'p_h_eff':>8}  "
                f"{'ps_fac':>7}  {'ltfrac':>7}  {'boil':>7}")
        nsep = "-" * len(nhdr)
        lines += [f"\n[{label}] normalization components", nsep, nhdr, nsep]
        for _, row in ny_df.sort_values("run").iterrows():
            run = int(row["run"])
            lines.append(
                f"{run:>7}  "
                f"{float(row['charge']):>9.4f}  "
                f"{float(row['h_esing_eff']):>8.5f}  "
                f"{float(row['p_hadron_eff']):>8.5f}  "
                f"{float(row['ps_factor']):>7.4f}  "
                f"{float(row['livetime']):>7.5f}  "
                f"{float(row['boil_corr']):>7.5f}"
            )
        lines.append(nsep)

    logger.debug("\n".join(lines))


def _fmt(v) -> str:
    """Format a count value (int or float) for the table."""
    try:
        f = float(v)
        return "nan" if np.isnan(f) else f"{int(round(f)):d}"
    except (TypeError, ValueError):
        return "nan"


def _fmt_e(v) -> str:
    try:
        f = float(v)
        return "nan" if np.isnan(f) else f"{f:.4e}"
    except (TypeError, ValueError):
        return "nan"


def _fmt_pct(v) -> str:
    try:
        f = float(v)
        return "nan" if np.isnan(f) else f"{f:+.2f}%"
    except (TypeError, ValueError):
        return "nan"


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------

def run_pipeline(
    cfg: AnalysisConfig,
    config_dir: Optional[Path] = None,
) -> PipelineResult:
    """
    Execute the complete fill pipeline for *cfg*.

    Parameters
    ----------
    cfg : AnalysisConfig
        Loaded and validated analysis config.
    config_dir : Path or None
        Directory of the YAML file, used to resolve the runlist CSV path.
        Defaults to the current working directory.

    Returns
    -------
    PipelineResult
    """
    # ---- runlist ----
    csv_path = cfg.resolve_runlist_path(config_dir)
    df_all = load_runlist(csv_path)

    setting = cfg.to_setting()
    df_setting = select_runs(df_all, setting, cfg.target)
    logger.info("Setting: %s  target: %s  → %d total rows",
                setting, cfg.target, len(df_setting))

    # ---- weight tables ----
    weight_tables: dict[str, WeightTableResult] = {}
    file_skips:    dict[str, list[int]] = {}
    charge_column = cfg.normalization.charge_column

    # Signal (e⁻) runs
    df_signal = _apply_run_filter(
        get_signal_runs(df_setting),
        cfg.runs_include, cfg.runs_exclude, "signal",
    )
    logger.info("Signal runs: %d", len(df_signal))
    weight_tables["signal"] = build_weight_table(
        df_signal,
        charge_column=charge_column,
        apply_boil_corr=cfg.apply_boil_corr,
        use_helicity_gated_charge=cfg.normalization.use_helicity_gated_charge,
        charge_hp_column=cfg.normalization.charge_hp_column,
        charge_hm_column=cfg.normalization.charge_hm_column,
    )

    # e⁺ background runs
    if cfg.do_eplus_subtraction:
        df_eplus = _apply_run_filter(
            get_eplus_runs(df_setting),
            cfg.runs_include, cfg.runs_exclude, "eplus",
        )
        logger.info("e+ runs: %d", len(df_eplus))
        weight_tables["eplus"] = build_weight_table(
            df_eplus,
            charge_column=charge_column,
            apply_boil_corr=cfg.apply_boil_corr,
            use_helicity_gated_charge=cfg.normalization.use_helicity_gated_charge,
            charge_hp_column=cfg.normalization.charge_hp_column,
            charge_hm_column=cfg.normalization.charge_hm_column,
        )

    # Dummy runs
    if cfg.do_dummy_subtraction:
        df_dummy = _apply_run_filter(
            get_dummy_runs(df_all, setting),
            cfg.runs_include, cfg.runs_exclude, "dummy",
        )
        logger.info("Dummy runs: %d", len(df_dummy))
        weight_tables["dummy"] = build_weight_table(
            df_dummy,
            charge_column=charge_column,
            apply_boil_corr=False,
            use_helicity_gated_charge=cfg.normalization.use_helicity_gated_charge,
            charge_hp_column=cfg.normalization.charge_hp_column,
            charge_hm_column=cfg.normalization.charge_hm_column,
        )

    # ---- dummy scale ----
    dummy_scale = 1.0
    if cfg.do_dummy_subtraction:
        ratio = cfg.dummy_scale(config_dir)
        dummy_scale = 1.0 / ratio
        logger.info(
            "Dummy subtraction scale for %s (period %s): 1 / %.4f = %.6f",
            cfg.target, cfg.run_period, ratio, dummy_scale,
        )

    # ---- beam bunch spacing ----
    beam_bunch = cfg.beam_bunch_ns(config_dir)
    logger.info("Beam bunch spacing: %.3f ns", beam_bunch)

    # ---- fill ----
    ctime_hists:              dict[str, bh.Histogram]                    = {}
    ctime_run_windows:        dict[str, list[tuple[float, float]]]       = {}
    ctime_random_run_windows: dict[str, list[list[tuple[float, float]]]] = {}
    run_dfs:                  dict[str, pd.DataFrame]                    = {}
    normyield_per_run:        dict[str, pd.DataFrame]                    = {}
    raw_helicity:             dict[str, np.ndarray]                      = {}

    (signal_regs, signal_skips,
     ctime_hists["signal"], ctime_run_windows["signal"],
     ctime_random_run_windows["signal"],
     normyield_per_run["signal"],
     raw_helicity["signal"]) = (
        _fill_run_type(df_signal, weight_tables["signal"], cfg, "signal", beam_bunch, config_dir)
    )
    run_dfs["signal"] = df_signal
    file_skips["signal"] = signal_skips
    if signal_skips and len(signal_skips) == len(weight_tables["signal"].weights):
        raise FileNotFoundError(
            f"All {len(signal_skips)} signal ROOT files are missing. "
            f"Check rootfiles.directory in the config "
            f"(resolved to: {cfg.rootfiles.get_path(signal_skips[0], config_dir).parent})"
        )

    # Derive the ctime override from signal runs: mean ctmean and mean ctsigma.
    # These are applied to e⁺ and dummy runs so that all run types share the
    # same ctime windows (signal fits are reliable; e⁺/dummy fits often are not).
    #
    # When ctime_real_nsigma is null, the fallback window is always used and
    # ctsigma is irrelevant — pass None so _fill_run_type also uses the fallback.
    nsigma = cfg.cuts.ctime_real_nsigma
    sig_windows = ctime_run_windows["signal"]
    valid_sig   = [(c, h) for (c, h) in sig_windows if not np.isnan(c)]
    if valid_sig:
        mean_ctmean  = float(np.mean([c for c, _ in valid_sig]))
        # half_win = nsigma × ctsigma → invert to recover mean ctsigma.
        # When nsigma is null every half_win is the fallback; pass None so
        # e+/dummy runs also trigger the fallback path.
        mean_ctsigma = (float(np.mean([h for _, h in valid_sig])) / nsigma
                        if nsigma is not None else None)
    else:
        mean_ctmean  = float(cfg.cuts.ctime_real_center) if cfg.cuts.ctime_real_center != "auto" else 0.0
        mean_ctsigma = (cfg.cuts.ctime_real_window_fallback / nsigma
                        if nsigma is not None else None)
    signal_ctime_override = (mean_ctmean, mean_ctsigma)
    logger.info(
        "Signal ctime override for e⁺/dummy: ctmean=%.3f ns  ctsigma=%s  "
        "(derived from %d signal runs)",
        mean_ctmean,
        f"{mean_ctsigma:.3f} ns" if mean_ctsigma is not None else "null (using fallback)",
        len(valid_sig),
    )

    eplus_regs = None
    if cfg.do_eplus_subtraction:
        if df_eplus.empty:
            logger.warning(
                "do_eplus_subtraction=True but no e⁺ runs found in runlist "
                "— skipping e⁺ subtraction"
            )
        else:
            (eplus_regs, eplus_skips,
             ctime_hists["eplus"], ctime_run_windows["eplus"],
             ctime_random_run_windows["eplus"],
             normyield_per_run["eplus"],
             raw_helicity["eplus"]) = (
                _fill_run_type(df_eplus, weight_tables["eplus"], cfg, "eplus",
                               beam_bunch, config_dir, ctime_override=signal_ctime_override)
            )
            run_dfs["eplus"] = df_eplus
            file_skips["eplus"] = eplus_skips

    dummy_regs = None
    if cfg.do_dummy_subtraction:
        if df_dummy.empty:
            logger.warning(
                "do_dummy_subtraction=True but no dummy runs found in runlist "
                "— skipping dummy subtraction"
            )
        else:
            (dummy_regs, dummy_skips,
             ctime_hists["dummy"], ctime_run_windows["dummy"],
             ctime_random_run_windows["dummy"],
             normyield_per_run["dummy"],
             raw_helicity["dummy"]) = (
                _fill_run_type(df_dummy, weight_tables["dummy"], cfg, "dummy",
                               beam_bunch, config_dir, ctime_override=signal_ctime_override)
            )
            run_dfs["dummy"] = df_dummy
            file_skips["dummy"] = dummy_skips

    return PipelineResult(
        signal=signal_regs,
        eplus=eplus_regs,
        dummy=dummy_regs,
        dummy_scale=dummy_scale,
        ctime_hists=ctime_hists,
        ctime_run_windows=ctime_run_windows,
        ctime_random_run_windows=ctime_random_run_windows,
        beam_bunch_ns=beam_bunch,
        weight_tables=weight_tables,
        file_skips=file_skips,
        run_dfs=run_dfs,
        normyield_per_run=normyield_per_run,
        raw_helicity=raw_helicity,
        cfg=cfg,
    )


def run_pipeline_from_yaml(
    yaml_path: str | Path,
) -> PipelineResult:
    """Convenience wrapper: load config from *yaml_path* then run the pipeline."""
    yaml_path = Path(yaml_path)
    cfg = load_config(yaml_path)
    return run_pipeline(cfg, config_dir=yaml_path.parent)

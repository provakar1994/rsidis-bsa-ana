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
           random — events in the sideband, weight = w_run × window_scale
         where window_scale = real_half_win / random_half_win absorbs the
         normalisation difference between the two windows.
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
    build_random_mask,
    build_real_mask,
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
        Axis: 1200 bins, 0–120 ns (0.1 ns/bin).
    ctime_run_windows : dict[str, list[tuple[float, float]]]
        Per-run real coincidence-time window parameters used during filling.
        Each entry is ``(ctmean, half_win)`` in ns.  Keys match ctime_hists.
    weight_tables : dict[str, WeightTableResult]
        Keys are "signal", "eplus", "dummy" as applicable.
    file_skips : dict[str, list[int]]
        Runs skipped because their ROOT file was not found.
        Keys match weight_tables.
    cfg : AnalysisConfig
        The config used for this run.
    """
    signal:            FilledRegistries
    eplus:             Optional[FilledRegistries]
    dummy:             Optional[FilledRegistries]
    dummy_scale:       float
    ctime_hists:       dict[str, bh.Histogram]
    ctime_run_windows: dict[str, list[tuple[float, float]]]
    weight_tables:     dict[str, WeightTableResult]
    file_skips:        dict[str, list[int]]
    cfg:               AnalysisConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _real_half_win(row: pd.Series, cfg: AnalysisConfig) -> float:
    """Per-run real coincidence-time half-window in ns."""
    cuts = cfg.cuts
    if cuts.ctime_real_center == "auto":
        ctsigma = row.get("ctsigma", float("nan"))
        if pd.isna(ctsigma):
            return cuts.ctime_real_window_fallback
        return cuts.ctime_real_nsigma * float(ctsigma)
    return cuts.ctime_real_window_fallback


_CTIME_AXIS = bh.axis.Regular(1100, 0.0, 110.0)  # 0.1 ns/bin, covers both peaks


def _fill_run_type(
    df_runs: pd.DataFrame,
    wt_result: WeightTableResult,
    cfg: AnalysisConfig,
    label: str,
    config_dir: Optional[Path] = None,
) -> tuple[FilledRegistries, list[int], bh.Histogram, list[tuple[float, float]]]:
    """
    Fill real + random registries for all valid runs in *df_runs*.

    Also accumulates an unweighted coincidence-time histogram (PID cuts only,
    no ctime cut) and records the per-run real-window parameters used.

    Returns
    -------
    FilledRegistries
        Real and random histogram registries.
    list[int]
        Run numbers whose ROOT file was not found.
    bh.Histogram
        Unweighted ctime distribution (PID-masked, all runs summed).
    list[tuple[float, float]]
        Per-run ``(ctmean, half_win)`` in ns for valid runs (skipped runs
        are excluded).
    """
    branches = required_branches(cfg.histograms, cfg.cuts)
    real_reg   = build_histogram_registry(cfg.histograms)
    random_reg = build_histogram_registry(cfg.histograms)
    ctime_hist = bh.Histogram(_CTIME_AXIS, storage=bh.storage.Double())
    run_windows: list[tuple[float, float]] = []
    file_skips: list[int] = []

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

        ihwp      = str(row["IHWP"])
        ctmean    = row.get("ctmean",  None)
        ctsigma   = row.get("ctsigma", None)

        real_mask   = build_real_mask(arrays, cfg.cuts, ctmean, ctsigma)
        random_mask = build_random_mask(arrays, cfg.cuts)

        # Absorb the window-width scale into the random fill weight
        half_win      = _real_half_win(row, cfg)
        win_scale     = half_win / cfg.cuts.ctime_random_window
        random_weight = rw.weight * win_scale

        fill_run(arrays, real_mask,   rw.weight,     real_reg,   cfg.histograms, ihwp)
        fill_run(arrays, random_mask, random_weight, random_reg, cfg.histograms, ihwp)

        # Ctime diagnostic: fill with PID-only mask, unweighted
        p_mask = build_pid_mask(arrays, cfg.cuts)
        ctime_hist.fill(arrays[BRANCH_CTIME][p_mask])

        # Record per-run window parameters (ctmean may be None for non-auto mode)
        real_center = (float(ctmean) if ctmean is not None and not np.isnan(float(ctmean))
                       else float(cfg.cuts.ctime_real_center)
                       if cfg.cuts.ctime_real_center != "auto"
                       else np.nan)
        run_windows.append((real_center, half_win))

        logger.debug(
            "[%s] run %d  weight=%.4e  scale=%.4f  "
            "real_events=%d  random_events=%d",
            label, run_no, rw.weight, win_scale,
            int(real_mask.sum()), int(random_mask.sum()),
        )

    logger.info("[%s] filled %d runs (%d files missing)",
                label, len(wt_result.weights) - len(file_skips), len(file_skips))
    return FilledRegistries(real=real_reg, random=random_reg), file_skips, ctime_hist, run_windows


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

    # Signal (e⁻) runs
    df_signal = get_signal_runs(df_setting)
    logger.info("Signal runs: %d", len(df_signal))
    weight_tables["signal"] = build_weight_table(
        df_signal, apply_boil_corr=cfg.apply_boil_corr
    )

    # e⁺ background runs
    if cfg.do_eplus_subtraction:
        df_eplus = get_eplus_runs(df_setting)
        logger.info("e+ runs: %d", len(df_eplus))
        weight_tables["eplus"] = build_weight_table(df_eplus, apply_boil_corr=False)

    # Dummy runs
    if cfg.do_dummy_subtraction:
        df_dummy = get_dummy_runs(df_all, setting)
        logger.info("Dummy runs: %d", len(df_dummy))
        weight_tables["dummy"] = build_weight_table(
            df_dummy, apply_boil_corr=False
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

    # ---- fill ----
    ctime_hists:       dict[str, bh.Histogram]            = {}
    ctime_run_windows: dict[str, list[tuple[float, float]]] = {}

    signal_regs, signal_skips, ctime_hists["signal"], ctime_run_windows["signal"] = (
        _fill_run_type(df_signal, weight_tables["signal"], cfg, "signal", config_dir)
    )
    file_skips["signal"] = signal_skips
    if signal_skips and len(signal_skips) == len(weight_tables["signal"].weights):
        raise FileNotFoundError(
            f"All {len(signal_skips)} signal ROOT files are missing. "
            f"Check rootfiles.directory in the config "
            f"(resolved to: {cfg.rootfiles.get_path(signal_skips[0], config_dir).parent})"
        )

    eplus_regs = None
    if cfg.do_eplus_subtraction:
        eplus_regs, eplus_skips, ctime_hists["eplus"], ctime_run_windows["eplus"] = (
            _fill_run_type(df_eplus, weight_tables["eplus"], cfg, "eplus", config_dir)
        )
        file_skips["eplus"] = eplus_skips

    dummy_regs = None
    if cfg.do_dummy_subtraction:
        dummy_regs, dummy_skips, ctime_hists["dummy"], ctime_run_windows["dummy"] = (
            _fill_run_type(df_dummy, weight_tables["dummy"], cfg, "dummy", config_dir)
        )
        file_skips["dummy"] = dummy_skips

    return PipelineResult(
        signal=signal_regs,
        eplus=eplus_regs,
        dummy=dummy_regs,
        dummy_scale=dummy_scale,
        ctime_hists=ctime_hists,
        ctime_run_windows=ctime_run_windows,
        weight_tables=weight_tables,
        file_skips=file_skips,
        cfg=cfg,
    )


def run_pipeline_from_yaml(
    yaml_path: str | Path,
) -> PipelineResult:
    """Convenience wrapper: load config from *yaml_path* then run the pipeline."""
    yaml_path = Path(yaml_path)
    cfg = load_config(yaml_path)
    return run_pipeline(cfg, config_dir=yaml_path.parent)

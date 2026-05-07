"""
Diagnostic analysis driver.

Usage:
    python analysis.py config/example_C_z05_thpq2.yaml [--output diag.pdf]

Produces a multi-page PDF with:
  Page 1  : Coincidence-time distribution — full range + zoom on real peak
            with per-run window boundaries and cut regions shaded
  Page 2  : Normalized yield vs run number, raw count comparisons,
            signal run weights, and normalization/current vs run
  Page 3+ : Per-histogram background subtraction — real / random / e⁺
            overlaid, plus the final subtracted result
  Last    : Subtraction statistics table

All logging goes to stdout; set LOG_LEVEL env-var to DEBUG for per-run detail.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages

from rsidis_ssa.backgrounds import (
    SubtractionResult,
    subtract_dummy,
    subtract_eplus,
    subtract_randoms,
    subtraction_summary,
)
from rsidis_ssa.asymmetry import (
    AsymmetryResult,
    compute_asymmetry,
    find_asymmetry_pairs,
)
from rsidis_ssa.normalization import weight_table_to_df
from rsidis_ssa.pipeline import PipelineResult, run_pipeline_from_yaml

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)


def _configure_logging(verbose: bool) -> None:
    """
    Default (no --verbose): root at INFO — shows high-level summary from both
    analysis.py and rsidis_ssa.pipeline; per-run detail tables and the
    background subtraction table (DEBUG) are suppressed.
    --verbose/-v: rsidis_ssa and __main__ go to DEBUG so detail tables appear;
    root stays at INFO so third-party libs (fsspec, asyncio…) stay quiet.
    LOG_LEVEL env-var sets root level directly (overrides both modes).
    """
    fmt  = logging.Formatter("%(levelname)-8s %(name)s: %(message)s")
    root = logging.getLogger()
    root.handlers.clear()

    env_level = os.environ.get("LOG_LEVEL")
    if env_level:
        root_level = getattr(logging, env_level.upper(), logging.INFO)
        root.setLevel(root_level)
        for name in ("__main__", "rsidis_ssa"):
            logging.getLogger(name).setLevel(logging.NOTSET)
    else:
        root.setLevel(logging.INFO)
        detail_level = logging.DEBUG if verbose else logging.INFO
        for name in ("__main__", "rsidis_ssa"):
            logging.getLogger(name).setLevel(detail_level)

    root.addHandler(_make_handler(fmt))
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


def _make_handler(fmt: logging.Formatter) -> logging.StreamHandler:
    h = logging.StreamHandler()
    h.setFormatter(fmt)
    return h

# ---------------------------------------------------------------------------
# Plot style
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "figure.dpi":        150,
    "axes.labelsize":    10,
    "axes.titlesize":    10,
    "xtick.labelsize":    8,
    "ytick.labelsize":    8,
    "legend.fontsize":    8,
    "axes.linewidth":   0.8,
    "lines.linewidth":  1.2,
})

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bin_centers(h) -> np.ndarray:
    return h.axes[0].centers


def _plot_hist_step(ax, h, label, color, alpha=1.0, lw=1.2):
    """Draw a step histogram with error bars from a boost_histogram."""
    centers   = _bin_centers(h)
    values    = h.values()
    errors    = np.sqrt(np.maximum(h.variances(), 0.0))
    ax.errorbar(centers, values, yerr=errors,
                fmt="none", color=color, alpha=alpha, lw=lw)
    ax.step(np.append(h.axes[0].edges[:-1], h.axes[0].edges[-1]),
            np.append(values, values[-1]),
            where="post", color=color, alpha=alpha, lw=lw, label=label)


def _make_summary_text(sub_results: list[SubtractionResult]) -> str:
    lines = [subtraction_summary(sub_results), ""]
    for r in sub_results:
        lines.append(f"  {r.histogram_name}  [{r.label}]  "
                     f"max|pull| = {r.max_abs_pull:.2f}")
    return "\n".join(lines)


def _ps5_runs(result: PipelineResult) -> set[int]:
    """Return the set of run numbers whose active prescale slot is ps5."""
    ps5_runs: set[int] = set()
    for df in result.run_dfs.values():
        if df is None or df.empty or "ps5" not in df.columns or "ps6" not in df.columns:
            continue
        for _, row in df.iterrows():
            ps5 = pd.to_numeric(row.get("ps5"), errors="coerce")
            ps6 = pd.to_numeric(row.get("ps6"), errors="coerce")
            ps5_on = pd.notna(ps5) and ps5 > 0
            ps6_on = pd.notna(ps6) and ps6 > 0
            if ps5_on and not ps6_on:
                ps5_runs.add(int(row["run"]))
    return ps5_runs


def _set_run_xticks(ax, x_positions, run_numbers, result: PipelineResult, fontsize: int = 6) -> None:
    """
    Set run-number x ticks and color ps5-active runs red.
    """
    ax.set_xticks(x_positions)
    ax.set_xticklabels(run_numbers, rotation=45, ha="right", fontsize=fontsize)
    ps5_runs = _ps5_runs(result)
    for tick, run in zip(ax.get_xticklabels(), run_numbers):
        tick.set_color("crimson" if int(run) in ps5_runs else "black")


# ---------------------------------------------------------------------------
# Weight distributions
# ---------------------------------------------------------------------------

def _draw_weight_panel(ax, label: str, result: PipelineResult) -> None:
    color = {"signal": "steelblue", "eplus": "tomato"}.get(label, "gray")
    wt = result.weight_tables.get(label)

    if wt is None or wt.n_valid == 0:
        ax.set_visible(False)
        return

    df_w = weight_table_to_df(wt.weights)
    weights = df_w["weight"].values
    ax.hist(weights, bins=20, color=color, edgecolor="white", alpha=0.85)
    ax.set_xlabel("per-event weight  $w_r$")
    ax.set_ylabel("runs")
    ax.set_title(f"{label}  ({wt.n_valid} runs)")
    ax.axvline(weights.mean(), color="black", ls="--", lw=1,
               label=f"mean = {weights.mean():.3e}")
    ax.legend()

    if wt.n_excluded:
        ax.text(0.98, 0.97, f"{wt.n_excluded} run(s) excluded",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=7, color="crimson")


def _plot_normcomponents_and_current_vs_run(ax, result: PipelineResult) -> None:
    """
    Plot per-run normalization components and beam current on twin y-axes.

    Left axis  — normalization components
    Right axis — beam current, from the CSV column derived from
                 normalization.charge_column by replacing ``_Q`` with ``_I``.
    """
    charge_col = result.cfg.normalization.charge_column
    current_col = charge_col.replace("_Q", "_I")

    series = [
        ("signal", "o", "steelblue"),
        ("eplus",  "s", "tomato"),
        ("dummy",  "^", "darkorange"),
    ]

    run_data: dict[str, pd.DataFrame] = {}
    all_runs: list[int] = []
    missing_current_labels: list[str] = []

    for label, _, _ in series:
        wt = result.weight_tables.get(label)
        df_runs = result.run_dfs.get(label)
        if wt is None or df_runs is None or df_runs.empty:
            continue

        if current_col not in df_runs.columns:
            missing_current_labels.append(label)
            continue

        df_w = weight_table_to_df(wt.weights)[[
            "run",
            "h_esing_Eff",
            "p_hadron_Eff",
            "comp_livetime",
            "boil_corr",
            "ps_factor",
        ]]
        df_plot = df_runs[["run", current_col]].copy()
        df_plot[current_col] = pd.to_numeric(df_plot[current_col], errors="coerce")
        df_plot = df_plot.merge(df_w, on="run", how="inner").sort_values("run")
        if df_plot.empty:
            continue

        ok = (
            np.isfinite(df_plot["h_esing_Eff"].values)
            & np.isfinite(df_plot["p_hadron_Eff"].values)
            & np.isfinite(df_plot["comp_livetime"].values)
            & np.isfinite(df_plot["boil_corr"].values)
            & np.isfinite(df_plot[current_col].values)
        )
        if not ok.any():
            continue

        run_data[label] = df_plot.loc[
            ok,
            ["run", "h_esing_Eff", "p_hadron_Eff", "comp_livetime",
             "boil_corr", "ps_factor", current_col],
        ].copy()
        all_runs.extend(run_data[label]["run"].astype(int).tolist())

    if not run_data:
        ax.set_visible(False)
        return

    sorted_runs = sorted(set(all_runs))
    run_to_x = {run: idx for idx, run in enumerate(sorted_runs)}
    ax_r = ax.twinx()
    component_specs = [
        ("h_esing_Eff",   "h_esing",  "steelblue",  "-",  5),
        ("p_hadron_Eff",  "p_hadron", "seagreen",   "--", 4),
        ("comp_livetime", "livetime", "darkorange", "-.", 3),
        ("boil_corr",     "boil",     "purple",     ":",  2),
    ]

    for label, marker, _ in series:
        df_plot = run_data.get(label)
        if df_plot is None:
            continue

        runs = df_plot["run"].astype(int).values
        x = np.array([run_to_x[r] for r in runs])
        current_vals = df_plot[current_col].values

        for comp_col, comp_label, comp_color, comp_ls, comp_z in component_specs:
            ax.plot(
                x,
                df_plot[comp_col].values,
                marker=marker,
                color=comp_color,
                ms=4,
                lw=0.9,
                ls=comp_ls,
                alpha=0.9,
                zorder=comp_z,
                label=f"{comp_label} ({label})",
            )
        ax_r.plot(
            x,
            current_vals,
            marker=marker,
            color="black",
            ms=4.5,
            lw=1.2,
            ls="--",
            mfc="white",
            mec="black",
            alpha=0.95,
            zorder=4,
            label=f"current ({label})",
        )

    _set_run_xticks(ax, np.arange(len(sorted_runs)), sorted_runs, result, fontsize=6)
    ax.set_xlim(-0.5, len(sorted_runs) - 0.5)
    ax.set_xlabel("run number")
    ax.set_ylabel("normalization components")
    ax_r.set_ylabel(f"{current_col}  (in uA from CSV)")
    ax_r.tick_params(axis="y", colors="black")
    ax_r.spines["right"].set_color("black")
    ax.set_title("Normalization components and beam current vs run")

    quantity_handles = [
        Line2D([0], [0], color="steelblue", lw=1.2, ls="-",  marker=None, label="h_esing"),
        Line2D([0], [0], color="seagreen", lw=1.2, ls="--", marker=None, label="p_hadron"),
        Line2D([0], [0], color="darkorange", lw=1.2, ls="-.", marker=None, label="livetime"),
        Line2D([0], [0], color="purple", lw=1.2, ls=":", marker=None, label="boil_corr"),
        Line2D([0], [0], color="black", lw=1.2, ls="--", marker=None, label="current"),
    ]
    run_type_handles = [
        Line2D([0], [0], color="0.25", lw=0, marker="o", ms=5, label="signal"),
        Line2D([0], [0], color="0.25", lw=0, marker="s", ms=5, label="eplus"),
        Line2D([0], [0], color="0.25", lw=0, marker="^", ms=5, label="dummy"),
    ]
    legend_quantities = ax.legend(
        handles=quantity_handles,
        fontsize=6,
        ncol=3,
        loc="center left",
        title="quantity",
        title_fontsize=6,
    )
    ax.add_artist(legend_quantities)
    ax.legend(
        handles=run_type_handles,
        fontsize=6,
        ncol=3,
        loc="center right",
        #bbox_to_anchor=(0.0, 0.84),
        title="run type",
        title_fontsize=6,
    )


# ---------------------------------------------------------------------------
# Page 1: coincidence-time distribution
# ---------------------------------------------------------------------------

def _page_ctime(pdf: PdfPages, result: PipelineResult) -> None:
    """
    Three-panel coincidence-time diagnostic page.

    Left   — full range showing both the real peak and random sideband.
    Centre — zoom on the real peak with per-run window boundaries.
    Right  — ctmean vs run number (from CSV) for signal and dummy runs,
             with ctsigma error bars and a weighted-mean constant fit.
    """
    cfg  = result.cfg
    cuts = cfg.cuts

    h_sig = result.ctime_hists["signal"]
    centers = h_sig.axes[0].centers
    sig_vals = h_sig.values()

    # Per-run window parameters for signal
    windows        = result.ctime_run_windows["signal"]
    rand_windows   = result.ctime_random_run_windows["signal"]
    valid_ctmeans  = [c for (c, _) in windows if not np.isnan(c)]
    valid_halfwins = [h for (_, h) in windows]
    mean_ctmean    = float(np.mean(valid_ctmeans))  if valid_ctmeans  else 0.0
    mean_half_win  = float(np.mean(valid_halfwins)) if valid_halfwins else cuts.ctime_real_window_fallback

    fig, (ax_full, ax_zoom, ax_run) = plt.subplots(1, 3, figsize=(17, 4.5))
    fig.suptitle("Coincidence-time distribution", fontsize=11, fontweight="bold")

    # ---- helper: draw on one axis ----
    def _draw_dist(ax):
        ax.step(centers, sig_vals, where="mid", color="black", lw=1.2, label="signal")
        if "eplus" in result.ctime_hists:
            ep_vals = result.ctime_hists["eplus"].values()
            ax.step(centers, ep_vals, where="mid", color="tomato",
                    lw=1.0, alpha=0.7, label="e⁺")
        ax.set_xlabel("Coincidence time (ns)")
        ax.set_ylabel("counts / 0.1 ns")

    # ---- left panel: full range ----
    _draw_dist(ax_full)
    # Real peak shading
    ax_full.axvspan(mean_ctmean - mean_half_win, mean_ctmean + mean_half_win,
                    color="limegreen", alpha=0.18,
                    label=f"real: {mean_ctmean:.2f} ± {mean_half_win:.2f} ns")
    ax_full.axvline(mean_ctmean - mean_half_win, color="limegreen", lw=0.9, ls="--")
    ax_full.axvline(mean_ctmean + mean_half_win, color="limegreen", lw=0.9, ls="--")
    # Discrete random-peak shading: one axvspan per peak index, using mean center across runs.
    # rand_windows is list[list[tuple[float,float]]]: outer = runs, inner = peaks per run.
    n_rand_peaks = cuts.ctime_random_n_peaks_lo + cuts.ctime_random_n_peaks_hi
    for p in range(n_rand_peaks):
        peak_cs = [rand_windows[i][p][0]
                   for i in range(len(rand_windows))
                   if len(rand_windows[i]) > p and not np.isnan(rand_windows[i][p][0])]
        peak_hs = [rand_windows[i][p][1]
                   for i in range(len(rand_windows))
                   if len(rand_windows[i]) > p]
        if not peak_cs:
            continue
        mc = float(np.mean(peak_cs))
        mh = float(np.mean(peak_hs))
        ax_full.axvspan(mc - mh, mc + mh, color="royalblue", alpha=0.15,
                        label="random peaks (mean)" if p == 0 else "_nolegend_")
        ax_full.axvline(mc - mh, color="royalblue", lw=0.9, ls="--")
        ax_full.axvline(mc + mh, color="royalblue", lw=0.9, ls="--")
    # Per-run peak boundaries (thin lines)
    for run_peaks in rand_windows:
        for rc, rhw in run_peaks:
            if np.isnan(rc):
                continue
            ax_full.axvline(rc - rhw, color="royalblue", lw=0.4, alpha=0.3)
            ax_full.axvline(rc + rhw, color="royalblue", lw=0.4, alpha=0.3)
    ax_full.set_title("Full range")
    ax_full.legend(fontsize=7)

    # ---- right panel: zoom on real peak ----
    _draw_dist(ax_zoom)
    zoom_hw = 5.0 * mean_half_win
    ax_zoom.set_xlim(mean_ctmean - zoom_hw, mean_ctmean + zoom_hw)
    # Per-run window lines
    for ctmean_r, hw_r in windows:
        if np.isnan(ctmean_r):
            continue
        ax_zoom.axvline(ctmean_r - hw_r, color="gray", lw=0.5, alpha=0.4)
        ax_zoom.axvline(ctmean_r + hw_r, color="gray", lw=0.5, alpha=0.4)
    # Mean window shading
    ax_zoom.axvspan(mean_ctmean - mean_half_win, mean_ctmean + mean_half_win,
                    color="limegreen", alpha=0.18)
    ax_zoom.axvline(mean_ctmean - mean_half_win, color="limegreen", lw=1.2, ls="--")
    ax_zoom.axvline(mean_ctmean + mean_half_win, color="limegreen", lw=1.2, ls="--")
    ax_zoom.axvline(mean_ctmean, color="limegreen", lw=1.0, ls=":")

    # Config info text box
    n_runs = len(windows)
    center_mode = cfg.cuts.ctime_real_center
    if center_mode == "auto":
        mode_str = f"auto  (±{cuts.ctime_real_nsigma}σ, fallback {cuts.ctime_real_window_fallback} ns)"
    else:
        mode_str = f"fixed = {center_mode} ns"
    info = (f"center: {mode_str}\n"
            f"mean ctmean = {mean_ctmean:.3f} ns\n"
            f"mean half-win = {mean_half_win:.3f} ns\n"
            f"n_skip = {cuts.ctime_random_n_skip}  "
            f"n_peaks = {cuts.ctime_random_n_peaks_lo}+{cuts.ctime_random_n_peaks_hi}\n"
            f"win_scale = 1/{n_rand_peaks}\n"
            f"N runs = {n_runs}")
    ax_zoom.text(0.02, 0.97, info, transform=ax_zoom.transAxes,
                 va="top", ha="left", fontsize=7, family="monospace",
                 bbox=dict(boxstyle="round", fc="0.96", ec="0.8"))
    ax_zoom.set_title("Zoom: real peak  (gray = per-run windows)")

    # ---- right panel: ctmean vs run number (from CSV) ----
    _plot_ctmean_vs_run(ax_run, result)

    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _plot_ctmean_vs_run(ax, result: PipelineResult) -> None:
    """
    Plot ctmean ± ctsigma vs run number (from CSV) for signal and dummy runs.
    Overlays a chi²-weighted constant fit (weighted mean ± error).

    All run types share the same x-axis, ordered by run number globally so
    signal and dummy points never land on the same tick.
    """
    series = [
        ("signal", "o", "steelblue"),
        ("dummy",  "s", "darkorange"),
    ]

    # ── Step 1: collect all valid run numbers across all active run types ──
    run_data: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    all_runs: list[int] = []
    for label, _, _ in series:
        df = result.run_dfs.get(label)
        if df is None or df.empty:
            continue
        runs = df["run"].values.astype(int)
        ct   = pd.to_numeric(df["ctmean"],  errors="coerce").values
        sig  = pd.to_numeric(df["ctsigma"], errors="coerce").values
        ok   = np.isfinite(ct) & np.isfinite(sig) & (sig > 0)
        if not ok.any():
            continue
        run_data[label] = (runs, ct, sig, ok)
        all_runs.extend(runs.tolist())

    if not all_runs:
        ax.set_visible(False)
        return

    # Global sorted unique run list → x position map
    sorted_runs = sorted(set(all_runs))
    run_to_x    = {r: i for i, r in enumerate(sorted_runs)}

    # ── Step 2: plot each run type ──
    all_ctmeans:  list[np.ndarray] = []
    all_ctsigmas: list[np.ndarray] = []

    for label, marker, color in series:
        if label not in run_data:
            continue
        runs, ct, sig, ok = run_data[label]
        x_pos = np.array([run_to_x[r] for r in runs[ok]])
        ax.errorbar(x_pos, ct[ok], yerr=sig[ok],
                    fmt=marker, color=color, ms=5, lw=1.0, capsize=3,
                    label=label, zorder=3)
        all_ctmeans.append(ct[ok])
        all_ctsigmas.append(sig[ok])

    # ── Step 3: x-axis ticks — one tick per run, label = run number ──
    x_all = np.arange(len(sorted_runs))
    _set_run_xticks(ax, x_all, sorted_runs, result, fontsize=6)
    ax.set_xlim(-0.5, len(sorted_runs) - 0.5)

    # ── Step 4: weighted-mean fit (signal only, so dummy runs don't bias it) ──
    if "signal" in run_data:
        _, ct_s, sig_s, ok_s = run_data["signal"]
        w       = 1.0 / sig_s[ok_s]**2
        wmean   = float(np.sum(w * ct_s[ok_s]) / np.sum(w))
        wmean_err = float(1.0 / np.sqrt(np.sum(w)))

        ax.axhline(wmean, color="black", lw=1.2, ls="--",
                   label=f"sig wtd mean = {wmean:.3f} ± {wmean_err:.3f} ns")
        ax.axhspan(wmean - wmean_err, wmean + wmean_err,
                   color="black", alpha=0.10)

        n_sig = int(ok_s.sum())
        info  = (f"wtd mean = {wmean:.3f} ns\n"
                 f"± {wmean_err:.3f} ns\n"
                 f"N(sig) = {n_sig}")
        ax.text(0.98, 0.97, info, transform=ax.transAxes,
                va="top", ha="right", fontsize=7, family="monospace",
                bbox=dict(boxstyle="round", fc="0.96", ec="0.8"))

    ax.set_xlabel("run number")
    ax.set_ylabel("ctmean (ns)")
    ax.set_title("ctmean vs run  (CSV values)")
    ax.legend(fontsize=7)


# ---------------------------------------------------------------------------
# Page 3: helicity & IHWP diagnostics
# ---------------------------------------------------------------------------

def _page_helicity(pdf: PdfPages, result: PipelineResult) -> None:
    """
    3×2 helicity diagnostic page (signal runs only):

    Row 1: raw T_helicity_hel distribution | IHWP state vs run
    Row 2: h+ vs h− event counts per run   | n_h+ / n_h− ratio
    Row 3: Q_hp vs Q_hm charge per run     | Q_hp / Q_hm ratio
           (charge/2 for each when use_helicity_gated_charge=False)
    """
    ny = result.normyield_per_run.get("signal")
    raw_hel = result.raw_helicity.get("signal")
    if ny is None or ny.empty:
        return

    use_hel_charge = result.cfg.normalization.use_helicity_gated_charge

    ny = ny.sort_values("run").reset_index(drop=True)
    runs = ny["run"].astype(str).tolist()
    x    = np.arange(len(ny))

    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    hel_charge_note = ("helicity-gated charge active"
                       if use_hel_charge else "helicity-gated charge inactive")
    fig.suptitle(f"Helicity & IHWP diagnostics  (signal runs)  [{hel_charge_note}]",
                 fontsize=11, fontweight="bold")
    ax_rawh  = axes[0, 0]
    ax_ihwp  = axes[0, 1]
    ax_split = axes[1, 0]
    ax_ratio = axes[1, 1]
    ax_qsplit = axes[2, 0]
    ax_qratio = axes[2, 1]

    # ── Row 1 left: raw helicity distribution ────────────────────────────────
    if raw_hel is not None and len(raw_hel) > 0:
        unique_vals = np.unique(raw_hel)
        counts      = [int((raw_hel == v).sum()) for v in unique_vals]
        bar_colors  = ["steelblue" if v > 0 else "tomato" for v in unique_vals]
        ax_rawh.bar([str(int(v)) for v in unique_vals], counts,
                    color=bar_colors, edgecolor="black", linewidth=0.7)
        for bar, cnt in zip(ax_rawh.patches, counts):
            ax_rawh.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                         f"{cnt:,}", ha="center", va="bottom", fontsize=9)
        total = len(raw_hel)
        ax_rawh.set_xlabel("T_helicity_hel (raw DAQ value)")
        ax_rawh.set_ylabel("Events (real window, all runs)")
        ax_rawh.set_title(f"Raw helicity distribution  (N = {total:,})")
    else:
        ax_rawh.text(0.5, 0.5, "No helicity data", transform=ax_rawh.transAxes,
                     ha="center", va="center")
        ax_rawh.set_title("Raw helicity distribution")

    # ── Row 1 right: IHWP state vs run number ────────────────────────────────
    ihwp_vals  = ny["IHWP"].str.upper() if "IHWP" in ny.columns else pd.Series(["?"] * len(ny))
    bar_colors = ["steelblue" if s == "OUT" else "tomato" for s in ihwp_vals]
    ax_ihwp.bar(x, 1, color=bar_colors, edgecolor="black", linewidth=0.5, width=0.7)
    ax_ihwp.set_xticks(x)
    ax_ihwp.set_xticklabels(runs, rotation=45, ha="right", fontsize=7)
    ax_ihwp.set_yticks([0.5])
    ax_ihwp.set_yticklabels([""])
    ax_ihwp.set_ylim(0, 1.3)
    ax_ihwp.set_xlabel("Run number")
    ax_ihwp.set_title("IHWP state per run")
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color="steelblue", label="OUT"),
        plt.Rectangle((0, 0), 1, 1, color="tomato",    label="IN"),
    ]
    ax_ihwp.legend(handles=legend_handles, fontsize=8, loc="upper right")
    for xi, state in zip(x, ihwp_vals):
        ax_ihwp.text(xi, 0.5, state, ha="center", va="center",
                     fontsize=7, fontweight="bold", color="white")

    # ── Row 2 left: h+ vs h− counts per run ──────────────────────────────────
    if "n_hplus" in ny.columns and "n_hminus" in ny.columns:
        ax_split.bar(x - 0.2, ny["n_hplus"],  0.4, label="h+",
                     color="steelblue", edgecolor="black", linewidth=0.5)
        ax_split.bar(x + 0.2, ny["n_hminus"], 0.4, label="h−",
                     color="tomato",    edgecolor="black", linewidth=0.5)
        ax_split.set_xticks(x)
        ax_split.set_xticklabels(runs, rotation=45, ha="right", fontsize=7)
        ax_split.set_ylabel("Events (real window)")
        ax_split.set_title("Effective h+ / h− counts per run")
        ax_split.legend(fontsize=8)
    else:
        ax_split.text(0.5, 0.5, "n_hplus/n_hminus not available",
                      transform=ax_split.transAxes, ha="center", va="center")
        ax_split.set_title("h+ / h− counts per run")

    # ── Row 2 right: h+ / h− count ratio per run ─────────────────────────────
    if "n_hplus" in ny.columns and "n_hminus" in ny.columns:
        denom = ny["n_hminus"].replace(0, np.nan)
        ratio = ny["n_hplus"] / denom
        ax_ratio.axhline(1.0, color="gray", ls="--", lw=0.9, zorder=1)
        ax_ratio.plot(x, ratio, "o-", color="purple", ms=6, lw=1.2, zorder=3)
        ax_ratio.set_xticks(x)
        ax_ratio.set_xticklabels(runs, rotation=45, ha="right", fontsize=7)
        ax_ratio.set_ylabel("n_h+ / n_h−")
        ax_ratio.set_title("Event count helicity balance per run")
        n_hp_tot = int(ny["n_hplus"].sum())
        n_hm_tot = int(ny["n_hminus"].sum())
        global_ratio = n_hp_tot / n_hm_tot if n_hm_tot > 0 else float("nan")
        ax_ratio.text(0.97, 0.97,
                      f"Total h+ = {n_hp_tot:,}\nTotal h− = {n_hm_tot:,}\n"
                      f"Global ratio = {global_ratio:.4f}",
                      transform=ax_ratio.transAxes, va="top", ha="right",
                      fontsize=8, family="monospace",
                      bbox=dict(boxstyle="round", fc="0.96", ec="0.8"))
    else:
        ax_ratio.text(0.5, 0.5, "n_hplus/n_hminus not available",
                      transform=ax_ratio.transAxes, ha="center", va="center")
        ax_ratio.set_title("Event count helicity balance per run")

    # ── Row 3 left: Q_hp vs Q_hm charge per run ──────────────────────────────
    if "charge_hp" in ny.columns and "charge_hm" in ny.columns:
        ax_qsplit.bar(x - 0.2, ny["charge_hp"], 0.4, label="Q_h+",
                      color="steelblue", edgecolor="black", linewidth=0.5)
        ax_qsplit.bar(x + 0.2, ny["charge_hm"], 0.4, label="Q_h−",
                      color="tomato",    edgecolor="black", linewidth=0.5)
        ax_qsplit.set_xticks(x)
        ax_qsplit.set_xticklabels(runs, rotation=45, ha="right", fontsize=7)
        ax_qsplit.set_ylabel("Charge [mC]")
        note = "" if use_hel_charge else "  (= Q_tot/2, flag disabled)"
        ax_qsplit.set_title(f"Helicity-gated charge per run{note}")
        ax_qsplit.legend(fontsize=8)
    else:
        ax_qsplit.text(0.5, 0.5, "charge_hp/charge_hm not available",
                       transform=ax_qsplit.transAxes, ha="center", va="center")
        ax_qsplit.set_title("Helicity-gated charge per run")

    # ── Row 3 right: Q_hp / Q_hm ratio per run ───────────────────────────────
    if "charge_hp" in ny.columns and "charge_hm" in ny.columns:
        q_denom = ny["charge_hm"].replace(0, np.nan)
        q_ratio = ny["charge_hp"] / q_denom
        ax_qratio.axhline(1.0, color="gray", ls="--", lw=0.9, zorder=1)
        ax_qratio.plot(x, q_ratio, "o-", color="darkorange", ms=6, lw=1.2, zorder=3)
        ax_qratio.set_xticks(x)
        ax_qratio.set_xticklabels(runs, rotation=45, ha="right", fontsize=7)
        ax_qratio.set_ylabel("Q_h+ / Q_h−")
        note = "" if use_hel_charge else "  (= 1 by construction, flag disabled)"
        ax_qratio.set_title(f"Charge helicity balance per run{note}")
        q_hp_tot = ny["charge_hp"].sum()
        q_hm_tot = ny["charge_hm"].sum()
        global_q_ratio = q_hp_tot / q_hm_tot if q_hm_tot > 0 else float("nan")
        ax_qratio.text(0.97, 0.97,
                       f"Total Q_h+ = {q_hp_tot:.3f} mC\n"
                       f"Total Q_h− = {q_hm_tot:.3f} mC\n"
                       f"Global ratio = {global_q_ratio:.4f}",
                       transform=ax_qratio.transAxes, va="top", ha="right",
                       fontsize=8, family="monospace",
                       bbox=dict(boxstyle="round", fc="0.96", ec="0.8"))
    else:
        ax_qratio.text(0.5, 0.5, "charge_hp/charge_hm not available",
                       transform=ax_qratio.transAxes, ha="center", va="center")
        ax_qratio.set_title("Charge helicity balance per run")

    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


# Page 4: normalized yield vs run number
# ---------------------------------------------------------------------------

def _page_normyield(pdf: PdfPages, result: PipelineResult) -> None:
    """
    Three-row diagnostic page per active run type.

    Row 1 — normalized yield: workflow vs CSV normyield per run.
    Row 2 — raw event counts: workflow (n_real, n_random, n_rsc)
             vs CSV (coin, randoms, ransubcoin) per run.
             This row isolates where the discrepancy originates.
    Row 3 — signal run-weight distribution, plus normalization components
             and beam current vs run number for signal/eplus/dummy.

    Flagged runs (|residual| > 2%) are marked with a red × on row 1.
    """
    labels = [l for l in ("signal", "eplus", "dummy")
              if l in result.normyield_per_run
              and not result.normyield_per_run[l].empty]
    if not labels:
        return

    n_label_cols = len(labels)
    n_cols = max(n_label_cols, 2)
    fig, all_axes = plt.subplots(3, n_cols,
                                 figsize=(5.5 * n_cols, 12.0),
                                 sharey=False)
    # Ensure 2-D indexing even for a single column
    if n_cols == 1:
        all_axes = [[all_axes[0]], [all_axes[1]], [all_axes[2]]]
    top_axes, mid_axes, bot_axes = all_axes[0], all_axes[1], all_axes[2]

    fig.suptitle("Normalized yield, raw counts, run weights, and beam current",
                 fontsize=11, fontweight="bold")

    run_colors = {"signal": "steelblue", "eplus": "tomato", "dummy": "darkorange"}

    for idx, label in enumerate(labels):
        ny_df = result.normyield_per_run[label].sort_values("run")
        wt    = result.weight_tables[label]
        color = run_colors.get(label, "gray")

        runs = ny_df["run"].values
        x    = np.arange(len(runs))

        # ── Row 1: normyield comparison ──────────────────────────────────────
        ax = top_axes[idx]

        ax.plot(x, ny_df["normyield_workflow"].values,
                "o-", color=color, ms=4, lw=1.2,
                label="workflow" if idx == 0 else "_nolegend_")

        csv_vals = ny_df.get("normyield_csv", pd.Series(dtype=float)).values
        if not np.all(np.isnan(csv_vals.astype(float))):
            ax.plot(x, csv_vals,
                    "s--", color="black", ms=4, lw=0.9, alpha=0.7,
                    label="CSV normyield" if idx == 0 else "_nolegend_")

        flagged_mask = ny_df["flagged"].fillna(False).values
        if flagged_mask.any():
            ax.scatter(x[flagged_mask], ny_df["normyield_workflow"].values[flagged_mask],
                       marker="x", color="crimson", s=80, zorder=5, lw=2,
                       label="flagged (>2%)" if idx == 0 else "_nolegend_")

        _set_run_xticks(ax, x, runs, result, fontsize=6)
        ax.set_xlabel("run number")
        ax.set_ylabel("normyield  (counts mC⁻¹)")
        ax.set_title(f"{label}  ({wt.n_valid} runs,  Q_tot = {wt.Q_tot:.1f} mC)")

        res = ny_df["residual_pct"].dropna()
        if len(res) > 0:
            n_flagged  = int(flagged_mask.sum())
            median_res = float(res.median())
            max_res    = float(res.abs().max())
            info = (f"median Δ = {median_res:+.2f}%\n"
                    f"max|Δ|   = {max_res:.2f}%\n"
                    f"flagged  = {n_flagged}")
            ax.text(0.98, 0.97, info, transform=ax.transAxes,
                    va="top", ha="right", fontsize=7, family="monospace",
                    bbox=dict(boxstyle="round", fc="0.96", ec="0.8"))

        if idx == 0:
            ax.legend(fontsize=7)

        # ── Row 2: raw count comparison ──────────────────────────────────────
        ax2 = mid_axes[idx]

        has_csv_counts = all(c in ny_df.columns for c in
                             ("csv_coin", "csv_randoms", "csv_ransubcoin"))

        # Workflow counts
        ax2.plot(x, ny_df["n_real"].values,
                 "o-",  color=color, ms=4, lw=1.2,
                 label="n_real (wf)" if idx == 0 else "_nolegend_")
        ax2.plot(x, ny_df["n_rand_scaled"].values,
                 "^--", color=color, ms=4, lw=1.0, alpha=0.6,
                 label="n_rand×scale (wf)" if idx == 0 else "_nolegend_")
        ax2.plot(x, ny_df["n_rsc"].values,
                 "D-",  color=color, ms=4, lw=1.4,
                 label="n_rsc (wf)" if idx == 0 else "_nolegend_")

        if has_csv_counts:
            ax2.plot(x, ny_df["csv_coin"].values,
                     "o--",  color="black", ms=4, lw=1.2, alpha=0.7,
                     label="coin (CSV)" if idx == 0 else "_nolegend_")
            ax2.plot(x, ny_df["csv_randoms"].values,
                     "^:",   color="black", ms=4, lw=1.0, alpha=0.5,
                     label="randoms (CSV)" if idx == 0 else "_nolegend_")
            ax2.plot(x, ny_df["csv_ransubcoin"].values,
                     "D--",  color="black", ms=4, lw=1.4, alpha=0.7,
                     label="ransubcoin (CSV)" if idx == 0 else "_nolegend_")

            # Annotate the ratio n_real / coin for each run
            coin_vals = ny_df["csv_coin"].values.astype(float)
            nreal_vals = ny_df["n_real"].values.astype(float)
            ok = (coin_vals != 0) & np.isfinite(coin_vals) & np.isfinite(nreal_vals)
            if ok.any():
                ratios = nreal_vals[ok] / coin_vals[ok]
                med_r  = float(np.median(ratios))
                ax2.text(0.02, 0.97,
                         f"median n_real/coin = {med_r:.3f}",
                         transform=ax2.transAxes, va="top", ha="left",
                         fontsize=7, family="monospace",
                         bbox=dict(boxstyle="round", fc="0.96", ec="0.8"))

        _set_run_xticks(ax2, x, runs, result, fontsize=6)
        ax2.set_xlabel("run number")
        ax2.set_ylabel("event counts")
        ax2.set_title(f"{label} — raw count comparison")

        if idx == 0:
            ax2.legend(fontsize=6)

    for idx in range(n_label_cols, n_cols):
        top_axes[idx].set_visible(False)
        mid_axes[idx].set_visible(False)

    if len(bot_axes) >= 1:
        _draw_weight_panel(bot_axes[0], "signal", result)
    if len(bot_axes) >= 2:
        _plot_normcomponents_and_current_vs_run(bot_axes[1], result)
    for ax in bot_axes[2:]:
        ax.set_visible(False)

    skips = {k: len(v) for k, v in result.file_skips.items() if v}
    if skips:
        note = "Files not found: " + ", ".join(f"{k}={n}" for k, n in skips.items())
        fig.text(0.5, 0.01, note, ha="center", fontsize=7, color="crimson")

    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Pages 3+: per-histogram subtraction overview
# ---------------------------------------------------------------------------

def _page_histogram(
    pdf: PdfPages,
    hname: str,
    xlabel: str,
    h_sig_real:   object,
    h_sig_random: object,
    h_ep_real:    object | None,
    h_ep_random:  object | None,
    h_dum_ran:    object | None,
    h_after_ran:  object,
    h_after_ep:   object | None,
    h_after_dum:  object | None,
    sub_results:  list[SubtractionResult],
    dummy_scale:  float = 1.0,
) -> None:
    """One figure per histogram showing all subtraction steps."""
    has_ep  = h_ep_real is not None
    has_dum = h_dum_ran is not None
    # Scale the dummy histogram to the fraction actually subtracted.
    # h_dum_ran carries the full dummy yield; only dummy_scale × that is removed.
    h_dum_scaled = (h_dum_ran * dummy_scale) if has_dum else None
    dum_label = (f"dummy×{dummy_scale:.3f} − random"
                 if abs(dummy_scale - 1.0) > 1e-6 else "dummy − random")
    n_cols  = 2 + int(has_ep) + int(has_dum)
    fig, axes = plt.subplots(1, n_cols, figsize=(4.5 * n_cols, 4.0), sharey=False)
    if n_cols == 1:
        axes = [axes]
    fig.suptitle(f"Histogram: {hname}", fontsize=11, fontweight="bold")

    col = 0  # current panel index

    # Panel 1 — signal real vs random (before subtraction)
    ax = axes[col]; col += 1
    _plot_hist_step(ax, h_sig_real,   "signal real",         "black")
    _plot_hist_step(ax, h_sig_random, "signal random×scale", "royalblue", alpha=0.7)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("yield  [a.u. / mC]")
    ax.set_title("Before random subtraction")
    ax.legend(fontsize=7)

    # Panel 2 — after random subtraction (+ e⁺ / dummy overlay as applicable)
    ax = axes[col]; col += 1
    _plot_hist_step(ax, h_sig_real,  "signal real",     "black", alpha=0.35)
    _plot_hist_step(ax, h_after_ran, "signal − random", "black")
    if has_ep:
        _plot_hist_step(ax, h_ep_real,   "e⁺ real",         "tomato", alpha=0.7)
        _plot_hist_step(ax, h_ep_random, "e⁺ random×scale", "tomato", alpha=0.4)
    elif has_dum:
        _plot_hist_step(ax, h_dum_scaled, dum_label, "darkorange", alpha=0.7)
    ax.set_xlabel(xlabel)
    ax.set_title("After random subtraction")
    ax.legend(fontsize=7)

    # Panel 3 — after e⁺ subtraction (if applicable)
    if has_ep and h_after_ep is not None:
        ax = axes[col]; col += 1
        _plot_hist_step(ax, h_after_ran, "signal − random", "black",  alpha=0.35)
        ep_sub_ran = subtract_randoms(h_ep_real, h_ep_random, scale=1.0)
        _plot_hist_step(ax, ep_sub_ran,  "e⁺ − random",     "tomato", alpha=0.7)
        _plot_hist_step(ax, h_after_ep,  "after e⁺ sub",    "black")
        if has_dum:
            _plot_hist_step(ax, h_dum_scaled, dum_label, "darkorange", alpha=0.6)
        ax.set_xlabel(xlabel)
        ax.set_title("After e⁺ subtraction")
        ax.legend(fontsize=7)

    # Panel 4 — after dummy subtraction (if applicable)
    if has_dum and h_after_dum is not None:
        ax = axes[col]; col += 1
        prev = h_after_ep if (has_ep and h_after_ep is not None) else h_after_ran
        _plot_hist_step(ax, prev,         "before dummy sub", "black",      alpha=0.35)
        _plot_hist_step(ax, h_dum_scaled, dum_label,         "darkorange", alpha=0.7)
        _plot_hist_step(ax, h_after_dum, "final",            "black")
        ax.set_xlabel(xlabel)
        ax.set_title("After dummy subtraction")
        ax.legend(fontsize=7)

    # Pull annotation — map each label to its result panel index
    label_to_col = {"random": 1, "eplus": 2 if has_ep else None,
                    "dummy": 2 + int(has_ep) if has_dum else None}
    for r in sub_results:
        if r.histogram_name != hname:
            continue
        panel_idx = label_to_col.get(r.label)
        if panel_idx is None or panel_idx >= len(axes):
            continue
        axes[panel_idx].text(
            0.98, 0.97,
            f"{r.label}: {r.fraction_subtracted*100:.1f}% sub'd  "
            f"max|pull|={r.max_abs_pull:.1f}",
            transform=axes[panel_idx].transAxes, ha="right", va="top",
            fontsize=6, color="dimgray",
        )

    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2-D histogram page
# ---------------------------------------------------------------------------

def _page_histogram_2d(
    pdf:     "PdfPages",
    hname:   str,
    hcfg:    object,
    h_final: object,
) -> None:
    """One figure for a 2-D histogram showing the final subtracted yield."""
    h = h_final
    x_edges = h.axes[0].edges      # type: ignore[attr-defined]
    y_edges = h.axes[1].edges      # type: ignore[attr-defined]
    values = h.values()             # type: ignore[attr-defined]  shape (bins_x, bins_y)

    if hname == "pt_vs_phi":
        _page_pt_vs_phi_2d(pdf, hcfg, x_edges, y_edges, values)
        return

    fig, ax = plt.subplots(figsize=(7.0, 5.5))
    fig.suptitle(f"Histogram: {hname}  [2D]", fontsize=11, fontweight="bold")

    im = ax.pcolormesh(x_edges, y_edges, values.T, cmap="viridis", shading="auto")
    fig.colorbar(im, ax=ax, label="yield  [a.u. / mC]")
    ax.set_xlabel(getattr(hcfg, "xlabel", None) or hname)
    ax.set_ylabel(getattr(hcfg, "ylabel", None) or getattr(hcfg, "branch_y", "y"))
    ax.set_title(hname)

    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _draw_pt_vs_phi_2d(
    ax:       object,
    hcfg:     object,
    x_edges:  np.ndarray,
    y_edges:  np.ndarray,
    values:   np.ndarray,
) -> None:
    """Draw the P_T vs phi polar-style Cartesian plot onto an existing axis."""
    plot_values = values.T.copy()
    plot_values[(plot_values == 0.0) | ~np.isfinite(plot_values)] = np.nan
    cmap = plt.get_cmap("jet").copy()
    cmap.set_bad("white")

    im = ax.pcolormesh(
        x_edges, y_edges, plot_values,
        cmap=cmap, shading="auto", zorder=2,
    )
    im.set_rasterized(True)

    xmin = float(getattr(hcfg, "xmin", np.min(x_edges)))
    xmax = float(getattr(hcfg, "xmax", np.max(x_edges)))
    ymin = float(getattr(hcfg, "ymin", np.min(y_edges)))
    ymax = float(getattr(hcfg, "ymax", np.max(y_edges)))
    r_frame = max(abs(xmin), abs(xmax), abs(ymin), abs(ymax))
    xmin, xmax = -r_frame, r_frame
    ymin, ymax = -r_frame, r_frame

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("")
    ax.set_ylabel("")

    ax.axhline(0.0, color="0.55", lw=0.7, zorder=3)
    ax.axvline(0.0, color="0.55", lw=0.7, zorder=3)

    ring_values = np.arange(0.1, 0.92 * r_frame + 1.0e-9, 0.1)
    for r in ring_values:
        ax.add_patch(plt.Circle(
            (0.0, 0.0), r,
            fill=False, ec="red", ls=":", lw=0.85, alpha=0.9, zorder=4,
        ))
        label_angle = np.deg2rad(24.0)
        if np.isclose((r * 10) % 2, 0.0):
            ax.text(
                r * np.cos(label_angle) + 0.015 * r_frame,
                r * np.sin(label_angle) + 0.015 * r_frame,
                rf"$P_T = {r:.1f}\ \mathrm{{GeV}}$",
                color="red", fontsize=8, fontweight="bold",
                ha="left", va="center", zorder=5,
            )

    ax.text(0.5, 1.02, r"$\phi_h = 90^\circ$", transform=ax.transAxes,
            ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.text(1.01, 0.5, r"$\phi_h = 0^\circ$", transform=ax.transAxes,
            ha="left", va="center", fontsize=10)
    ax.text(0.5, -0.02, r"$\phi_h = 270^\circ$", transform=ax.transAxes,
            ha="center", va="top", fontsize=10)
    ax.text(-0.01, 0.5, r"$\phi_h = 180^\circ$", transform=ax.transAxes,
            ha="right", va="center", fontsize=10)

    major = 0.2 if r_frame <= 1.1 else 0.5
    minor = major / 4.0
    ax.set_xticks(np.arange(np.ceil(xmin / major) * major, xmax + 0.5 * major, major))
    ax.set_yticks(np.arange(np.ceil(ymin / major) * major, ymax + 0.5 * major, major))
    ax.set_xticks(np.arange(np.ceil(xmin / minor) * minor, xmax + 0.5 * minor, minor), minor=True)
    ax.set_yticks(np.arange(np.ceil(ymin / minor) * minor, ymax + 0.5 * minor, minor), minor=True)
    ax.tick_params(which="major", direction="in", length=6, width=0.6,
                   labelbottom=False, labelleft=False, labelright=False, labeltop=False)
    ax.tick_params(which="minor", direction="in", length=3, width=0.4)

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("0.25")

def _page_pt_vs_phi_2d(
    pdf:      "PdfPages",
    hcfg:     object,
    x_edges:  np.ndarray,
    y_edges:  np.ndarray,
    values:   np.ndarray,
) -> None:
    """Draw P_T vs phi in a polar-style Cartesian canvas (standalone page)."""
    fig, ax = plt.subplots(figsize=(6.4, 5.1))
    _draw_pt_vs_phi_2d(ax, hcfg, x_edges, y_edges, values)
    fig.tight_layout(pad=1.7)
    pdf.savefig(fig)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Asymmetry page
# ---------------------------------------------------------------------------

def _page_asymmetry(
    pdf:        PdfPages,
    result:     PipelineResult,
    final:      dict[str, object],
    config_dir: Path | None = None,
    pairs:      list[tuple[str, str, str]] | None = None,
    subtitle:   str | None = None,
) -> list[AsymmetryResult]:
    """
    Compute and plot the beam SSA (A_LU^sinφ) for every helicity-split pair.

    For each pair (foo_hplus, foo_hminus) found in *final*, one subplot is
    produced showing:
      - per-bin A_phys ± σ  (blue circles with error bars)
      - fitted A × sin(φ) curve  (red line)
      - zero reference line  (gray dashed)
      - text box: fitted amplitude, uncertainty, χ²/ndf, P_beam

    Parameters
    ----------
    pairs : list of (base, hplus_name, hminus_name) or None
        If None, auto-detected from *final* keys via find_asymmetry_pairs().
        Pass an explicit list to restrict to a specific subset of pairs.
    subtitle : str or None
        Optional second line added to the page suptitle (e.g. kinematic bin label).

    Returns the list of AsymmetryResult objects for use by the statistics page.
    Returns an empty list and skips the page when no pairs are found or
    beam_polarization is not configured.
    """
    cfg = result.cfg
    if pairs is None:
        pairs = find_asymmetry_pairs(list(final.keys()))
    if not pairs:
        return []

    try:
        P_beam = cfg.beam_polarization(config_dir)
    except (FileNotFoundError, KeyError) as exc:
        logger.warning("Skipping asymmetry page: %s", exc)
        return []

    asym_results: list[AsymmetryResult] = []
    n_pairs = len(pairs)
    ZOOM_LO, ZOOM_HI = -0.1, 0.1   # y-axis limits for the zoomed column

    # Layout: one row per pair, two columns (full + zoomed)
    fig, axes_grid = plt.subplots(
        n_pairs, 2,
        figsize=(11.0, 4.5 * n_pairs),
        sharey=False,
    )
    # Normalise to 2-D array regardless of n_pairs
    if n_pairs == 1:
        axes_grid = axes_grid.reshape(1, 2)

    _title = (
        f"Beam SSA  —  $A_{{LU}}^{{\\sin\\phi}}$   "
        f"($P_{{\\rm beam}} = {P_beam:.2f}$)"
    )
    if subtitle:
        _title += f"\n{subtitle}"
    fig.suptitle(_title, fontsize=11, fontweight="bold")

    phi_fit = np.linspace(-np.pi, np.pi, 200)

    for row, (base, hplus_name, hminus_name) in enumerate(pairs):
        h_plus  = final[hplus_name]
        h_minus = final[hminus_name]

        ar = compute_asymmetry(h_plus, h_minus, P_beam, base)
        asym_results.append(ar)

        valid = np.isfinite(ar.A_phys) & np.isfinite(ar.A_phys_err)
        phi_v = ar.phi_centers[valid]
        A_v   = ar.A_phys[valid]
        dA_v  = ar.A_phys_err[valid]

        # Bin half-width for horizontal error bars
        hw = 0.5 * (ar.phi_centers[1] - ar.phi_centers[0]) if len(ar.phi_centers) > 1 else 0.0

        fit_curve = ar.amplitude * np.sin(phi_fit) if np.isfinite(ar.amplitude) else None

        ax_full = axes_grid[row, 0]
        ax_zoom = axes_grid[row, 1]

        _text_kw = dict(va="top", ha="right", fontsize=9, family="monospace",
                        zorder=10,
                        bbox=dict(boxstyle="round", fc="0.96", ec="0.8"))

        if np.isfinite(ar.amplitude):
            info = (
                rf"$A_{{LU}}^{{\sin\phi}} = {ar.amplitude:+.4f} \pm {ar.amplitude_err:.4f}$"
                f"\n$\\chi^2/\\mathrm{{ndf}} = {ar.chi2_ndf:.2f}$"
                f"\n$N_{{\\rm bins}} = {ar.n_bins_used}$"
                f"\n$P_{{\\rm beam}} = {P_beam:.2f}$"
            )
            info_short = (
                rf"$A_{{LU}}^{{\sin\phi}} = {ar.amplitude:+.4f} \pm {ar.amplitude_err:.4f}$"
                f"\n$\\chi^2/\\mathrm{{ndf}} = {ar.chi2_ndf:.2f}$"
            )
        else:
            info = info_short = "fit: insufficient bins"

        # ── Full panel ───────────────────────────────────────────────────────
        ax_full.axhline(0.0, color="gray", lw=0.9, ls="--", zorder=1)
        ax_full.errorbar(phi_v, A_v, yerr=dA_v, xerr=hw,
                         fmt="o", color="steelblue", ms=5, lw=1.2, capsize=3,
                         label=r"$A_{\rm phys}(\phi_k)$", zorder=3)
        if fit_curve is not None:
            ax_full.plot(phi_fit, fit_curve,
                         color="tomato", lw=1.5, zorder=2,
                         label=rf"$A \sin\phi$  fit")
        ax_full.set_xlim(-np.pi * 1.05, np.pi * 1.05)
        ax_full.set_xlabel(r"$\phi_{pq}$ (rad)")
        ax_full.set_ylabel(r"$A_{LU}$")
        ax_full.set_title(base)
        ax_full.legend(fontsize=7)
        ax_full.text(0.97, 0.97, info, transform=ax_full.transAxes, **_text_kw)

        # ── Zoom panel ───────────────────────────────────────────────────────
        ax_zoom.axhline(0.0, color="gray", lw=0.9, ls="--", zorder=1)
        ax_zoom.errorbar(phi_v, A_v, yerr=dA_v, xerr=hw,
                         fmt="o", color="steelblue", ms=5, lw=1.2, capsize=3,
                         zorder=3)
        if fit_curve is not None:
            ax_zoom.plot(phi_fit, fit_curve,
                         color="tomato", lw=1.5, zorder=2)
        ax_zoom.set_xlim(-np.pi * 1.05, np.pi * 1.05)
        ax_zoom.set_xlabel(r"$\phi_{pq}$ (rad)")
        ax_zoom.set_ylabel(r"$A_{LU}$")
        ax_zoom.set_ylim(ZOOM_LO, ZOOM_HI)
        ax_zoom.set_title(f"{base}  [zoom: {ZOOM_LO}, {ZOOM_HI}]", fontsize=9)
        ax_zoom.text(0.36, 0.97, info_short, transform=ax_zoom.transAxes, **_text_kw)

        logger.info(
            "Asymmetry [%s]  A_LU^sinphi = %+.4f ± %.4f  chi2/ndf = %.2f  "
            "(%d bins, P_beam=%.2f)",
            base,
            ar.amplitude if np.isfinite(ar.amplitude) else float("nan"),
            ar.amplitude_err if np.isfinite(ar.amplitude_err) else float("nan"),
            ar.chi2_ndf if np.isfinite(ar.chi2_ndf) else float("nan"),
            ar.n_bins_used, P_beam,
        )

    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)
    return asym_results


# ---------------------------------------------------------------------------
# Asymmetry CSV export
# ---------------------------------------------------------------------------

def _write_asymmetry_csv(path: Path, asym_results: list[AsymmetryResult]) -> None:
    """Write phi bin centers, A_phys, and A_phys_err for all pairs to a CSV."""
    rows = []
    for ar in asym_results:
        for phi, A, dA in zip(ar.phi_centers, ar.A_phys, ar.A_phys_err):
            rows.append({
                "histogram": ar.histogram_name,
                "phi_center": phi,
                "A_phys":     A,
                "A_phys_err": dA,
            })
    df = pd.DataFrame(rows, columns=["histogram", "phi_center", "A_phys", "A_phys_err"])
    df.to_csv(path, index=False)
    logger.info("Asymmetry CSV → %s", path)


_PARTICLE_LABEL = {"PI-SIDIS": "pi-", "PI+SIDIS": "pi+"}


# ---------------------------------------------------------------------------
# ROOT file export
# ---------------------------------------------------------------------------

def _write_root_file(
    path: Path,
    result: PipelineResult,
    signal_after_random: dict[str, object],
    signal_after_eplus:  dict[str, object],
    dummy_subtracted:    dict[str, object],
    final:               dict[str, object],
) -> None:
    """
    Write all per-histogram yields to a ROOT file with flat key names.

    Naming convention  (<hname> = histogram name from config)
    -----------------
    <hname>                  fully background-subtracted yield  (= "final")
    <hname>_sig_real         raw signal real-window yield
    <hname>_sig_rand         raw signal random-window yield (scaled)
    <hname>_sig_rsc          signal after random subtraction
    <hname>_sig_ep           signal after e⁺ subtraction       (if e⁺ enabled)
    <hname>_ep_real          raw e⁺ real-window yield           (if e⁺ enabled)
    <hname>_ep_rand          raw e⁺ random-window yield         (if e⁺ enabled)
    <hname>_dum_real         raw dummy real-window yield        (if dummy enabled)
    <hname>_dum_rsc          dummy after random subtraction     (if dummy enabled)

    Flat keys are required so that uproot sets each TH1's internal ROOT name
    (fName) equal to the full key string.  With subdirectory keys uproot sets
    fName to only the last path component, causing all same-named histograms
    in different directories to share a single ROOT name — ROOT's gDirectory
    then collapses them to one object in memory, breaking Draw("SAME").

    Each histogram is deep-copied before writing to ensure independent numpy
    storage: several intermediate histograms share the same Python object
    (e.g. final == signal_after_eplus when dummy subtraction is disabled),
    and without copying they map to the same underlying numpy arrays.
    """
    import copy
    import uproot

    cfg = result.cfg
    with uproot.recreate(str(path)) as out:
        for hcfg in cfg.effective_histograms():
            hname  = hcfg.name
            ndim   = "2D" if hcfg.is_2d else "1D"

            out[f"{hname}"]          = copy.deepcopy(final[hname])
            out[f"{hname}_sig_real"] = copy.deepcopy(result.signal.real[hname])
            out[f"{hname}_sig_rand"] = copy.deepcopy(result.signal.random[hname])
            out[f"{hname}_sig_rsc"]  = copy.deepcopy(signal_after_random[hname])

            if result.eplus is not None:
                out[f"{hname}_sig_ep"]  = copy.deepcopy(signal_after_eplus[hname])
                out[f"{hname}_ep_real"] = copy.deepcopy(result.eplus.real[hname])
                out[f"{hname}_ep_rand"] = copy.deepcopy(result.eplus.random[hname])

            if result.dummy is not None:
                out[f"{hname}_dum_real"] = copy.deepcopy(result.dummy.real[hname])
                out[f"{hname}_dum_rsc"]  = copy.deepcopy(dummy_subtracted[hname])

            logger.debug("  wrote %s  [%s]  sum=%.3g", hname, ndim,
                         float(final[hname].values().sum()))

    logger.info("ROOT file → %s", path)


def _write_kinematic_summary(
    path: Path,
    cfg,
    asym_results: list[AsymmetryResult],
) -> None:
    """Write one row per asymmetry result with kinematic metadata and fit values."""
    try:
        run_period = int(cfg.run_period.split("_")[-1])
    except (ValueError, IndexError):
        run_period = cfg.run_period

    rows = []
    for ar in asym_results:
        rows.append({
            #"run_period":    run_period,
            "target":   cfg.target,
            "particle": _PARTICLE_LABEL.get(cfg.setting.run_type, cfg.setting.run_type),
            "ebeam":    cfg.setting.ebeam,
            "x":        cfg.setting.x,
            "q2":       cfg.setting.Q2,
            "z":        cfg.setting.z,
            "thpq":     cfg.setting.thpq,
            "histogram": ar.histogram_name,
            "asym":     ar.amplitude     if np.isfinite(ar.amplitude)     else np.nan,
            "asym_err": ar.amplitude_err if np.isfinite(ar.amplitude_err) else np.nan,
            "chi2_ndf": ar.chi2_ndf      if np.isfinite(ar.chi2_ndf)      else np.nan,
            "n_bins":   ar.n_bins_used,
        })
    cols = ["target", "particle", "ebeam", "x", "q2", "z", "thpq",
            "histogram", "asym", "asym_err", "chi2_ndf", "n_bins"]
    pd.DataFrame(rows, columns=cols).to_csv(path, index=False)
    logger.info("Kinematic summary → %s", path)


# ---------------------------------------------------------------------------
# Binned asymmetry CSV and summary plot
# ---------------------------------------------------------------------------

def _write_binned_asymmetry_csv(
    path:        Path,
    bin_groups:  list[tuple[str, list[tuple[AsymmetryResult, float, float, float]]]],
) -> None:
    """
    Write per-bin φ asymmetry to CSV.

    Parameters
    ----------
    bin_groups
        List of (variable_name, [(ar, lo, hi, center), ...]) — one entry per
        histogram that has a bin_in block.  ``variable_name`` is the branch
        name (e.g. 'pt'); lo/hi are the bin edges; center = (lo+hi)/2.
    """
    rows = []
    for variable, bin_results in bin_groups:
        for ar, lo, hi, center in bin_results:
            for phi, A, dA in zip(ar.phi_centers, ar.A_phys, ar.A_phys_err):
                rows.append({
                    "variable":   variable,
                    "hmin":       lo,
                    "hmax":       hi,
                    "bin_center": round(center, 3),
                    "histogram":  ar.histogram_name,
                    "phi_center": phi,
                    "A_phys":     A,
                    "A_phys_err": dA,
                })
    cols = ["variable", "hmin", "hmax", "bin_center",
            "histogram", "phi_center", "A_phys", "A_phys_err"]
    pd.DataFrame(rows, columns=cols).to_csv(path, index=False)
    logger.info("Binned asymmetry CSV → %s", path)


def _page_binned_summary(
    pdf:         PdfPages,
    bin_results: list[tuple[AsymmetryResult, float, float, float]],
    variable:    str,
    histo_name:  str,
    h_pt_vs_phi: object | None = None,
    hcfg_pt:     object | None = None,
) -> None:
    """
    Plot A_LU^sinphi vs the binned kinematic variable.

    If *h_pt_vs_phi* and *hcfg_pt* are provided, draws a two-column layout
    with the pt_vs_phi 2-D yield plot on the left.
    """
    if not bin_results:
        return

    centers    = np.array([entry[3] for entry in bin_results])
    amplitudes = np.array([entry[0].amplitude     for entry in bin_results])
    amp_errs   = np.array([entry[0].amplitude_err for entry in bin_results])
    lo_vals    = np.array([entry[1] for entry in bin_results])
    hi_vals    = np.array([entry[2] for entry in bin_results])
    xerr_lo    = centers - lo_vals
    xerr_hi    = hi_vals - centers
    valid = np.isfinite(amplitudes) & np.isfinite(amp_errs)

    has_2d = h_pt_vs_phi is not None and hcfg_pt is not None
    if has_2d:
        fig, (ax_2d, ax_ssa) = plt.subplots(
            1, 2, figsize=(14, 6.0),
            gridspec_kw={"width_ratios": [1, 1.2]},
        )
        x_edges = h_pt_vs_phi.axes[0].edges   # type: ignore[attr-defined]
        y_edges = h_pt_vs_phi.axes[1].edges   # type: ignore[attr-defined]
        values  = h_pt_vs_phi.values()        # type: ignore[attr-defined]
        _draw_pt_vs_phi_2d(ax_2d, hcfg_pt, x_edges, y_edges, values)
    else:
        fig, ax_ssa = plt.subplots(figsize=(8, 5))

    ax_ssa.axhline(0.0, color="gray", lw=0.9, ls="--", zorder=1)
    if valid.any():
        ax_ssa.errorbar(
            centers[valid], amplitudes[valid],
            yerr=amp_errs[valid],
            xerr=[xerr_lo[valid], xerr_hi[valid]],
            fmt="o", color="steelblue", ms=6, lw=1.2, capsize=4, zorder=3,
            label=r"$A_{LU}^{\sin\phi}$",
        )
    ax_ssa.set_xlabel(variable)
    ax_ssa.set_ylabel(r"$A_{LU}^{\sin\phi}$")
    ax_ssa.set_title(
        rf"$A_{{LU}}^{{\sin\phi}}$ vs {variable}  —  {histo_name}",
        fontsize=11, fontweight="bold",
    )
    ax_ssa.legend(fontsize=8)
    fig.tight_layout(pad=1.7)
    pdf.savefig(fig)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Last page: statistics table
# ---------------------------------------------------------------------------

def _style_table(
    tbl,
    header_color: str = "#c8ddf0",
    fontsize: int = 8,
    pull_cols: list[int] | None = None,
    pull_data: list[list[str]] | None = None,
    pull_warn: float = 5.0,
) -> None:
    """Apply consistent styling to a matplotlib Table.
    Optionally highlights pull columns where |pull| > pull_warn in light red.
    """
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(fontsize)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("0.75")
        cell.set_linewidth(0.4)
        if r == 0:
            cell.set_facecolor(header_color)
            cell.set_text_props(fontweight="bold")
        else:
            cell.set_facecolor("#f5f5f5" if r % 2 == 1 else "#ffffff")
    # Highlight large pulls
    if pull_cols and pull_data:
        for ri, row in enumerate(pull_data):
            for ci in pull_cols:
                if ci < len(row):
                    try:
                        if float(row[ci]) > pull_warn:
                            tbl[ri + 1, ci].set_facecolor("#ffdddd")
                    except (ValueError, TypeError):
                        pass


def _page_statistics(
    pdf: PdfPages,
    result: PipelineResult,
    sub_results: list[SubtractionResult],
    yaml_path: Path,
) -> None:
    from datetime import datetime as _dt
    from collections import defaultdict as _defaultdict

    cfg  = result.cfg
    cuts = cfg.cuts
    norm = cfg.normalization
    st   = cfg.setting

    # ── Subtraction pivot: pivot[hname][step] = (sub_pct, max_pull) ──────
    steps_present: list[str] = list(dict.fromkeys(sr.label for sr in sub_results))
    scale_by_step: dict[str, float] = {}
    pivot: dict[str, dict[str, tuple[float, float]]] = _defaultdict(dict)
    for sr in sub_results:
        pivot[sr.histogram_name][sr.label] = (
            sr.fraction_subtracted * 100.0,
            sr.max_abs_pull,
        )
        scale_by_step.setdefault(sr.label, sr.scale)
    hist_names: list[str] = list(dict.fromkeys(sr.histogram_name for sr in sub_results))

    # ── Figure ─────────────────────────────────────────────────────────────
    n_h = len(hist_names)
    fig_h = max(9.0, 0.25 * n_h + 6.5)
    fig = plt.figure(figsize=(14, fig_h))
    gs  = fig.add_gridspec(
        4, 1, hspace=0.27,
        height_ratios=[0.5, 2.2, 1.5, max(3.0, 0.25 * n_h + 1.0)],
    )
    fig.suptitle("Run Summary", fontsize=12, fontweight="bold", y=0.995)

    _HC = "#c8ddf0"   # header colour shared by all tables
    _FS = 8           # base font size

    # ── A: Header (config file + timestamp) ───────────────────────────────
    ax0 = fig.add_subplot(gs[0])
    ax0.axis("off")
    ax0.text(0.0, 0.95, f"Config:    {yaml_path}",
             transform=ax0.transAxes, fontsize=9, family="monospace", va="top")
    ax0.text(0.0, 0.35, f"Generated  {_dt.now():%Y-%m-%d %H:%M}",
             transform=ax0.transAxes, fontsize=8, color="0.45", va="top")

    # ── B: Setting & Normalization ─────────────────────────────────────────
    ax1 = fig.add_subplot(gs[1])
    ax1.axis("off")
    ax1.set_title("Setting & Normalization", loc="left",
                  fontsize=9, fontweight="bold", pad=3)

    # B3: per-run-type charge totals + exclusions (one row per run type).
    # Built first so B2/B1 can be stacked above it without overlap.
    b3_cols = ["run_type", "n_valid", "n_excl", "Q_tot [mC]",
               "Q_hp [mC]", "Q_hm [mC]", "excluded runs"]
    b3_rows = []
    for lbl, wt in result.weight_tables.items():
        excl_str = ("  ".join(f"{ex.run}({ex.column})" for ex in wt.excluded)
                    or "—")
        b3_rows.append([
            lbl,
            str(wt.n_valid),
            str(wt.n_excluded),
            f"{wt.Q_tot:.2f}",
            f"{wt.Q_hp_tot:.2f}",
            f"{wt.Q_hm_tot:.2f}",
            excl_str,
        ])
    n_b3    = len(b3_rows)
    b3_h    = 0.09 * (n_b3 + 1)   # ~0.09 per row (header + data)
    b3_bot  = 0.02
    tbl_b3  = ax1.table(cellText=b3_rows, colLabels=b3_cols,
                        loc="upper center",
                        bbox=[0, b3_bot, 1.0, b3_h])
    _style_table(tbl_b3, _HC, _FS)

    # B2: normalization scheme (1 data row) — placed above B3
    b2_h    = 0.18
    b2_bot  = b3_bot + b3_h + 0.04
    b2_cols = ["weight_scheme", "charge_col", "heli_gated",
               "hp_col", "hm_col"]
    b2_rows = [[
        norm.weight_scheme,
        norm.charge_column,
        "yes" if norm.use_helicity_gated_charge else "no",
        norm.charge_hp_column if norm.use_helicity_gated_charge else "—",
        norm.charge_hm_column if norm.use_helicity_gated_charge else "—",
    ]]
    tbl_b2  = ax1.table(cellText=b2_rows, colLabels=b2_cols,
                        loc="upper center", bbox=[0, b2_bot, 0.65, b2_h])
    _style_table(tbl_b2, _HC, _FS)

    # B1: kinematic setting (1 data row) — placed above B2
    b1_h    = 0.18
    b1_bot  = b2_bot + b2_h + 0.04
    b1_cols = ["target", "run_period", "ebeam [GeV]", "x", "Q² [GeV²]",
               "z", "θpq [°]", "run type"]
    b1_rows = [[
        cfg.target, cfg.run_period,
        f"{st.ebeam:.4f}", f"{st.x:.3f}", f"{st.Q2:.2f}",
        f"{st.z:.2f}", f"{st.thpq:.2f}", st.run_type,
    ]]
    tbl_b1  = ax1.table(cellText=b1_rows, colLabels=b1_cols,
                        loc="upper center", bbox=[0, b1_bot, 1.0, b1_h])
    _style_table(tbl_b1, _HC, _FS)

    # ── C: Cuts ────────────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[2])
    ax2.axis("off")
    ax2.set_title("Cuts", loc="left", fontsize=9, fontweight="bold", pad=3)

    # C1: acceptance + PID (1 data row)
    hgc_thr = (f"{cuts.phgc_p_threshold:.2f} GeV"
               if cuts.phgc_p_threshold is not None else "unconditional")
    c1_cols = ["HMS δ [%]", "SHMS δ [%]", "HCer NPE",
               "SHSsum", "Aero NPE", "HGC NPE", "SHScal E/p", "HGC thr"]
    c1_rows = [[
        f"{cuts.hsdelta_lo:.1f} → {cuts.hsdelta_hi:.1f}",
        f"{cuts.psdelta_lo:.1f} → {cuts.psdelta_hi:.1f}",
        f"> {cuts.hcer_npe_min:.1f}",
        f"> {cuts.hsshsum_min:.2f}",
        f"> {cuts.paero_npe_min:.1f}",
        f"> {cuts.phgc_npe_min:.1f}",
        f"< {cuts.psshsum_max:.2f}",
        hgc_thr,
    ]]
    tbl_c1 = ax2.table(cellText=c1_rows, colLabels=c1_cols,
                       loc="upper center", bbox=[0, 0.52, 1.0, 0.44])
    _style_table(tbl_c1, _HC, _FS)

    # C2: coincidence-time window (1 data row)
    ctr_str = ("auto" if cuts.ctime_real_center == "auto"
               else f"{cuts.ctime_real_center:.3f} ns")
    nsigma_str = (f"{cuts.ctime_real_nsigma:.1f}σ"
                  if cuts.ctime_real_nsigma is not None else "— (always fallback)")
    c2_cols = ["ctime center", "ctime nsigma", "fallback [ns]",
               "rand n_skip", "rand peaks lo+hi"]
    c2_rows = [[
        ctr_str, nsigma_str,
        f"{cuts.ctime_real_window_fallback:.2f}",
        str(cuts.ctime_random_n_skip),
        f"{cuts.ctime_random_n_peaks_lo}+{cuts.ctime_random_n_peaks_hi}",
    ]]
    tbl_c2 = ax2.table(cellText=c2_rows, colLabels=c2_cols,
                       loc="upper center", bbox=[0, 0.02, 0.65, 0.44])
    _style_table(tbl_c2, _HC, _FS)

    # ── D: Background Subtraction (one row per histogram) ─────────────────
    ax3 = fig.add_subplot(gs[3])
    ax3.axis("off")
    ax3.set_title("Background Subtraction", loc="left",
                  fontsize=9, fontweight="bold", pad=3)

    d_step_cols: list[str] = []
    pull_col_indices: list[int] = []
    for i, step in enumerate(steps_present):
        scl = scale_by_step.get(step, 1.0)
        d_step_cols.append(f"{step} sub%\n(scale={scl:.3f})")
        d_step_cols.append(f"{step} |pull|")
        pull_col_indices.append(1 + 2 * i + 1)   # 0=hname, then pairs

    d_cols = ["histogram"] + d_step_cols
    d_rows: list[list[str]] = []
    for hname in hist_names:
        row: list[str] = [hname]
        for step in steps_present:
            if step in pivot[hname]:
                sub_pct, pull = pivot[hname][step]
                row.extend([f"{sub_pct:.1f}", f"{pull:.2f}"])
            else:
                row.extend(["—", "—"])
        d_rows.append(row)

    tbl_d = ax3.table(cellText=d_rows, colLabels=d_cols,
                      loc="upper center", bbox=[0, 0, 1, 1])
    _style_table(tbl_d, _HC, _FS - 1,
                 pull_cols=pull_col_indices, pull_data=d_rows)

    pdf.savefig(fig)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def make_diagnostic_pdf(yaml_path: Path, output_pdf: Path | None = None) -> None:
    if output_pdf is None:
        stem = yaml_path.stem
        out_dir = Path("output") / stem
        out_dir.mkdir(parents=True, exist_ok=True)
        output_pdf = out_dir / f"{stem}.pdf"
    else:
        output_pdf.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Loading config: %s", yaml_path)
    result = run_pipeline_from_yaml(yaml_path)
    cfg    = result.cfg

    # ---- apply subtractions ----
    sub_results: list[SubtractionResult] = []

    signal_after_random: dict[str, object] = {}
    signal_after_eplus:  dict[str, object] = {}
    dummy_subtracted:    dict[str, object] = {}  # h_dum_ran per histogram
    final:               dict[str, object] = {}

    for _hcfg in cfg.effective_histograms():
        hname = _hcfg.name
        h_sig_real   = result.signal.real[hname]
        h_sig_random = result.signal.random[hname]

        h_after_ran = subtract_randoms(h_sig_real, h_sig_random, scale=1.0)
        signal_after_random[hname] = h_after_ran
        sub_results.append(SubtractionResult(
            label="random", histogram_name=hname,
            h_before=h_sig_real, h_background=h_sig_random,
            h_after=h_after_ran, scale=1.0,
        ))

        h_final = h_after_ran
        if result.eplus is not None:
            h_ep_ran = subtract_randoms(
                result.eplus.real[hname],
                result.eplus.random[hname],
                scale=1.0,
            )
            h_final = subtract_eplus(h_after_ran, h_ep_ran)
            sub_results.append(SubtractionResult(
                label="eplus", histogram_name=hname,
                h_before=h_after_ran, h_background=h_ep_ran,
                h_after=h_final, scale=1.0,
            ))
        signal_after_eplus[hname] = h_final  # equals h_after_ran when no e⁺

        if result.dummy is not None:
            h_dum_ran = subtract_randoms(
                result.dummy.real[hname],
                result.dummy.random[hname],
                scale=1.0,
            )
            dummy_subtracted[hname] = h_dum_ran
            h_prev   = h_final
            h_final  = subtract_dummy(h_final, h_dum_ran, scale=result.dummy_scale)
            sub_results.append(SubtractionResult(
                label="dummy", histogram_name=hname,
                h_before=h_prev, h_background=h_dum_ran,
                h_after=h_final, scale=result.dummy_scale,
            ))

        final[hname] = h_final

    logger.debug(subtraction_summary(sub_results))

    # ---- generate PDF ----
    logger.info("Writing diagnostic PDF: %s", output_pdf)
    with PdfPages(output_pdf) as pdf:

        _page_helicity(pdf, result)
        _page_ctime(pdf, result)
        _page_normyield(pdf, result)

        for hcfg in cfg.histograms:
            hname = hcfg.name
            if hcfg.is_2d:
                _page_histogram_2d(pdf, hname, hcfg, final[hname])
                continue

            h_ep_real   = result.eplus.real[hname]   if result.eplus else None
            h_ep_random = result.eplus.random[hname] if result.eplus else None
            h_after_ep  = signal_after_eplus[hname]  if result.eplus else None
            h_dum_ran   = dummy_subtracted.get(hname)
            h_after_dum = final[hname]                if result.dummy else None

            _page_histogram(
                pdf, hname, hcfg.xlabel or hname,
                h_sig_real   = result.signal.real[hname],
                h_sig_random = result.signal.random[hname],
                h_ep_real    = h_ep_real,
                h_ep_random  = h_ep_random,
                h_dum_ran    = h_dum_ran,
                h_after_ran  = signal_after_random[hname],
                h_after_ep   = h_after_ep,
                h_after_dum  = h_after_dum,
                sub_results  = sub_results,
                dummy_scale  = result.dummy_scale,
            )

        # Overall SSA page (pairs from original histograms only)
        overall_names = {h.name for h in cfg.histograms}
        overall_pairs = find_asymmetry_pairs([k for k in final if k in overall_names])
        asym_results = _page_asymmetry(
            pdf, result, final,
            config_dir=yaml_path.parent,
            pairs=overall_pairs,
        )

        # Per-bin SSA pages: one page per bin for each histogram with bin_in
        # bin_groups collects (variable, [(ar, lo, hi, center), ...]) for CSV export
        _hcfg_pt_vs_phi = next((h for h in cfg.histograms if h.name == "pt_vs_phi"), None)
        _h_pt_vs_phi    = final.get("pt_vs_phi") if _hcfg_pt_vs_phi is not None else None

        bin_groups: list[tuple[str, list[tuple[AsymmetryResult, float, float, float]]]] = []
        for _hcfg_bin in cfg.histograms:
            if _hcfg_bin.bin_in is None:
                continue
            _bin_in = _hcfg_bin.bin_in
            _this_group: list[tuple[AsymmetryResult, float, float, float]] = []

            for _i, (_lo, _hi) in enumerate(_bin_in.intervals()):
                _base       = f"{_hcfg_bin.name}_{_bin_in.label(_i)}"
                _hplus_name = f"{_base}_hplus"
                _hminus_name = f"{_base}_hminus"
                if _hplus_name not in final or _hminus_name not in final:
                    continue
                _subtitle = f"{_bin_in.branch} ∈ [{_lo:.3f}, {_hi:.3f})"
                _bin_ars = _page_asymmetry(
                    pdf, result, final,
                    config_dir=yaml_path.parent,
                    pairs=[(_base, _hplus_name, _hminus_name)],
                    subtitle=_subtitle,
                )
                if _bin_ars:
                    _center = 0.5 * (_lo + _hi)
                    _this_group.append((_bin_ars[0], _lo, _hi, _center))

            if _this_group:
                _page_binned_summary(
                    pdf, _this_group, _bin_in.branch, _hcfg_bin.name,
                    h_pt_vs_phi=_h_pt_vs_phi, hcfg_pt=_hcfg_pt_vs_phi,
                )
                bin_groups.append((_bin_in.branch, _this_group))

        _page_statistics(pdf, result, sub_results, yaml_path)

    root_path = output_pdf.with_suffix(".root")
    _write_root_file(root_path, result,
                     signal_after_random, signal_after_eplus,
                     dummy_subtracted, final)

    if asym_results:
        csv_path = output_pdf.with_name(output_pdf.stem + ".csv")
        _write_asymmetry_csv(csv_path, asym_results)
        summary_path = output_pdf.with_name(output_pdf.stem + "_summary.csv")
        _write_kinematic_summary(summary_path, cfg, asym_results)

    if bin_groups:
        binned_csv_path = output_pdf.with_name(output_pdf.stem + "_binned.csv")
        _write_binned_asymmetry_csv(binned_csv_path, bin_groups)

    logger.info("Done → %s  (%d pages)", output_pdf,
                len(cfg.histograms) + 5)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RSIDIS SSA diagnostic plots"
    )
    parser.add_argument("config", type=Path, help="Path to YAML config file")
    parser.add_argument(
        "--output", "-o", type=Path, default=None,
        help="Output PDF path (default: <config-stem>_diagnostics.pdf)"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Show all INFO messages from rsidis_ssa submodules (default: top-level only)",
    )
    args = parser.parse_args()

    _configure_logging(args.verbose)

    yaml_path  = args.config
    output_pdf = Path(args.output) if args.output else None
    make_diagnostic_pdf(yaml_path, output_pdf)


if __name__ == "__main__":
    main()

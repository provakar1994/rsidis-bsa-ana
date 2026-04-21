"""
Diagnostic analysis driver.

Usage:
    python analysis.py config/example_C_z05_thpq2.yaml [--output diag.pdf]

Produces a multi-page PDF with:
  Page 1  : Run weight distributions (signal + e⁺)
  Page 2  : Coincidence-time distribution — full range + zoom on real peak
            with per-run window boundaries and cut regions shaded
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
from matplotlib.backends.backend_pdf import PdfPages

from rsidis_ssa.backgrounds import (
    SubtractionResult,
    subtract_dummy,
    subtract_eplus,
    subtract_randoms,
    subtraction_summary,
)
from rsidis_ssa.normalization import weight_table_to_df
from rsidis_ssa.pipeline import PipelineResult, run_pipeline_from_yaml

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(levelname)-8s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

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


# ---------------------------------------------------------------------------
# Page 1: weight distributions
# ---------------------------------------------------------------------------

def _page_weights(pdf: PdfPages, result: PipelineResult) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle("Run weight distributions", fontsize=11, fontweight="bold")

    labels  = ["signal", "eplus"]
    colors  = ["steelblue", "tomato"]
    tables  = [result.weight_tables.get("signal"),
               result.weight_tables.get("eplus")]

    for ax, label, color, wt in zip(axes, labels, colors, tables):
        if wt is None or wt.n_valid == 0:
            ax.set_visible(False)
            continue
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

    skips = {k: len(v) for k, v in result.file_skips.items() if v}
    if skips:
        note = "Files not found: " + ", ".join(f"{k}={n}" for k, n in skips.items())
        fig.text(0.5, 0.01, note, ha="center", fontsize=7, color="crimson")

    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Page 2: coincidence-time distribution
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
    mean_rand_center  = float(np.mean([c for (c, _) in rand_windows if not np.isnan(c)])) \
                        if rand_windows else mean_ctmean - cuts.ctime_random_offset
    mean_rand_hw      = float(np.mean([h for (_, h) in rand_windows])) \
                        if rand_windows else mean_half_win * cuts.ctime_random_wscale

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
    # Random sideband shading (mean over runs; per-run windows shown as thin lines)
    rand_lo = mean_rand_center - mean_rand_hw
    rand_hi = mean_rand_center + mean_rand_hw
    ax_full.axvspan(rand_lo, rand_hi, color="royalblue", alpha=0.15,
                    label=f"random: {mean_rand_center:.2f} ± {mean_rand_hw:.2f} ns")
    ax_full.axvline(rand_lo, color="royalblue", lw=0.9, ls="--")
    ax_full.axvline(rand_hi, color="royalblue", lw=0.9, ls="--")
    for rc, rhw in rand_windows:
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
            f"random offset = {cuts.ctime_random_offset:.1f} ns\n"
            f"random wscale = {cuts.ctime_random_wscale:.1f}  (win_scale = 1/{cuts.ctime_random_wscale:.0f})\n"
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
    ax.set_xticks(x_all)
    ax.set_xticklabels(sorted_runs, rotation=45, ha="right", fontsize=6)
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
# Page 3: normalized yield vs run number
# ---------------------------------------------------------------------------

def _page_normyield(pdf: PdfPages, result: PipelineResult) -> None:
    """
    Two-row diagnostic page per active run type.

    Row 1 — normalized yield: workflow vs CSV normyield per run.
    Row 2 — raw event counts: workflow (n_real, n_random, n_rsc)
             vs CSV (coin, randoms, ransubcoin) per run.
             This row isolates where the discrepancy originates.

    Flagged runs (|residual| > 2%) are marked with a red × on row 1.
    """
    labels = [l for l in ("signal", "eplus", "dummy")
              if l in result.normyield_per_run
              and not result.normyield_per_run[l].empty]
    if not labels:
        return

    n_cols = len(labels)
    # 2 rows: top = normyield, bottom = raw counts
    fig, all_axes = plt.subplots(2, n_cols,
                                 figsize=(5.5 * n_cols, 8.5),
                                 sharey=False)
    # Ensure 2-D indexing even for a single column
    if n_cols == 1:
        all_axes = [[all_axes[0]], [all_axes[1]]]
    top_axes, bot_axes = all_axes[0], all_axes[1]

    fig.suptitle("Normalized yield vs run number", fontsize=11, fontweight="bold")

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

        ax.set_xticks(x)
        ax.set_xticklabels(runs, rotation=45, ha="right", fontsize=6)
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
        ax2 = bot_axes[idx]

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

        ax2.set_xticks(x)
        ax2.set_xticklabels(runs, rotation=45, ha="right", fontsize=6)
        ax2.set_xlabel("run number")
        ax2.set_ylabel("event counts")
        ax2.set_title(f"{label} — raw count comparison")

        if idx == 0:
            ax2.legend(fontsize=6)

    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Pages 4+: per-histogram subtraction overview
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
) -> None:
    """One figure per histogram showing all subtraction steps."""
    has_ep  = h_ep_real is not None
    has_dum = h_dum_ran is not None
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
        _plot_hist_step(ax, h_dum_ran, "dummy − random", "darkorange", alpha=0.7)
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
            _plot_hist_step(ax, h_dum_ran, "dummy − random", "darkorange", alpha=0.6)
        ax.set_xlabel(xlabel)
        ax.set_title("After e⁺ subtraction")
        ax.legend(fontsize=7)

    # Panel 4 — after dummy subtraction (if applicable)
    if has_dum and h_after_dum is not None:
        ax = axes[col]; col += 1
        prev = h_after_ep if (has_ep and h_after_ep is not None) else h_after_ran
        _plot_hist_step(ax, prev,        "before dummy sub", "black",      alpha=0.35)
        _plot_hist_step(ax, h_dum_ran,   "dummy − random",   "darkorange", alpha=0.7)
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
# Last page: statistics table
# ---------------------------------------------------------------------------

def _page_statistics(
    pdf: PdfPages,
    result: PipelineResult,
    sub_results: list[SubtractionResult],
) -> None:
    fig, ax = plt.subplots(figsize=(11, max(4, 0.35 * len(sub_results) + 2)))
    ax.axis("off")
    fig.suptitle("Background subtraction statistics", fontsize=11, fontweight="bold")

    # Weight table exclusion summary
    lines = []
    for label, wt in result.weight_tables.items():
        lines.append(f"  {label:8s}  {wt.n_valid:4d} runs accepted"
                     f"  {wt.n_excluded:3d} excluded (bad normalization)")
        for ex in wt.excluded:
            lines.append(f"             ↳ run {ex.run}  ({ex.column}): {ex.reason}")

    for label, skips in result.file_skips.items():
        if skips:
            lines.append(f"  {label:8s}  {len(skips)} ROOT file(s) not found: "
                         + ", ".join(str(r) for r in sorted(skips)))

    lines.append("")
    lines.append(subtraction_summary(sub_results))

    ax.text(0.02, 0.98, "\n".join(lines),
            transform=ax.transAxes, va="top", ha="left",
            fontsize=8, family="monospace",
            bbox=dict(boxstyle="round", fc="0.96", ec="0.8"))

    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def make_diagnostic_pdf(yaml_path: Path, output_pdf: Path) -> None:
    logger.info("Loading config: %s", yaml_path)
    result = run_pipeline_from_yaml(yaml_path)
    cfg    = result.cfg

    # ---- apply subtractions ----
    sub_results: list[SubtractionResult] = []

    signal_after_random: dict[str, object] = {}
    signal_after_eplus:  dict[str, object] = {}
    dummy_subtracted:    dict[str, object] = {}  # h_dum_ran per histogram
    final:               dict[str, object] = {}

    for hname in cfg.histogram_names():
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

    logger.info(subtraction_summary(sub_results))

    # ---- generate PDF ----
    logger.info("Writing diagnostic PDF: %s", output_pdf)
    with PdfPages(output_pdf) as pdf:

        _page_weights(pdf, result)
        _page_ctime(pdf, result)
        _page_normyield(pdf, result)

        for hcfg in cfg.histograms:
            hname = hcfg.name
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
            )

        _page_statistics(pdf, result, sub_results)

    logger.info("Done → %s  (%d pages)", output_pdf,
                len(cfg.histograms) + 4)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RSIDIS SSA diagnostic plots"
    )
    parser.add_argument("config", type=Path, help="Path to YAML config file")
    parser.add_argument(
        "--output", "-o", type=Path, default=None,
        help="Output PDF path (default: <config-stem>_diagnostics.pdf)"
    )
    args = parser.parse_args()

    yaml_path  = args.config
    output_pdf = args.output or yaml_path.with_suffix("").name + "_diagnostics.pdf"
    make_diagnostic_pdf(yaml_path, Path(output_pdf))


if __name__ == "__main__":
    main()

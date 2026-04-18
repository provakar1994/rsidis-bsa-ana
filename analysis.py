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
    Two-panel coincidence-time diagnostic page.

    Left  — full range showing both the real peak and random sideband.
    Right — zoom on the real peak with per-run window boundaries.
    """
    cfg  = result.cfg
    cuts = cfg.cuts

    h_sig = result.ctime_hists["signal"]
    centers = h_sig.axes[0].centers
    sig_vals = h_sig.values()

    # Per-run window parameters for signal
    windows = result.ctime_run_windows["signal"]
    valid_ctmeans  = [c for (c, _) in windows if not np.isnan(c)]
    valid_halfwins = [h for (_, h) in windows]
    mean_ctmean   = float(np.mean(valid_ctmeans))  if valid_ctmeans  else 0.0
    mean_half_win = float(np.mean(valid_halfwins)) if valid_halfwins else cuts.ctime_real_window_fallback

    fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(12, 4.5))
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
    # Random sideband shading
    rand_lo = cuts.ctime_random_center - cuts.ctime_random_window
    rand_hi = cuts.ctime_random_center + cuts.ctime_random_window
    ax_full.axvspan(rand_lo, rand_hi, color="royalblue", alpha=0.15,
                    label=f"random: {cuts.ctime_random_center:.2f} ± {cuts.ctime_random_window:.2f} ns")
    ax_full.axvline(rand_lo, color="royalblue", lw=0.9, ls="--")
    ax_full.axvline(rand_hi, color="royalblue", lw=0.9, ls="--")
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
            f"N runs = {n_runs}")
    ax_zoom.text(0.02, 0.97, info, transform=ax_zoom.transAxes,
                 va="top", ha="left", fontsize=7, family="monospace",
                 bbox=dict(boxstyle="round", fc="0.96", ec="0.8"))
    ax_zoom.set_title("Zoom: real peak  (gray = per-run windows)")

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
                len(cfg.histograms) + 3)


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

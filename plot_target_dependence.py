#!/usr/bin/env python3
"""
Compare SSA across targets and compute differences from a reference target.

Reads all *_binned_summary.csv files in output/combined/ and produces a
multi-page PDF:

  Pages 1–2 — Target comparison grid (pip / pim):
               rows = targets, cols = p_T bins, each panel: A vs z
  Pages 3–4 — Difference grid (pip / pim):
               rows = non-reference targets, cols = p_T bins,
               each panel: A_target − A_reference vs z

Must be run from the project root.

Usage
-----
    python plot_target_dependence.py
    python plot_target_dependence.py --reference LH2
    python plot_target_dependence.py --targets C Cu LD2 LH2
    python plot_target_dependence.py --output my_plot.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

COMBINED_DIR  = Path("output/combined")
PLOTS_DIR     = Path("output/plots")

_PARTICLE_COL = {"pip": "pi+", "pim": "pi-"}
_TARGET_ORDER = ["LH2", "LD2", "C", "Cu", "Al"]   # preferred display order


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_all(variable: str) -> pd.DataFrame:
    """Load all *_binned_summary.csv files, both particles."""
    frames = []
    for csv in sorted(COMBINED_DIR.glob("*_binned_summary.csv")):
        df  = pd.read_csv(csv)
        sub = df[df["variable"] == variable]
        if not sub.empty:
            frames.append(sub)
    if not frames:
        raise FileNotFoundError(
            f"No *_binned_summary.csv files found in {COMBINED_DIR}"
        )
    return pd.concat(frames, ignore_index=True)


def _ordered_targets(df: pd.DataFrame, requested: list[str] | None) -> list[str]:
    available = df["target"].unique().tolist()
    pool      = requested if requested else _TARGET_ORDER
    return [t for t in pool if t in available]


def _pt_bins(df: pd.DataFrame) -> pd.DataFrame:
    """Unique p_T bins sorted by hmin."""
    return (df.drop_duplicates("histogram")
              .sort_values("hmin")[["histogram", "hmin", "hmax", "bin_center"]]
              .reset_index(drop=True))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _particle_tex(particle: str) -> str:
    return r"$\pi^+$" if particle == "pip" else r"$\pi^-$"


def _ax_style(ax: plt.Axes, *, xlabel: bool, ylabel: bool) -> None:
    ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.tick_params(which="both", direction="in", top=True, right=True)
    if not xlabel:
        ax.tick_params(labelbottom=False)
    if not ylabel:
        ax.tick_params(labelleft=False)


def _page_suptitle(df: pd.DataFrame, particle: str, suffix: str = "") -> str:
    x   = df["x"].iloc[0]
    q2  = df["q2"].iloc[0]
    return (f"SSA $A_{{LU}}^{{\\sin\\phi}}${suffix} — "
            f"{_particle_tex(particle)},  "
            f"$x = {x:.2f}$,  $Q^2 = {q2:.1f}$ GeV$^2$")


# ---------------------------------------------------------------------------
# Difference computation
# ---------------------------------------------------------------------------

def _compute_diff(df: pd.DataFrame, reference: str) -> pd.DataFrame:
    """Subtract reference-target asymmetry from every other target."""
    ref = (df[df["target"] == reference]
           .rename(columns={"asym": "_asym_ref", "asym_err": "_asym_err_ref"})
           [["particle", "z", "histogram", "_asym_ref", "_asym_err_ref"]])

    other  = df[df["target"] != reference].copy()
    merged = other.merge(ref, on=["particle", "z", "histogram"], how="inner")

    merged["asym"]     = merged["asym"] - merged["_asym_ref"]
    merged["asym_err"] = np.sqrt(merged["asym_err"] ** 2
                                 + merged["_asym_err_ref"] ** 2)
    return merged.drop(columns=["_asym_ref", "_asym_err_ref"])


# ---------------------------------------------------------------------------
# Grid figure
# ---------------------------------------------------------------------------

def _fig_grid(
    df:       pd.DataFrame,
    particle: str,
    targets:  list[str],
    bins:     pd.DataFrame,
    suptitle: str,
    diff_ref: str | None = None,
    ylim:     tuple[float, float] | None = None,
    fmt:      str = "ks",
) -> plt.Figure:
    """
    Grid of panels: rows = targets, cols = p_T bins.
    All panels share x- and y-axes; no gap between columns or rows.

    diff_ref: when set, each row's y-axis label reads
              A_LU|_<target> - A_LU|_<diff_ref>.
    """
    n_rows = len(targets)
    n_cols = len(bins)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        sharex=True, sharey=True,
        figsize=(3.5 * n_cols, 3.0 * n_rows),
        squeeze=False,
    )
    fig.subplots_adjust(top=0.93, wspace=0, hspace=0)
    fig.suptitle(suptitle, fontsize=10)

    for r, target in enumerate(targets):
        is_bottom = (r == n_rows - 1)
        is_top    = (r == 0)

        for c, brow in enumerate(bins.itertuples()):
            ax      = axes[r, c]
            is_left = (c == 0)

            grp = (df[(df["target"]    == target)
                      & (df["histogram"] == brow.histogram)]
                   .sort_values("z"))

            if grp.empty:
                ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                        ha="center", va="center", fontsize=8, color="gray")
            else:
                ax.errorbar(
                    grp["z"], grp["asym"], yerr=grp["asym_err"],
                    fmt=fmt, capsize=3, markersize=4, elinewidth=1.0, lw=0,
                )

            ax.axhline(0, color="gray", lw=0.8, ls="--", zorder=0)

            if ylim is not None:
                ax.set_ylim(*ylim)

            # target label — top-right corner of every panel
            ax.text(0.97, 0.95, target,
                    transform=ax.transAxes, va="top", ha="right",
                    fontsize=9, fontweight="bold")

            # p_T bin label — inside top-left of top-row panels
            if is_top:
                ax.text(0.05, 0.95,
                        f"$\\langle P_T \\rangle = {brow.bin_center:.2f}$ GeV/$c$",
                        transform=ax.transAxes, va="top", ha="left", fontsize=9)

            if is_left:
                if diff_ref is not None:
                    yl = (rf"$A_{{LU}}^{{\sin\phi}}|_{{\rm {target}}}"
                          rf" - A_{{LU}}^{{\sin\phi}}|_{{\rm {diff_ref}}}$")
                else:
                    yl = r"$A_{LU}^{\sin\phi}$"
                ax.set_ylabel(yl, fontsize=8)

            if is_bottom:
                ax.set_xlabel("$z$", fontsize=10)

            _ax_style(ax, xlabel=is_bottom, ylabel=is_left)

    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Target-dependence of SSA: comparison and difference grids",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--targets", nargs="+", default=None, metavar="TARGET",
        help="Targets to include (default: all found, ordered LH2 LD2 C Cu Al)",
    )
    parser.add_argument(
        "--reference", default="LH2", metavar="TARGET",
        help="Reference target for difference plots (default: LH2)",
    )
    parser.add_argument(
        "--variable", default="pt", metavar="VAR",
        help="Binning variable in summary CSV (default: pt)",
    )
    parser.add_argument(
        "--ylim", nargs=2, type=float, default=None, metavar=("YMIN", "YMAX"),
        help="Y-axis range for comparison pages (1-2), e.g. --ylim -0.05 0.15",
    )
    parser.add_argument(
        "--ylim-diff", nargs=2, type=float, default=None, metavar=("YMIN", "YMAX"),
        help="Y-axis range for difference pages (3-4), e.g. --ylim-diff -0.05 0.05",
    )
    parser.add_argument(
        "--output", default=None, metavar="PDF",
        help="Output PDF (default: output/plots/target_dependence.pdf)",
    )
    args      = parser.parse_args()
    ylim      = tuple(args.ylim)      if args.ylim      else None
    ylim_diff = tuple(args.ylim_diff) if args.ylim_diff else None

    df_all  = _load_all(args.variable)
    targets = _ordered_targets(df_all, args.targets)
    bins    = _pt_bins(df_all)

    if not targets:
        print("No matching targets found — nothing to plot.")
        return

    out = (Path(args.output) if args.output
           else PLOTS_DIR / "target_dependence.pdf")
    out.parent.mkdir(parents=True, exist_ok=True)

    diff_targets = [t for t in targets if t != args.reference]
    has_diff     = args.reference in df_all["target"].values and diff_targets

    with PdfPages(out) as pdf:
        for particle in ("pip", "pim"):
            col   = _PARTICLE_COL[particle]
            df_p  = df_all[df_all["particle"] == col]
            if df_p.empty:
                continue
            tgts  = [t for t in targets if t in df_p["target"].values]

            # ── comparison grid ──────────────────────────────────────────────
            title = _page_suptitle(df_p, particle)
            fig   = _fig_grid(df_p, particle, tgts, bins, title, ylim=ylim)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        # ── difference grids ─────────────────────────────────────────────────
        if has_diff:
            ref = args.reference
            for particle in ("pip", "pim"):
                col   = _PARTICLE_COL[particle]
                df_p  = df_all[df_all["particle"] == col]
                if df_p.empty:
                    continue
                dtgts = [t for t in diff_targets if t in df_p["target"].values]
                if not dtgts:
                    continue

                df_diff = _compute_diff(df_p, ref)
                title   = _page_suptitle(df_p, particle, suffix=rf" $-$ {ref}")
                fig     = _fig_grid(df_diff, particle, dtgts, bins, title,
                                    diff_ref=ref, ylim=ylim_diff, fmt="rs")
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

    n_pages = 2 + (2 if has_diff else 0)
    print(f"Saved: {out}  ({n_pages} pages)")


if __name__ == "__main__":
    main()

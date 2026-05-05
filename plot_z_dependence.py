#!/usr/bin/env python3
"""
Plot z-dependence of the SSA for a given target and hadron.

Reads all *_binned_summary.csv files in output/combined/, filters by
target and particle, and produces a two-page PDF:

  Page 1 — A_LU^sinφ vs p_T (one curve per z) and vs z (one curve per p_T bin)
  Page 2 — A_LU^sinφ vs z, one panel per p_T bin, shared y-axis, no gap

Must be run from the project root.

Usage
-----
    python plot_z_dependence.py --target C --particle pip
    python plot_z_dependence.py --target LH2 --particle pim
    python plot_z_dependence.py --target C --particle pip --output my_plot.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd

COMBINED_DIR = Path("output/combined")
PLOTS_DIR    = Path("output/plots")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

_PARTICLE_COL = {"pip": "pi+", "pim": "pi-"}


def _load(target: str, particle: str, variable: str) -> pd.DataFrame:
    """Concatenate all matching *_binned_summary.csv rows."""
    particle_col = _PARTICLE_COL[particle]
    frames = []
    for csv in sorted(COMBINED_DIR.glob("*_binned_summary.csv")):
        df   = pd.read_csv(csv)
        mask = (
            (df["target"]   == target)       &
            (df["particle"] == particle_col) &
            (df["variable"] == variable)
        )
        sub = df[mask]
        if not sub.empty:
            frames.append(sub)

    if not frames:
        raise FileNotFoundError(
            f"No rows found in {COMBINED_DIR}/*_binned_summary.csv "
            f"for target={target!r} particle={particle!r} variable={variable!r}"
        )
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _particle_tex(particle: str) -> str:
    return r"$\pi^+$" if particle == "pip" else r"$\pi^-$"


def _marker_style(n: int) -> str:
    styles = ["o", "s", "^", "D", "v", "p", "*", "h"]
    return styles[n % len(styles)]



def _pt_bin_label(hmin: float, hmax: float) -> str:
    return f"$p_T \\in [{hmin:.2f},\\,{hmax:.2f})$ GeV/$c$"


def _z_colors(z_vals: list[float]) -> dict[float, tuple]:
    cmap = plt.colormaps["tab10"].resampled(max(len(z_vals), 2))
    return {z: cmap(i / max(len(z_vals) - 1, 1)) for i, z in enumerate(z_vals)}


def _pt_colors(n: int) -> list[tuple]:
    cmap = plt.colormaps["Set1"].resampled(max(n, 2))
    return [cmap(i / max(n - 1, 1)) for i in range(n)]


def _pt_bins(df: pd.DataFrame) -> pd.DataFrame:
    """Unique pt bins sorted by hmin."""
    return (df.drop_duplicates("histogram")
              .sort_values("hmin")[["histogram", "hmin", "hmax", "bin_center"]]
              .reset_index(drop=True))


def _suptitle(df: pd.DataFrame, particle: str) -> str:
    target = df["target"].iloc[0]
    x      = df["x"].iloc[0]
    q2     = df["q2"].iloc[0]
    return (f"SSA $A_{{LU}}^{{\\sin\\phi}}$ — "
            f"{target}, {_particle_tex(particle)},  "
            f"$x = {x:.2f}$,  $Q^2 = {q2:.1f}$ GeV$^2$")


def _ax_style(ax: plt.Axes, ylabel: bool = True) -> None:
    ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.tick_params(which="both", direction="in", top=True, right=True)
    if not ylabel:
        ax.tick_params(labelleft=False)


# ---------------------------------------------------------------------------
# Page 1: overview — A vs p_T (left) and A vs z (right)
# ---------------------------------------------------------------------------

def _fig_overview(df: pd.DataFrame, particle: str) -> plt.Figure:
    z_vals  = sorted(df["z"].unique())
    z_colors = _z_colors(z_vals)
    bins    = _pt_bins(df)
    pt_cols = _pt_colors(len(bins))

    fig, (ax_pt, ax_z) = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle(_suptitle(df, particle), fontsize=10)

    # left: A vs p_T, one curve per z
    for i, z in enumerate(z_vals):
        grp = df[df["z"] == z].sort_values("bin_center")
        ax_pt.errorbar(
            grp["bin_center"], grp["asym"], yerr=grp["asym_err"],
            fmt=_marker_style(i), color=z_colors[z], label=f"$z = {z}$",
            capsize=3, markersize=5, lw=1.3, elinewidth=1.0,
        )
    ax_pt.axhline(0, color="k", lw=0.7, ls="--", zorder=0)
    ax_pt.set_xlabel("$p_T$ (GeV/$c$)", fontsize=10)
    ax_pt.set_ylabel(r"$A_{LU}^{\sin\phi}$", fontsize=10)
    ax_pt.set_title("Asymmetry vs $p_T$", fontsize=10)
    ax_pt.legend(fontsize=8, framealpha=0.7)
    _ax_style(ax_pt)

    # right: A vs z, one curve per p_T bin
    for i, row in bins.iterrows():
        grp   = df[df["histogram"] == row["histogram"]].sort_values("z")
        label = _pt_bin_label(row["hmin"], row["hmax"])
        ax_z.errorbar(
            grp["z"], grp["asym"], yerr=grp["asym_err"],
            fmt=_marker_style(i), color=pt_cols[i], label=label,
            capsize=3, markersize=5, lw=1.3, elinewidth=1.0,
        )
    ax_z.axhline(0, color="k", lw=0.7, ls="--", zorder=0)
    ax_z.set_xlabel("$z$", fontsize=10)
    ax_z.set_ylabel(r"$A_{LU}^{\sin\phi}$", fontsize=10)
    ax_z.set_title("Asymmetry vs $z$", fontsize=10)
    ax_z.legend(fontsize=8, framealpha=0.7)
    _ax_style(ax_z)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Page 2: A vs z  (one panel per p_T bin, shared y-axis)
# ---------------------------------------------------------------------------

def _fig_vs_z_panels(df: pd.DataFrame, particle: str) -> plt.Figure:
    bins = _pt_bins(df)
    n    = len(bins)

    fig, axes = plt.subplots(
        1, n,
        sharey=True,
        figsize=(3.5 * n, 4.5),
    )
    fig.subplots_adjust(wspace=0)
    if n == 1:
        axes = [axes]

    fig.suptitle(_suptitle(df, particle), fontsize=10)

    for i, (ax, row) in enumerate(zip(axes, bins.itertuples())):
        grp = df[df["histogram"] == row.histogram].sort_values("z")

        ax.errorbar(
            grp["z"], grp["asym"], yerr=grp["asym_err"],
            fmt="ks", capsize=3, markersize=5, elinewidth=1.0, lw=0,
        )
        ax.axhline(0, color="gray", lw=0.8, ls="--", zorder=0)

        ax.text(
            0.05, 0.95,
            f"$\\langle P_T \\rangle = {row.bin_center:.2f}$ GeV/$c$",
            transform=ax.transAxes, va="top", ha="left", fontsize=9,
        )

        ax.set_xlabel("$z$", fontsize=10)
        if i == 0:
            ax.set_ylabel(r"$A_{LU}^{\sin\phi}$", fontsize=10)
        _ax_style(ax, ylabel=(i == 0))

    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot z-dependence of SSA (two-page PDF)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--target",   required=True, metavar="TARGET",
                        help="Target material: C  Cu  LD2  LH2  Al")
    parser.add_argument("--particle", required=True, choices=["pip", "pim"],
                        help="Hadron: pip or pim")
    parser.add_argument("--variable", default="pt", metavar="VAR",
                        help="Binning variable in summary CSV (default: pt)")
    parser.add_argument("--output",   default=None, metavar="PDF",
                        help="Output PDF (default: output/plots/<target>_<particle>_z_dependence.pdf)")
    args = parser.parse_args()

    df = _load(args.target, args.particle, args.variable)

    out = (Path(args.output) if args.output
           else PLOTS_DIR / f"{args.target}_{args.particle}_z_dependence.pdf")
    out.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(out) as pdf:
        fig1 = _fig_overview(df, args.particle)
        pdf.savefig(fig1, bbox_inches="tight")
        plt.close(fig1)

        fig2 = _fig_vs_z_panels(df, args.particle)
        pdf.savefig(fig2, bbox_inches="tight")
        plt.close(fig2)

    print(f"Saved: {out}  (2 pages)")


if __name__ == "__main__":
    main()

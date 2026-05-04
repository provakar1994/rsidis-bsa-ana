#!/usr/bin/env python3
"""
Plot z-dependence of the SSA for a given target and hadron.

Reads all *_binned_summary.csv files in output/combined/, filters by
target and particle, and produces a two-panel figure:

  Left  — A_LU^sinφ vs p_T  (one curve per z value)
  Right — A_LU^sinφ vs z    (one curve per p_T bin)

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
    styles = ['o', 's', '^', 'D', 'v', 'p', '*', 'h', '<', '>']
    return styles[n % len(styles)]


def _pt_bin_label(hmin: float, hmax: float) -> str:
    return f"$p_T \\in [{hmin:.2f},\\,{hmax:.2f})$ GeV/$c$"


def _z_colors(z_vals: list[float]) -> dict[float, tuple]:
    cmap = plt.colormaps["tab10"].resampled(max(len(z_vals), 2))
    return {z: cmap(i / max(len(z_vals) - 1, 1)) for i, z in enumerate(z_vals)}


def _pt_colors(n: int) -> list[tuple]:
    cmap = plt.colormaps["Set1"].resampled(max(n, 2))
    return [cmap(i / max(n - 1, 1)) for i in range(n)]


# ---------------------------------------------------------------------------
# Individual panels
# ---------------------------------------------------------------------------

def _panel_vs_pt(ax: plt.Axes, df: pd.DataFrame, colors: dict) -> None:
    """Left panel: A vs p_T, one curve per z."""
    counter = 0
    for z, grp in df.groupby("z"):
        grp = grp.sort_values("bin_center")
        mstyle = _marker_style(counter)
        ax.errorbar(
            grp["bin_center"], grp["asym"], yerr=grp["asym_err"],
            fmt=mstyle, color=colors[z], label=f"$z = {z}$",
            capsize=3, markersize=5, lw=1.3, elinewidth=1.0,
        )
        counter += 1
    ax.axhline(0, color="k", lw=0.7, ls="--", zorder=0)
    ax.set_xlabel("$p_T$ (GeV/$c$)", fontsize=10)
    ax.set_ylabel(r"$A_{LU}^{\sin\phi}$", fontsize=10)
    ax.set_title("Asymmetry vs $p_T$", fontsize=10)
    ax.legend(fontsize=8, framealpha=0.7)
    ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.tick_params(which="both", direction="in", top=True, right=True)


def _panel_vs_z(ax: plt.Axes, df: pd.DataFrame) -> None:
    """Right panel: A vs z, one curve per p_T bin."""
    bins   = (df.drop_duplicates("histogram")
                .sort_values("hmin")[["histogram", "hmin", "hmax"]]
                .reset_index(drop=True))
    colors = _pt_colors(len(bins))

    for i, row in bins.iterrows():
        grp   = df[df["histogram"] == row["histogram"]].sort_values("z")
        label = _pt_bin_label(row["hmin"], row["hmax"])
        mstyle = _marker_style(i)
        ax.errorbar(
            grp["z"], grp["asym"], yerr=grp["asym_err"],
            fmt=mstyle, color=colors[i], label=label,
            capsize=3, markersize=5, lw=1.3, elinewidth=1.0,
        )
    ax.axhline(0, color="k", lw=0.7, ls="--", zorder=0)
    ax.set_xlabel("$z$", fontsize=10)
    ax.set_ylabel(r"$A_{LU}^{\sin\phi}$", fontsize=10)
    ax.set_title("Asymmetry vs $z$", fontsize=10)
    ax.legend(fontsize=8, framealpha=0.7)
    ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.tick_params(which="both", direction="in", top=True, right=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot z-dependence of SSA: A_LU^sinφ vs p_T and vs z",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--target",   required=True, metavar="TARGET",
                        help="Target material: C  Cu  LD2  LH2  Al")
    parser.add_argument("--particle", required=True, choices=["pip", "pim"],
                        help="Hadron: pip or pim")
    parser.add_argument("--variable", default="pt", metavar="VAR",
                        help="Binning variable used in summary CSV (default: pt)")
    parser.add_argument("--output",   default=None, metavar="PDF",
                        help="Output PDF path (default: output/plots/<target>_<particle>_z_dependence.pdf)")
    args = parser.parse_args()

    df = _load(args.target, args.particle, args.variable)

    z_vals = sorted(df["z"].unique())
    colors = _z_colors(z_vals)

    # ── figure ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    x_val  = df["x"].iloc[0]
    q2_val = df["q2"].iloc[0]
    fig.suptitle(
        f"SSA $A_{{LU}}^{{\\sin\\phi}}$ — "
        f"{args.target}, {_particle_tex(args.particle)},  "
        f"$x = {x_val:.2f}$,  $Q^2 = {q2_val:.1f}$ GeV$^2$",
        fontsize=11,
    )

    _panel_vs_pt(axes[0], df, colors)
    _panel_vs_z(axes[1], df)

    fig.tight_layout()

    out = (Path(args.output) if args.output
           else PLOTS_DIR / f"{args.target}_{args.particle}_z_dependence.pdf")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()

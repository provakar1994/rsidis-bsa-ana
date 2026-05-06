#!/usr/bin/env python3
"""
Compare SSA across targets and compute differences from a reference target.

Reads all *_binned_summary.csv files in output/combined/ and produces a
multi-page PDF.  π⁺ and π⁻ are overlaid on the same panels with distinct
colours and markers.  Without --epsilon, pages cover A_LU^sinφ only.
With --epsilon, additional pages show F_LU^sinφ/F_UU:

    F_LU^sinφ / F_UU = A_LU^sinφ / sqrt(2 ε (1-ε))

Page layout
-----------
  1 — A_LU^sinφ, all targets vs z (both π)
  2 — A_LU^sinφ difference (A_target − A_ref) (both π)
  3 — F_LU/F_UU, all targets vs z              (only with --epsilon)
  4 — F_LU/F_UU difference                     (only with --epsilon)

Must be run from the project root.

Usage
-----
    python plot_target_dependence.py
    python plot_target_dependence.py --epsilon 0.7
    python plot_target_dependence.py --epsilon 0.7 \\
        --ylim -0.02 0.12  --ylim-diff -0.06 0.06 \\
        --ylim-flu -0.05 0.20  --ylim-flu-diff -0.10 0.10
    python plot_target_dependence.py --targets C Cu LD2 LH2 --reference LH2
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

COMBINED_DIR  = Path("output/combined")
PLOTS_DIR     = Path("output/plots")

_TARGET_ORDER = ["LH2", "LD2", "C", "Cu", "Al"]   # preferred display order

# LaTeX quantity symbols (without outer $) used to build axis labels
_QTY_A = r"A_{LU}^{\sin\phi}"
_QTY_F = r"F_{LU}^{\sin\phi}/F_{UU}"

# Per-particle plot styles
_PARTICLE_STYLES = {
    "pi+": dict(marker="o", color="black", label=r"$\pi^+$"),
    "pi-": dict(marker="s", color="red", label=r"$\pi^-$"),
}


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
# Derived observables
# ---------------------------------------------------------------------------

def _to_flu_fuu(df: pd.DataFrame, epsilon: float) -> pd.DataFrame:
    """Scale asym → F_LU^sinφ / F_UU = asym / sqrt(2 ε (1−ε))."""
    scale          = 1.0 / np.sqrt(2.0 * epsilon * (1.0 - epsilon))
    df             = df.copy()
    df["asym"]     = df["asym"]     * scale
    df["asym_err"] = df["asym_err"] * scale
    return df


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ax_style(ax: plt.Axes, *, xlabel: bool, ylabel: bool) -> None:
    ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.tick_params(which="both", direction="in", top=True, right=True)
    if not xlabel:
        ax.tick_params(labelbottom=False)
    if not ylabel:
        ax.tick_params(labelleft=False)


def _page_suptitle(df: pd.DataFrame, qty: str,
                   suffix: str = "", epsilon: float | None = None) -> str:
    x   = df["x"].iloc[0]
    q2  = df["q2"].iloc[0]
    eps = f",  $\\varepsilon = {epsilon:.3f}$" if epsilon is not None else ""
    return (f"${qty}${suffix} — "
            f"$x = {x:.2f}$,  $Q^2 = {q2:.1f}$ GeV$^2${eps}")


def _ylabel(qty: str, target: str | None, diff_ref: str | None) -> str:
    if diff_ref is None:
        return f"${qty}$"
    return (rf"${qty}|_{{\rm {target}}}"
            rf" - {qty}|_{{\rm {diff_ref}}}$")


# ---------------------------------------------------------------------------
# Difference computation
# ---------------------------------------------------------------------------

def _compute_diff(df: pd.DataFrame, reference: str) -> pd.DataFrame:
    """Subtract reference-target values from every other target (all particles)."""
    ref = (df[df["target"] == reference]
           .rename(columns={"asym": "_ref", "asym_err": "_ref_err"})
           [["particle", "z", "histogram", "_ref", "_ref_err"]])

    other  = df[df["target"] != reference].copy()
    merged = other.merge(ref, on=["particle", "z", "histogram"], how="inner")

    merged["asym"]     = merged["asym"] - merged["_ref"]
    merged["asym_err"] = np.sqrt(merged["asym_err"] ** 2 + merged["_ref_err"] ** 2)
    return merged.drop(columns=["_ref", "_ref_err"])


# ---------------------------------------------------------------------------
# Grid figure  (both particles overlaid)
# ---------------------------------------------------------------------------

def _fig_grid(
    df:       pd.DataFrame,
    targets:  list[str],
    bins:     pd.DataFrame,
    suptitle: str,
    qty:      str = _QTY_A,
    diff_ref: str | None = None,
    ylim:     tuple[float, float] | None = None,
) -> plt.Figure:
    """
    Grid: rows = targets, cols = p_T bins.
    Both π⁺ and π⁻ are overlaid per panel with distinct colour/marker.
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

            for particle, style in _PARTICLE_STYLES.items():
                grp = (df[(df["target"]    == target)
                          & (df["particle"]   == particle)
                          & (df["histogram"]  == brow.histogram)]
                       .sort_values("z"))
                if grp.empty:
                    continue
                ax.errorbar(
                    grp["z"], grp["asym"], yerr=grp["asym_err"],
                    fmt=style["marker"], color=style["color"],
                    capsize=3, markersize=4, elinewidth=1.0, lw=0,
                )

            ax.axhline(0, color="gray", lw=0.8, ls="--", zorder=0)

            if ylim is not None:
                ax.set_ylim(*ylim)

            # target label — top-right corner
            ax.text(0.97, 0.95, target,
                    transform=ax.transAxes, va="top", ha="right",
                    fontsize=9, fontweight="bold")

            # p_T bin label — top-left of top-row panels
            if is_top:
                ax.text(0.05, 0.95,
                        f"$\\langle P_T \\rangle = {brow.bin_center:.2f}$ GeV/$c$",
                        transform=ax.transAxes, va="top", ha="left", fontsize=9)

            if is_left:
                ax.set_ylabel(_ylabel(qty, target, diff_ref), fontsize=8)

            if is_bottom:
                ax.set_xlabel("$z$", fontsize=10)

            _ax_style(ax, xlabel=is_bottom, ylabel=is_left)

    # legend — bottom-left corner of the top-left panel
    handles = [
        Line2D([], [], marker=st["marker"], color=st["color"],
               ls="", markersize=5, label=st["label"])
        for st in _PARTICLE_STYLES.values()
    ]
    axes[0, 0].legend(handles=handles, loc="lower left",
                      fontsize=8, framealpha=0.7)

    return fig


# ---------------------------------------------------------------------------
# Helper: emit comparison + difference pages for one observable
# ---------------------------------------------------------------------------

def _emit_pages(
    pdf:          PdfPages,
    df_all:       pd.DataFrame,
    targets:      list[str],
    diff_targets: list[str],
    bins:         pd.DataFrame,
    has_diff:     bool,
    reference:    str,
    qty:          str,
    ylim:         tuple | None,
    ylim_diff:    tuple | None,
    epsilon:      float | None = None,
) -> int:
    """Write one comparison page + one difference page. Returns page count."""
    n = 0

    # ── comparison ───────────────────────────────────────────────────────────
    title = _page_suptitle(df_all, qty, epsilon=epsilon)
    fig   = _fig_grid(df_all, targets, bins, title, qty=qty, ylim=ylim)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    n += 1

    # ── difference ───────────────────────────────────────────────────────────
    if has_diff:
        df_diff = _compute_diff(df_all, reference)
        title   = _page_suptitle(df_all, qty,
                                 suffix=rf" $-$ {reference}", epsilon=epsilon)
        fig     = _fig_grid(df_diff, diff_targets, bins, title,
                            qty=qty, diff_ref=reference, ylim=ylim_diff)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        n += 1

    return n


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Target-dependence of SSA/structure-function ratio vs z",
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
    # ── A_LU ylim ────────────────────────────────────────────────────────────
    parser.add_argument(
        "--ylim", nargs=2, type=float, default=None, metavar=("YMIN", "YMAX"),
        help="Y range for A_LU comparison page",
    )
    parser.add_argument(
        "--ylim-diff", nargs=2, type=float, default=None, metavar=("YMIN", "YMAX"),
        help="Y range for A_LU difference page",
    )
    # ── F_LU/F_UU ────────────────────────────────────────────────────────────
    parser.add_argument(
        "--epsilon", type=float, default=None, metavar="EPS",
        help="Virtual-photon depolarisation ε ∈ (0,1); enables F_LU/F_UU pages",
    )
    parser.add_argument(
        "--ylim-flu", nargs=2, type=float, default=None, metavar=("YMIN", "YMAX"),
        help="Y range for F_LU/F_UU comparison page",
    )
    parser.add_argument(
        "--ylim-flu-diff", nargs=2, type=float, default=None, metavar=("YMIN", "YMAX"),
        help="Y range for F_LU/F_UU difference page",
    )
    # ── output ───────────────────────────────────────────────────────────────
    parser.add_argument(
        "--output", default=None, metavar="PDF",
        help="Output PDF (default: output/plots/target_dependence.pdf)",
    )
    args = parser.parse_args()

    if args.epsilon is not None and not (0.0 < args.epsilon < 1.0):
        parser.error("--epsilon must be in (0, 1)")

    ylim          = tuple(args.ylim)          if args.ylim          else None
    ylim_diff     = tuple(args.ylim_diff)     if args.ylim_diff     else None
    ylim_flu      = tuple(args.ylim_flu)      if args.ylim_flu      else None
    ylim_flu_diff = tuple(args.ylim_flu_diff) if args.ylim_flu_diff else None

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
    has_diff     = args.reference in df_all["target"].values and bool(diff_targets)

    n_pages = 0
    with PdfPages(out) as pdf:
        # ── A_LU^sinφ pages ──────────────────────────────────────────────────
        n_pages += _emit_pages(
            pdf, df_all, targets, diff_targets, bins, has_diff,
            reference = args.reference,
            qty       = _QTY_A,
            ylim      = ylim,
            ylim_diff = ylim_diff,
        )

        # ── F_LU^sinφ / F_UU pages (only when --epsilon is given) ────────────
        if args.epsilon is not None:
            df_flu = _to_flu_fuu(df_all, args.epsilon)
            n_pages += _emit_pages(
                pdf, df_flu, targets, diff_targets, bins, has_diff,
                reference = args.reference,
                qty       = _QTY_F,
                ylim      = ylim_flu,
                ylim_diff = ylim_flu_diff,
                epsilon   = args.epsilon,
            )

    print(f"Saved: {out}  ({n_pages} pages)")


if __name__ == "__main__":
    main()

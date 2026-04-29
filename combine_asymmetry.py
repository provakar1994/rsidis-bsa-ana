"""
Combine asymmetry CSVs via inverse-variance weighted averaging and re-fit.

Usage:
    python combine_asymmetry.py file1.csv file2.csv ... [--output PATH] [--label LABEL]

Reads any number of per-config asymmetry CSVs produced by analysis.py
(columns: histogram, phi_center, A_phys, A_phys_err), combines them
bin-by-bin with inverse-variance weighting, re-fits A × sin(φ), and writes:

  <output>.csv  — combined bin values (same format as individual CSVs)
  <output>.pdf  — individual + combined points with fit overlay

If --output is omitted, outputs are written to the current directory as
"combined_asymmetry.csv" / "combined_asymmetry.pdf".
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D

from rsidis_ssa.asymmetry import fit_sinphi

logging.basicConfig(level="INFO", format="%(levelname)-8s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Combine
# ---------------------------------------------------------------------------

def combine_bins(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Inverse-variance weighted average of A_phys across input DataFrames,
    grouped by (histogram, phi_center).

    Bins where A_phys_err is zero or non-finite are dropped before averaging.
    """
    all_rows = pd.concat(dfs, ignore_index=True)
    all_rows["A_phys"]     = pd.to_numeric(all_rows["A_phys"],     errors="coerce")
    all_rows["A_phys_err"] = pd.to_numeric(all_rows["A_phys_err"], errors="coerce")

    ok = np.isfinite(all_rows["A_phys"]) & np.isfinite(all_rows["A_phys_err"]) & (all_rows["A_phys_err"] > 0)
    all_rows = all_rows.loc[ok].copy()

    records = []
    for (hname, phi), grp in all_rows.groupby(["histogram", "phi_center"], sort=True):
        inv_var  = 1.0 / grp["A_phys_err"].values**2
        A_comb   = float(np.sum(grp["A_phys"].values * inv_var) / np.sum(inv_var))
        dA_comb  = float(1.0 / np.sqrt(np.sum(inv_var)))
        records.append({"histogram": hname, "phi_center": phi,
                        "A_phys": A_comb, "A_phys_err": dA_comb})

    return pd.DataFrame(records, columns=["histogram", "phi_center", "A_phys", "A_phys_err"])


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

_INPUT_COLORS = [
    "steelblue", "tomato", "seagreen", "darkorange",
    "purple",    "sienna", "teal",     "crimson",
]


def _plot_histogram(ax_full, ax_zoom, hname: str,
                    combined: pd.DataFrame,
                    inputs: list[tuple[str, pd.DataFrame]]) -> None:
    """
    Two panels for one histogram:
      left  — full auto-scale with individual points + combined + fit
      right — zoomed to [-0.1, 0.1] (same content)
    """
    phi_fit = np.linspace(-np.pi, np.pi, 300)
    ZOOM_LO, ZOOM_HI = -0.1, 0.1

    sub = combined[combined["histogram"] == hname].sort_values("phi_center")
    phi_c = sub["phi_center"].values
    A_c   = sub["A_phys"].values
    dA_c  = sub["A_phys_err"].values

    amplitude, amplitude_err, chi2_ndf, n_used = fit_sinphi(phi_c, A_c, dA_c)

    hw = 0.5 * (phi_c[1] - phi_c[0]) if len(phi_c) > 1 else 0.0

    for ax in (ax_full, ax_zoom):
        ax.axhline(0.0, color="gray", lw=0.8, ls="--", zorder=1)

        # Individual input points
        for (label, df_in), color in zip(inputs, _INPUT_COLORS):
            sub_in = df_in[df_in["histogram"] == hname].sort_values("phi_center")
            if sub_in.empty:
                continue
            ax.errorbar(sub_in["phi_center"].values,
                        sub_in["A_phys"].values,
                        yerr=sub_in["A_phys_err"].values,
                        fmt="o", color=color, ms=4, lw=0.8,
                        alpha=0.45, capsize=2, zorder=2,
                        label=label)

        # Combined points
        ax.errorbar(phi_c, A_c, yerr=dA_c, xerr=hw,
                    fmt="D", color="black", ms=6, lw=1.4,
                    capsize=3, zorder=4, label="combined")

        # Fit curve
        if np.isfinite(amplitude):
            ax.plot(phi_fit, amplitude * np.sin(phi_fit),
                    color="tomato", lw=1.8, zorder=3, label=r"$A\sin\phi$ fit")

        ax.set_xlim(-np.pi * 1.05, np.pi * 1.05)
        ax.set_xlabel(r"$\phi_{pq}$ (rad)")
        ax.set_ylabel(r"$A_{LU}$")

    # Full panel — title + info box
    ax_full.set_title(hname)
    ax_full.legend(fontsize=7)
    if np.isfinite(amplitude):
        info = (
            rf"$A_{{LU}}^{{\sin\phi}} = {amplitude:+.4f} \pm {amplitude_err:.4f}$"
            f"\n$\\chi^2/\\mathrm{{ndf}} = {chi2_ndf:.2f}$"
            f"\n$N_{{\\rm bins}} = {n_used}$"
            f"\n$N_{{\\rm inputs}} = {len(inputs)}$"
        )
    else:
        info = "fit: insufficient bins"
    ax_full.text(0.97, 0.97, info, transform=ax_full.transAxes,
                 va="top", ha="right", fontsize=7, family="monospace",
                 bbox=dict(boxstyle="round", fc="0.96", ec="0.8"))

    # Zoom panel
    ax_zoom.set_ylim(ZOOM_LO, ZOOM_HI)
    ax_zoom.set_title(f"{hname}  [zoom {ZOOM_LO}, {ZOOM_HI}]", fontsize=9)


def write_plot(pdf_path: Path,
               combined: pd.DataFrame,
               inputs: list[tuple[str, pd.DataFrame]]) -> None:
    histograms = combined["histogram"].unique().tolist()
    n = len(histograms)
    fig, axes_grid = plt.subplots(n, 2, figsize=(11.0, 4.5 * n), sharey=False)
    if n == 1:
        axes_grid = axes_grid.reshape(1, 2)

    fig.suptitle(
        r"Combined beam SSA  —  $A_{LU}^{\sin\phi}$",
        fontsize=11, fontweight="bold",
    )

    for row, hname in enumerate(histograms):
        _plot_histogram(axes_grid[row, 0], axes_grid[row, 1],
                        hname, combined, inputs)

    fig.tight_layout()
    with PdfPages(pdf_path) as pdf:
        pdf.savefig(fig)
    plt.close(fig)
    logger.info("Plot → %s", pdf_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combine asymmetry CSVs via inverse-variance weighting"
    )
    parser.add_argument("csvfiles", nargs="+", type=Path,
                        help="Input asymmetry CSV files (>=2 recommended)")
    parser.add_argument("--output", "-o", type=Path, default=None,
                        help="Output path stem (default: combined_asymmetry)")
    parser.add_argument("--label", "-l", type=str, default=None,
                        help="Override label used in plot title")
    args = parser.parse_args()

    if len(args.csvfiles) < 2:
        logger.warning("Only one input CSV — combined result equals input.")

    # Load inputs
    inputs: list[tuple[str, pd.DataFrame]] = []
    for path in args.csvfiles:
        if not path.exists():
            logger.error("File not found: %s", path)
            sys.exit(1)
        df = pd.read_csv(path)
        for col in ("histogram", "phi_center", "A_phys", "A_phys_err"):
            if col not in df.columns:
                logger.error("Missing column '%s' in %s", col, path)
                sys.exit(1)
        inputs.append((path.stem, df))
        logger.info("Loaded %d rows from %s", len(df), path)

    # Output stem
    out_stem = args.output or Path("combined_asymmetry")
    out_stem.parent.mkdir(parents=True, exist_ok=True)

    # Combine
    combined = combine_bins([df for _, df in inputs])
    logger.info("Combined: %d rows across %d histogram(s)",
                len(combined), combined["histogram"].nunique())

    # Log fit results
    for hname, grp in combined.groupby("histogram"):
        phi = grp["phi_center"].values
        A   = grp["A_phys"].values
        dA  = grp["A_phys_err"].values
        amp, amp_err, chi2_ndf, n_used = fit_sinphi(phi, A, dA)
        logger.info(
            "[%s]  A_LU^sinφ = %+.4f ± %.4f  χ²/ndf = %.2f  (%d bins)",
            hname,
            amp if np.isfinite(amp) else float("nan"),
            amp_err if np.isfinite(amp_err) else float("nan"),
            chi2_ndf if np.isfinite(chi2_ndf) else float("nan"),
            n_used,
        )

    # Write CSV
    csv_path = out_stem.with_suffix(".csv")
    combined.to_csv(csv_path, index=False)
    logger.info("CSV  → %s", csv_path)

    # Write plot
    pdf_path = out_stem.with_suffix(".pdf")
    write_plot(pdf_path, combined, inputs)


if __name__ == "__main__":
    main()

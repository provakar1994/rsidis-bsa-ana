"""
Combine asymmetry CSVs via inverse-variance weighted averaging and re-fit.

Two run modes
-------------
Direct mode — list CSV files explicitly:

    python combine_asymmetry.py file1.csv file2.csv ... [--output PATH]

Config mode — locate CSVs automatically from a base YAML + thpq values:

    python combine_asymmetry.py \\
        --base_config config/C_pim_e10p7_x0p25_q23p3_z0p5_base.yaml \\
        --thpq -0.8 2.0 [--is_binned] [--output PATH]

    Looks up output/<stem>/<stem>[_binned].csv for each thpq.
    Default output: output/combined/<setting>_thpq<t1>AND<t2>[_binned].

Accepts both overall and binned asymmetry CSVs produced by analysis.py.
Format is auto-detected from column names:

  Overall  (columns: histogram, phi_center, A_phys, A_phys_err)
  Binned   (columns: variable, hmin, hmax, bin_center, histogram, phi_center,
                     A_phys, A_phys_err)

Outputs
-------
  <output>.csv         — combined bin values (same format as input)
  <output>.pdf         — individual + combined points with fit overlay
  <output>_summary.csv — one row per histogram (overall) or per
                         (variable, hmin, hmax, histogram) (binned) with fit amplitude

All inputs must be the same format; mixing overall and binned raises an error.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from rsidis_ssa.asymmetry import fit_sinphi

logging.basicConfig(level="INFO", format="%(levelname)-8s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Kinematic stem parser
# ---------------------------------------------------------------------------

_STEM_RE = re.compile(
    r"^(?P<target>[A-Za-z0-9]+)_"
    r"(?P<particle>pi[mp])_"
    r"e(?P<ebeam>[0-9mp]+)_"
    r"x(?P<x>[0-9mp]+)_"
    r"q2(?P<q2>[0-9mp]+)_"
    r"z(?P<z>[0-9mp]+)_"
    r"(?:(?P<process>[A-Za-z0-9_]+)_)?"
    r"thpq(?P<thpq>[0-9mp]+)$"
)
_PARTICLE_LABEL = {"pim": "pi-", "pip": "pi+"}


def _decode_float(s: str) -> float:
    """Decode encoded float: 'p' → '.', 'm' → '-'."""
    return float(s.replace("m", "-").replace("p", "."))


def _encode_float(v: float) -> str:
    """Encode float to filename format: '.' → 'p', '-' → 'm'.  E.g. -0.8 → 'm0p8'."""
    s = f"{v:g}"
    if "." not in s:
        s += ".0"
    return s.replace("-", "m").replace(".", "p")


def _parse_kinematic_stem(stem: str) -> dict | None:
    """
    Parse kinematic fields from a config filename stem.

    Returns a dict with keys target, particle, ebeam, x, q2, z, thpq,
    or None if the stem does not match the expected pattern.
    """
    # Strip known suffixes added by analysis.py (_binned, _summary, etc.)
    stem = re.sub(r"_(binned|summary)$", "", stem)
    m = _STEM_RE.match(stem)
    if not m:
        return None
    return {
        "target":   m.group("target"),
        "particle": _PARTICLE_LABEL.get(m.group("particle"), m.group("particle")),
        "ebeam":    _decode_float(m.group("ebeam")),
        "x":        _decode_float(m.group("x")),
        "q2":       _decode_float(m.group("q2")),
        "z":        _decode_float(m.group("z")),
        "process":  m.group("process") or "sidis",
        "thpq":     _decode_float(m.group("thpq")),
    }


def _build_inputs_from_config(
    base_config: Path,
    thpq_vals:   list[float],
    is_binned:   bool,
) -> tuple[list[tuple[str, pd.DataFrame]], Path]:
    """
    Discover CSV files from a base YAML config + thpq values.

    Expects analysis.py output in:
        output/<stem>/<stem>.csv          (overall)
        output/<stem>/<stem>_binned.csv   (binned)

    where <stem> = <base_config_stem_without_base>_thpq<encoded_thpq>.
    Process-tagged bases are also supported, e.g.
    <base_config_stem_without_base>_thpq... for
    C_pip_..._z0p9_exclusive_base.yaml.

    Returns (inputs, default_output_stem).
    """
    setting_stem = re.sub(r"_base$", "", base_config.stem)
    thpq_vals    = sorted(thpq_vals)
    inputs: list[tuple[str, pd.DataFrame]] = []
    for thpq in thpq_vals:
        stem     = f"{setting_stem}_thpq{_encode_float(thpq)}"
        suffix   = "_binned" if is_binned else ""
        csv_path = Path("output") / stem / f"{stem}{suffix}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(
                f"CSV not found: {csv_path}\n"
                f"  (expected analysis.py output for thpq={thpq})"
            )
        df = pd.read_csv(csv_path)
        for col in ("histogram", "phi_center", "A_phys", "A_phys_err"):
            if col not in df.columns:
                raise ValueError(f"Missing column '{col}' in {csv_path}")
        inputs.append((stem, df))
        logger.info("Loaded %d rows from %s", len(df), csv_path)

    thpq_str    = "AND".join(_encode_float(t) for t in thpq_vals)
    out_suffix  = "_binned" if is_binned else ""
    default_out = Path("output") / "combined" / f"{setting_stem}_thpq{thpq_str}{out_suffix}"
    return inputs, default_out


def _kin_header(inputs: list[tuple[str, pd.DataFrame]]) -> tuple[dict, str]:
    """Return (first_kin_dict, thpq_joined_string) from input stems."""
    kin_list  = [_parse_kinematic_stem(stem) for stem, _ in inputs]
    first_kin = next((k for k in kin_list if k is not None), {})
    thpq_vals = [str(k["thpq"]) for k in kin_list if k is not None]
    thpq_str  = "|".join(dict.fromkeys(thpq_vals))
    if not first_kin:
        logger.warning("Could not parse kinematics from any input stem — "
                       "kinematic columns will be empty.")
    return first_kin, thpq_str


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------

_BINNED_COLS = {"variable", "hmin", "hmax", "bin_center"}
_COUNT_COLS = ("N_plus", "N_minus")


def _is_binned_csv(df: pd.DataFrame) -> bool:
    """Return True when *df* has the extra columns produced by binned-mode analysis."""
    return _BINNED_COLS.issubset(df.columns)


def _has_count_cols(df: pd.DataFrame) -> bool:
    """Return True when per-bin unweighted helicity-count columns are present."""
    return all(col in df.columns for col in _COUNT_COLS)


# ---------------------------------------------------------------------------
# Overall combine
# ---------------------------------------------------------------------------

def combine_bins(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Inverse-variance weighted average of A_phys across input DataFrames,
    grouped by (histogram, phi_center).

    Bins where A_phys_err is zero or non-finite are dropped before averaging.
    Counts sum only contributing rows; any missing count keeps the total unknown.
    """
    all_rows = pd.concat(dfs, ignore_index=True)
    all_rows["A_phys"]     = pd.to_numeric(all_rows["A_phys"],     errors="coerce")
    all_rows["A_phys_err"] = pd.to_numeric(all_rows["A_phys_err"], errors="coerce")
    has_counts = all(_has_count_cols(df) for df in dfs)
    if has_counts:
        for col in _COUNT_COLS:
            all_rows[col] = pd.to_numeric(all_rows[col], errors="coerce")

    ok = (np.isfinite(all_rows["A_phys"])
          & np.isfinite(all_rows["A_phys_err"])
          & (all_rows["A_phys_err"] > 0))
    all_rows = all_rows.loc[ok].copy()

    records = []
    for (hname, phi), grp in all_rows.groupby(["histogram", "phi_center"], sort=True):
        inv_var = 1.0 / grp["A_phys_err"].values**2
        A_comb  = float(np.sum(grp["A_phys"].values * inv_var) / np.sum(inv_var))
        dA_comb = float(1.0 / np.sqrt(np.sum(inv_var)))
        record = {"histogram": hname, "phi_center": phi,
                  "A_phys": A_comb, "A_phys_err": dA_comb}
        if has_counts:
            record["N_plus"] = float(grp["N_plus"].sum(skipna=False))
            record["N_minus"] = float(grp["N_minus"].sum(skipna=False))
        records.append(record)

    cols = ["histogram", "phi_center", "A_phys", "A_phys_err"]
    if has_counts:
        cols += list(_COUNT_COLS)
    return pd.DataFrame(records, columns=cols)


# ---------------------------------------------------------------------------
# Binned combine
# ---------------------------------------------------------------------------

def combine_bins_binned(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Inverse-variance weighted average for binned CSVs, grouped by
    (variable, hmin, hmax, histogram, phi_center).

    Counts sum only contributing rows; any missing count keeps the total unknown.
    bin_center is taken from the first contributing row per group (all rows
    with the same hmin/hmax share the same center by construction).
    """
    all_rows = pd.concat(dfs, ignore_index=True)
    for col in ("A_phys", "A_phys_err", "hmin", "hmax", "bin_center"):
        all_rows[col] = pd.to_numeric(all_rows[col], errors="coerce")
    has_counts = all(_has_count_cols(df) for df in dfs)
    if has_counts:
        for col in _COUNT_COLS:
            all_rows[col] = pd.to_numeric(all_rows[col], errors="coerce")

    ok = (np.isfinite(all_rows["A_phys"])
          & np.isfinite(all_rows["A_phys_err"])
          & (all_rows["A_phys_err"] > 0))
    all_rows = all_rows.loc[ok].copy()

    group_cols = ["variable", "hmin", "hmax", "histogram", "phi_center"]
    records = []
    for keys, grp in all_rows.groupby(group_cols, sort=True):
        variable, hmin, hmax, hname, phi = keys
        bin_center = round(float(grp["bin_center"].iloc[0]), 3)
        inv_var    = 1.0 / grp["A_phys_err"].values**2
        A_comb     = float(np.sum(grp["A_phys"].values * inv_var) / np.sum(inv_var))
        dA_comb    = float(1.0 / np.sqrt(np.sum(inv_var)))
        records.append({
            "variable":   variable,
            "hmin":       hmin,
            "hmax":       hmax,
            "bin_center": bin_center,
            "histogram":  hname,
            "phi_center": phi,
            "A_phys":     A_comb,
            "A_phys_err": dA_comb,
        })
        if has_counts:
            records[-1]["N_plus"] = float(grp["N_plus"].sum(skipna=False))
            records[-1]["N_minus"] = float(grp["N_minus"].sum(skipna=False))

    cols = ["variable", "hmin", "hmax", "bin_center",
            "histogram", "phi_center", "A_phys", "A_phys_err"]
    if has_counts:
        cols += list(_COUNT_COLS)
    return pd.DataFrame(records, columns=cols)


# ---------------------------------------------------------------------------
# Shared plot helpers
# ---------------------------------------------------------------------------

_INPUT_COLORS = [
    "steelblue", "tomato", "seagreen", "darkorange",
    "purple",    "sienna", "teal",     "crimson",
]

#_ZOOM_LO, _ZOOM_HI = -0.1, 0.1
_ZOOM_LO, _ZOOM_HI = -0.2, 0.2


def _draw_phi_panels(
    ax_full,
    ax_zoom,
    phi_c:  np.ndarray,
    A_c:    np.ndarray,
    dA_c:   np.ndarray,
    inputs_filtered: list[tuple[str, pd.DataFrame]],
    title:  str,
    n_inputs: int,
) -> None:
    """
    Fill two axes (full + zoomed) with individual inputs, combined points, and fit.

    *inputs_filtered* contains (label, sub_df) where sub_df is already sliced to
    the relevant histogram (and kinematic bin for binned mode).
    """
    phi_fit = np.linspace(-np.pi, np.pi, 300)
    amplitude, amplitude_err, chi2_ndf, n_used = fit_sinphi(phi_c, A_c, dA_c)
    hw = 0.5 * (phi_c[1] - phi_c[0]) if len(phi_c) > 1 else 0.0

    # ── Full panel: all individual inputs + combined + fit ───────────────────
    ax_full.axhline(0.0, color="gray", lw=0.8, ls="--", zorder=1)
    for (label, sub_in), color in zip(inputs_filtered, _INPUT_COLORS):
        if sub_in.empty:
            continue
        ax_full.errorbar(sub_in["phi_center"].values,
                         sub_in["A_phys"].values,
                         yerr=sub_in["A_phys_err"].values,
                         fmt="o", color=color, ms=4, lw=0.8,
                         alpha=0.45, capsize=2, zorder=2, label=label)
    ax_full.errorbar(phi_c, A_c, yerr=dA_c, xerr=hw,
                     fmt="D", color="black", ms=6, lw=1.4,
                     capsize=3, zorder=4, label="combined")
    if np.isfinite(amplitude):
        ax_full.plot(phi_fit, amplitude * np.sin(phi_fit),
                     color="tomato", lw=1.8, zorder=3, label=r"$A\sin\phi$ fit")
    ax_full.set_xlim(-np.pi * 1.05, np.pi * 1.05)
    ax_full.set_xlabel(r"$\phi_{pq}$ (rad)")
    ax_full.set_ylabel(r"$A_{LU}$")
    ax_full.set_title(title, fontsize=9)
    ax_full.legend(fontsize=7)

    # ── Zoom panel: combined points only + fit ───────────────────────────────
    ax_zoom.axhline(0.0, color="gray", lw=0.8, ls="--", zorder=1)
    ax_zoom.errorbar(phi_c, A_c, yerr=dA_c, xerr=hw,
                     fmt="D", color="black", ms=6, lw=1.4,
                     capsize=3, zorder=4, label="combined")
    if np.isfinite(amplitude):
        ax_zoom.plot(phi_fit, amplitude * np.sin(phi_fit),
                     color="tomato", lw=1.8, zorder=3, label=r"$A\sin\phi$ fit")
    ax_zoom.set_xlim(-np.pi * 1.05, np.pi * 1.05)
    ax_zoom.set_xlabel(r"$\phi_{pq}$ (rad)")
    ax_zoom.set_ylabel(r"$A_{LU}$")
    ax_zoom.set_ylim(_ZOOM_LO, _ZOOM_HI)
    ax_zoom.set_title(f"{title}  [zoom]", fontsize=9)

    # ── Fit info text ────────────────────────────────────────────────────────
    if np.isfinite(amplitude):
        info = (
            rf"$A_{{LU}}^{{\sin\phi}} = {amplitude:+.4f} \pm {amplitude_err:.4f}$"
            f"\n$\\chi^2/\\mathrm{{ndf}} = {chi2_ndf:.2f}$"
            f"\n$N_{{\\rm bins}} = {n_used}$"
            f"\n$N_{{\\rm inputs}} = {n_inputs}$"
        )
        info_short = (
            rf"$A_{{LU}}^{{\sin\phi}} = {amplitude:+.4f} \pm {amplitude_err:.4f}$"
            f"\n$\\chi^2/\\mathrm{{ndf}} = {chi2_ndf:.2f}$"
        )
    else:
        info = "fit: insufficient bins"
        info_short = info
    _text_kw = dict(va="top", ha="right", fontsize=9, family="monospace",
                    zorder=10,
                    bbox=dict(boxstyle="round", fc="0.96", ec="0.8"))
    ax_full.text(0.97, 0.97, info,       transform=ax_full.transAxes, **_text_kw)
    ax_zoom.text(0.36, 0.97, info_short, transform=ax_zoom.transAxes, **_text_kw)


# ---------------------------------------------------------------------------
# Overall plot
# ---------------------------------------------------------------------------

def _plot_histogram(ax_full, ax_zoom, hname: str,
                    combined: pd.DataFrame,
                    inputs: list[tuple[str, pd.DataFrame]]) -> None:
    sub = combined[combined["histogram"] == hname].sort_values("phi_center")
    phi_c = sub["phi_center"].values
    A_c   = sub["A_phys"].values
    dA_c  = sub["A_phys_err"].values

    inputs_filtered = [
        (label, df[df["histogram"] == hname].sort_values("phi_center"))
        for label, df in inputs
    ]
    _draw_phi_panels(ax_full, ax_zoom, phi_c, A_c, dA_c,
                     inputs_filtered, title=hname, n_inputs=len(inputs))
    ax_zoom.set_title(f"{hname}  [zoom {_ZOOM_LO}, {_ZOOM_HI}]", fontsize=9)


def write_plot(pdf_path: Path,
               combined: pd.DataFrame,
               inputs: list[tuple[str, pd.DataFrame]]) -> None:
    histograms = combined["histogram"].unique().tolist()
    n = len(histograms)
    fig, axes_grid = plt.subplots(n, 2, figsize=(11.0, 4.5 * n), sharey=False)
    if n == 1:
        axes_grid = axes_grid.reshape(1, 2)

    fig.suptitle(r"Combined beam SSA  —  $A_{LU}^{\sin\phi}$",
                 fontsize=11, fontweight="bold")

    for row, hname in enumerate(histograms):
        _plot_histogram(axes_grid[row, 0], axes_grid[row, 1],
                        hname, combined, inputs)

    fig.tight_layout()
    with PdfPages(pdf_path) as pdf:
        pdf.savefig(fig)
    plt.close(fig)
    logger.info("Plot → %s", pdf_path)


# ---------------------------------------------------------------------------
# Binned plot
# ---------------------------------------------------------------------------

def _strip_bin_suffix(hname: str, variable: str) -> str:
    """Remove the _{variable}bin{N} suffix to recover the base histogram name."""
    return re.sub(rf"_{re.escape(variable)}bin\d+$", "", hname)


def write_plot_binned(
    pdf_path: Path,
    combined: pd.DataFrame,
    inputs:   list[tuple[str, pd.DataFrame]],
) -> None:
    """
    Write the binned-mode PDF:
      - One page per kinematic bin interval: φ asymmetry (individual + combined + fit)
      - One summary page per (variable, base-histogram): A_LU^sinphi vs bin variable
    """
    with PdfPages(pdf_path) as pdf:
        for variable in sorted(combined["variable"].unique()):
            sub_var = combined[combined["variable"] == variable]

            # Ordered list of (hmin, hmax) intervals
            intervals = (sub_var[["hmin", "hmax"]]
                         .drop_duplicates()
                         .sort_values("hmin")
                         .itertuples(index=False))

            # ── Per-bin-interval φ asymmetry pages ──────────────────────────
            for interval in intervals:
                hmin, hmax = float(interval.hmin), float(interval.hmax)
                sub_bin = sub_var[
                    (sub_var["hmin"] == hmin) & (sub_var["hmax"] == hmax)
                ]
                hnames = sorted(sub_bin["histogram"].unique())
                n = len(hnames)
                if n == 0:
                    continue

                fig, axes_grid = plt.subplots(n, 2, figsize=(11.0, 4.5 * n),
                                              sharey=False)
                if n == 1:
                    axes_grid = axes_grid.reshape(1, 2)
                fig.suptitle(
                    rf"Combined beam SSA  —  {variable} $\in$ [{hmin:.3f}, {hmax:.3f})",
                    fontsize=11, fontweight="bold",
                )

                for row, hname in enumerate(hnames):
                    sub_c = sub_bin[sub_bin["histogram"] == hname].sort_values("phi_center")
                    phi_c = sub_c["phi_center"].values
                    A_c   = sub_c["A_phys"].values
                    dA_c  = sub_c["A_phys_err"].values

                    inputs_filtered = [
                        (label,
                         df[
                             (df.get("histogram", pd.Series(dtype=str)) == hname)
                             & (pd.to_numeric(df.get("hmin", pd.Series(dtype=float)), errors="coerce") == hmin)
                             & (pd.to_numeric(df.get("hmax", pd.Series(dtype=float)), errors="coerce") == hmax)
                         ].sort_values("phi_center"))
                        for label, df in inputs
                    ]
                    _draw_phi_panels(
                        axes_grid[row, 0], axes_grid[row, 1],
                        phi_c, A_c, dA_c,
                        inputs_filtered,
                        title=f"{hname}  ({variable} ∈ [{hmin:.3f}, {hmax:.3f}))",
                        n_inputs=len(inputs),
                    )

                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)

            # ── A_LU^sinphi vs bin variable summary pages ────────────────────
            # Group histogram names by their base (strip the _ptbin{N} suffix)
            all_hnames = sorted(sub_var["histogram"].unique())
            base_map: dict[str, list[str]] = {}
            for hname in all_hnames:
                base = _strip_bin_suffix(hname, variable)
                base_map.setdefault(base, []).append(hname)

            for base_hname, hnames in sorted(base_map.items()):
                # One point per histogram = one kinematic bin
                centers, lo_vals, hi_vals, amplitudes, amp_errs = [], [], [], [], []
                for hname in hnames:
                    grp = sub_var[sub_var["histogram"] == hname]
                    if grp.empty:
                        continue
                    phi = grp["phi_center"].values
                    A   = grp["A_phys"].values
                    dA  = grp["A_phys_err"].values
                    amp, amp_err, _, _ = fit_sinphi(phi, A, dA)
                    centers.append(float(grp["bin_center"].iloc[0]))
                    lo_vals.append(float(grp["hmin"].iloc[0]))
                    hi_vals.append(float(grp["hmax"].iloc[0]))
                    amplitudes.append(amp)
                    amp_errs.append(amp_err)

                if not centers:
                    continue

                cx  = np.array(centers)
                amp = np.array(amplitudes)
                err = np.array(amp_errs)
                lox = cx - np.array(lo_vals)
                hix = np.array(hi_vals) - cx
                valid = np.isfinite(amp) & np.isfinite(err)

                fig, ax = plt.subplots(figsize=(8, 5))
                ax.axhline(0.0, color="gray", lw=0.9, ls="--", zorder=1)
                if valid.any():
                    ax.errorbar(cx[valid], amp[valid],
                                yerr=err[valid],
                                xerr=[lox[valid], hix[valid]],
                                fmt="o", color="steelblue", ms=6, lw=1.2,
                                capsize=4, zorder=3,
                                label=r"$A_{LU}^{\sin\phi}$ (combined)")
                ax.set_xlabel(variable)
                ax.set_ylabel(r"$A_{LU}^{\sin\phi}$")
                ax.set_title(
                    rf"Combined $A_{{LU}}^{{\sin\phi}}$ vs {variable}  —  {base_hname}",
                    fontsize=11, fontweight="bold",
                )
                ax.legend(fontsize=8)
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)

    logger.info("Plot → %s", pdf_path)


# ---------------------------------------------------------------------------
# Kinematic summary CSVs
# ---------------------------------------------------------------------------

def _write_combined_summary(
    path:    Path,
    combined: pd.DataFrame,
    inputs:  list[tuple[str, pd.DataFrame]],
) -> None:
    """One row per histogram with kinematic metadata and combined fit values."""
    first_kin, thpq_str = _kin_header(inputs)

    rows = []
    for hname, grp in combined.groupby("histogram", sort=True):
        phi = grp["phi_center"].values
        A   = grp["A_phys"].values
        dA  = grp["A_phys_err"].values
        amp, amp_err, chi2_ndf, n_used = fit_sinphi(phi, A, dA)
        rows.append({
            "target":    first_kin.get("target"),
            "particle":  first_kin.get("particle"),
            "ebeam":     first_kin.get("ebeam"),
            "x":         first_kin.get("x"),
            "q2":        first_kin.get("q2"),
            "z":         first_kin.get("z"),
            "process":   first_kin.get("process", "sidis"),
            "thpq":      thpq_str,
            "histogram": hname,
            "asym":      amp      if np.isfinite(amp)      else np.nan,
            "asym_err":  amp_err  if np.isfinite(amp_err)  else np.nan,
            "chi2_ndf":  chi2_ndf if np.isfinite(chi2_ndf) else np.nan,
            "n_bins":    n_used,
            "N_plus":    grp["N_plus"].sum(skipna=False)  if "N_plus"  in grp.columns else np.nan,
            "N_minus":   grp["N_minus"].sum(skipna=False) if "N_minus" in grp.columns else np.nan,
        })

    cols = ["target", "particle", "ebeam", "x", "q2", "z", "process", "thpq",
            "histogram", "asym", "asym_err", "chi2_ndf", "n_bins",
            "N_plus", "N_minus"]
    pd.DataFrame(rows, columns=cols).to_csv(path, index=False)
    logger.info("Kinematic summary → %s", path)


def _write_combined_binned_summary(
    path:    Path,
    combined: pd.DataFrame,
    inputs:  list[tuple[str, pd.DataFrame]],
) -> None:
    """One row per (variable, hmin, hmax, histogram) with combined fit values."""
    first_kin, thpq_str = _kin_header(inputs)

    rows = []
    group_cols = ["variable", "hmin", "hmax", "histogram"]
    for (variable, hmin, hmax, hname), grp in combined.groupby(group_cols, sort=True):
        phi = grp["phi_center"].values
        A   = grp["A_phys"].values
        dA  = grp["A_phys_err"].values
        amp, amp_err, chi2_ndf, n_used = fit_sinphi(phi, A, dA)
        bin_center = round(float(grp["bin_center"].iloc[0]), 3)
        rows.append({
            "target":     first_kin.get("target"),
            "particle":   first_kin.get("particle"),
            "ebeam":      first_kin.get("ebeam"),
            "x":          first_kin.get("x"),
            "q2":         first_kin.get("q2"),
            "z":          first_kin.get("z"),
            "process":    first_kin.get("process", "sidis"),
            "thpq":       thpq_str,
            "variable":   variable,
            "hmin":       hmin,
            "hmax":       hmax,
            "bin_center": bin_center,
            "histogram":  hname,
            "asym":       amp      if np.isfinite(amp)      else np.nan,
            "asym_err":   amp_err  if np.isfinite(amp_err)  else np.nan,
            "chi2_ndf":   chi2_ndf if np.isfinite(chi2_ndf) else np.nan,
            "n_bins":     n_used,
            "N_plus":     grp["N_plus"].sum(skipna=False)  if "N_plus"  in grp.columns else np.nan,
            "N_minus":    grp["N_minus"].sum(skipna=False) if "N_minus" in grp.columns else np.nan,
        })

    cols = ["target", "particle", "ebeam", "x", "q2", "z", "process", "thpq",
            "variable", "hmin", "hmax", "bin_center", "histogram",
            "asym", "asym_err", "chi2_ndf", "n_bins", "N_plus", "N_minus"]
    pd.DataFrame(rows, columns=cols).to_csv(path, index=False)
    logger.info("Kinematic summary → %s", path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combine asymmetry CSVs via inverse-variance weighting"
    )
    parser.add_argument("csvfiles", nargs="*", type=Path,
                        help="Input asymmetry CSV files (alternative to --base_config)")
    parser.add_argument("--output", "-o", type=Path, default=None,
                        help="Output path stem (auto-generated when using --base_config)")
    parser.add_argument("--label", "-l", type=str, default=None,
                        help="Override label used in plot title")
    parser.add_argument("--base_config", type=Path, default=None,
                        help="Base YAML config; use with --thpq to locate CSVs automatically")
    parser.add_argument("--thpq", nargs="+", type=float, default=None,
                        help="thpq values to combine (required with --base_config)")
    parser.add_argument("--is_binned", action="store_true",
                        help="Use _binned.csv files (used with --base_config)")
    args = parser.parse_args()

    # ── Input resolution ─────────────────────────────────────────────────────
    if args.base_config and args.csvfiles:
        parser.error("Provide either CSV files or --base_config, not both.")
    if not args.base_config and not args.csvfiles:
        parser.error("Provide CSV files or use --base_config with --thpq.")
    if args.base_config and not args.thpq:
        parser.error("--thpq is required when using --base_config.")

    if args.base_config:
        try:
            inputs, auto_out = _build_inputs_from_config(
                args.base_config, args.thpq, args.is_binned
            )
        except (FileNotFoundError, ValueError) as exc:
            logger.error("%s", exc)
            sys.exit(1)
        binned_mode = args.is_binned
        out_stem    = args.output or auto_out
    else:
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

        binned_flags = [_is_binned_csv(df) for _, df in inputs]
        if any(binned_flags) and not all(binned_flags):
            logger.error(
                "Mixed input formats: some CSVs are binned (have 'variable' column) "
                "and some are not.  All inputs must be the same format."
            )
            sys.exit(1)
        binned_mode = all(binned_flags)
        out_stem    = args.output or Path("combined_asymmetry")

    if len(inputs) < 2:
        logger.warning("Only one input CSV — combined result equals input.")
    logger.info("Mode: %s", "binned" if binned_mode else "overall")

    out_stem.parent.mkdir(parents=True, exist_ok=True)

    # ── Combine ──────────────────────────────────────────────────────────────
    if binned_mode:
        combined = combine_bins_binned([df for _, df in inputs])
        logger.info(
            "Combined: %d rows across %d histogram(s) in %d kinematic bin(s)",
            len(combined),
            combined["histogram"].nunique(),
            combined[["variable", "hmin", "hmax"]].drop_duplicates().shape[0],
        )
    else:
        combined = combine_bins([df for _, df in inputs])
        logger.info("Combined: %d rows across %d histogram(s)",
                    len(combined), combined["histogram"].nunique())

    # ── Log fit results ──────────────────────────────────────────────────────
    if binned_mode:
        for (variable, hmin, hmax, hname), grp in combined.groupby(
            ["variable", "hmin", "hmax", "histogram"], sort=True
        ):
            phi = grp["phi_center"].values
            A   = grp["A_phys"].values
            dA  = grp["A_phys_err"].values
            amp, amp_err, chi2_ndf, n_used = fit_sinphi(phi, A, dA)
            logger.info(
                "[%s  %s∈[%.3f,%.3f)]  A_LU^sinφ = %+.4f ± %.4f  χ²/ndf = %.2f  (%d bins)",
                hname, variable, hmin, hmax,
                amp if np.isfinite(amp) else float("nan"),
                amp_err if np.isfinite(amp_err) else float("nan"),
                chi2_ndf if np.isfinite(chi2_ndf) else float("nan"),
                n_used,
            )
    else:
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

    # ── Write CSV ────────────────────────────────────────────────────────────
    csv_path = out_stem.with_suffix(".csv")
    combined.to_csv(csv_path, index=False)
    logger.info("CSV  → %s", csv_path)

    # ── Write plot ───────────────────────────────────────────────────────────
    pdf_path = out_stem.with_suffix(".pdf")
    if binned_mode:
        write_plot_binned(pdf_path, combined, inputs)
    else:
        write_plot(pdf_path, combined, inputs)

    # ── Write kinematic summary ───────────────────────────────────────────────
    summary_path = out_stem.with_name(out_stem.name + "_summary.csv")
    if binned_mode:
        _write_combined_binned_summary(summary_path, combined, inputs)
    else:
        _write_combined_summary(summary_path, combined, inputs)


if __name__ == "__main__":
    main()


# ── Direct mode (explicit CSV paths) ──────────────────────────────────────
# python combine_asymmetry.py \
#   output/C_pim_e10p7_x0p25_q23p3_z0p5_thpqm0p8/C_pim_e10p7_x0p25_q23p3_z0p5_thpqm0p8.csv \
#   output/C_pim_e10p7_x0p25_q23p3_z0p5_thpq2p0/C_pim_e10p7_x0p25_q23p3_z0p5_thpq2p0.csv
#
# python combine_asymmetry.py \
#   output/C_pim_e10p7_x0p25_q23p3_z0p5_thpqm0p8/C_pim_e10p7_x0p25_q23p3_z0p5_thpqm0p8_binned.csv \
#   output/C_pim_e10p7_x0p25_q23p3_z0p5_thpq2p0/C_pim_e10p7_x0p25_q23p3_z0p5_thpq2p0_binned.csv
#
# ── Config mode (locate CSVs automatically from base YAML) ─────────────────
# python combine_asymmetry.py \
#   --base_config config/C_pim_e10p7_x0p25_q23p3_z0p5_base.yaml \
#   --thpq -0.8 2.0
#
# python combine_asymmetry.py \
#   --base_config config/C_pim_e10p7_x0p25_q23p3_z0p5_base.yaml \
#   --thpq -0.8 2.0 --is_binned

#!/usr/bin/env python3
"""
RSIDIS SSA pipeline automation.

Reads settings CSV(s), generates YAML configs, runs analysis.py per setting,
and combines asymmetries across thpq values within each family.

Must be run from the project root (same directory as analysis.py).

Usage examples
--------------
Full pipeline for all settings:
    python run_pipeline.py --settings data/settings/rpr1_pip_settings.csv \\
                                      data/settings/rpr1_pim_settings.csv

Just one target and z:
    python run_pipeline.py --settings ... --target LH2 --z 0.5

Generate configs only:
    python run_pipeline.py --settings ... --steps generate

Analyze a single thpq, then combine all:
    python run_pipeline.py --settings ... --target C --z 0.5 --thpq -0.8 --steps analyze
    python run_pipeline.py --settings ... --target C --z 0.5 --steps combine

Dry run (print every command, execute nothing):
    python run_pipeline.py --settings ... --dry-run
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------

CONFIG_DIR   = Path("config")
OUTPUT_DIR   = Path("output")
CRYO_TARGETS = {"LH2", "LD2"}
_PYTHON      = sys.executable   # use the same interpreter that launched this script

# ---------------------------------------------------------------------------
# Float encoding (matches existing config filename convention)
# ---------------------------------------------------------------------------

def _fenc(v: float, decimals: int) -> str:
    """
    Encode a float to filename-safe form: '.' → 'p', '-' → 'm'.

    Examples: 10.6716 (1dp) → '10p7',  -0.8 (1dp) → 'm0p8',
              0.25 (2dp) → '0p25',      0.9 (2dp) → '0p9'
    """
    s = f"{round(v, decimals):.{decimals}f}".rstrip("0").rstrip(".")
    if "." not in s:
        s += ".0"
    return s.replace("-", "m").replace(".", "p")


def _setting_stem(target: str, ebeam: float, x: float, Q2: float,
                  z: float, thpq: float, run_type: str) -> str:
    particle = "pip" if run_type == "PI+SIDIS" else "pim"
    return (f"{target}_{particle}"
            f"_e{_fenc(ebeam,1)}_x{_fenc(x,2)}_q2{_fenc(Q2,1)}"
            f"_z{_fenc(z,2)}_thpq{_fenc(thpq,1)}")


def _base_stem(target: str, ebeam: float, x: float, Q2: float,
               z: float, run_type: str) -> str:
    particle = "pip" if run_type == "PI+SIDIS" else "pim"
    return (f"{target}_{particle}"
            f"_e{_fenc(ebeam,1)}_x{_fenc(x,2)}_q2{_fenc(Q2,1)}"
            f"_z{_fenc(z,2)}_base")


# ---------------------------------------------------------------------------
# Settings loading and filtering
# ---------------------------------------------------------------------------

def load_settings(*csv_paths: Path) -> pd.DataFrame:
    return pd.concat([pd.read_csv(p) for p in csv_paths], ignore_index=True)


def filter_settings(
    df:        pd.DataFrame,
    targets:   list[str]   | None = None,
    z_vals:    list[float] | None = None,
    thpq_vals: list[float] | None = None,
) -> pd.DataFrame:
    if targets:
        df = df[df["target"].isin(targets)]
    if z_vals:
        df = df[df["z"].isin(z_vals)]
    if thpq_vals:
        df = df[df["thpq"].isin(thpq_vals)]
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# YAML templates
# ---------------------------------------------------------------------------

# Standard histogram block — identical across all generated base configs.
# Cuts and bin_in edges can be tuned in specific configs afterwards.
_HISTOGRAMS_YAML = """\
histograms:
  # HMS acceptance
  - name:   hsp
    branch: H_gtr_p
    bins:   50
    xmin:   2.0
    xmax:   5.0
    xlabel: 'HMS p (GeV/c)'

  - name:   hsdelta
    branch: H_gtr_dp
    bins:   16
    xmin:   -8.0
    xmax:    8.0
    xlabel: 'HMS $\\delta$ (%)'

  - name:   hsxptar
    branch: H_gtr_th
    bins:   50
    xmin:   -0.1
    xmax:    0.1
    xlabel: "HMS xptar (rad)"

  - name:   hsyptar
    branch: H_gtr_ph
    bins:   50
    xmin:   -0.06
    xmax:    0.06
    xlabel: "HMS yptar (rad)"

  - name:   hsytar
    branch: H_gtr_y
    bins:   50
    xmin:   -5.0
    xmax:    5.0
    xlabel: "HMS ytar (cm)"

  # SHMS acceptance
  - name:   psp
    branch: P_gtr_p
    bins:   50
    xmin:   2.0
    xmax:   5.0
    xlabel: 'SHMS p (GeV/c)'

  - name:   psdelta
    branch: P_gtr_dp
    bins:   30
    xmin:   -10.0
    xmax:    20.0
    xlabel: 'SHMS $\\delta$ (%)'

  - name:   psxptar
    branch: P_gtr_th
    bins:   50
    xmin:   -0.1
    xmax:    0.1
    xlabel: "SHMS xptar (rad)"

  - name:   psyptar
    branch: P_gtr_ph
    bins:   50
    xmin:   -0.06
    xmax:    0.06
    xlabel: "SHMS yptar (rad)"

  - name:   psytar
    branch: P_gtr_y
    bins:   50
    xmin:   -5.0
    xmax:    5.0
    xlabel: "SHMS ytar (cm)"

  # Physics quantities
  - name:   W
    branch: H_kin_primary_W
    bins:   50
    xmin:    2.0
    xmax:    4.0
    xlabel: "W (GeV)"

  - name:   Q2
    branch: H_kin_primary_Q2
    bins:   50
    xmin:    2.0
    xmax:    6.0
    xlabel: '$Q^{2}$ (GeV$^{2}$)'

  - name:   xbj
    branch: H_kin_primary_x_bj
    bins:   50
    xmin:    0.1
    xmax:    0.4
    xlabel: '$x_{Bj}$'

  - name:   zhad
    branch: z
    bins:   45
    xmin:    0.1
    xmax:    1.0
    xlabel: '$z_{had}$'

  - name:   Pt
    branch: pt
    bins:   50
    xmin:    0.0
    xmax:    0.5
    xlabel: '$P_T$ (GeV/c)'

  # Azimuthal angle (full + helicity-split)
  - name:   phipq
    branch: P_kin_secondary_ph_xq
    bins:   16
    xmin:   -3.14159265
    xmax:    3.14159265
    xlabel: '$\\phi_{pq}$ (rad)'
    bin_in:
      branch: pt
      edges: [0.0, 0.1, 0.2, 0.3]

  - name:   phipq_hplus
    branch: P_kin_secondary_ph_xq
    bins:   16
    xmin:   -3.14159265
    xmax:    3.14159265
    xlabel: '$\\phi_{pq}$ (rad)  [$h^{+}$]'
    helicity_cut: positive

  - name:   phipq_hminus
    branch: P_kin_secondary_ph_xq
    bins:   16
    xmin:   -3.14159265
    xmax:    3.14159265
    xlabel: '$\\phi_{pq}$ (rad)  [$h^{-}$]'
    helicity_cut: negative

  # 2D histogram of Pt vs phi for acceptance studies
  - name:     pt_vs_phi
    branch:   ptx
    bins:     50
    xmin:     -0.5
    xmax:     0.5
    xlabel:   '$P_T\\cos(\\phi_{pq})$ (GeV/c)'
    branch_y: pty
    bins_y:   50
    ymin:     -0.5
    ymax:     0.5
    ylabel:   '$P_T\\sin(\\phi_{pq})$ (GeV/c)'
"""


def _make_base_yaml(target: str, ebeam: float, x: float,
                    Q2: float, z: float, run_type: str) -> str:
    base_name = _base_stem(target, ebeam, x, Q2, z, run_type)
    cryo      = target in CRYO_TARGETS
    return f"""\
# ============================================================
# RSIDIS SSA base configuration — {target}, {run_type}
# Kinematic point: e={ebeam}, x={x}, Q2={Q2}, z={z}
#
# THIS FILE IS A BASE — do not pass it directly to analysis.py.
# Use a daughter YAML that specifies _base and overrides thpq:
#
#   _base: {base_name}.yaml
#   setting:
#     thpq: <value>
# ============================================================

setting:
  ebeam:    {ebeam}
  x:        {x}
  Q2:       {Q2}
  z:        {z}
  thpq:     0.0        # placeholder — must be overridden in daughter
  run_type: {run_type}

target: {target}
run_period: period_1
do_eplus_subtraction: true
do_dummy_subtraction: {"true" if cryo else "false"}

runs_include: null
runs_exclude: []

rootfiles:
  directory:  ../data/rootfiles_pass0p1
  pattern:    skimmed_coin_replay_production_{{run}}_-1.root
  treename:   T

runlist:
  csv: ../data/rsidis_bigtable_pass0p1.csv

normalization:
  charge_column:  BCM2_Q
  weight_scheme:  charge_only

cuts:
  # HMS electron PID
  hsdelta_lo:    -8.0    # %
  hsdelta_hi:     8.0    # %
  hcer_npe_min:   1.0
  hsshsum_min:    0.7

  # SHMS pion PID
  psdelta_lo:   -10.0    # %
  psdelta_hi:    20.0    # %
  paero_npe_min:  2.0
  phgc_npe_min:   1.0
  phgc_p_threshold: 2.9   # GeV/c — require HGC only above this momentum
  psshsum_max:    0.8

  # Coincidence time — real peak
  ctime_real_center:          auto
  ctime_real_nsigma:          null
  ctime_real_window_fallback: 2.0

  # Coincidence time — random sideband
  ctime_random_n_skip:     1
  ctime_random_n_peaks_lo: 6
  ctime_random_n_peaks_hi: 0

{_HISTOGRAMS_YAML}"""


def _make_override_yaml(base_filename: str, thpq: float) -> str:
    return f"""\
# thpq = {thpq}
_base: {base_filename}

setting:
  thpq: {thpq}
"""


# ---------------------------------------------------------------------------
# Pipeline helpers
# ---------------------------------------------------------------------------

_FAMILY_COLS = ["target", "ebeam", "x", "Q2", "z", "run_type"]


def _run(cmd: list[str], dry_run: bool) -> bool:
    """Print command and optionally execute it. Returns True on success."""
    display = " ".join(str(c) for c in cmd)
    if dry_run:
        print(f"  [DRY RUN] {display}")
        return True
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"  ERROR (exit {result.returncode}): {display}", file=sys.stderr)
        return False
    return True


def _write_file(path: Path, content: str, force: bool, dry_run: bool) -> None:
    """Write content to path, respecting force/dry_run/skip-existing logic."""
    if path.exists() and not force:
        print(f"  exists  {path.name}")
        return
    if dry_run:
        verb = "overwrite" if path.exists() else "write"
        print(f"  [DRY RUN] {verb} {path}")
        return
    path.write_text(content)
    print(f"  wrote   {path.name}")


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def step_generate(df: pd.DataFrame, force: bool, dry_run: bool) -> None:
    CONFIG_DIR.mkdir(exist_ok=True)
    for keys, group in df.groupby(_FAMILY_COLS):
        target, ebeam, x, Q2, z, run_type = keys
        thpq_list = sorted(group["thpq"].tolist())

        base_path = CONFIG_DIR / f"{_base_stem(target, ebeam, x, Q2, z, run_type)}.yaml"
        _write_file(base_path, _make_base_yaml(target, ebeam, x, Q2, z, run_type),
                    force, dry_run)

        for thpq in thpq_list:
            override_path = CONFIG_DIR / f"{_setting_stem(target, ebeam, x, Q2, z, thpq, run_type)}.yaml"
            _write_file(override_path, _make_override_yaml(base_path.name, thpq),
                        force, dry_run)


def step_analyze(df: pd.DataFrame, force: bool, dry_run: bool) -> None:
    for _, row in df.iterrows():
        stem        = _setting_stem(row.target, row.ebeam, row.x, row.Q2, row.z,
                                    row.thpq, row.run_type)
        config_path = CONFIG_DIR / f"{stem}.yaml"
        output_csv  = OUTPUT_DIR / stem / f"{stem}.csv"

        if not config_path.exists():
            print(f"  SKIP (no config)  {stem}")
            continue
        if output_csv.exists() and not force:
            print(f"  SKIP (done)       {stem}")
            continue

        print(f"  analyze           {stem}")
        _run([_PYTHON, "analysis.py", str(config_path)], dry_run)


def step_combine(df: pd.DataFrame, binned: bool, force: bool, dry_run: bool) -> None:
    for keys, group in df.groupby(_FAMILY_COLS):
        target, ebeam, x, Q2, z, run_type = keys
        thpq_list = sorted(group["thpq"].tolist())
        particle  = "pip" if run_type == "PI+SIDIS" else "pim"

        if len(thpq_list) < 2:
            stem = _base_stem(target, ebeam, x, Q2, z, run_type)
            print(f"  SKIP (1 thpq)     {stem}")
            continue

        base_path = CONFIG_DIR / f"{_base_stem(target, ebeam, x, Q2, z, run_type)}.yaml"
        if not base_path.exists():
            print(f"  SKIP (no base)    {base_path.name}")
            continue

        thpq_enc = "AND".join(_fenc(t, 1) for t in thpq_list)
        out_base = (f"{target}_{particle}"
                    f"_e{_fenc(ebeam,1)}_x{_fenc(x,2)}_q2{_fenc(Q2,1)}"
                    f"_z{_fenc(z,2)}_thpq{thpq_enc}")

        for is_binned in ([False, True] if binned else [False]):
            suffix  = "_binned" if is_binned else ""
            out_csv = OUTPUT_DIR / "combined" / f"{out_base}{suffix}.csv"

            if out_csv.exists() and not force:
                print(f"  SKIP (done)       {out_base}{suffix}")
                continue

            cmd = ([_PYTHON, "combine_asymmetry.py",
                    "--base_config", str(base_path),
                    "--thpq"] + [str(t) for t in thpq_list]
                   + (["--is_binned"] if is_binned else []))
            print(f"  combine           {out_base}{suffix}")
            _run(cmd, dry_run)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="RSIDIS SSA pipeline: generate configs, run analysis, combine results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--settings", nargs="+", required=True, type=Path, metavar="CSV",
        help="Settings CSV file(s)",
    )
    parser.add_argument(
        "--steps", nargs="+", default=["generate", "analyze", "combine"],
        choices=["generate", "analyze", "combine"], metavar="STEP",
        help="Steps to run (default: all three)",
    )
    parser.add_argument(
        "--target", nargs="+", default=None, metavar="TARGET",
        help="Filter to target(s): C Cu LD2 LH2 Al",
    )
    parser.add_argument(
        "--z", nargs="+", type=float, default=None, metavar="Z",
        help="Filter to z value(s): 0.36 0.5 0.67 0.9",
    )
    parser.add_argument(
        "--thpq", nargs="+", type=float, default=None, metavar="THPQ",
        help="Filter to thpq value(s): -0.8 2.0 5.2",
    )
    parser.add_argument(
        "--binned", action="store_true",
        help="Also combine binned (_binned.csv) results in the combine step",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print every action without executing anything",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-generate / re-run even when output already exists",
    )
    args = parser.parse_args()

    df = load_settings(*args.settings)
    df = filter_settings(df, args.target, args.z, args.thpq)

    if df.empty:
        print("No settings match the given filters — nothing to do.")
        sys.exit(0)

    n_families = df.groupby(_FAMILY_COLS).ngroups
    print(f"Settings: {len(df)} row(s) across "
          f"{n_families} famil{'y' if n_families == 1 else 'ies'}")
    if args.dry_run:
        print("(dry run — no files will be written or commands executed)")

    if "generate" in args.steps:
        print("\n── generate configs ─────────────────────────────────────────")
        step_generate(df, args.force, args.dry_run)

    if "analyze" in args.steps:
        print("\n── analyze ──────────────────────────────────────────────────")
        step_analyze(df, args.force, args.dry_run)

    if "combine" in args.steps:
        print("\n── combine ──────────────────────────────────────────────────")
        step_combine(df, args.binned, args.force, args.dry_run)


if __name__ == "__main__":
    main()

"""
Runlist loading and setting selection for the RSIDIS SSA analysis.

A kinematic setting is the unique combination of:
    (ebeam, x, Q2, z, thpq, run_type)

Within a setting and target:
    hms_p < 0  →  e⁻ signal runs
    hms_p > 0  →  e⁺ charge-symmetric background runs
    target == 'Dummy'  →  dummy target (cryo subtraction only)

The CSV uses -999 as a sentinel for missing/invalid values; these are
replaced with NaN on load so that downstream code can use pandas NA
semantics uniformly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SENTINEL: float = -999.0
FLOAT_TOL: float = 0.02   # tolerance when matching floating-point setting values

SIDIS_RUN_TYPES: frozenset[str] = frozenset({"PI+SIDIS", "PI-SIDIS"})

CRYO_TARGETS: frozenset[str] = frozenset({"LH2", "LD2"})

# Columns that must be present in the CSV.
REQUIRED_COLUMNS: frozenset[str] = frozenset({
    "run", "ebeam", "target", "hms_p", "hms_th",
    "shms_p", "shms_th", "run_type",
    "x", "Q2", "z", "thpq",
    "BCM2_Q",
    "h_esing_Eff", "p_hadron_Eff",
    "ps5", "ps6",
    "comp_livetime",
    "boil_corr",
    "ctmean", "ctsigma",
    "IHWP",
})


# ---------------------------------------------------------------------------
# Setting dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Setting:
    """
    Uniquely identifies a kinematic setting.

    A setting is matched against a runlist row using float tolerance on
    the five kinematic quantities; run_type is matched exactly.
    """
    ebeam:    float
    x:        float
    Q2:       float
    z:        float
    thpq:     float
    run_type: str

    def __str__(self) -> str:
        return (
            f"{self.run_type} | ebeam={self.ebeam:.4f}  "
            f"x={self.x:.3f}  Q2={self.Q2:.2f}  "
            f"z={self.z:.2f}  thpq={self.thpq:.2f}"
        )


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_runlist(csv_path: str | Path) -> pd.DataFrame:
    """
    Load the run table CSV and return a typed DataFrame.

    Sentinel values (-999) are replaced with NaN for all numeric columns.
    The 'run' column is cast to int and 'IHWP' is kept as a string.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Runlist not found: {csv_path}")

    df = pd.read_csv(csv_path, dtype={"IHWP": str})

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Runlist CSV is missing required columns: {sorted(missing)}"
        )

    # Replace sentinels in every numeric column with NaN
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].replace(SENTINEL, np.nan)

    # Ensure run is integer (may have been read as float if mixed)
    df["run"] = df["run"].astype(int)

    return df


# ---------------------------------------------------------------------------
# Setting discovery
# ---------------------------------------------------------------------------

def get_settings(
    df: pd.DataFrame,
    run_types: frozenset[str] = SIDIS_RUN_TYPES,
) -> list[Setting]:
    """
    Return the sorted list of unique kinematic settings found in *df*.

    Only rows whose run_type is in *run_types* and whose kinematic
    columns (x, Q2, z, thpq) are non-NaN contribute to the list.
    """
    sidis = df[df["run_type"].isin(run_types)]
    valid = sidis.dropna(subset=["ebeam", "x", "Q2", "z", "thpq"])

    setting_cols = ["ebeam", "x", "Q2", "z", "thpq", "run_type"]
    unique_rows = valid[setting_cols].drop_duplicates()

    settings = [
        Setting(
            ebeam=row["ebeam"],
            x=row["x"],
            Q2=row["Q2"],
            z=row["z"],
            thpq=row["thpq"],
            run_type=row["run_type"],
        )
        for _, row in unique_rows.iterrows()
    ]

    return sorted(settings, key=lambda s: (s.run_type, s.ebeam, s.z, s.thpq))


# ---------------------------------------------------------------------------
# Run selection
# ---------------------------------------------------------------------------

def _setting_mask(df: pd.DataFrame, setting: Setting, tol: float) -> pd.Series:
    """
    Return a boolean Series selecting rows that match *setting* within *tol*.

    Rows with NaN in any kinematic column are excluded.
    """
    kin_valid = (
        df["x"].notna()
        & df["Q2"].notna()
        & df["z"].notna()
        & df["thpq"].notna()
    )
    return (
        kin_valid
        & (df["run_type"] == setting.run_type)
        & (np.abs(df["ebeam"] - setting.ebeam) < tol)
        & (np.abs(df["x"]    - setting.x)     < tol)
        & (np.abs(df["Q2"]   - setting.Q2)    < tol)
        & (np.abs(df["z"]    - setting.z)     < tol)
        & (np.abs(df["thpq"] - setting.thpq)  < tol)
    )


def select_runs(
    df: pd.DataFrame,
    setting: Setting,
    target: str,
    tol: float = FLOAT_TOL,
) -> pd.DataFrame:
    """
    Return all rows in *df* that match *setting* and *target*.

    The result may contain both e⁻ and e⁺ runs.  Use
    :func:`get_signal_runs` and :func:`get_eplus_runs` to separate them.
    """
    mask = _setting_mask(df, setting, tol) & (df["target"] == target)
    return df[mask].reset_index(drop=True)


def get_signal_runs(df_setting: pd.DataFrame) -> pd.DataFrame:
    """Return the e⁻ (signal) subset: rows where hms_p < 0."""
    return df_setting[df_setting["hms_p"] < 0].reset_index(drop=True)


def get_eplus_runs(df_setting: pd.DataFrame) -> pd.DataFrame:
    """Return the e⁺ (charge-symmetric background) subset: rows where hms_p > 0."""
    return df_setting[df_setting["hms_p"] > 0].reset_index(drop=True)


def get_dummy_runs(
    df: pd.DataFrame,
    setting: Setting,
    tol: float = FLOAT_TOL,
) -> pd.DataFrame:
    """
    Return Dummy-target rows matching *setting*.

    Only meaningful for cryo targets (LH2, LD2).  The result may contain
    both e⁻ and e⁺ dummy runs; use :func:`get_signal_runs` /
    :func:`get_eplus_runs` on the return value to separate them.
    """
    return select_runs(df, setting, target="Dummy", tol=tol)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def is_cryo(target: str) -> bool:
    """Return True when *target* requires a boiling correction (LH2/LD2)."""
    return target in CRYO_TARGETS


def print_settings_table(settings: list[Setting]) -> None:
    """Print a human-readable table of settings — useful for interactive use."""
    header = f"{'#':>3}  {'run_type':<10}  {'ebeam':>8}  {'x':>6}  {'Q2':>6}  {'z':>5}  {'thpq':>7}"
    print(header)
    print("-" * len(header))
    for i, s in enumerate(settings):
        print(
            f"{i:>3}  {s.run_type:<10}  {s.ebeam:>8.4f}  "
            f"{s.x:>6.3f}  {s.Q2:>6.2f}  {s.z:>5.2f}  {s.thpq:>7.2f}"
        )

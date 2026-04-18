"""
Shared pytest fixtures for the rsidis_ssa test suite.
"""

from pathlib import Path
import pytest
from rsidis_ssa.runlist import load_runlist, Setting

# Path to the real CSV — tests run against the actual data
CSV_PATH = Path(__file__).parent.parent / "data" / "rsidis_bigtable_pass0p1.csv"


@pytest.fixture(scope="session")
def runlist_df():
    """Loaded runlist DataFrame — shared across the entire test session."""
    return load_runlist(CSV_PATH)


# ---------------------------------------------------------------------------
# Known settings derived from the real CSV for use in multiple test files.
# These are verified by hand against the CSV before being used as fixtures.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def setting_C_z05_thpq2():
    """
    Carbon, PI-SIDIS, z=0.5, thpq=2.0 at ebeam=8.5831.
    Expected: 14 e⁻ runs (hms_p=-1.531) + 2 e⁺ runs (hms_p=+1.531).
    Run numbers verified from CSV:
        e-: 24120, 24121, 24122, 24123, 24125, 24138-24141, 24162-24166
        e+: 24589, 24590
    """
    return Setting(ebeam=8.5831, x=0.25, Q2=3.3, z=0.5, thpq=2.0, run_type="PI-SIDIS")


@pytest.fixture(scope="session")
def setting_LH2_z05_thpq5p2():
    """
    LH2, PI-SIDIS, z=0.5, thpq=5.2 at ebeam=8.5831.
    LH2 is a cryo target — dummy runs are expected.
    """
    return Setting(ebeam=8.5831, x=0.25, Q2=3.3, z=0.5, thpq=5.2, run_type="PI-SIDIS")

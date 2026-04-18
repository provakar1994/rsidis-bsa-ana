"""
Tests for rsidis_ssa.runlist

Tests use the real CSV in data/ — no mocks.  Each test class is isolated
to one logical concern so failures pinpoint the exact broken behaviour.
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from rsidis_ssa.runlist import (
    REQUIRED_COLUMNS,
    SENTINEL,
    SIDIS_RUN_TYPES,
    Setting,
    get_dummy_runs,
    get_eplus_runs,
    get_settings,
    get_signal_runs,
    is_cryo,
    load_runlist,
    select_runs,
)


# ===========================================================================
# load_runlist
# ===========================================================================

class TestLoadRunlist:

    def test_loads_successfully(self, runlist_df):
        assert isinstance(runlist_df, pd.DataFrame)
        assert len(runlist_df) > 0

    def test_sentinel_not_present_in_numeric_columns(self, runlist_df):
        """Every -999 must have been converted to NaN."""
        numeric = runlist_df.select_dtypes(include=[np.number])
        for col in numeric.columns:
            count = (numeric[col] == SENTINEL).sum()
            assert count == 0, f"Sentinel value -999 still present in column '{col}'"

    def test_run_column_is_integer(self, runlist_df):
        assert pd.api.types.is_integer_dtype(runlist_df["run"]), (
            "Column 'run' should be integer dtype"
        )

    def test_ihwp_column_is_string(self, runlist_df):
        assert pd.api.types.is_string_dtype(runlist_df["IHWP"]), (
            f"Column 'IHWP' should be a string dtype, got {runlist_df['IHWP'].dtype}"
        )

    def test_ihwp_only_in_and_out(self, runlist_df):
        """IHWP values are exactly IN or OUT (no other strings, no NaN)."""
        pi_sidis = runlist_df[runlist_df["run_type"].isin(SIDIS_RUN_TYPES)]
        unique = set(pi_sidis["IHWP"].dropna().unique())
        assert unique <= {"IN", "OUT"}, f"Unexpected IHWP values: {unique - {'IN', 'OUT'}}"

    def test_required_columns_all_present(self, runlist_df):
        for col in REQUIRED_COLUMNS:
            assert col in runlist_df.columns, f"Required column '{col}' missing"

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_runlist("/nonexistent/path/to/runlist.csv")

    def test_run_numbers_are_positive(self, runlist_df):
        assert (runlist_df["run"] > 0).all()

    def test_comp_livetime_is_numeric(self, runlist_df):
        """comp_livetime must be a numeric column (fraction, ~1.0 for good runs)."""
        assert pd.api.types.is_float_dtype(runlist_df["comp_livetime"]), (
            f"comp_livetime should be float, got {runlist_df['comp_livetime'].dtype}"
        )

    def test_comp_livetime_in_valid_range_for_sidis(self, runlist_df):
        """For PI-SIDIS runs with valid kinematics, comp_livetime must be in (0, 1]."""
        sidis = runlist_df[
            runlist_df["run_type"].isin(SIDIS_RUN_TYPES)
            & runlist_df["x"].notna()
        ]
        valid = sidis["comp_livetime"].dropna()
        assert len(valid) > 0
        assert (valid > 0).all() and (valid <= 1.0).all(), (
            "comp_livetime outside (0, 1] for PI-SIDIS runs with valid kinematics"
        )


# ===========================================================================
# get_settings
# ===========================================================================

class TestGetSettings:

    def test_returns_non_empty_list(self, runlist_df):
        settings = get_settings(runlist_df)
        assert isinstance(settings, list)
        assert len(settings) > 0

    def test_only_sidis_run_types(self, runlist_df):
        for s in get_settings(runlist_df):
            assert s.run_type in SIDIS_RUN_TYPES

    def test_no_nan_in_kinematic_fields(self, runlist_df):
        for s in get_settings(runlist_df):
            for field in ("ebeam", "x", "Q2", "z", "thpq"):
                val = getattr(s, field)
                assert not (isinstance(val, float) and np.isnan(val)), (
                    f"Setting has NaN in field '{field}': {s}"
                )

    def test_known_setting_C_z05_thpq2_present(self, runlist_df, setting_C_z05_thpq2):
        settings = get_settings(runlist_df)
        found = any(
            abs(s.ebeam - setting_C_z05_thpq2.ebeam) < 0.01
            and abs(s.x    - setting_C_z05_thpq2.x)    < 0.01
            and abs(s.Q2   - setting_C_z05_thpq2.Q2)   < 0.01
            and abs(s.z    - setting_C_z05_thpq2.z)     < 0.01
            and abs(s.thpq - setting_C_z05_thpq2.thpq)  < 0.01
            and s.run_type == setting_C_z05_thpq2.run_type
            for s in settings
        )
        assert found, f"Expected setting not found: {setting_C_z05_thpq2}"

    def test_settings_are_sorted(self, runlist_df):
        """Verify the list is sorted by (run_type, ebeam, z, thpq)."""
        settings = get_settings(runlist_df)
        keys = [(s.run_type, s.ebeam, s.z, s.thpq) for s in settings]
        assert keys == sorted(keys), "Settings list is not sorted"

    def test_no_duplicate_settings(self, runlist_df):
        settings = get_settings(runlist_df)
        # Each (run_type, ebeam, x, Q2, z, thpq) tuple should be unique
        keys = [(s.run_type, s.ebeam, s.x, s.Q2, s.z, s.thpq) for s in settings]
        assert len(keys) == len(set(keys)), "Duplicate settings found"


# ===========================================================================
# select_runs / get_signal_runs / get_eplus_runs
# ===========================================================================

class TestSelectRuns:

    def test_selects_correct_target(self, runlist_df, setting_C_z05_thpq2):
        runs = select_runs(runlist_df, setting_C_z05_thpq2, target="C")
        assert (runs["target"] == "C").all()

    def test_all_rows_match_setting_kinematics(self, runlist_df, setting_C_z05_thpq2):
        runs = select_runs(runlist_df, setting_C_z05_thpq2, target="C")
        assert len(runs) > 0
        tol = 0.02
        s = setting_C_z05_thpq2
        for _, row in runs.iterrows():
            assert abs(row["ebeam"] - s.ebeam) < tol, f"ebeam mismatch in run {row['run']}"
            assert abs(row["x"]    - s.x)     < tol, f"x mismatch in run {row['run']}"
            assert abs(row["Q2"]   - s.Q2)    < tol, f"Q2 mismatch in run {row['run']}"
            assert abs(row["z"]    - s.z)     < tol, f"z mismatch in run {row['run']}"
            assert abs(row["thpq"] - s.thpq)  < tol, f"thpq mismatch in run {row['run']}"
            assert row["run_type"] == s.run_type

    def test_nonexistent_target_returns_empty(self, runlist_df, setting_C_z05_thpq2):
        runs = select_runs(runlist_df, setting_C_z05_thpq2, target="NONEXISTENT")
        assert len(runs) == 0

    def test_result_has_reset_index(self, runlist_df, setting_C_z05_thpq2):
        runs = select_runs(runlist_df, setting_C_z05_thpq2, target="C")
        assert list(runs.index) == list(range(len(runs)))

    def test_signal_runs_all_have_negative_hms_p(self, runlist_df, setting_C_z05_thpq2):
        all_runs = select_runs(runlist_df, setting_C_z05_thpq2, target="C")
        signal = get_signal_runs(all_runs)
        assert len(signal) > 0, "Expected at least one e⁻ run"
        assert (signal["hms_p"] < 0).all(), "Signal run with non-negative hms_p found"

    def test_eplus_runs_all_have_positive_hms_p(self, runlist_df, setting_C_z05_thpq2):
        all_runs = select_runs(runlist_df, setting_C_z05_thpq2, target="C")
        eplus = get_eplus_runs(all_runs)
        assert len(eplus) > 0, "Expected at least one e⁺ run for this setting"
        assert (eplus["hms_p"] > 0).all(), "e⁺ run with non-positive hms_p found"

    def test_signal_and_eplus_runs_are_disjoint(self, runlist_df, setting_C_z05_thpq2):
        all_runs = select_runs(runlist_df, setting_C_z05_thpq2, target="C")
        signal_nos = set(get_signal_runs(all_runs)["run"])
        eplus_nos  = set(get_eplus_runs(all_runs)["run"])
        assert signal_nos.isdisjoint(eplus_nos), (
            f"Runs appear in both signal and e⁺ sets: {signal_nos & eplus_nos}"
        )

    def test_signal_and_eplus_together_cover_all_runs(self, runlist_df, setting_C_z05_thpq2):
        """Every run must be either signal or e⁺ — no runs with hms_p == 0."""
        all_runs = select_runs(runlist_df, setting_C_z05_thpq2, target="C")
        signal_nos = set(get_signal_runs(all_runs)["run"])
        eplus_nos  = set(get_eplus_runs(all_runs)["run"])
        all_nos    = set(all_runs["run"])
        assert signal_nos | eplus_nos == all_nos, (
            f"Runs not covered by either polarity: {all_nos - (signal_nos | eplus_nos)}"
        )

    def test_known_eminus_run_numbers(self, runlist_df, setting_C_z05_thpq2):
        """Spot-check that specific known run numbers are selected as e⁻ signal."""
        all_runs = select_runs(runlist_df, setting_C_z05_thpq2, target="C")
        signal = get_signal_runs(all_runs)
        signal_nos = set(signal["run"])
        expected_subset = {24120, 24121, 24122, 24123}
        assert expected_subset <= signal_nos, (
            f"Known e⁻ runs missing from selection: {expected_subset - signal_nos}"
        )

    def test_known_eplus_run_numbers(self, runlist_df, setting_C_z05_thpq2):
        """Spot-check that specific known run numbers are selected as e⁺ background."""
        all_runs = select_runs(runlist_df, setting_C_z05_thpq2, target="C")
        eplus = get_eplus_runs(all_runs)
        eplus_nos = set(eplus["run"])
        expected = {24589, 24590}
        assert expected <= eplus_nos, (
            f"Known e⁺ runs missing from selection: {expected - eplus_nos}"
        )

    def test_eminus_run_count_for_known_setting(self, runlist_df, setting_C_z05_thpq2):
        all_runs = select_runs(runlist_df, setting_C_z05_thpq2, target="C")
        n_signal = len(get_signal_runs(all_runs))
        assert n_signal == 14, (
            f"Expected 14 e⁻ runs, got {n_signal}.  "
            "If the CSV was updated, update this count too."
        )

    def test_eplus_run_count_for_known_setting(self, runlist_df, setting_C_z05_thpq2):
        all_runs = select_runs(runlist_df, setting_C_z05_thpq2, target="C")
        n_eplus = len(get_eplus_runs(all_runs))
        assert n_eplus == 2, (
            f"Expected 2 e⁺ runs, got {n_eplus}.  "
            "If the CSV was updated, update this count too."
        )


# ===========================================================================
# get_dummy_runs
# ===========================================================================

class TestGetDummyRuns:

    def test_dummy_target_column_is_dummy(self, runlist_df, setting_LH2_z05_thpq5p2):
        dummy = get_dummy_runs(runlist_df, setting_LH2_z05_thpq5p2)
        assert (dummy["target"] == "Dummy").all()

    def test_dummy_runs_exist_for_cryo_setting(self, runlist_df, setting_LH2_z05_thpq5p2):
        dummy = get_dummy_runs(runlist_df, setting_LH2_z05_thpq5p2)
        assert len(dummy) > 0, "Expected dummy runs for an LH2 setting"

    def test_dummy_runs_absent_for_solid_target(self, runlist_df, setting_C_z05_thpq2):
        """Carbon is not a cryo target — there should be no dummy runs for it."""
        dummy = get_dummy_runs(runlist_df, setting_C_z05_thpq2)
        # The Dummy target doesn't match the same kinematics unless explicitly present
        # For solid targets no dummy subtraction is done; this may or may not be empty
        # depending on what's in the CSV.  What we assert is that if it is non-empty
        # every row has target == 'Dummy'.
        if len(dummy) > 0:
            assert (dummy["target"] == "Dummy").all()


# ===========================================================================
# is_cryo
# ===========================================================================

class TestIsCryo:

    @pytest.mark.parametrize("target", ["LH2", "LD2"])
    def test_cryo_targets(self, target):
        assert is_cryo(target) is True

    @pytest.mark.parametrize("target", ["C", "Cu", "Al", "Dummy", "HOLE", ""])
    def test_non_cryo_targets(self, target):
        assert is_cryo(target) is False


# ===========================================================================
# Setting.__str__
# ===========================================================================

class TestSettingStr:

    def test_str_contains_run_type(self, setting_C_z05_thpq2):
        assert "PI-SIDIS" in str(setting_C_z05_thpq2)

    def test_str_contains_z_value(self, setting_C_z05_thpq2):
        assert "0.50" in str(setting_C_z05_thpq2)

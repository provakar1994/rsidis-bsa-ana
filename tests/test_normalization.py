"""
Tests for rsidis_ssa.normalization

All tests that use the CSV run against the real data in data/.
Tests that check error conditions build minimal pd.Series objects
inline — no mocks, no fixtures needed for those.
"""

import math
import re
import tempfile
from pathlib import Path

import pandas as pd
import numpy as np
import pytest

from rsidis_ssa.runlist import (
    Setting,
    load_runlist,
    select_runs,
    get_signal_runs,
    get_eplus_runs,
)
from rsidis_ssa.normalization import (
    NormalizationError,
    RunExclusion,
    RunWeight,
    WeightTableResult,
    build_weight_table,
    check_normyield_consistency,
    compute_run_weight,
    get_prescale_factor,
    weight_table_to_df,
)

CSV = Path(__file__).parent.parent / "data" / "rsidis_bigtable_pass0p1.csv"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_row(**kwargs) -> pd.Series:
    """Minimal valid runlist row for unit-testing individual functions."""
    defaults = dict(
        run=99999,
        BCM2_Q=50.0,
        h_esing_Eff=0.9998,
        p_hadron_Eff=0.9300,
        ps5=float("nan"),
        ps6=1.0,
        comp_livetime=1.0,
        boil_corr=float("nan"),
    )
    defaults.update(kwargs)
    return pd.Series(defaults)


def _setting_C_z05():
    return Setting(ebeam=8.5831, x=0.25, Q2=3.3, z=0.5, thpq=2.0, run_type="PI-SIDIS")


# ---------------------------------------------------------------------------
# get_prescale_factor
# ---------------------------------------------------------------------------

class TestGetPrescaleFactor:

    def test_ps6_active(self):
        assert get_prescale_factor(_make_row(ps6=1.0, ps5=float("nan"))) == 1.0

    def test_ps5_active_fallback(self):
        assert get_prescale_factor(_make_row(ps6=float("nan"), ps5=1.0)) == 1.0

    def test_ps6_takes_precedence_when_both_set(self):
        assert get_prescale_factor(_make_row(ps6=2.0, ps5=4.0)) == 2.0

    def test_neither_active_raises(self):
        with pytest.raises(NormalizationError, match="no active prescale"):
            get_prescale_factor(_make_row(ps6=float("nan"), ps5=float("nan")))

    def test_neither_active_error_carries_column(self):
        with pytest.raises(NormalizationError) as exc_info:
            get_prescale_factor(_make_row(ps6=float("nan"), ps5=float("nan")))
        assert exc_info.value.column == "ps5/ps6"

    def test_negative_treated_as_inactive(self):
        # -1.0 (un-NaN-ified sentinel) should not be treated as active
        assert get_prescale_factor(_make_row(ps6=-1.0, ps5=1.0)) == 1.0

    def test_prescale_factor_gt_1(self):
        assert get_prescale_factor(_make_row(ps6=3.0)) == 3.0


# ---------------------------------------------------------------------------
# compute_run_weight — unit tests with synthetic rows
# ---------------------------------------------------------------------------

class TestComputeRunWeightUnit:

    def test_basic_weight_formula(self):
        row = _make_row(BCM2_Q=50.0, h_esing_Eff=0.9998, p_hadron_Eff=0.930,
                        ps6=1.0, comp_livetime=1.0)
        rw = compute_run_weight(row)
        expected = 1.0 / (50.0 * 0.9998 * 0.930 * 1.0)
        assert math.isclose(rw.weight, expected, rel_tol=1e-9)

    def test_boil_corr_in_numerator(self):
        """boil_corr > 1 must increase the weight (less luminosity → more cross-section)."""
        boil = 1.025
        row = _make_row(BCM2_Q=50.0, h_esing_Eff=0.9998, p_hadron_Eff=0.930,
                        ps6=1.0, comp_livetime=1.0, boil_corr=boil)
        rw_with    = compute_run_weight(row, apply_boil_corr=True)
        rw_without = compute_run_weight(row, apply_boil_corr=False)
        expected_with = boil / (50.0 * 0.9998 * 0.930 * 1.0)
        assert math.isclose(rw_with.weight, expected_with, rel_tol=1e-9)
        assert rw_with.weight > rw_without.weight

    def test_boil_corr_nan_raises_normalization_error_when_cryo(self):
        with pytest.raises(NormalizationError) as exc_info:
            compute_run_weight(_make_row(boil_corr=float("nan")), apply_boil_corr=True)
        assert exc_info.value.column == "boil_corr"

    def test_boil_corr_nan_ok_when_not_cryo(self):
        rw = compute_run_weight(_make_row(boil_corr=float("nan")), apply_boil_corr=False)
        assert rw.boil_corr == 1.0

    def test_zero_charge_raises_with_column(self):
        with pytest.raises(NormalizationError) as exc_info:
            compute_run_weight(_make_row(BCM2_Q=0.0))
        assert exc_info.value.column == "BCM2_Q"

    def test_negative_charge_raises(self):
        with pytest.raises(NormalizationError, match="BCM2_Q"):
            compute_run_weight(_make_row(BCM2_Q=-5.0))

    def test_nan_charge_raises(self):
        with pytest.raises(NormalizationError, match="BCM2_Q"):
            compute_run_weight(_make_row(BCM2_Q=float("nan")))

    def test_h_esing_eff_out_of_range_raises(self):
        with pytest.raises(NormalizationError) as exc_info:
            compute_run_weight(_make_row(h_esing_Eff=1.05))
        assert exc_info.value.column == "h_esing_Eff"

    def test_p_hadron_eff_zero_raises(self):
        with pytest.raises(NormalizationError) as exc_info:
            compute_run_weight(_make_row(p_hadron_Eff=0.0))
        assert exc_info.value.column == "p_hadron_Eff"

    def test_livetime_out_of_range_raises(self):
        with pytest.raises(NormalizationError) as exc_info:
            compute_run_weight(_make_row(comp_livetime=1.5))
        assert exc_info.value.column == "comp_livetime"

    def test_negative_livetime_raises(self):
        with pytest.raises(NormalizationError) as exc_info:
            compute_run_weight(_make_row(comp_livetime=-0.1))
        assert exc_info.value.column == "comp_livetime"

    def test_runweight_is_frozen(self):
        rw = compute_run_weight(_make_row())
        with pytest.raises((AttributeError, TypeError)):
            rw.weight = 999.0  # type: ignore

    def test_norm_factor_is_reciprocal_of_weight(self):
        rw = compute_run_weight(_make_row(BCM2_Q=40.0, h_esing_Eff=0.9995, p_hadron_Eff=0.95))
        assert math.isclose(rw.norm_factor, 1.0 / rw.weight, rel_tol=1e-12)

    def test_prescale_gt_1_increases_weight(self):
        rw1 = compute_run_weight(_make_row(ps6=1.0, BCM2_Q=50.0))
        rw2 = compute_run_weight(_make_row(ps6=2.0, BCM2_Q=50.0))
        assert math.isclose(rw2.weight, 2.0 * rw1.weight, rel_tol=1e-9)

    def test_run_number_preserved(self):
        assert compute_run_weight(_make_row(run=24120)).run == 24120

    def test_normalization_error_carries_run_number(self):
        with pytest.raises(NormalizationError) as exc_info:
            compute_run_weight(_make_row(run=12345, BCM2_Q=0.0))
        assert exc_info.value.run == 12345


# ---------------------------------------------------------------------------
# compute_run_weight — against real CSV rows
# ---------------------------------------------------------------------------

class TestComputeRunWeightReal:

    @pytest.fixture(scope="class")
    def df(self):
        return load_runlist(CSV)

    def _cross_check(self, row, tol=0.02):
        rw = compute_run_weight(row, apply_boil_corr=False)
        ny_csv     = row["normyield"]
        ransubcoin = row["ransubcoin"]
        if pd.isna(ny_csv) or pd.isna(ransubcoin) or ny_csv == 0:
            pytest.skip("normyield or ransubcoin not available for this run")
        ny_computed = ransubcoin * rw.weight
        residual = abs(ny_computed - ny_csv) / abs(ny_csv)
        assert residual < tol, (
            f"Run {rw.run}: normyield mismatch "
            f"(csv={ny_csv:.4f}, computed={ny_computed:.4f}, "
            f"residual={100*residual:.2f}%)"
        )

    def test_carbon_run_24120(self, df):
        self._cross_check(df[df["run"] == 24120].iloc[0])

    def test_carbon_run_24165(self, df):
        self._cross_check(df[df["run"] == 24165].iloc[0])

    def test_lh2_run_no_boil(self, df):
        lh2 = df[(df["target"] == "LH2") & df["boil_corr"].notna() & df["normyield"].notna()]
        self._cross_check(lh2.iloc[0])

    def test_ld2_run_no_boil(self, df):
        ld2 = df[(df["target"] == "LD2") & df["boil_corr"].notna() & df["normyield"].notna()]
        self._cross_check(ld2.iloc[0])

    def test_weight_increases_with_lower_charge(self, df):
        runs = get_signal_runs(select_runs(df, _setting_C_z05(), "C"))
        result = build_weight_table(runs)
        wdf = weight_table_to_df(result.weights)
        corr = wdf["BCM2_Q"].corr(wdf["weight"], method="spearman")
        assert corr < -0.9, (
            f"Expected strong negative charge–weight correlation, got {corr:.3f}"
        )


# ---------------------------------------------------------------------------
# build_weight_table — return type and valid runs
# ---------------------------------------------------------------------------

class TestBuildWeightTable:

    @pytest.fixture(scope="class")
    def df(self):
        return load_runlist(CSV)

    @pytest.fixture(scope="class")
    def result_C(self, df):
        runs = get_signal_runs(select_runs(df, _setting_C_z05(), "C"))
        return build_weight_table(runs)

    def test_returns_weight_table_result(self, result_C):
        assert isinstance(result_C, WeightTableResult)

    def test_weights_are_run_weight_objects(self, result_C):
        assert all(isinstance(v, RunWeight) for v in result_C.weights.values())

    def test_keys_are_run_numbers(self, df, result_C):
        runs = get_signal_runs(select_runs(df, _setting_C_z05(), "C"))
        assert set(result_C.weights.keys()) == set(runs["run"])

    def test_all_weights_positive(self, result_C):
        assert all(rw.weight > 0 for rw in result_C.weights.values())

    def test_all_weights_finite(self, result_C):
        assert all(math.isfinite(rw.weight) for rw in result_C.weights.values())

    def test_no_exclusions_for_clean_C_setting(self, result_C):
        assert result_C.n_excluded == 0, (
            f"Unexpected exclusions:\n{result_C.summary()}"
        )

    def test_n_valid_and_n_excluded_sum_to_input_count(self, df):
        runs = get_signal_runs(select_runs(df, _setting_C_z05(), "C"))
        result = build_weight_table(runs)
        assert result.n_valid + result.n_excluded == len(runs)

    def test_eplus_runs_same_structure(self, df):
        eplus = get_eplus_runs(select_runs(df, _setting_C_z05(), "C"))
        result = build_weight_table(eplus)
        assert isinstance(result, WeightTableResult)
        assert result.n_valid == len(eplus)


# ---------------------------------------------------------------------------
# build_weight_table — exclusion behaviour
# ---------------------------------------------------------------------------

class TestBuildWeightTableExclusions:

    @pytest.fixture(scope="class")
    def df(self):
        return load_runlist(CSV)

    def test_nan_boil_corr_excluded_not_raised(self, df):
        """Run 23861 has NaN boil_corr — must appear in excluded, not raise."""
        lh2_run = df[df["run"] == 23861]
        result = build_weight_table(lh2_run, apply_boil_corr=True)
        assert result.n_valid == 0
        assert result.n_excluded == 1
        assert result.excluded[0].run == 23861

    def test_excluded_run_carries_column_name(self, df):
        lh2_run = df[df["run"] == 23861]
        result = build_weight_table(lh2_run, apply_boil_corr=True)
        assert result.excluded[0].column == "boil_corr"

    def test_excluded_run_carries_reason_string(self, df):
        lh2_run = df[df["run"] == 23861]
        result = build_weight_table(lh2_run, apply_boil_corr=True)
        assert "23861" in result.excluded[0].reason
        assert "boil_corr" in result.excluded[0].reason

    def test_mixed_batch_separates_valid_and_excluded(self, df):
        """Batch of LH2 runs: 23861 (no boil_corr) + others (valid boil_corr)."""
        setting = Setting(ebeam=8.5831, x=0.25, Q2=3.3, z=0.5, thpq=5.2,
                          run_type="PI-SIDIS")
        lh2_runs = get_signal_runs(select_runs(df, setting, "LH2"))
        assert len(lh2_runs) > 1, "Need more than one LH2 run for this test"

        result = build_weight_table(lh2_runs, apply_boil_corr=True)
        excluded_nos = {ex.run for ex in result.excluded}
        valid_nos    = set(result.weights.keys())

        # Excluded and valid sets must be disjoint and together cover all input runs
        assert excluded_nos.isdisjoint(valid_nos)
        assert excluded_nos | valid_nos == set(lh2_runs["run"])

        # Run 23861 specifically must be in excluded
        assert 23861 in excluded_nos

        # Valid runs must all have finite positive weights
        assert all(math.isfinite(rw.weight) and rw.weight > 0
                   for rw in result.weights.values())

    def test_exclusion_with_invalid_charge(self):
        """Inject a bad row into a DataFrame and verify exclusion."""
        bad_row = _make_row(run=11111, BCM2_Q=0.0)
        good_row = _make_row(run=22222, BCM2_Q=50.0)
        df_fake = pd.DataFrame([bad_row, good_row])
        result = build_weight_table(df_fake)
        assert 11111 in {ex.run for ex in result.excluded}
        assert 22222 in result.weights
        assert result.excluded[0].column == "BCM2_Q"

    def test_summary_string_lists_excluded_runs(self, df):
        lh2_run = df[df["run"] == 23861]
        result = build_weight_table(lh2_run, apply_boil_corr=True)
        summary = result.summary()
        assert "23861" in summary
        assert "excluded" in summary.lower()

    def test_cryo_valid_boil_succeeds_with_no_exclusions(self, df):
        lh2_valid = df[
            (df["target"] == "LH2") & df["boil_corr"].notna()
            & (df["run_type"] == "PI-SIDIS") & df["x"].notna()
        ].head(5)
        result = build_weight_table(lh2_valid, apply_boil_corr=True)
        assert result.n_excluded == 0
        assert all(rw.boil_corr > 1.0 for rw in result.weights.values())

    def test_boil_corr_increases_weight_vs_no_boil(self, df):
        lh2_valid = df[
            (df["target"] == "LH2") & df["boil_corr"].notna()
            & (df["run_type"] == "PI-SIDIS") & df["x"].notna()
        ].head(3)
        if len(lh2_valid) == 0:
            pytest.skip("No suitable LH2 rows found")
        r_with    = build_weight_table(lh2_valid, apply_boil_corr=True)
        r_without = build_weight_table(lh2_valid, apply_boil_corr=False)
        for run_no in r_with.weights:
            assert r_with.weights[run_no].weight > r_without.weights[run_no].weight


# ---------------------------------------------------------------------------
# build_weight_table — log file
# ---------------------------------------------------------------------------

class TestBuildWeightTableLog:

    @pytest.fixture(scope="class")
    def df(self):
        return load_runlist(CSV)

    def test_log_file_created(self, df):
        runs = get_signal_runs(select_runs(df, _setting_C_z05(), "C"))
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test_norm.log"
            build_weight_table(runs, log_path=log_path)
            assert log_path.exists(), "Log file was not created"

    def test_log_file_created_even_with_no_exclusions(self, df):
        """An empty exclusion section should still produce a log file."""
        runs = get_signal_runs(select_runs(df, _setting_C_z05(), "C"))
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "norm.log"
            result = build_weight_table(runs, log_path=log_path)
            assert result.n_excluded == 0
            assert log_path.exists()
            content = log_path.read_text()
            assert "No runs were excluded" in content

    def test_log_contains_all_accepted_run_numbers(self, df):
        runs = get_signal_runs(select_runs(df, _setting_C_z05(), "C"))
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "norm.log"
            result = build_weight_table(runs, log_path=log_path)
            content = log_path.read_text()
            for run_no in result.weights:
                assert str(run_no) in content, (
                    f"Run {run_no} not found in log file"
                )

    def test_log_contains_excluded_run_number_and_column(self, df):
        lh2_run = df[df["run"] == 23861]
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "excl.log"
            build_weight_table(lh2_run, apply_boil_corr=True, log_path=log_path)
            content = log_path.read_text()
            assert "23861" in content
            assert "boil_corr" in content

    def test_log_contains_timestamp(self, df):
        runs = get_signal_runs(select_runs(df, _setting_C_z05(), "C")).head(1)
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "norm.log"
            build_weight_table(runs, log_path=log_path)
            content = log_path.read_text()
            # Timestamp pattern: YYYY-MM-DD HH:MM:SS
            assert re.search(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", content)

    def test_log_parent_directory_created_if_missing(self, df):
        runs = get_signal_runs(select_runs(df, _setting_C_z05(), "C")).head(1)
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "subdir" / "deep" / "norm.log"
            build_weight_table(runs, log_path=log_path)
            assert log_path.exists()

    def test_no_log_file_without_log_path(self, df):
        """Passing no log_path must not create any file."""
        runs = get_signal_runs(select_runs(df, _setting_C_z05(), "C")).head(1)
        # Just verify it doesn't raise — can't check for absent file path
        result = build_weight_table(runs)
        assert isinstance(result, WeightTableResult)


# ---------------------------------------------------------------------------
# weight_table_to_df
# ---------------------------------------------------------------------------

class TestWeightTableToDf:

    @pytest.fixture(scope="class")
    def weights(self, df=None):
        _df = load_runlist(CSV)
        runs = get_signal_runs(select_runs(_df, _setting_C_z05(), "C"))
        return build_weight_table(runs).weights

    def test_returns_dataframe(self, weights):
        assert isinstance(weight_table_to_df(weights), pd.DataFrame)

    def test_expected_columns(self, weights):
        cols = set(weight_table_to_df(weights).columns)
        assert {"run", "weight", "norm_factor", "BCM2_Q",
                "h_esing_Eff", "p_hadron_Eff", "ps_factor",
                "comp_livetime", "boil_corr"} <= cols

    def test_row_count_matches(self, weights):
        assert len(weight_table_to_df(weights)) == len(weights)

    def test_sorted_by_run(self, weights):
        out = weight_table_to_df(weights)
        assert list(out["run"]) == sorted(out["run"])

    def test_weight_times_norm_factor_is_one(self, weights):
        out = weight_table_to_df(weights)
        assert (abs(out["weight"] * out["norm_factor"] - 1.0) < 1e-10).all()


# ---------------------------------------------------------------------------
# check_normyield_consistency
# ---------------------------------------------------------------------------

class TestCheckNormyieldConsistency:

    @pytest.fixture(scope="class")
    def df(self):
        return load_runlist(CSV)

    @pytest.fixture(scope="class")
    def weights_C(self, df):
        runs = get_signal_runs(select_runs(df, _setting_C_z05(), "C"))
        return build_weight_table(runs).weights   # pass plain dict

    def test_returns_dataframe(self, weights_C, df):
        assert isinstance(check_normyield_consistency(weights_C, df), pd.DataFrame)

    def test_expected_columns(self, weights_C, df):
        cols = set(check_normyield_consistency(weights_C, df).columns)
        assert {"run", "normyield_csv", "normyield_computed",
                "residual_pct", "flagged"} <= cols

    def test_no_runs_flagged_for_C_setting(self, weights_C, df):
        out = check_normyield_consistency(weights_C, df, rtol=0.02)
        flagged = out[out["flagged"]]
        assert len(flagged) == 0, (
            f"Runs with normyield discrepancy > 2%:\n{flagged.to_string()}"
        )

    def test_residuals_all_small(self, weights_C, df):
        out = check_normyield_consistency(weights_C, df)
        assert (out["residual_pct"].abs() < 2.0).all(), (
            f"Residuals > 2%:\n{out[out['residual_pct'].abs() >= 2.0].to_string()}"
        )

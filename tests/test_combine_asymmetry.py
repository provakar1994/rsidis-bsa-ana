"""Regression coverage for diagnostic count aggregation, without ROOT inputs."""

import numpy as np
import pandas as pd
import pytest

from combine_asymmetry import (
    combine_bins,
    combine_bins_binned,
    _write_combined_summary,
    _write_combined_binned_summary,
)


def _frame(count):
    return pd.DataFrame({
        "histogram": ["phipq"], "phi_center": [1.0],
        "A_phys": [0.2], "A_phys_err": [0.1],
        "N_plus": [count], "N_minus": [count],
        "variable": ["pt"], "hmin": [0.0], "hmax": [0.1],
        "bin_center": [0.05],
    })


@pytest.mark.parametrize("combine", [combine_bins, combine_bins_binned])
@pytest.mark.parametrize("counts, expected", [
    ([10.0, 20.0], 30.0),
    ([0.0, 0.0], 0.0),
    ([np.nan, np.nan], np.nan),
    ([10.0, np.nan], np.nan),
])
def test_combined_counts_preserve_missing_values(combine, counts, expected):
    result = combine([_frame(count) for count in counts])
    for col in ("N_plus", "N_minus"):
        assert result[col].iloc[0] == pytest.approx(expected, nan_ok=True)
    assert result.A_phys.iloc[0] == pytest.approx(0.2)
    assert result.A_phys_err.iloc[0] == pytest.approx(0.1 / np.sqrt(2))


@pytest.mark.parametrize("combine", [combine_bins, combine_bins_binned])
def test_legacy_inputs_without_counts_still_combine(combine):
    result = combine([_frame(10), _frame(20).drop(columns=["N_plus", "N_minus"])])
    assert "N_plus" not in result
    assert result.A_phys.iloc[0] == pytest.approx(0.2)


@pytest.mark.parametrize("combine", [combine_bins, combine_bins_binned])
def test_counts_include_only_valid_asymmetry_contributors(combine):
    invalid = _frame(100)
    invalid["A_phys_err"] = 0.0
    result = combine([_frame(10), invalid])
    assert result.N_plus.iloc[0] == 10


@pytest.mark.parametrize("writer", [_write_combined_summary, _write_combined_binned_summary])
def test_summary_does_not_hide_missing_counts(writer, tmp_path):
    frame = pd.concat([_frame(10), _frame(np.nan)], ignore_index=True)
    frame["phi_center"] = [1.0, 2.0]
    output = tmp_path / "summary.csv"
    writer(output, frame, [("LH2_pip_e10p7_x0p25_q23p3_z0p5_thpq2p0", frame)])
    summary = pd.read_csv(output)
    assert summary.N_plus.isna().all()
    assert summary.N_minus.isna().all()

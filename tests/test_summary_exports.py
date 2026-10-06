"""Summary fit values and bin metadata remain distinct from phi-bin data."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from analysis import _write_kinematic_summary
from combine_asymmetry import _write_combined_binned_summary
from rsidis_ssa.config_loader import load_config


def test_single_setting_summary_includes_pt_and_overall(tmp_path):
    cfg = load_config('tests/fixtures/example_C_z05_thpq2.yaml')
    result = SimpleNamespace(histogram_name='phipq', amplitude=0.12,
                             amplitude_err=0.03, chi2_ndf=1.1, n_bins_used=9,
                             N_plus=None, N_minus=None)
    path = tmp_path / 'summary.csv'
    _write_kinematic_summary(path, cfg, [result],
                             [('pt', [(result, 0.1, 0.2, 0.15)])])
    df = pd.read_csv(path)
    assert df.variable.tolist() == ['overall', 'pt']
    assert np.isnan(df.hmin.iloc[0])
    assert df.hmin.iloc[1] == 0.1
    assert df.hmax.iloc[1] == 0.2
    assert df.bin_center.iloc[1] == 0.15
    assert df.A_phys.tolist() == [0.12, 0.12]
    assert df.A_phys_error.tolist() == [0.03, 0.03]
    pd.testing.assert_series_equal(df.A_phys, df.asym, check_names=False)
    pd.testing.assert_series_equal(df.A_phys_error, df.asym_err, check_names=False)


def test_combined_pt_summary_exports_fitted_amplitude(tmp_path):
    phi = np.array([-2.0, -1.0, 1.0, 2.0])
    frames = []
    for lo, amp in [(0.0, 0.1), (0.1, -0.2)]:
        frames.append(pd.DataFrame(dict(variable='pt', hmin=lo, hmax=lo+0.1,
            bin_center=lo+0.05, histogram='phipq', phi_center=phi,
            A_phys=amp*np.sin(phi), A_phys_err=0.02)))
    df = pd.concat(frames)
    path = tmp_path / 'summary.csv'
    _write_combined_binned_summary(path, df,
        [('LH2_pip_e10p7_x0p25_q23p3_z0p5_thpq2p0', df)])
    summary = pd.read_csv(path)
    assert summary.A_phys.tolist() == pytest.approx([0.1, -0.2])
    assert summary.A_phys_error.tolist() == pytest.approx(
        [0.02 / np.sqrt(np.sum(np.sin(phi)**2))]*2)
    assert summary.hmin.tolist() == [0.0, 0.1]

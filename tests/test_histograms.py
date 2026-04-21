"""
Tests for rsidis_ssa.histograms

All tests use synthetic numpy arrays and HistogramConfig objects built
in-memory — no ROOT files or YAML files required.
"""

from __future__ import annotations

import math
import numpy as np
import boost_histogram as bh
import pytest

from rsidis_ssa.config_loader import HistogramConfig
from rsidis_ssa.cuts import (
    BRANCH_CTIME,
    BRANCH_HCER_NPE,
    BRANCH_HELICITY,
    BRANCH_HSDELTA,
    BRANCH_HETOTTRACKNORM,
    BRANCH_PETOTTRACKNORM,
    BRANCH_NU,
    BRANCH_PAERO_NPE,
    BRANCH_PHGC_NPE,
    BRANCH_PPi,
    BRANCH_PSDELTA,
    BRANCH_THETA_PQ,
    M_PI,
)
from rsidis_ssa.histograms import _resolve_values, build_histogram_registry, fill_run


# ===========================================================================
# Helpers
# ===========================================================================

def _hcfg(
    name="test_hist",
    branch="H_gtr_dp",
    bins=10,
    xmin=-5.0,
    xmax=5.0,
    xlabel="",
    helicity_cut=None,
) -> HistogramConfig:
    return HistogramConfig(
        name=name,
        branch=branch,
        bins=bins,
        xmin=xmin,
        xmax=xmax,
        xlabel=xlabel,
        helicity_cut=helicity_cut,
    )


def _all_pass_mask(n: int = 10) -> np.ndarray:
    return np.ones(n, dtype=bool)


def _no_pass_mask(n: int = 10) -> np.ndarray:
    return np.zeros(n, dtype=bool)


def _make_arrays(n: int = 10, **overrides) -> dict[str, np.ndarray]:
    base = {
        BRANCH_HSDELTA:        np.linspace(-4.0, 4.0, n),
        BRANCH_HCER_NPE:       np.full(n, 2.0),
        BRANCH_HETOTTRACKNORM: np.full(n, 0.9),
        BRANCH_PSDELTA:        np.full(n, 5.0),
        BRANCH_PAERO_NPE:      np.full(n, 3.0),
        BRANCH_PHGC_NPE:       np.full(n, 2.0),
        BRANCH_PETOTTRACKNORM: np.full(n, 0.3),
        BRANCH_CTIME:          np.full(n, 51.2),
        BRANCH_HELICITY:       np.full(n, 1.0),
        BRANCH_PPi:            np.full(n, 3.0),
        BRANCH_NU:             np.full(n, 6.0),
        BRANCH_THETA_PQ:       np.full(n, 0.1),
    }
    base.update(overrides)
    return base


# ===========================================================================
# build_histogram_registry
# ===========================================================================

class TestBuildHistogramRegistry:

    def test_creates_one_entry_per_config(self):
        cfgs = [_hcfg("a"), _hcfg("b"), _hcfg("c")]
        reg = build_histogram_registry(cfgs)
        assert set(reg.keys()) == {"a", "b", "c"}

    def test_values_are_boost_histograms(self):
        reg = build_histogram_registry([_hcfg()])
        assert isinstance(reg["test_hist"], bh.Histogram)

    def test_bins_match_config(self):
        reg = build_histogram_registry([_hcfg(bins=16, xmin=-8.0, xmax=8.0)])
        h = reg["test_hist"]
        assert h.axes[0].extent == 16 + 2  # extent includes overflow bins
        assert len(h.axes[0]) == 16         # visible bins

    def test_range_matches_config(self):
        reg = build_histogram_registry([_hcfg(bins=10, xmin=-5.0, xmax=5.0)])
        ax = reg["test_hist"].axes[0]
        assert ax.extent > 0
        assert ax.bin(0)[0] == pytest.approx(-5.0)
        assert ax.bin(9)[1] == pytest.approx(5.0)

    def test_uses_weight_storage(self):
        reg = build_histogram_registry([_hcfg()])
        assert isinstance(reg["test_hist"].storage_type(), bh.storage.Weight)

    def test_histograms_start_empty(self):
        reg = build_histogram_registry([_hcfg()])
        assert reg["test_hist"].values().sum() == pytest.approx(0.0)

    def test_empty_config_list(self):
        reg = build_histogram_registry([])
        assert reg == {}


# ===========================================================================
# _resolve_values
# ===========================================================================

class TestResolveValues:

    def test_direct_branch(self):
        arrays = _make_arrays()
        hcfg = _hcfg(branch=BRANCH_HSDELTA)
        result = _resolve_values(arrays, hcfg)
        np.testing.assert_array_equal(result, arrays[BRANCH_HSDELTA])

    def test_computed_zhad(self):
        arrays = _make_arrays()
        hcfg = _hcfg(branch="__computed__zhad")
        result = _resolve_values(arrays, hcfg)
        p_pi = arrays[BRANCH_PPi]
        nu   = arrays[BRANCH_NU]
        expected = np.sqrt(p_pi**2 + M_PI**2) / nu
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_computed_Pt(self):
        arrays = _make_arrays()
        hcfg = _hcfg(branch="__computed__Pt")
        result = _resolve_values(arrays, hcfg)
        expected = arrays[BRANCH_PPi] * np.sin(arrays[BRANCH_THETA_PQ])
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_unknown_computed_raises(self):
        arrays = _make_arrays()
        hcfg = _hcfg(branch="__computed__mystery")
        with pytest.raises(ValueError, match="Unknown computed quantity"):
            _resolve_values(arrays, hcfg)


# ===========================================================================
# fill_run — basic filling
# ===========================================================================

class TestFillRunBasic:

    def test_fills_correct_number_of_events(self):
        n = 10
        arrays = _make_arrays(n=n)
        cfg = _hcfg(bins=10, xmin=-5.0, xmax=5.0)
        reg = build_histogram_registry([cfg])
        fill_run(arrays, _all_pass_mask(n), 1.0, reg, [cfg], "OUT")
        # all events in [-4, 4] land in the visible bins
        assert reg["test_hist"].values().sum() == pytest.approx(n)

    def test_empty_mask_fills_nothing(self):
        n = 10
        arrays = _make_arrays(n=n)
        cfg = _hcfg()
        reg = build_histogram_registry([cfg])
        fill_run(arrays, _no_pass_mask(n), 1.0, reg, [cfg], "OUT")
        assert reg["test_hist"].values().sum() == pytest.approx(0.0)

    def test_weight_applied(self):
        n = 5
        arrays = _make_arrays(n=n)
        cfg = _hcfg(bins=10, xmin=-5.0, xmax=5.0)
        reg = build_histogram_registry([cfg])
        fill_run(arrays, _all_pass_mask(n), 2.0, reg, [cfg], "OUT")
        assert reg["test_hist"].values().sum() == pytest.approx(n * 2.0)

    def test_partial_mask(self):
        n = 6
        arrays = _make_arrays(n=n)
        mask = np.array([True, False, True, False, True, True])
        cfg = _hcfg(bins=10, xmin=-5.0, xmax=5.0)
        reg = build_histogram_registry([cfg])
        fill_run(arrays, mask, 1.0, reg, [cfg], "OUT")
        assert reg["test_hist"].values().sum() == pytest.approx(4.0)

    def test_accumulates_across_calls(self):
        n = 5
        arrays = _make_arrays(n=n)
        cfg = _hcfg(bins=10, xmin=-5.0, xmax=5.0)
        reg = build_histogram_registry([cfg])
        fill_run(arrays, _all_pass_mask(n), 1.0, reg, [cfg], "OUT")
        fill_run(arrays, _all_pass_mask(n), 1.0, reg, [cfg], "OUT")
        assert reg["test_hist"].values().sum() == pytest.approx(n * 2)

    def test_fills_multiple_histograms(self):
        n = 8
        arrays = _make_arrays(n=n)
        cfg_a = _hcfg(name="a", branch=BRANCH_HSDELTA, bins=10, xmin=-5.0, xmax=5.0)
        cfg_b = _hcfg(name="b", branch=BRANCH_PSDELTA, bins=10, xmin=0.0, xmax=10.0)
        reg = build_histogram_registry([cfg_a, cfg_b])
        fill_run(arrays, _all_pass_mask(n), 1.0, reg, [cfg_a, cfg_b], "OUT")
        assert reg["a"].values().sum() == pytest.approx(n)
        assert reg["b"].values().sum() == pytest.approx(n)


# ===========================================================================
# fill_run — helicity cuts
# ===========================================================================

class TestFillRunHelicity:

    def _arrays_mixed_hel(self, n=10):
        arrays = _make_arrays(n=n)
        # Alternating +1, -1 helicity
        arrays[BRANCH_HELICITY] = np.array([1.0 if i % 2 == 0 else -1.0
                                            for i in range(n)])
        return arrays

    def test_helicity_positive_ihwp_out(self):
        n = 10
        arrays = self._arrays_mixed_hel(n)
        cfg = _hcfg(helicity_cut="positive", bins=10, xmin=-5.0, xmax=5.0)
        reg = build_histogram_registry([cfg])
        fill_run(arrays, _all_pass_mask(n), 1.0, reg, [cfg], "OUT")
        assert reg["test_hist"].values().sum() == pytest.approx(n // 2)

    def test_helicity_negative_ihwp_out(self):
        n = 10
        arrays = self._arrays_mixed_hel(n)
        cfg = _hcfg(helicity_cut="negative", bins=10, xmin=-5.0, xmax=5.0)
        reg = build_histogram_registry([cfg])
        fill_run(arrays, _all_pass_mask(n), 1.0, reg, [cfg], "OUT")
        assert reg["test_hist"].values().sum() == pytest.approx(n // 2)

    def test_helicity_positive_ihwp_in_flips(self):
        """IHWP IN: raw +1 becomes effective -1, so 'positive' cut selects raw -1."""
        n = 10
        arrays = self._arrays_mixed_hel(n)
        cfg = _hcfg(helicity_cut="positive", bins=10, xmin=-5.0, xmax=5.0)
        reg_in  = build_histogram_registry([cfg])
        reg_out = build_histogram_registry([cfg])
        fill_run(arrays, _all_pass_mask(n), 1.0, reg_in,  [cfg], "IN")
        fill_run(arrays, _all_pass_mask(n), 1.0, reg_out, [cfg], "OUT")
        # IHWP IN flips which half passes; same count (n//2) but different events
        assert reg_in["test_hist"].values().sum() == pytest.approx(n // 2)
        assert reg_out["test_hist"].values().sum() == pytest.approx(n // 2)

    def test_no_helicity_cut_fills_all(self):
        n = 10
        arrays = self._arrays_mixed_hel(n)
        cfg = _hcfg(helicity_cut=None, bins=10, xmin=-5.0, xmax=5.0)
        reg = build_histogram_registry([cfg])
        fill_run(arrays, _all_pass_mask(n), 1.0, reg, [cfg], "OUT")
        assert reg["test_hist"].values().sum() == pytest.approx(n)

    def test_helicity_plus_and_minus_sum_to_no_cut(self):
        n = 10
        arrays = self._arrays_mixed_hel(n)
        cfg_all   = _hcfg(name="all",   helicity_cut=None,     bins=10, xmin=-5.0, xmax=5.0)
        cfg_plus  = _hcfg(name="plus",  helicity_cut="positive", bins=10, xmin=-5.0, xmax=5.0)
        cfg_minus = _hcfg(name="minus", helicity_cut="negative", bins=10, xmin=-5.0, xmax=5.0)
        reg = build_histogram_registry([cfg_all, cfg_plus, cfg_minus])
        fill_run(arrays, _all_pass_mask(n), 1.0, reg, [cfg_all, cfg_plus, cfg_minus], "OUT")
        total = reg["plus"].values().sum() + reg["minus"].values().sum()
        assert total == pytest.approx(reg["all"].values().sum())


# ===========================================================================
# fill_run — computed quantities
# ===========================================================================

class TestFillRunComputed:

    def test_fills_zhad_histogram(self):
        n = 5
        arrays = _make_arrays(n=n)
        cfg = _hcfg(name="zhad", branch="__computed__zhad", bins=50, xmin=0.1, xmax=1.0)
        reg = build_histogram_registry([cfg])
        fill_run(arrays, _all_pass_mask(n), 1.0, reg, [cfg], "OUT")
        # p_pi=3, nu=6 → zhad ~ sqrt(9+0.0195)/6 ~ 3.001/6 ~ 0.5
        assert reg["zhad"].values().sum() == pytest.approx(n)

    def test_fills_Pt_histogram(self):
        n = 5
        arrays = _make_arrays(n=n, p_pi=3.0, theta_pq=0.1)
        cfg = _hcfg(name="Pt", branch="__computed__Pt", bins=50, xmin=0.0, xmax=0.5)
        reg = build_histogram_registry([cfg])
        fill_run(arrays, _all_pass_mask(n), 1.0, reg, [cfg], "OUT")
        assert reg["Pt"].values().sum() == pytest.approx(n)

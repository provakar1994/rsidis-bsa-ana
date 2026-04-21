"""
Tests for rsidis_ssa.cuts

All tests use synthetic numpy arrays — no ROOT files required.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from rsidis_ssa.config_loader import CutsConfig
from rsidis_ssa.cuts import (
    M_PI,
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
    build_random_mask,
    build_real_mask,
    compute_Pt,
    compute_zhad,
    effective_helicity,
    ihwp_sign,
    pid_mask,
    random_ctime_mask,
    real_ctime_mask,
)


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def cuts_cfg() -> CutsConfig:
    """A standard CutsConfig for testing."""
    return CutsConfig(
        hsdelta_lo=-8.0,
        hsdelta_hi=8.0,
        hcer_npe_min=1.0,
        hsshsum_min=0.7,
        psdelta_lo=-10.0,
        psdelta_hi=20.0,
        paero_npe_min=2.0,
        phgc_npe_min=1.0,
        psshsum_max=0.8,
        ctime_real_center="auto",
        ctime_real_nsigma=3.0,
        ctime_real_window_fallback=2.0,
        ctime_random_n_skip=1,
        ctime_random_n_peaks_lo=3,
        ctime_random_n_peaks_hi=3,
    )


def _make_arrays(
    n: int = 10,
    hsdelta: float = 0.0,
    hcer_npe: float = 2.0,
    hsshsum: float = 0.9,
    psdelta: float = 5.0,
    paero_npe: float = 3.0,
    phgc_npe: float = 2.0,
    psshsum: float = 0.3,   # SHMS cal E/p — below psshsum_max=0.8 by default
    ctime: float = 51.2,
    hel: float = 1.0,
    p_pi: float = 3.0,
    nu: float = 6.0,
    theta_pq: float = 0.1,
) -> dict[str, np.ndarray]:
    """
    Build a minimal arrays dict where every event has the same value.
    Override individual branches via keyword arguments.
    """
    return {
        BRANCH_HSDELTA:          np.full(n, hsdelta),
        BRANCH_HCER_NPE:         np.full(n, hcer_npe),
        BRANCH_HETOTTRACKNORM:   np.full(n, hsshsum),
        BRANCH_PSDELTA:          np.full(n, psdelta),
        BRANCH_PAERO_NPE:        np.full(n, paero_npe),
        BRANCH_PHGC_NPE:         np.full(n, phgc_npe),
        BRANCH_PETOTTRACKNORM:   np.full(n, psshsum),
        BRANCH_CTIME:            np.full(n, ctime),
        BRANCH_HELICITY:         np.full(n, hel),
        BRANCH_PPi:              np.full(n, p_pi),
        BRANCH_NU:               np.full(n, nu),
        BRANCH_THETA_PQ:         np.full(n, theta_pq),
    }


# ===========================================================================
# IHWP sign
# ===========================================================================

class TestIhwpSign:

    def test_out_returns_plus_one(self):
        assert ihwp_sign("OUT") == +1

    def test_in_returns_minus_one(self):
        assert ihwp_sign("IN") == -1

    def test_case_insensitive_out(self):
        assert ihwp_sign("out") == +1

    def test_case_insensitive_in(self):
        assert ihwp_sign("in") == -1

    def test_whitespace_stripped(self):
        assert ihwp_sign("  OUT  ") == +1

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown IHWP"):
            ihwp_sign("SIDEWAYS")


# ===========================================================================
# Effective helicity
# ===========================================================================

class TestEffectiveHelicity:

    def test_ihwp_out_no_flip(self):
        arrays = _make_arrays(hel=1.0)
        result = effective_helicity(arrays, "OUT")
        assert np.all(result == 1.0)

    def test_ihwp_in_flips_sign(self):
        arrays = _make_arrays(hel=1.0)
        result = effective_helicity(arrays, "IN")
        assert np.all(result == -1.0)

    def test_mixed_helicity_ihwp_in(self):
        arrays = _make_arrays()
        arrays[BRANCH_HELICITY] = np.array([1.0, -1.0, 1.0, -1.0])
        result = effective_helicity(arrays, "IN")
        expected = np.array([-1.0, 1.0, -1.0, 1.0])
        np.testing.assert_array_equal(result, expected)

    def test_mixed_helicity_ihwp_out(self):
        arrays = _make_arrays()
        arrays[BRANCH_HELICITY] = np.array([1.0, -1.0, 1.0])
        result = effective_helicity(arrays, "OUT")
        np.testing.assert_array_equal(result, np.array([1.0, -1.0, 1.0]))


# ===========================================================================
# PID mask
# ===========================================================================

class TestPidMask:

    def test_all_pass(self, cuts_cfg):
        arrays = _make_arrays(n=5)
        mask = pid_mask(arrays, cuts_cfg)
        assert mask.all()

    def test_hsdelta_too_low(self, cuts_cfg):
        arrays = _make_arrays(hsdelta=-9.0)
        assert not pid_mask(arrays, cuts_cfg).any()

    def test_hsdelta_too_high(self, cuts_cfg):
        arrays = _make_arrays(hsdelta=9.0)
        assert not pid_mask(arrays, cuts_cfg).any()

    def test_hsdelta_at_boundary_passes(self, cuts_cfg):
        # boundary is inclusive: |hsdelta| == 8.0 should pass
        arrays = _make_arrays(hsdelta=-8.0)
        assert pid_mask(arrays, cuts_cfg).all()
        arrays = _make_arrays(hsdelta=8.0)
        assert pid_mask(arrays, cuts_cfg).all()

    def test_hcer_npe_below_min(self, cuts_cfg):
        arrays = _make_arrays(hcer_npe=0.5)
        assert not pid_mask(arrays, cuts_cfg).any()

    def test_hcer_npe_at_min_passes(self, cuts_cfg):
        arrays = _make_arrays(hcer_npe=1.0)
        assert pid_mask(arrays, cuts_cfg).all()

    def test_hsshsum_below_min(self, cuts_cfg):
        arrays = _make_arrays(hsshsum=0.5)
        assert not pid_mask(arrays, cuts_cfg).any()

    def test_psdelta_too_low(self, cuts_cfg):
        arrays = _make_arrays(psdelta=-11.0)
        assert not pid_mask(arrays, cuts_cfg).any()

    def test_psdelta_too_high(self, cuts_cfg):
        arrays = _make_arrays(psdelta=21.0)
        assert not pid_mask(arrays, cuts_cfg).any()

    def test_paero_npe_below_min(self, cuts_cfg):
        arrays = _make_arrays(paero_npe=1.5)
        assert not pid_mask(arrays, cuts_cfg).any()

    def test_phgc_npe_below_min(self, cuts_cfg):
        arrays = _make_arrays(phgc_npe=0.5)
        assert not pid_mask(arrays, cuts_cfg).any()

    def test_partial_pass(self, cuts_cfg):
        arrays = _make_arrays(n=4)
        # event 0: bad hsdelta; events 1-3: good
        arrays[BRANCH_HSDELTA] = np.array([-9.0, 0.0, 1.0, -1.0])
        mask = pid_mask(arrays, cuts_cfg)
        assert not mask[0]
        assert mask[1:].all()

    def test_returns_bool_array(self, cuts_cfg):
        mask = pid_mask(_make_arrays(), cuts_cfg)
        assert mask.dtype == bool


# ===========================================================================
# Real coincidence-time mask
# ===========================================================================

class TestRealCtimeMask:

    def test_auto_within_window_passes(self, cuts_cfg):
        # center=51.2, nsigma=3, ctsigma=0.5 → half_win=1.5 ns
        arrays = _make_arrays(ctime=51.2)
        mask = real_ctime_mask(arrays, cuts_cfg, ctmean=51.2, ctsigma=0.5)
        assert mask.all()

    def test_auto_outside_window_fails(self, cuts_cfg):
        # ctime = 53.0 → |53.0 − 51.2| = 1.8 > 1.5
        arrays = _make_arrays(ctime=53.0)
        mask = real_ctime_mask(arrays, cuts_cfg, ctmean=51.2, ctsigma=0.5)
        assert not mask.any()

    def test_auto_at_edge_passes(self, cuts_cfg):
        # ctime exactly at center + half_win (<=, inclusive)
        ctmean, ctsigma = 51.2, 0.5
        half_win = cuts_cfg.ctime_real_nsigma * ctsigma  # 1.5
        arrays = _make_arrays(ctime=ctmean + half_win)
        mask = real_ctime_mask(arrays, cuts_cfg, ctmean=ctmean, ctsigma=ctsigma)
        assert mask.all()

    def test_auto_nan_ctsigma_uses_fallback(self, cuts_cfg):
        # fallback = 2.0 ns
        arrays = _make_arrays(ctime=52.5)  # |52.5 − 51.2| = 1.3 < 2.0 → pass
        mask = real_ctime_mask(arrays, cuts_cfg, ctmean=51.2, ctsigma=float("nan"))
        assert mask.all()

    def test_auto_none_ctsigma_uses_fallback(self, cuts_cfg):
        arrays = _make_arrays(ctime=53.3)  # |53.3 − 51.2| = 2.1 > 2.0 → fail
        mask = real_ctime_mask(arrays, cuts_cfg, ctmean=51.2, ctsigma=None)
        assert not mask.any()

    def test_fixed_center_uses_fallback_window(self):
        cfg = CutsConfig(
            hsdelta_lo=-8.0,
            hsdelta_hi=8.0,
            hcer_npe_min=1.0,
            hsshsum_min=0.7,
            psdelta_lo=-10.0,
            psdelta_hi=20.0,
            paero_npe_min=2.0,
            phgc_npe_min=1.0,
            psshsum_max=0.8,
            ctime_real_center=51.2,
            ctime_real_nsigma=3.0,
            ctime_real_window_fallback=2.0,
            ctime_random_n_skip=1,
            ctime_random_n_peaks_lo=3,
            ctime_random_n_peaks_hi=3,
        )
        arrays = _make_arrays(ctime=51.2)
        mask = real_ctime_mask(arrays, cfg, ctmean=None, ctsigma=None)
        assert mask.all()

    def test_fixed_center_outside_window_fails(self):
        cfg = CutsConfig(
            hsdelta_lo=-8.0,
            hsdelta_hi=8.0,
            hcer_npe_min=1.0,
            hsshsum_min=0.7,
            psdelta_lo=-10.0,
            psdelta_hi=20.0,
            paero_npe_min=2.0,
            phgc_npe_min=1.0,
            psshsum_max=0.8,
            ctime_real_center=51.2,
            ctime_real_nsigma=3.0,
            ctime_real_window_fallback=2.0,
            ctime_random_n_skip=1,
            ctime_random_n_peaks_lo=3,
            ctime_random_n_peaks_hi=3,
        )
        arrays = _make_arrays(ctime=54.0)  # |54.0 − 51.2| = 2.8 > 2.0 → fail
        mask = real_ctime_mask(arrays, cfg, ctmean=None, ctsigma=None)
        assert not mask.any()


# ===========================================================================
# Random coincidence-time mask
# ===========================================================================

class TestRandomCtimeMask:
    """
    n_skip=1, n_peaks_lo=3, n_peaks_hi=3 (from cuts_cfg fixture)
    ctmean=51.2, real_half_win=1.0, beam_bunch_ns=4.0

    Lo peak centers: 51.2 − (1+k)×4  for k=1..3  → 43.2, 39.2, 35.2
    Hi peak centers: 51.2 + (1+k)×4  for k=1..3  → 59.2, 63.2, 67.2
    Each window: ±1.0 ns.  Windows do NOT overlap (half_win < beam_bunch_ns/2).
    Skipped adjacent peaks (n_skip=1): 47.2 (lo) and 55.2 (hi).
    """
    _CTMEAN  = 51.2
    _REAL_HW = 1.0
    _BBN     = 4.0

    def test_at_first_lo_peak_passes(self, cuts_cfg):
        arrays = _make_arrays(ctime=43.2)
        mask = random_ctime_mask(arrays, cuts_cfg, self._CTMEAN, self._REAL_HW, self._BBN)
        assert mask.all()

    def test_at_first_hi_peak_passes(self, cuts_cfg):
        arrays = _make_arrays(ctime=59.2)
        mask = random_ctime_mask(arrays, cuts_cfg, self._CTMEAN, self._REAL_HW, self._BBN)
        assert mask.all()

    def test_at_third_lo_peak_passes(self, cuts_cfg):
        arrays = _make_arrays(ctime=35.2)
        mask = random_ctime_mask(arrays, cuts_cfg, self._CTMEAN, self._REAL_HW, self._BBN)
        assert mask.all()

    def test_real_peak_fails(self, cuts_cfg):
        """Real peak center is not in any random window."""
        arrays = _make_arrays(ctime=self._CTMEAN)
        mask = random_ctime_mask(arrays, cuts_cfg, self._CTMEAN, self._REAL_HW, self._BBN)
        assert not mask.any()

    def test_skipped_adjacent_lo_fails(self, cuts_cfg):
        """Immediately adjacent lo peak (51.2 − 1×4 = 47.2) is skipped (n_skip=1)."""
        arrays = _make_arrays(ctime=47.2)
        mask = random_ctime_mask(arrays, cuts_cfg, self._CTMEAN, self._REAL_HW, self._BBN)
        assert not mask.any()

    def test_at_lo_peak_edge_passes(self, cuts_cfg):
        """Edge of first lo peak window: 43.2 + 1.0 = 44.2 (≤, inclusive)."""
        arrays = _make_arrays(ctime=43.2 + self._REAL_HW)
        mask = random_ctime_mask(arrays, cuts_cfg, self._CTMEAN, self._REAL_HW, self._BBN)
        assert mask.all()

    def test_just_beyond_lo_peak_fails(self, cuts_cfg):
        arrays = _make_arrays(ctime=43.2 + self._REAL_HW + 0.001)
        mask = random_ctime_mask(arrays, cuts_cfg, self._CTMEAN, self._REAL_HW, self._BBN)
        assert not mask.any()

    def test_between_lo_peaks_fails(self, cuts_cfg):
        """37.2 lies between peaks 35.2 and 39.2, outside both windows."""
        arrays = _make_arrays(ctime=37.2)
        mask = random_ctime_mask(arrays, cuts_cfg, self._CTMEAN, self._REAL_HW, self._BBN)
        assert not mask.any()

    def test_multiple_events_selects_correct_ones(self, cuts_cfg):
        """Events at peak centers pass; events at real peak or skipped peak fail."""
        ctimes   = np.array([43.2, 51.2, 59.2, 47.2, 35.2])
        expected = np.array([True, False, True, False, True])
        arrays = _make_arrays(n=5)
        arrays[BRANCH_CTIME] = ctimes
        mask = random_ctime_mask(arrays, cuts_cfg, self._CTMEAN, self._REAL_HW, self._BBN)
        np.testing.assert_array_equal(mask, expected)


# ===========================================================================
# Combined masks
# ===========================================================================

class TestBuildRealMask:

    def test_all_pass(self, cuts_cfg):
        arrays = _make_arrays(ctime=51.2)
        mask = build_real_mask(arrays, cuts_cfg, ctmean=51.2, ctsigma=0.5)
        assert mask.all()

    def test_bad_pid_fails(self, cuts_cfg):
        arrays = _make_arrays(ctime=51.2, hcer_npe=0.0)
        mask = build_real_mask(arrays, cuts_cfg, ctmean=51.2, ctsigma=0.5)
        assert not mask.any()

    def test_bad_ctime_fails(self, cuts_cfg):
        arrays = _make_arrays(ctime=99.0)
        mask = build_real_mask(arrays, cuts_cfg, ctmean=51.2, ctsigma=0.5)
        assert not mask.any()

    def test_partial_events(self, cuts_cfg):
        arrays = _make_arrays(n=4)
        arrays[BRANCH_CTIME] = np.array([51.2, 51.2, 99.0, 51.2])
        arrays[BRANCH_HCER_NPE] = np.array([2.0, 0.0, 2.0, 2.0])
        mask = build_real_mask(arrays, cuts_cfg, ctmean=51.2, ctsigma=0.5)
        # event 0 passes, event 1 fails (hcer), event 2 fails (ctime), event 3 passes
        np.testing.assert_array_equal(mask, [True, False, False, True])


class TestBuildRandomMask:
    """Combined PID + discrete random-ctime mask."""
    _CTMEAN  = 51.2
    _REAL_HW = 1.0
    _BBN     = 4.0

    def test_all_pass(self, cuts_cfg):
        """Good PID + ctime at first lo random peak (43.2) → passes."""
        arrays = _make_arrays(ctime=43.2)
        mask = build_random_mask(arrays, cuts_cfg, self._CTMEAN, self._REAL_HW, self._BBN)
        assert mask.all()

    def test_bad_pid_fails(self, cuts_cfg):
        """Ctime inside a random window but bad PID → fails."""
        arrays = _make_arrays(ctime=43.2, hcer_npe=0.0)
        mask = build_random_mask(arrays, cuts_cfg, self._CTMEAN, self._REAL_HW, self._BBN)
        assert not mask.any()

    def test_real_ctime_not_selected(self, cuts_cfg):
        """Real peak ctime is not in any random window."""
        arrays = _make_arrays(ctime=self._CTMEAN)
        mask = build_random_mask(arrays, cuts_cfg, self._CTMEAN, self._REAL_HW, self._BBN)
        assert not mask.any()


# ===========================================================================
# Computed quantities
# ===========================================================================

class TestComputeZhad:

    def test_known_value(self):
        arrays = _make_arrays(p_pi=3.0, nu=6.0)
        expected = math.sqrt(3.0**2 + M_PI**2) / 6.0
        result = compute_zhad(arrays)
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_shape_preserved(self):
        arrays = _make_arrays(n=7)
        result = compute_zhad(arrays)
        assert result.shape == (7,)

    def test_varies_with_ppi(self):
        p_values = np.array([1.0, 2.0, 3.0, 4.0])
        arrays = _make_arrays(n=4)
        arrays[BRANCH_PPi] = p_values
        arrays[BRANCH_NU] = np.full(4, 6.0)
        result = compute_zhad(arrays)
        expected = np.sqrt(p_values**2 + M_PI**2) / 6.0
        np.testing.assert_allclose(result, expected, rtol=1e-10)


class TestComputePt:

    def test_known_value(self):
        arrays = _make_arrays(p_pi=3.0, theta_pq=0.1)
        expected = 3.0 * math.sin(0.1)
        result = compute_Pt(arrays)
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_zero_angle(self):
        arrays = _make_arrays(p_pi=3.0, theta_pq=0.0)
        result = compute_Pt(arrays)
        np.testing.assert_allclose(result, 0.0, atol=1e-15)

    def test_shape_preserved(self):
        arrays = _make_arrays(n=5)
        assert compute_Pt(arrays).shape == (5,)

    def test_varies_with_angle(self):
        angles = np.array([0.0, 0.1, 0.2, 0.3])
        arrays = _make_arrays(n=4)
        arrays[BRANCH_THETA_PQ] = angles
        arrays[BRANCH_PPi] = np.full(4, 2.0)
        result = compute_Pt(arrays)
        np.testing.assert_allclose(result, 2.0 * np.sin(angles), rtol=1e-10)

"""
Tests for rsidis_ssa.asymmetry

All tests use synthetic boost_histogram objects — no ROOT files required.
"""

from __future__ import annotations

import math
import numpy as np
import pytest
import boost_histogram as bh

from rsidis_ssa.asymmetry import (
    AsymmetryResult,
    compute_asymmetry,
    compute_raw_asymmetry,
    find_asymmetry_pairs,
    fit_sinphi,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_hist(values: np.ndarray, variances: np.ndarray) -> bh.Histogram:
    """Build a 1-D Weight-storage histogram with explicit values and variances."""
    n = len(values)
    h = bh.Histogram(bh.axis.Regular(n, -math.pi, math.pi),
                     storage=bh.storage.Weight())
    h.view().value    = values
    h.view().variance = variances
    return h


def _phi_centers(n_bins: int = 16) -> np.ndarray:
    edges = np.linspace(-math.pi, math.pi, n_bins + 1)
    return 0.5 * (edges[:-1] + edges[1:])


# ===========================================================================
# compute_raw_asymmetry
# ===========================================================================

class TestComputeRawAsymmetry:

    def test_zero_asymmetry(self):
        """Equal yields → A = 0 in every bin."""
        n = 16
        vals = np.ones(n) * 100.0
        h_p = _make_hist(vals, vals)
        h_m = _make_hist(vals, vals)
        _, A, dA = compute_raw_asymmetry(h_p, h_m)
        np.testing.assert_allclose(A, 0.0, atol=1e-14)

    def test_full_asymmetry_plus(self):
        """h_minus = 0 → A = +1 in all bins."""
        n = 16
        vals = np.ones(n) * 50.0
        h_p = _make_hist(vals, vals)
        h_m = _make_hist(np.zeros(n), np.zeros(n))
        _, A, _ = compute_raw_asymmetry(h_p, h_m)
        np.testing.assert_allclose(A, 1.0, atol=1e-14)

    def test_full_asymmetry_minus(self):
        """h_plus = 0 → A = -1 in all bins."""
        n = 16
        vals = np.ones(n) * 50.0
        h_p = _make_hist(np.zeros(n), np.zeros(n))
        h_m = _make_hist(vals, vals)
        _, A, _ = compute_raw_asymmetry(h_p, h_m)
        np.testing.assert_allclose(A, -1.0, atol=1e-14)

    def test_known_asymmetry_single_bin(self):
        """Manual check: W_+ = 60, W_- = 40 → A = 0.2."""
        n = 4
        vp = np.array([60.0, 0.0, 0.0, 0.0])
        vm = np.array([40.0, 0.0, 0.0, 0.0])
        h_p = _make_hist(vp, vp)
        h_m = _make_hist(vm, vm)
        _, A, dA = compute_raw_asymmetry(h_p, h_m)
        assert abs(A[0] - 0.2) < 1e-12
        # σ²_A = 4(60²×40 + 40²×60) / 100⁴ = 4×(144000+96000)/1e8 = 960000/1e8
        expected_var = 4.0 * (60.0**2 * 40.0 + 40.0**2 * 60.0) / 100.0**4
        assert abs(dA[0]**2 - expected_var) < 1e-15

    def test_empty_bins_are_nan(self):
        """Bins where W_+ + W_- = 0 should give NaN, not crash."""
        n = 4
        vp = np.array([10.0, 0.0, 5.0, 0.0])
        vm = np.array([10.0, 0.0, 5.0, 0.0])
        h_p = _make_hist(vp, vp)
        h_m = _make_hist(vm, vm)
        _, A, dA = compute_raw_asymmetry(h_p, h_m)
        assert np.isnan(A[1]) and np.isnan(A[3])
        assert not np.isnan(A[0]) and not np.isnan(A[2])

    def test_returns_phi_centers_matching_axis(self):
        n = 8
        h_p = _make_hist(np.ones(n), np.ones(n))
        h_m = _make_hist(np.ones(n), np.ones(n))
        phi, _, _ = compute_raw_asymmetry(h_p, h_m)
        expected = h_p.axes[0].centers
        np.testing.assert_allclose(phi, expected, rtol=1e-12)

    def test_uncertainty_zero_when_variances_zero(self):
        """If both variances are 0 (pure weight-1 fill), σ_A = 0."""
        n = 4
        vp = np.full(n, 5.0)
        vm = np.full(n, 3.0)
        h_p = _make_hist(vp, np.zeros(n))
        h_m = _make_hist(vm, np.zeros(n))
        _, _, dA = compute_raw_asymmetry(h_p, h_m)
        np.testing.assert_allclose(dA, 0.0, atol=1e-14)


# ===========================================================================
# fit_sinphi
# ===========================================================================

class TestFitSinphi:

    def _inject_signal(self, amplitude: float, n_bins: int = 16,
                       err_per_bin: float = 0.01) -> tuple:
        """Build noiseless A_phys = amplitude × sin(φ) arrays."""
        phi = _phi_centers(n_bins)
        A   = amplitude * np.sin(phi)
        dA  = np.full(n_bins, err_per_bin)
        return phi, A, dA

    def test_zero_amplitude_recovered(self):
        phi, A, dA = self._inject_signal(0.0)
        amp, amp_err, chi2_ndf, n = fit_sinphi(phi, A, dA)
        assert abs(amp) < 1e-12
        assert amp_err > 0
        assert n == 16

    def test_positive_amplitude_recovered(self):
        phi, A, dA = self._inject_signal(0.10, err_per_bin=0.001)
        amp, amp_err, _, _ = fit_sinphi(phi, A, dA)
        assert abs(amp - 0.10) < 1e-10

    def test_negative_amplitude_recovered(self):
        phi, A, dA = self._inject_signal(-0.05, err_per_bin=0.001)
        amp, amp_err, _, _ = fit_sinphi(phi, A, dA)
        assert abs(amp - (-0.05)) < 1e-10

    def test_chi2_ndf_near_zero_for_exact_signal(self):
        """A noiseless sin signal should give χ²/ndf ≈ 0."""
        phi, A, dA = self._inject_signal(0.08, err_per_bin=0.01)
        _, _, chi2_ndf, _ = fit_sinphi(phi, A, dA)
        assert chi2_ndf < 1e-18

    def test_fewer_than_two_valid_bins_returns_nan(self):
        phi = _phi_centers(4)
        A   = np.full(4, np.nan)
        dA  = np.full(4, np.nan)
        amp, amp_err, chi2_ndf, n = fit_sinphi(phi, A, dA)
        assert math.isnan(amp)
        assert math.isnan(amp_err)
        assert math.isnan(chi2_ndf)
        assert n == 0

    def test_one_valid_bin_returns_nan(self):
        phi = _phi_centers(4)
        A   = np.array([0.1, np.nan, np.nan, np.nan])
        dA  = np.array([0.01, np.nan, np.nan, np.nan])
        amp, _, _, n = fit_sinphi(phi, A, dA)
        assert math.isnan(amp)
        assert n == 1

    def test_skips_zero_error_bins(self):
        """Bins with dA = 0 are excluded from fit even if A is finite."""
        phi = _phi_centers(4)
        A   = np.array([0.1, 0.05, -0.1, -0.05])
        dA  = np.array([0.01, 0.0, 0.01, 0.01])  # bin 1 has dA=0
        _, _, _, n = fit_sinphi(phi, A, dA)
        assert n == 3

    def test_amplitude_err_positive(self):
        phi, A, dA = self._inject_signal(0.03)
        _, amp_err, _, _ = fit_sinphi(phi, A, dA)
        assert amp_err > 0

    def test_n_used_equals_valid_bins(self):
        phi = _phi_centers(8)
        A   = np.array([0.1, np.nan, 0.05, 0.0, -0.05, np.nan, -0.1, 0.0])
        dA  = np.array([0.01, np.nan, 0.01, 0.01, 0.01, np.nan, 0.01, 0.01])
        _, _, _, n = fit_sinphi(phi, A, dA)
        assert n == 6


# ===========================================================================
# compute_asymmetry
# ===========================================================================

class TestComputeAsymmetry:

    def _uniform_hists(self, n_bins: int = 16, amplitude: float = 0.05,
                       total_counts: float = 1000.0, P: float = 0.85):
        """
        Construct h_+ and h_- with injected sin(φ) asymmetry.

        N_+(φ) = N_0/2 × (1 + A_raw × sin(φ))
        N_-(φ) = N_0/2 × (1 − A_raw × sin(φ))
        where A_raw = amplitude × P  (so A_phys = amplitude after correction)
        """
        phi = _phi_centers(n_bins)
        A_raw = amplitude * P
        N0  = total_counts / n_bins
        Np  = N0 * (1.0 + A_raw * np.sin(phi))
        Nm  = N0 * (1.0 - A_raw * np.sin(phi))
        h_p = _make_hist(Np, Np)
        h_m = _make_hist(Nm, Nm)
        return h_p, h_m

    def test_returns_asymmetry_result(self):
        h_p, h_m = self._uniform_hists()
        r = compute_asymmetry(h_p, h_m, 0.85, "phipq")
        assert isinstance(r, AsymmetryResult)

    def test_histogram_name_stored(self):
        h_p, h_m = self._uniform_hists()
        r = compute_asymmetry(h_p, h_m, 0.85, "phipq")
        assert r.histogram_name == "phipq"

    def test_beam_polarization_stored(self):
        h_p, h_m = self._uniform_hists()
        r = compute_asymmetry(h_p, h_m, 0.85, "phipq")
        assert abs(r.beam_polarization - 0.85) < 1e-12

    def test_amplitude_recovered_from_injected_signal(self):
        """Injected A_phys = 0.10 should be recovered within 1e-8 (noiseless)."""
        h_p, h_m = self._uniform_hists(amplitude=0.10, P=0.85,
                                        total_counts=1e6)
        r = compute_asymmetry(h_p, h_m, 0.85, "phipq")
        assert abs(r.amplitude - 0.10) < 1e-8

    def test_zero_amplitude_gives_zero(self):
        h_p, h_m = self._uniform_hists(amplitude=0.0)
        r = compute_asymmetry(h_p, h_m, 0.85, "phipq")
        assert abs(r.amplitude) < 1e-12

    def test_A_phys_equals_A_raw_over_polarization(self):
        h_p, h_m = self._uniform_hists(amplitude=0.08, P=0.85)
        r = compute_asymmetry(h_p, h_m, 0.85, "phipq")
        valid = np.isfinite(r.A_raw)
        np.testing.assert_allclose(
            r.A_phys[valid],
            r.A_raw[valid] / 0.85,
            rtol=1e-12,
        )

    def test_n_bins_used_equals_nonempty_bins(self):
        h_p, h_m = self._uniform_hists(n_bins=16)
        r = compute_asymmetry(h_p, h_m, 0.85, "phipq")
        assert r.n_bins_used == 16

    def test_invalid_polarization_raises(self):
        h_p, h_m = self._uniform_hists()
        with pytest.raises(ValueError, match="beam_polarization"):
            compute_asymmetry(h_p, h_m, 0.0, "phipq")
        with pytest.raises(ValueError, match="beam_polarization"):
            compute_asymmetry(h_p, h_m, -0.5, "phipq")
        with pytest.raises(ValueError, match="beam_polarization"):
            compute_asymmetry(h_p, h_m, 1.01, "phipq")

    def test_chi2_ndf_near_zero_for_exact_signal(self):
        h_p, h_m = self._uniform_hists(amplitude=0.05, P=1.0,
                                        total_counts=1e8)
        r = compute_asymmetry(h_p, h_m, 1.0, "phipq")
        assert r.chi2_ndf < 1e-6

    def test_arrays_have_correct_length(self):
        n = 12
        h_p, h_m = self._uniform_hists(n_bins=n)
        r = compute_asymmetry(h_p, h_m, 0.85, "phipq")
        assert len(r.phi_centers) == n
        assert len(r.A_raw)       == n
        assert len(r.A_raw_err)   == n
        assert len(r.A_phys)      == n
        assert len(r.A_phys_err)  == n

    def test_optional_counts_are_stored(self):
        h_p, h_m = self._uniform_hists(n_bins=4)
        N_plus = np.array([10, 20, 30, 40])
        N_minus = np.array([11, 21, 31, 41])
        r = compute_asymmetry(h_p, h_m, 0.85, "phipq", N_plus, N_minus)
        np.testing.assert_array_equal(r.N_plus, N_plus)
        np.testing.assert_array_equal(r.N_minus, N_minus)


# ===========================================================================
# find_asymmetry_pairs
# ===========================================================================

class TestFindAsymmetryPairs:

    def test_finds_standard_pair(self):
        names = ["phipq", "phipq_hplus", "phipq_hminus", "hsdelta"]
        pairs = find_asymmetry_pairs(names)
        assert len(pairs) == 1
        base, hp, hm = pairs[0]
        assert base == "phipq"
        assert hp   == "phipq_hplus"
        assert hm   == "phipq_hminus"

    def test_multiple_pairs(self):
        names = [
            "phipq_hplus", "phipq_hminus",
            "zhad_hplus",  "zhad_hminus",
            "hsdelta",
        ]
        pairs = find_asymmetry_pairs(names)
        bases = [p[0] for p in pairs]
        assert sorted(bases) == ["phipq", "zhad"]

    def test_unpaired_hplus_ignored(self):
        names = ["phipq_hplus", "hsdelta"]
        pairs = find_asymmetry_pairs(names)
        assert pairs == []

    def test_unpaired_hminus_ignored(self):
        names = ["phipq_hminus", "hsdelta"]
        pairs = find_asymmetry_pairs(names)
        assert pairs == []

    def test_no_duplicates_from_repeated_names(self):
        names = ["phipq_hplus", "phipq_hminus",
                 "phipq_hplus", "phipq_hminus"]
        pairs = find_asymmetry_pairs(names)
        assert len(pairs) == 1

    def test_empty_list(self):
        assert find_asymmetry_pairs([]) == []

    def test_sorted_by_base_name(self):
        names = ["z_hplus", "z_hminus", "a_hplus", "a_hminus", "m_hplus", "m_hminus"]
        pairs = find_asymmetry_pairs(names)
        bases = [p[0] for p in pairs]
        assert bases == sorted(bases)

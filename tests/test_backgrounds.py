"""
Tests for rsidis_ssa.backgrounds

All tests use synthetic boost_histogram objects filled with known values
so expected outputs can be computed analytically.
"""

from __future__ import annotations

import numpy as np
import boost_histogram as bh
import pytest

from rsidis_ssa.backgrounds import (
    SubtractionResult,
    subtract_dummy,
    subtract_eplus,
    subtract_randoms,
    subtraction_summary,
)


# ===========================================================================
# Helpers
# ===========================================================================

def _make_hist(values: list[float], variances: list[float] | None = None) -> bh.Histogram:
    """
    Build a 1-D Weight-storage histogram from explicit bin values/variances.
    Bins cover [0, len(values)) with unit width; overflow bins are empty.
    """
    n = len(values)
    h = bh.Histogram(bh.axis.Regular(n, 0.0, float(n)), storage=bh.storage.Weight())
    # Fill each bin by inserting one event per bin with the desired weight.
    # For Weight storage: fill(x, weight=w) → sum_w += w, sum_w2 += w²
    # To set arbitrary (value, variance) pairs we manipulate the view directly.
    var_list = variances if variances is not None else [abs(v) for v in values]
    view = h.view(flow=True)  # shape (n+2,): underflow, n bins, overflow
    view["value"][1:-1]    = values
    view["variance"][1:-1] = var_list
    return h


def _uniform(n: int, value: float, variance: float | None = None) -> bh.Histogram:
    """All bins equal to *value* with variance *variance* (defaults to value)."""
    return _make_hist(
        [value] * n,
        [variance if variance is not None else value] * n,
    )


# ===========================================================================
# subtract_randoms
# ===========================================================================

class TestSubtractRandoms:

    def test_subtracts_scaled_values(self):
        # real: 10 per bin, random: 6 per bin, scale=0.5 → result: 10 - 3 = 7
        h_real   = _uniform(4, 10.0)
        h_random = _uniform(4,  6.0)
        result = subtract_randoms(h_real, h_random, scale=0.5)
        np.testing.assert_allclose(result.values(), [7.0] * 4)

    def test_scale_1_is_simple_subtraction(self):
        h_real   = _uniform(4, 10.0)
        h_random = _uniform(4,  3.0)
        result = subtract_randoms(h_real, h_random, scale=1.0)
        np.testing.assert_allclose(result.values(), [7.0] * 4)

    def test_variance_propagated(self):
        # Var(result) = Var(real) + scale² × Var(random)
        # With Var(real)=4, Var(random)=9, scale=2: Var(result) = 4 + 4*9 = 40
        h_real   = _make_hist([10.0], [4.0])
        h_random = _make_hist([3.0],  [9.0])
        result = subtract_randoms(h_real, h_random, scale=2.0)
        expected_var = 4.0 + 2.0**2 * 9.0  # = 40
        np.testing.assert_allclose(result.variances(), [expected_var])

    def test_returns_new_histogram(self):
        h_real   = _uniform(3, 10.0)
        h_random = _uniform(3,  2.0)
        result = subtract_randoms(h_real, h_random, scale=1.0)
        assert result is not h_real
        assert result is not h_random

    def test_inputs_not_mutated(self):
        h_real   = _uniform(3, 10.0)
        h_random = _uniform(3,  2.0)
        original_real = h_real.values().copy()
        subtract_randoms(h_real, h_random, scale=1.0)
        np.testing.assert_array_equal(h_real.values(), original_real)

    def test_zero_scale_raises(self):
        h = _uniform(3, 5.0)
        with pytest.raises(ValueError, match="scale must be > 0"):
            subtract_randoms(h, h, scale=0.0)

    def test_negative_scale_raises(self):
        h = _uniform(3, 5.0)
        with pytest.raises(ValueError, match="scale must be > 0"):
            subtract_randoms(h, h, scale=-1.0)

    def test_partial_subtraction(self):
        # real=[10, 8, 6], random=[3, 2, 1], scale=1/3 → result=[9, 7.33, 5.67]
        h_real   = _make_hist([10.0, 8.0, 6.0])
        h_random = _make_hist([ 3.0, 2.0, 1.0])
        result = subtract_randoms(h_real, h_random, scale=1.0 / 3.0)
        np.testing.assert_allclose(result.values(), [9.0, 7.333, 5.667], rtol=1e-3)

    def test_typical_window_ratio(self):
        """Real ±2 ns / random ±6 ns → scale = 2/6."""
        scale = 2.0 / 6.0
        h_real   = _uniform(5, 100.0)
        h_random = _uniform(5,  30.0)
        result = subtract_randoms(h_real, h_random, scale=scale)
        expected = 100.0 - scale * 30.0
        np.testing.assert_allclose(result.values(), [expected] * 5, rtol=1e-10)


# ===========================================================================
# subtract_eplus
# ===========================================================================

class TestSubtractEplus:

    def test_subtracts_eplus(self):
        h_signal = _uniform(4, 10.0)
        h_eplus  = _uniform(4,  3.0)
        result = subtract_eplus(h_signal, h_eplus)
        np.testing.assert_allclose(result.values(), [7.0] * 4)

    def test_variance_propagated(self):
        # Var(result) = Var(signal) + Var(eplus)
        h_signal = _make_hist([10.0], [4.0])
        h_eplus  = _make_hist([ 3.0], [2.0])
        result = subtract_eplus(h_signal, h_eplus)
        np.testing.assert_allclose(result.variances(), [6.0])

    def test_returns_new_histogram(self):
        h_signal = _uniform(3, 10.0)
        h_eplus  = _uniform(3,  2.0)
        result = subtract_eplus(h_signal, h_eplus)
        assert result is not h_signal
        assert result is not h_eplus

    def test_inputs_not_mutated(self):
        h_signal = _uniform(3, 10.0)
        h_eplus  = _uniform(3,  2.0)
        original = h_signal.values().copy()
        subtract_eplus(h_signal, h_eplus)
        np.testing.assert_array_equal(h_signal.values(), original)

    def test_zero_eplus(self):
        h_signal = _uniform(4, 10.0)
        h_eplus  = _uniform(4,  0.0, variance=0.0)
        result = subtract_eplus(h_signal, h_eplus)
        np.testing.assert_allclose(result.values(), [10.0] * 4)


# ===========================================================================
# subtract_dummy
# ===========================================================================

class TestSubtractDummy:

    def test_subtracts_dummy(self):
        h_data  = _uniform(4, 10.0)
        h_dummy = _uniform(4,  2.0)
        result = subtract_dummy(h_data, h_dummy)
        np.testing.assert_allclose(result.values(), [8.0] * 4)

    def test_variance_propagated(self):
        h_data  = _make_hist([10.0], [5.0])
        h_dummy = _make_hist([ 2.0], [3.0])
        result = subtract_dummy(h_data, h_dummy)
        np.testing.assert_allclose(result.variances(), [8.0])

    def test_returns_new_histogram(self):
        h_data  = _uniform(3, 10.0)
        h_dummy = _uniform(3,  2.0)
        result = subtract_dummy(h_data, h_dummy)
        assert result is not h_data
        assert result is not h_dummy

    def test_inputs_not_mutated(self):
        h_data  = _uniform(3, 10.0)
        h_dummy = _uniform(3,  2.0)
        original = h_data.values().copy()
        subtract_dummy(h_data, h_dummy)
        np.testing.assert_array_equal(h_data.values(), original)


# ===========================================================================
# SubtractionResult
# ===========================================================================

class TestSubtractionResult:

    def _make_result(
        self,
        before_vals=(10.0, 10.0, 10.0),
        bg_vals=(2.0, 2.0, 2.0),
        scale=1.0,
        label="random",
    ) -> SubtractionResult:
        h_before     = _make_hist(list(before_vals))
        h_background = _make_hist(list(bg_vals))
        h_after      = h_before + h_background * (-scale)
        return SubtractionResult(
            label=label,
            histogram_name="test_hist",
            h_before=h_before,
            h_background=h_background,
            h_after=h_after,
            scale=scale,
        )

    # ---- fraction_subtracted ----

    def test_fraction_subtracted_correct(self):
        r = self._make_result(
            before_vals=(10.0, 10.0),
            bg_vals=(2.0, 2.0),
            scale=1.0,
        )
        # before total = 20, after total = 16, fraction = 4/20 = 0.2
        assert r.fraction_subtracted == pytest.approx(0.2)

    def test_fraction_subtracted_zero_background(self):
        r = self._make_result(bg_vals=(0.0, 0.0, 0.0), scale=1.0)
        assert r.fraction_subtracted == pytest.approx(0.0)

    def test_fraction_subtracted_empty_before(self):
        h_empty = _make_hist([0.0, 0.0], [0.0, 0.0])
        h_bg    = _make_hist([1.0, 1.0])
        h_after = subtract_eplus(h_empty, h_bg)
        r = SubtractionResult("eplus", "h", h_empty, h_bg, h_after, scale=1.0)
        assert r.fraction_subtracted == pytest.approx(0.0)

    def test_fraction_with_scale(self):
        # before=10, bg=6, scale=0.5 → after = 10 - 3 = 7; fraction = 3/10 = 0.3
        r = self._make_result(
            before_vals=(10.0,),
            bg_vals=(6.0,),
            scale=0.5,
        )
        assert r.fraction_subtracted == pytest.approx(0.3)

    # ---- pulls ----

    def test_pulls_shape(self):
        r = self._make_result()
        assert r.pulls.shape == (3,)

    def test_pulls_zero_when_no_background(self):
        r = self._make_result(bg_vals=(0.0, 0.0, 0.0))
        np.testing.assert_allclose(r.pulls, [0.0, 0.0, 0.0])

    def test_pulls_formula(self):
        # bg_scaled = 2.0 * 1.0 = 2.0; var_before = 10.0; pull = 2/sqrt(10)
        r = self._make_result(
            before_vals=(10.0,),
            bg_vals=(2.0,),
            scale=1.0,
        )
        expected = 2.0 / np.sqrt(10.0)
        np.testing.assert_allclose(r.pulls, [expected], rtol=1e-10)

    def test_pulls_with_scale(self):
        # bg_scaled = 6.0 * 0.5 = 3.0; var_before = 10.0; pull = 3/sqrt(10)
        r = self._make_result(
            before_vals=(10.0,),
            bg_vals=(6.0,),
            scale=0.5,
        )
        expected = 3.0 / np.sqrt(10.0)
        np.testing.assert_allclose(r.pulls, [expected], rtol=1e-10)

    def test_pulls_zero_variance_returns_zero(self):
        h_before     = _make_hist([0.0], [0.0])
        h_background = _make_hist([2.0], [2.0])
        h_after      = subtract_eplus(h_before, h_background)
        r = SubtractionResult("eplus", "h", h_before, h_background, h_after)
        np.testing.assert_allclose(r.pulls, [0.0])

    # ---- max_abs_pull ----

    def test_max_abs_pull(self):
        # Two bins: pulls = [2/sqrt(10), 1/sqrt(5)]
        h_before     = _make_hist([10.0, 5.0])
        h_background = _make_hist([ 2.0, 1.0])
        h_after      = subtract_eplus(h_before, h_background)
        r = SubtractionResult("eplus", "h", h_before, h_background, h_after)
        expected_max = max(2.0 / np.sqrt(10.0), 1.0 / np.sqrt(5.0))
        assert r.max_abs_pull == pytest.approx(expected_max)


# ===========================================================================
# subtraction_summary
# ===========================================================================

class TestSubtractionSummary:

    def _make_result(self, label, hname, frac, max_pull=0.5):
        before = 100.0
        bg     = before * frac
        h_before     = _uniform(5, before / 5.0)
        h_background = _uniform(5, bg / 5.0, variance=(bg / 5.0) * 0.01)
        h_after      = subtract_randoms(h_before, h_background, scale=1.0)
        return SubtractionResult(label, hname, h_before, h_background, h_after, scale=1.0)

    def test_summary_is_string(self):
        results = [self._make_result("random", "phipq", 0.10)]
        s = subtraction_summary(results)
        assert isinstance(s, str)

    def test_summary_contains_label(self):
        results = [self._make_result("random", "phipq", 0.10)]
        assert "random" in subtraction_summary(results)

    def test_summary_contains_histogram_name(self):
        results = [self._make_result("eplus", "zhad", 0.15)]
        assert "zhad" in subtraction_summary(results)

    def test_summary_contains_percentage(self):
        results = [self._make_result("random", "phipq", 0.10)]
        s = subtraction_summary(results)
        assert "%" in s

    def test_summary_multiple_steps(self):
        results = [
            self._make_result("random", "phipq", 0.08),
            self._make_result("eplus",  "phipq", 0.12),
        ]
        s = subtraction_summary(results)
        assert "random" in s
        assert "eplus" in s

    def test_empty_results(self):
        s = subtraction_summary([])
        assert isinstance(s, str)
        assert "summary" in s.lower()

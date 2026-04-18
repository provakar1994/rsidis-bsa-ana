"""
Helicity asymmetry calculation and fitting.

Phase 6 — not yet implemented.

Design notes (for Phase 6):
  - effective_helicity(hel_array, ihwp) -> np.ndarray
      Applies IHWP sign flip: hel * (-1 if ihwp == 'IN' else +1).
      This must be applied per-run at fill time, not after summing.

  - compute_asymmetry(h_plus, h_minus) -> tuple[np.ndarray, np.ndarray]
      Returns (A, dA) where:
          A   = (N+ - N-) / (N+ + N-)
          dA² = 4 * (N-² * σ+² + N+² * σ-²) / (N+ + N-)⁴
      Uses boost_histogram .counts() and .variances().

  - fit_sin(phi_centers, A, dA) -> FitResult
      Fits A(φ) = A_LT * sin(φ) via scipy.optimize.curve_fit with
      absolute_sigma=True.  Returns a FitResult dataclass with A_LT,
      its uncertainty, chi²/ndf, and the covariance matrix.
"""

raise NotImplementedError("asymmetry module is not yet implemented (Phase 6)")

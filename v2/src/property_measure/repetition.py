"""Weft Inner PAM v2 — repetition structure measurement (spec §4.4).

For a trajectory of length T, the self-similarity matrix is S_ij = cos(x_i, x_j).
Repetition structure is a pair (P, F):

  * Period P  — the off-diagonal offset k maximising mean_i[S_{i, i+k}] over
    k ∈ [1, T/2]. P undefined if no off-diagonal mean exceeds the noise floor.
  * Fidelity F — the mean cosine at the dominant period, mean_i[S_{i, i+P}].

Variant — coverage: the fraction of positions i with S_{i, i+P} > τ_R, i.e.
the fraction of the trajectory participating in the dominant pattern.

The autocorrelation profile m(k) = mean_i cos(x_i, x_{i+k}) is computed directly
(O(T·d) per lag); block-wise/lag-capped for long trajectories (§6.2 step 1).
The dominant period is the smallest lag whose m(k) is within `peak_tol` of the
profile maximum — this skips the continuity auto-correlation lobe at small k
(which decays below the peak) and recovers the fundamental rather than a
harmonic multiple of it.
"""

from __future__ import annotations

import numpy as np


def autocorrelation_profile(stream: np.ndarray, max_lag: int) -> np.ndarray:
    """Return m(k) = mean_i cos(x_i, x_{i+k}) for k = 1 .. max_lag."""
    x = stream / np.clip(np.linalg.norm(stream, axis=1, keepdims=True), 1e-12, None)
    T = x.shape[0]
    max_lag = int(min(max_lag, T - 1))
    m = np.empty(max_lag + 1, dtype=np.float64)
    m[0] = 1.0
    for k in range(1, max_lag + 1):
        m[k] = float(np.mean(np.sum(x[:-k] * x[k:], axis=1)))
    return m


def measure_repetition(
    stream: np.ndarray,
    tau_R: float,
    noise_floor: float,
    max_lag: int | None = None,
    peak_tol: float = 0.02,
) -> dict:
    """Return repetition structure {'period', 'fidelity', 'coverage', ...} (§4.4)."""
    if stream.ndim != 2 or stream.shape[0] < 4:
        raise ValueError(f"stream must be (T>=4, d); got {stream.shape}")
    x = stream / np.clip(np.linalg.norm(stream, axis=1, keepdims=True), 1e-12, None)
    T = x.shape[0]
    if max_lag is None:
        max_lag = T // 2
    m = autocorrelation_profile(x, max_lag)
    profile = m[1:]                                    # lags 1..max_lag
    lags = np.arange(1, profile.size + 1)
    peak = float(profile.max())

    if peak <= noise_floor:
        return {
            "period": None,
            "fidelity": float(peak),
            "coverage": 0.0,
            "peak_offdiag_mean": peak,
            "max_lag": int(max_lag),
        }

    # Smallest lag within peak_tol of the maximum = the fundamental period.
    candidates = lags[profile >= peak - peak_tol]
    period = int(candidates.min())
    fidelity = float(m[period])

    # Coverage: fraction of positions with cos(x_i, x_{i+P}) > τ_R.
    if period < T:
        cos_at_P = np.sum(x[:-period] * x[period:], axis=1)
        coverage = float(np.mean(cos_at_P > tau_R))
    else:
        coverage = 0.0

    return {
        "period": period,
        "fidelity": fidelity,
        "coverage": coverage,
        "peak_offdiag_mean": peak,
        "max_lag": int(max_lag),
    }

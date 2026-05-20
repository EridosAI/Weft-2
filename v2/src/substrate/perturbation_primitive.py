"""Weft Inner PAM v2 — perturbation primitive (spec §5.3).

Applied on top of a §5.2 repetition trajectory. At designated repetitions
k* ∈ K_pert and positions i* ∈ I_pert within each perturbed repetition, the
trajectory value is shifted:

    x_{kP + i} = y_i + ν_{k,i} + δ_{k,i} · 1[k ∈ K_pert ∧ i ∈ I_pert]

where δ_{k,i} is a random-isotropic shift vector (spec §5.3 pass-2 default)
controlling §4.1 magnitude M. Locality (§4.2) is controlled by the spread of
I_pert relative to the period P — supplied by the caller (the stream builder).

Magnitude calibration: for unit-norm reference y and isotropic δ of relative
norm ρ_M ⊥ y, cos(y, y + δ) ≈ 1 / sqrt(1 + ρ_M²). Targeting magnitude
M = 1 − cos gives ρ_M = sqrt(1/(1−M)² − 1). The measured M (§4.1) is verified
against the target in PRE-A.

Parameter-to-axis mapping (spec §5.3):
  * ||δ|| (§4.1 magnitude)            ← ρ_M from the target M.
  * |I_pert| / P (inverse §4.2)       ← caller-chosen perturbed-position set.
"""

from __future__ import annotations

import numpy as np


def magnitude_to_rel_shift(magnitude_M: float) -> float:
    """Relative shift norm ρ_M for a target perturbation magnitude M.

    M = 0 → ρ_M = 0 (bit-identical, control C3); M → 1 → ρ_M → ∞ (orthogonal).
    """
    if not (0.0 <= magnitude_M < 1.0):
        raise ValueError(f"magnitude_M must be in [0, 1); got {magnitude_M}")
    if magnitude_M == 0.0:
        return 0.0
    return float(np.sqrt(1.0 / (1.0 - magnitude_M) ** 2 - 1.0))


def apply_perturbation(
    stream: np.ndarray,
    period_P: int,
    perturbed_reps: np.ndarray,
    pert_positions: np.ndarray,
    magnitude_M: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Add isotropic shifts to (rep, position) pairs; return (stream, S_pert).

    Args:
        stream: (R*P, d) tiled trajectory from §5.2.
        period_P: P.
        perturbed_reps: array of repetition indices k ∈ [0, R) to perturb.
        pert_positions: array of within-period indices i ∈ [0, P) (= I_pert).
        magnitude_M: target §4.1 magnitude in [0, 1).
        seed: RNG seed for the shift directions.

    Returns:
        (perturbed_stream (R*P, d) re-normalised, S_pert sorted int array of
        absolute perturbed indices).
    """
    if stream.ndim != 2:
        raise ValueError(f"stream must be (N, d); got {stream.shape}")
    N, d = stream.shape
    rho_M = magnitude_to_rel_shift(magnitude_M)
    rng = np.random.default_rng(seed)

    out = stream.astype(np.float64).copy()
    s_pert: list[int] = []
    for k in np.asarray(perturbed_reps, dtype=np.int64):
        for i in np.asarray(pert_positions, dtype=np.int64):
            idx = int(k) * period_P + int(i)
            if idx >= N:
                continue
            s_pert.append(idx)
            if rho_M > 0:
                delta = rng.standard_normal(d)
                delta = rho_M * delta / np.linalg.norm(delta)   # isotropic, norm ρ_M
                out[idx] = out[idx] + delta

    norms = np.linalg.norm(out, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    out = out / norms
    return out, np.array(sorted(s_pert), dtype=np.int64)

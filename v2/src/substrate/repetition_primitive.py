"""Weft Inner PAM v2 — repetition primitive (spec §5.2).

A trajectory composed of a base loop segment {y_1, ..., y_P} repeated R times
with controlled fidelity:

    x_{kP + i} = y_i + ν_{k,i}

for k ∈ [0, R) and i ∈ [1, P], where ν_{k,i} is per-repetition isotropic noise
whose magnitude controls §4.4 fidelity F. The base segment y is drawn from the
§5.1 manifold-trajectory primitive at the desired manifold dimensionality and
continuity.

Parameter-to-axis mapping (spec §5.2):
  * period P (§4.4)              ← base segment length.
  * R·P                          ← trajectory length.
  * ν magnitude (§4.4 fidelity)  ← per-repetition noise.

Fidelity calibration: for unit-norm y_i with independent isotropic ν of relative
norm ρ_F ⊥ y, cos(y_i + ν_{k1}, y_i + ν_{k2}) ≈ 1 / (1 + ρ_F²). Targeting
fidelity F gives ρ_F = sqrt(1/F − 1). The measured F (§4.4, mean cosine at the
dominant period) is verified against the target in PRE-A.
"""

from __future__ import annotations

import numpy as np


def fidelity_to_rel_noise(fidelity_F: float) -> float:
    """Relative per-repetition noise norm ρ_F for a target fidelity F.

    F = 1 → ρ_F = 0 (bit-identical repetition); F → 0 → ρ_F → ∞.
    """
    if not (0.0 < fidelity_F <= 1.0):
        raise ValueError(f"fidelity_F must be in (0, 1]; got {fidelity_F}")
    return float(np.sqrt(1.0 / fidelity_F - 1.0))


def tile_with_fidelity(
    base_segment: np.ndarray,
    n_repetitions: int,
    fidelity_F: float,
    seed: int,
) -> np.ndarray:
    """Tile `base_segment` R times with per-repetition fidelity noise.

    Args:
        base_segment: (P, d) unit-normalised base loop from §5.1.
        n_repetitions: R.
        fidelity_F: target §4.4 fidelity in (0, 1].
        seed: RNG seed for the per-repetition noise.

    Returns:
        (R*P, d) float64 stream, each row re-normalised to unit norm.
    """
    if base_segment.ndim != 2:
        raise ValueError(f"base_segment must be (P, d); got {base_segment.shape}")
    P, d = base_segment.shape
    if n_repetitions < 1:
        raise ValueError(f"n_repetitions must be >= 1; got {n_repetitions}")

    rho_F = fidelity_to_rel_noise(fidelity_F)
    rng = np.random.default_rng(seed)

    stream = np.tile(base_segment, (n_repetitions, 1)).astype(np.float64)  # (R*P, d)
    if rho_F > 0:
        # Isotropic Gaussian scaled to per-vector relative norm ρ_F (norm of a
        # d-dim N(0, s²I) vector ≈ s·sqrt(d); set s = ρ_F / sqrt(d)).
        noise = (rho_F / np.sqrt(d)) * rng.standard_normal(stream.shape)
        stream = stream + noise

    norms = np.linalg.norm(stream, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return stream / norms

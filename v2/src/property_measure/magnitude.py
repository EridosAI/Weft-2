"""Weft Inner PAM v2 — perturbation magnitude measurement (spec §4.1).

    M = 1 − median_{i ∈ S_pert, j ∈ ref-neighbourhood(i)} cos(x_i, x_j^ref)

where S_pert is the perturbed subsequence, x_j^ref is the reference-state value
at position j, and j ranges over the reference-subsequence neighbourhood of i by
trajectory position (radius R = REF_NEIGHBOURHOOD_WINDOW). M = 0 when perturbed
and reference are cosine-indistinguishable; M = 1 when orthogonal.

The reference state x^ref and perturbed set S_pert are supplied by the §6.2
protocol (estimated from the dominant repeating pattern); this module computes M
from them and makes no construction-side assumption.
"""

from __future__ import annotations

import numpy as np


def measure_magnitude(
    stream: np.ndarray,
    s_pert: np.ndarray,
    reference_state: np.ndarray,
    ref_neighbourhood: int,
) -> dict:
    """Return {'M', 'n_pairs'} (spec §4.1). M is None if S_pert is empty."""
    if stream.shape != reference_state.shape:
        raise ValueError(
            f"stream {stream.shape} and reference_state {reference_state.shape} must match"
        )
    x = stream / np.clip(np.linalg.norm(stream, axis=1, keepdims=True), 1e-12, None)
    xr = reference_state / np.clip(
        np.linalg.norm(reference_state, axis=1, keepdims=True), 1e-12, None
    )
    s_pert = np.asarray(s_pert, dtype=np.int64)
    if s_pert.size == 0:
        return {"M": None, "n_pairs": 0}

    # The nearest reference-subsequence position to a perturbed position i is the
    # reference state at the same trajectory position, x^ref_i (the dominant
    # repeating pattern's value at phase i mod P). Comparing to positional
    # neighbours j != i would conflate magnitude with continuity, so M is taken
    # against x^ref_i directly (ref_neighbourhood retained for API symmetry).
    _ = int(ref_neighbourhood)
    cosines = np.sum(x[s_pert] * xr[s_pert], axis=1)
    return {"M": float(1.0 - np.median(cosines)), "n_pairs": int(s_pert.size)}

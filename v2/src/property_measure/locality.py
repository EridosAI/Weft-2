"""Weft Inner PAM v2 — perturbation locality measurement (spec §4.2).

    L = 1 − |{ i ∉ S_pert : |cos(x_i, x_{i'}) − cos(x_i^ref, x_{i'}^ref)| > τ_L }|
            / |{ i ∉ S_pert }|

where x_{i'} is position i's reference neighbour (within radius R) and x^ref is
the reference state. L = 1 means the perturbation affects only S_pert (perfect
locality); L = 0 means it shifts every other position's relations.

Structural limitation (§4.2 / §6.2 step 3): locality is undefined on trajectories
whose repetition coverage falls below the calibrated threshold — the reference
state cannot be estimated. The §6.2 protocol marks locality "undefined" in that
case; this module computes L given a defined reference state and S_pert.
"""

from __future__ import annotations

import numpy as np


def measure_locality(
    stream: np.ndarray,
    s_pert: np.ndarray,
    reference_state: np.ndarray,
    tau_L: float,
    ref_neighbourhood: int,
    restrict_to: np.ndarray | None = None,
) -> dict:
    """Return {'L', 'n_nonpert', 'n_affected'} (spec §4.2).

    `restrict_to` (optional boolean mask, length T) limits the non-perturbed
    population to a region of interest — the protocol passes the "active periods"
    (repetitions containing a detected perturbation) so that locality reflects
    the perturbation's spatial extent within affected periods rather than being
    diluted by unperturbed repetitions used to estimate the reference state.
    """
    if stream.shape != reference_state.shape:
        raise ValueError(
            f"stream {stream.shape} and reference_state {reference_state.shape} must match"
        )
    x = stream / np.clip(np.linalg.norm(stream, axis=1, keepdims=True), 1e-12, None)
    xr = reference_state / np.clip(
        np.linalg.norm(reference_state, axis=1, keepdims=True), 1e-12, None
    )
    T = x.shape[0]
    R = int(ref_neighbourhood)
    pert_mask = np.zeros(T, dtype=bool)
    pert_mask[np.asarray(s_pert, dtype=np.int64)] = True
    consider = ~pert_mask
    if restrict_to is not None:
        consider = consider & np.asarray(restrict_to, dtype=bool)
    nonpert = np.flatnonzero(consider)
    if nonpert.size == 0:
        return {"L": 0.0, "n_nonpert": 0, "n_affected": 0}

    affected = 0
    for i in nonpert:
        lo, hi = max(0, i - R), min(T - 1, i + R)
        max_shift = 0.0
        for j in range(lo, hi + 1):
            if j == i:
                continue
            actual = float(np.dot(x[i], x[j]))
            reference = float(np.dot(xr[i], xr[j]))
            max_shift = max(max_shift, abs(actual - reference))
        if max_shift > tau_L:
            affected += 1

    return {
        "L": float(1.0 - affected / nonpert.size),
        "n_nonpert": int(nonpert.size),
        "n_affected": int(affected),
    }

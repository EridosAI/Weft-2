"""Weft Inner PAM v2 — measurement protocol (spec §6.2, the 7-step procedure).

Applies the §4 property definitions to a single empirical trajectory, producing
a per-trajectory record on each of the five axes. The protocol makes no
construction-side assumption — it is the methodology's portable Output 2,
applicable to any (env, enc) pair's trajectory (§6.1).

Procedure (§6.2), for a trajectory τ:
  1. Self-similarity / autocorrelation profile (block-wise / lag-capped for long τ).
  2. Repetition detection — dominant period P, fidelity F, coverage (§4.4).
  3. Reference-state estimation — x^ref_i = median over repetitions at phase i mod P.
     If repetition coverage < threshold, reference is undefined and locality is
     marked "undefined" (§4.2 structural limitation).
  4. Perturbation detection — S_pert = { i : ||x_i − x^ref_i|| > τ_pert }.
  5. Magnitude and locality from S_pert and the reference state (§4.1, §4.2).
  6. Continuity C and C_curv (§4.3).
  7. Manifold dimensionality D_local, D_global (§4.5).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from v2.config import load_calibrated_thresholds
from v2.src.property_measure.continuity import measure_continuity
from v2.src.property_measure.locality import measure_locality
from v2.src.property_measure.magnitude import measure_magnitude
from v2.src.property_measure.manifold_dim import measure_manifold_dim
from v2.src.property_measure.repetition import measure_repetition


def _normalise(stream: np.ndarray) -> np.ndarray:
    return stream / np.clip(np.linalg.norm(stream, axis=1, keepdims=True), 1e-12, None)


def estimate_reference_state(stream: np.ndarray, period_P: int) -> np.ndarray:
    """x^ref_i = component-wise median over repetitions at phase (i mod P), renormalised.

    The median over repetitions recovers the unperturbed baseline as long as the
    perturbation occupies a minority of repetitions at each phase (spec §6.2 step 3).
    """
    N, d = stream.shape
    ref_phase = np.empty((period_P, d), dtype=np.float64)
    for phi in range(period_P):
        idx = np.arange(phi, N, period_P)
        ref_phase[phi] = np.median(stream[idx], axis=0)
    ref_phase = _normalise(ref_phase)
    return ref_phase[np.arange(N) % period_P]


def detect_perturbations(
    stream: np.ndarray, reference_state: np.ndarray, tau_pert: float
) -> np.ndarray:
    """S_pert = { i : ||x_i − x^ref_i|| > τ_pert } (spec §6.2 step 4)."""
    residual = np.linalg.norm(stream - reference_state, axis=1)
    return np.flatnonzero(residual > tau_pert).astype(np.int64)


def apply_protocol(
    stream: np.ndarray,
    thresholds: Optional[dict] = None,
    max_lag: Optional[int] = None,
) -> dict:
    """Run the §6.2 7-step protocol on one trajectory; return a per-trajectory record."""
    if stream.ndim != 2:
        raise ValueError(f"stream must be (T, d); got {stream.shape}")
    th = thresholds if thresholds is not None else load_calibrated_thresholds()
    x = _normalise(stream.astype(np.float64))
    T = x.shape[0]
    if max_lag is None:
        max_lag = min(T // 2, 2000)

    record: dict = {"trajectory_length": int(T)}

    # Steps 1-2: self-similarity + repetition detection.
    rep = measure_repetition(
        x,
        tau_R=th["TAU_R"],
        noise_floor=th["REPETITION_NOISE_FLOOR"],
        max_lag=max_lag,
    )
    record.update(
        {
            "period_P": rep["period"],
            "fidelity_F": rep["fidelity"],
            "repetition_coverage": rep["coverage"],
            "peak_offdiag_mean": rep["peak_offdiag_mean"],
        }
    )

    # Step 3: reference-state estimation (gated on repetition presence + coverage).
    period_P = rep["period"]
    coverage_ok = (
        period_P is not None
        and rep["coverage"] >= th["REPETITION_COVERAGE_THRESHOLD"]
    )
    reference_state = None
    s_pert = np.array([], dtype=np.int64)
    if period_P is not None and period_P >= 1 and T // period_P >= 2:
        reference_state = estimate_reference_state(x, period_P)
        # Step 4: perturbation detection.
        s_pert = detect_perturbations(x, reference_state, th["TAU_PERT"])
    record["n_perturbed_detected"] = int(s_pert.size)

    # Step 5: magnitude and locality.
    if reference_state is not None:
        mag = measure_magnitude(
            x, s_pert, reference_state, ref_neighbourhood=th["REF_NEIGHBOURHOOD_WINDOW"]
        )
        record["magnitude_M"] = mag["M"]
    else:
        record["magnitude_M"] = None

    if reference_state is not None and coverage_ok and s_pert.size > 0:
        # Active periods: repetitions containing a detected perturbation. Restrict
        # locality to these so it reflects spatial extent within affected periods
        # rather than being diluted by the clean repetitions (§4.2).
        active_reps = np.unique(s_pert // period_P)
        active_mask = np.isin(np.arange(T) // period_P, active_reps)
        loc = measure_locality(
            x,
            s_pert,
            reference_state,
            tau_L=th["TAU_L"],
            ref_neighbourhood=th["REF_NEIGHBOURHOOD_WINDOW"],
            restrict_to=active_mask,
        )
        record["locality_L"] = loc["L"]
        record["locality_defined"] = True
    else:
        record["locality_L"] = None
        record["locality_defined"] = coverage_ok and reference_state is not None

    # Step 6: continuity.
    cont = measure_continuity(x)
    record["continuity_C"] = cont["C"]
    record["continuity_C_curv"] = cont["C_curv"]

    # Step 7: manifold dimensionality.
    md = measure_manifold_dim(
        x, window=th["LOCAL_PCA_WINDOW"], subsample_rate=th["MANIFOLD_SUBSAMPLE_RATE"]
    )
    record["manifold_D_local"] = md["D_local"]
    record["manifold_D_global"] = md["D_global"]

    return record

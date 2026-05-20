"""Weft Inner PAM v2 — stream composition helper (spec §5.1-5.3).

Composes the three construction primitives into a single synthetic stream with
controlled values on all five §4 axes, and records the construction-side
"ground truth" each axis is verified against in PRE-A.

This is a thin composition layer over base_manifold_trajectory / repetition /
perturbation; it adds no new substrate mechanism. Locality is realised here:
the perturbed positions I_pert are spread over an extent E = round((1-L)·P)
with spacing ~= the reference-neighbourhood radius R, so that the §4.2
relation-shift footprint of the perturbation ≈ its extent (giving a locality
measurement that tracks the construction value).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from v2.config import (
    DEFAULT_N_REPETITIONS,
    EPS_NOISE_STD,
    REF_NEIGHBOURHOOD_WINDOW,
    SUBSTRATE_SEED,
    TAU_L,
)
from v2.src.property_measure.locality import measure_locality
from v2.src.substrate.base_manifold_trajectory import (
    analytic_continuity,
    base_loop_segment,
    centered_harmonics,
)
from v2.src.substrate.perturbation_primitive import apply_perturbation
from v2.src.substrate.repetition_primitive import tile_with_fidelity


@dataclass
class StreamParams:
    """Construction parameters for one synthetic stream (the five §4 axes)."""

    period_P: int = 256                 # §4.4 period
    manifold_dim_D: int = 16            # §4.5 dimensionality (distinct harmonics)
    continuity_center: int = 24         # central harmonic → §4.3 continuity
    fidelity_F: float = 0.97            # §4.4 fidelity
    magnitude_M: float = 0.10           # §4.1 perturbation magnitude
    locality_L: float = 0.90            # §4.2 locality (1 − extent/P)
    n_repetitions: int = DEFAULT_N_REPETITIONS
    eps_std: float = EPS_NOISE_STD
    seed: int = SUBSTRATE_SEED
    # Which repetitions carry the perturbation. Default: the middle third, so the
    # majority of repetitions are clean and the §6.2 reference-state estimate
    # (median over repetitions) recovers the unperturbed baseline.
    perturbed_reps: Optional[np.ndarray] = None


@dataclass
class BuiltStream:
    """A constructed stream plus its construction-side ground truth."""

    stream: np.ndarray                  # (N, d) L2-normalised
    params: StreamParams
    harmonics: np.ndarray
    construction: dict = field(default_factory=dict)  # per-axis construction values
    s_pert: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    i_pert: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))


def _locality_positions(period_P: int, locality_L: float, radius: int) -> np.ndarray:
    """I_pert positions spread over extent E = round((1-L)·P), spacing ~= radius.

    Spacing <= the neighbourhood radius makes the perturbation's relation-shift
    footprint ≈ its extent E, so the §4.2 measurement tracks L = 1 − E/P.
    """
    locality_L = float(np.clip(locality_L, 0.0, 1.0))
    extent = int(round((1.0 - locality_L) * period_P))
    extent = max(1, min(extent, period_P))
    step = max(1, radius)
    start = (period_P - extent) // 2
    positions = np.arange(start, start + extent, step, dtype=np.int64)
    if positions.size == 0:
        positions = np.array([period_P // 2], dtype=np.int64)
    return positions


def build_stream(params: StreamParams, U: np.ndarray) -> BuiltStream:
    """Build a synthetic stream from `params` using the §5.1-5.3 primitives."""
    harmonics = centered_harmonics(
        params.manifold_dim_D, params.period_P, params.continuity_center
    )
    base = base_loop_segment(
        P=params.period_P,
        D=params.manifold_dim_D,
        center=params.continuity_center,
        U=U,
        seed=params.seed,
        eps_std=params.eps_std,
    )
    tiled = tile_with_fidelity(
        base_segment=base,
        n_repetitions=params.n_repetitions,
        fidelity_F=params.fidelity_F,
        seed=params.seed + 1,
    )

    R = params.n_repetitions
    if params.perturbed_reps is None:
        perturbed_reps = np.arange(R // 3, max(R // 3 + 1, (2 * R) // 3), dtype=np.int64)
    else:
        perturbed_reps = np.asarray(params.perturbed_reps, dtype=np.int64)

    i_pert = _locality_positions(
        params.period_P, params.locality_L, REF_NEIGHBOURHOOD_WINDOW
    )
    stream, s_pert = apply_perturbation(
        stream=tiled,
        period_P=params.period_P,
        perturbed_reps=perturbed_reps,
        pert_positions=i_pert,
        magnitude_M=params.magnitude_M,
        seed=params.seed + 2,
    )

    # Ground-truth locality: the §4.2 locality the stream actually has, computed
    # with the TRUE reference (pre-perturbation tiled stream) and TRUE perturbed
    # set, over the active periods. The dial 1 − extent/P only approximates this
    # (the dial→property map is nonlinear); PRE-A verifies the protocol — which
    # estimates the reference and perturbed set — recovers this ground truth.
    if params.magnitude_M > 0.0 and s_pert.size > 0:
        active_reps = np.unique(s_pert // params.period_P)
        active_mask = np.isin(np.arange(stream.shape[0]) // params.period_P, active_reps)
        gt_loc = measure_locality(
            stream, s_pert, reference_state=tiled, tau_L=TAU_L,
            ref_neighbourhood=REF_NEIGHBOURHOOD_WINDOW, restrict_to=active_mask,
        )["L"]
    else:
        gt_loc = 1.0 - i_pert_extent_fraction(i_pert, params.period_P)

    construction = {
        "period_P": params.period_P,
        "manifold_dim_D": params.manifold_dim_D,
        "continuity_C": analytic_continuity(harmonics, params.period_P),
        "fidelity_F": params.fidelity_F,
        "magnitude_M": params.magnitude_M,
        "locality_L": gt_loc,
        "locality_L_dial": 1.0 - i_pert_extent_fraction(i_pert, params.period_P),
        "n_perturbed_positions_per_rep": int(i_pert.size),
        "n_perturbed_reps": int(perturbed_reps.size),
    }
    return BuiltStream(
        stream=stream,
        params=params,
        harmonics=harmonics,
        construction=construction,
        s_pert=s_pert,
        i_pert=i_pert,
    )


def i_pert_extent_fraction(i_pert: np.ndarray, period_P: int) -> float:
    """Extent (max-min span, inclusive) of I_pert as a fraction of P."""
    if i_pert.size == 0:
        return 0.0
    extent = int(i_pert.max() - i_pert.min()) + 1
    return float(extent) / float(period_P)

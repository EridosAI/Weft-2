"""V2-PRE-A unit tests — construction primitives (spec §5.1-5.3)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from v2.config import D_AMBIENT
from v2.src.substrate.base_manifold_trajectory import (
    analytic_continuity,
    base_loop_segment,
    centered_harmonics,
    load_or_create_U,
)
from v2.src.substrate.perturbation_primitive import (
    apply_perturbation,
    magnitude_to_rel_shift,
)
from v2.src.substrate.repetition_primitive import (
    fidelity_to_rel_noise,
    tile_with_fidelity,
)
from v2.src.substrate.stream_builder import StreamParams, build_stream


def test_U_reproducible_and_orthogonal():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "U.npy"
        U1 = load_or_create_U(seed=0, path=path, ambient=64)
        assert U1.shape == (64, 64)
        # Orthogonal: U U^T == I.
        assert np.abs(U1 @ U1.T - np.eye(64)).max() < 1e-10
        # Reload from disk is identical.
        U2 = load_or_create_U(seed=0, path=path, ambient=64)
        assert np.array_equal(U1, U2)
    # Fresh generation with the same seed is deterministic.
    with tempfile.TemporaryDirectory() as d:
        U3 = load_or_create_U(seed=0, path=Path(d) / "U.npy", ambient=64)
    assert np.allclose(U1, U3)


def test_base_loop_segment_shape_norm_and_closure():
    U = load_or_create_U(seed=0, path=Path(tempfile.mkdtemp()) / "U.npy", ambient=128)
    P, D = 50, 8
    seg = base_loop_segment(P=P, D=D, center=4, U=U, seed=7, eps_std=0.0)
    assert seg.shape == (P, 128)
    norms = np.linalg.norm(seg, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-6)
    # Closure: the loop closes — extrapolating one period returns to the start.
    # The harmonic phase at t=P equals t=0 (integer cycles), so seg would repeat.
    seg2 = base_loop_segment(P=P, D=D, center=4, U=U, seed=7, eps_std=0.0)
    assert np.array_equal(seg, seg2)  # deterministic given seed


def test_centered_harmonics_distinct_and_feasibility():
    h = centered_harmonics(D=8, P=256, center=24)
    assert h.shape == (8,)
    assert len(set(h.tolist())) == 8           # distinct
    assert h.min() >= 1 and h.max() <= 128      # within [1, P//2]
    # Infeasible request raises (D distinct harmonics cannot fit in [1, P//2]).
    try:
        centered_harmonics(D=200, P=256, center=24)
        assert False, "expected ValueError for D > P//2"
    except ValueError:
        pass


def test_analytic_continuity_monotone_in_center():
    P = 256
    c_low = analytic_continuity(centered_harmonics(8, P, 4), P)
    c_mid = analytic_continuity(centered_harmonics(8, P, 24), P)
    c_high = analytic_continuity(centered_harmonics(8, P, 60), P)
    assert 0.0 <= c_low < c_mid < c_high <= 2.0


def test_fidelity_and_magnitude_inversions():
    # F = 1/(1+ρ²)  ->  ρ = sqrt(1/F - 1)
    assert abs(fidelity_to_rel_noise(1.0) - 0.0) < 1e-12
    assert abs(fidelity_to_rel_noise(0.5) - 1.0) < 1e-12
    # M = 1 - 1/sqrt(1+ρ²)  ->  ρ = sqrt(1/(1-M)² - 1)
    assert abs(magnitude_to_rel_shift(0.0) - 0.0) < 1e-12
    rho = magnitude_to_rel_shift(0.5)
    assert abs((1.0 - 1.0 / np.sqrt(1.0 + rho ** 2)) - 0.5) < 1e-9


def test_tile_shape_and_perturbation_indices():
    U = load_or_create_U(seed=0, path=Path(tempfile.mkdtemp()) / "U.npy", ambient=128)
    seg = base_loop_segment(P=20, D=4, center=2, U=U, seed=7, eps_std=0.0)
    tiled = tile_with_fidelity(seg, n_repetitions=5, fidelity_F=0.99, seed=1)
    assert tiled.shape == (100, 128)
    assert np.allclose(np.linalg.norm(tiled, axis=1), 1.0, atol=1e-6)
    stream, s_pert = apply_perturbation(
        tiled, period_P=20, perturbed_reps=np.array([2]),
        pert_positions=np.array([5, 8, 11]), magnitude_M=0.5, seed=3,
    )
    assert stream.shape == (100, 128)
    # Perturbed absolute indices = rep*P + position.
    assert set(s_pert.tolist()) == {2 * 20 + 5, 2 * 20 + 8, 2 * 20 + 11}


def test_build_stream_satisfies_norm_contract():
    U = load_or_create_U()  # full 1024-d (persisted)
    bs = build_stream(StreamParams(), U)
    assert bs.stream.shape[1] == D_AMBIENT
    norms = np.linalg.norm(bs.stream, axis=1)
    assert np.all(np.abs(norms - 1.0) < 1e-5)   # v1 trainer stream contract

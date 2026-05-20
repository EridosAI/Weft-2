"""V2-PRE-A unit tests — property measurement on hand-crafted streams (spec §4).

Each test constructs a stream whose property value is known analytically and
checks the §4 measurement recovers it.
"""

from __future__ import annotations

import numpy as np

from v2.src.property_measure.continuity import measure_continuity
from v2.src.property_measure.locality import measure_locality
from v2.src.property_measure.magnitude import measure_magnitude
from v2.src.property_measure.manifold_dim import measure_manifold_dim
from v2.src.property_measure.repetition import measure_repetition


def _great_circle(alpha: float, T: int, d: int = 64, seed: int = 0) -> np.ndarray:
    """x_t = cos(αt)·a + sin(αt)·b for orthonormal a, b.

    cos(x_t, x_{t+1}) = cos(α) exactly, so continuity C = 1 − cos(α).
    """
    rng = np.random.default_rng(seed)
    a = rng.standard_normal(d)
    a /= np.linalg.norm(a)
    b = rng.standard_normal(d)
    b -= (b @ a) * a
    b /= np.linalg.norm(b)
    t = np.arange(T)[:, None]
    return np.cos(alpha * t) * a[None, :] + np.sin(alpha * t) * b[None, :]


def test_continuity_great_circle():
    for alpha in (0.05, 0.3, 1.0):
        x = _great_circle(alpha, T=500)
        out = measure_continuity(x)
        assert abs(out["C"] - (1.0 - np.cos(alpha))) < 1e-3
        # Uniform-speed great circle => near-zero curvature spread.
        assert out["C_curv"] < 1e-3


def test_repetition_perfect_period():
    # Spec §7.1 example: a periodic stream of length 100 with period 10 should
    # yield P=10, F≈1.0.
    rng = np.random.default_rng(1)
    base = rng.standard_normal((10, 64))
    base /= np.linalg.norm(base, axis=1, keepdims=True)
    stream = np.tile(base, (10, 1))                    # length 100, period 10
    out = measure_repetition(stream, tau_R=0.6, noise_floor=0.3, max_lag=50)
    assert out["period"] == 10
    assert out["fidelity"] > 0.999
    assert out["coverage"] > 0.999


def test_repetition_none_when_no_structure():
    rng = np.random.default_rng(2)
    stream = rng.standard_normal((300, 128))
    stream /= np.linalg.norm(stream, axis=1, keepdims=True)
    out = measure_repetition(stream, tau_R=0.6, noise_floor=0.3, max_lag=150)
    # Random high-dim vectors have ~0 off-diagonal mean => below noise floor.
    assert out["period"] is None


def test_manifold_dim_subspace():
    # Points spanning a k-dim subspace with equal variance => participation ratio ≈ k.
    rng = np.random.default_rng(3)
    for k in (2, 5, 10):
        basis = np.linalg.qr(rng.standard_normal((64, k)))[0]   # (64, k) orthonormal
        coeffs = rng.standard_normal((400, k))
        x = coeffs @ basis.T
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        out = measure_manifold_dim(x, window=40, subsample_rate=4)
        assert abs(out["D_global"] - k) < 0.2 * k + 0.5


def test_magnitude_known_shift():
    # Reference = unit vectors; perturbed = ref + δ (δ ⊥ ref, norm ρ).
    rng = np.random.default_rng(4)
    d, T = 256, 60
    ref = rng.standard_normal((T, d))
    ref /= np.linalg.norm(ref, axis=1, keepdims=True)
    for M in (0.1, 0.5):
        rho = np.sqrt(1.0 / (1.0 - M) ** 2 - 1.0)
        stream = ref.copy()
        s_pert = np.arange(10, 40)
        for i in s_pert:
            delta = rng.standard_normal(d)
            delta -= (delta @ ref[i]) * ref[i]            # orthogonalise to ref
            delta = rho * delta / np.linalg.norm(delta)
            stream[i] = ref[i] + delta
        out = measure_magnitude(stream, s_pert, ref, ref_neighbourhood=3)
        assert abs(out["M"] - M) < 0.02


def test_locality_perfect_when_no_neighbours_affected():
    # Reference == stream except a single perturbed position; with a reference
    # equal to stream everywhere (no perturbation), no relations shift => L = 1.
    rng = np.random.default_rng(5)
    x = rng.standard_normal((100, 64))
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    out = measure_locality(x, s_pert=np.array([], dtype=int), reference_state=x,
                           tau_L=0.05, ref_neighbourhood=3)
    assert out["L"] == 1.0

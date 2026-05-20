"""Weft Inner PAM v2 — base manifold trajectory primitive (spec §5.1).

    x_t = U · φ(θ_t) + ε_t

A closed-loop smooth path on a D-dimensional manifold embedded in R^1024 via a
fixed random orthogonal projection U (spec §5.5; persisted, reloaded across all
v2 runs). The parameterisation φ is a sinusoidal lift with D distinct integer
harmonics:

    c_t[k] = sin(2π · m_k · t / P + ψ_k),   k = 0 .. D-1,   t = 0 .. P-1
    x_t    = U[:, :D] · c_t           (then per-vector L2-normalised)

Because each harmonic m_k is an integer number of cycles over the period P, the
segment closes (c_P = c_0): the base segment is a loop, matching v0/v1's
loop-based exploration policy.

Parameter-to-axis mapping (spec §5.1):
  * manifold dimensionality D (§4.5)  ← number of distinct harmonics. Distinct
    frequencies are required for local-PCA rank > 2 (phase-shifted copies of one
    frequency span only a 2-D subspace locally).
  * continuity C (§4.3)               ← central harmonic frequency. With the D
    harmonics centred at `center`, E[cos(x_t, x_{t+1})] ≈ cos(2π·center/P), so
    C ≈ 1 − cos(2π·center/P). Centring keeps C ~independent of D.
  * period P (§4.4)                   ← loop length.

U is generated once (QR of a Gaussian, sign-fixed for determinism) and persisted
to `v2/data/embedding_U.npy`. Dimensionality D uses the first D columns U[:, :D]
(orthonormal R^D → R^1024).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from v2.config import (
    D_AMBIENT,
    EMBEDDING_U_PATH,
    EMBEDDING_U_SEED,
    EPS_NOISE_STD,
    SUBSTRATE_SEED,
)


def load_or_create_U(
    seed: int = EMBEDDING_U_SEED,
    path: Path = EMBEDDING_U_PATH,
    ambient: int = D_AMBIENT,
) -> np.ndarray:
    """Return the fixed random orthogonal embedding U ∈ R^{ambient×ambient}.

    Generated once (QR of a seeded Gaussian, sign-fixed so the factorisation is
    deterministic), persisted to `path`, and reloaded on subsequent calls
    (spec §5.5 reproducibility commitment).
    """
    path = Path(path)
    if path.exists():
        U = np.load(path)
        if U.shape != (ambient, ambient):
            raise ValueError(
                f"persisted U at {path} has shape {U.shape}; expected "
                f"{(ambient, ambient)} — delete to regenerate."
            )
        return U
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((ambient, ambient))
    Q, R = np.linalg.qr(A)
    # Sign-fix: make the diagonal of R non-negative so QR is unique/deterministic.
    Q = Q * np.sign(np.diag(R))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, Q)
    return Q


def centered_harmonics(D: int, P: int, center: int) -> np.ndarray:
    """Return D distinct positive integer harmonics centred at `center`.

    Harmonics lie in [1, P//2] (above Nyquist they alias). Centring the band
    keeps the mean frequency — and hence continuity — ~independent of D.

    Raises if D distinct harmonics cannot fit in [1, P//2] (the manifold
    dimensionality requested exceeds what period P can host); the caller
    surfaces this as a sweep-grid feasibility note (instr §7.1).
    """
    max_h = P // 2
    if D < 1:
        raise ValueError(f"D must be >= 1; got {D}")
    if D > max_h:
        raise ValueError(
            f"cannot fit D={D} distinct harmonics in [1, P//2={max_h}] for P={P}; "
            f"need P >= 2*D. Reduce D or increase P (sweep-grid feasibility, instr §7.1)."
        )
    lo = int(round(center)) - D // 2
    # Shift the contiguous block so it fits within [1, max_h].
    lo = max(1, min(lo, max_h - D + 1))
    return np.arange(lo, lo + D, dtype=np.int64)


def analytic_continuity(harmonics: np.ndarray, P: int) -> float:
    """Analytic continuity C = 1 − mean_k cos(2π m_k / P) (spec §4.3).

    Exact for the un-noised, equal-amplitude harmonic loop in the large-D /
    concentration limit; the measured C (§4.3) is verified against it in PRE-A.
    """
    return float(1.0 - np.mean(np.cos(2.0 * np.pi * harmonics / P)))


def base_loop_segment(
    P: int,
    D: int,
    center: int,
    U: np.ndarray,
    seed: int = SUBSTRATE_SEED,
    eps_std: float = EPS_NOISE_STD,
) -> np.ndarray:
    """Return one closed-loop base segment of shape (P, ambient), L2-normalised.

    Args:
        P: loop length (period). Also the §4.4 period.
        D: manifold dimensionality (number of distinct harmonics).
        center: central harmonic (sets continuity).
        U: embedding matrix from `load_or_create_U`.
        seed: phase / noise seed (SCAFFOLDING).
        eps_std: per-position noise magnitude (spec §5.1 ε_t).

    Returns:
        (P, ambient) float64 array; each row L2-normalised to unit norm so the
        downstream stream satisfies the v1 trainer's contract (norms 1 ± 1e-5).
        All §4 cosine-based properties are invariant to this per-vector scaling.
    """
    ambient = U.shape[0]
    rng = np.random.default_rng(seed)
    harmonics = centered_harmonics(D, P, center)               # (D,)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=D)             # (D,)

    t = np.arange(P)[:, None]                                   # (P, 1)
    theta = 2.0 * np.pi * harmonics[None, :] * t / P + phases[None, :]  # (P, D)
    coords = np.sin(theta)                                      # φ(θ_t), (P, D)

    x = coords @ U[:, :D].T                                     # (P, ambient)
    if eps_std > 0:
        x = x + eps_std * rng.standard_normal(x.shape)

    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms

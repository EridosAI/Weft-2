"""Weft Inner PAM v2 — manifold dimensionality measurement (spec §4.5).

Effective dimensionality of the trajectory's local neighbourhood structure, via
the participation ratio of local-PCA eigenvalues:

    D = E_t[ (Σ_k λ_k^{(t)})² / Σ_k (λ_k^{(t)})² ]

where λ_k^{(t)} are eigenvalues of the covariance of trajectory positions in a
local window around t. Variant — global dimensionality: participation ratio of
the full-trajectory covariance.

Implementation: for a window of w positions in d=1024 space (w << d), the
non-zero covariance eigenvalues equal those of the (w×w) Gram matrix of the
centred window, so the participation ratio is computed from the small Gram
matrix (O(w²d + w³) per window) rather than the (d×d) covariance. The global
estimate uses the singular values of the centred trajectory.
"""

from __future__ import annotations

import numpy as np


def _participation_ratio(eigvals: np.ndarray) -> float:
    """(Σλ)² / Σλ² over the non-negative eigenvalues."""
    lam = np.clip(eigvals, 0.0, None)
    s1 = lam.sum()
    s2 = np.square(lam).sum()
    if s2 <= 0:
        return 0.0
    return float(s1 * s1 / s2)


def _local_pr(window_vectors: np.ndarray) -> float:
    """Participation ratio of the local window via its centred Gram matrix."""
    Xc = window_vectors - window_vectors.mean(axis=0, keepdims=True)
    gram = Xc @ Xc.T                                    # (w, w), same non-zero spectrum as cov
    eigvals = np.linalg.eigvalsh(gram)
    return _participation_ratio(eigvals)


def measure_manifold_dim(
    stream: np.ndarray,
    window: int,
    subsample_rate: int,
) -> dict:
    """Return {'D_local', 'D_global', 'n_windows'} (spec §4.5)."""
    if stream.ndim != 2:
        raise ValueError(f"stream must be (T, d); got {stream.shape}")
    T, d = stream.shape
    window = int(min(window, T))
    half = window // 2

    centres = range(half, T - half, max(1, subsample_rate))
    prs = [_local_pr(stream[c - half : c - half + window]) for c in centres]
    d_local = float(np.mean(prs)) if prs else 0.0

    # Global: participation ratio of singular-value² of the centred trajectory.
    Xc = stream - stream.mean(axis=0, keepdims=True)
    sv = np.linalg.svd(Xc, full_matrices=False, compute_uv=False)
    d_global = _participation_ratio(sv ** 2)

    return {"D_local": d_local, "D_global": d_global, "n_windows": len(prs)}

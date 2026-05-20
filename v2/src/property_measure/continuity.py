"""Weft Inner PAM v2 — continuity / smoothness measurement (spec §4.3).

    C       = 1 − E_t[cos(x_t, x_{t+1})]
    C_curv  = std_t[1 − cos(x_t, x_{t+1})]

Both are direct computations on consecutive trajectory positions; no reference
state or perturbation detection required (§4.3 measurability note). Inputs are
L2-normalised, so cos(x_t, x_{t+1}) is the dot product.
"""

from __future__ import annotations

import numpy as np


def measure_continuity(stream: np.ndarray) -> dict:
    """Return {'C', 'C_curv'} for a trajectory (spec §4.3)."""
    if stream.ndim != 2 or stream.shape[0] < 2:
        raise ValueError(f"stream must be (T>=2, d); got {stream.shape}")
    # Per-vector L2-normalise defensively so cos == dot even if caller passed raw.
    x = stream / np.clip(np.linalg.norm(stream, axis=1, keepdims=True), 1e-12, None)
    cos_consec = np.sum(x[:-1] * x[1:], axis=1)        # (T-1,)
    dist = 1.0 - cos_consec
    return {"C": float(np.mean(dist)), "C_curv": float(np.std(dist))}

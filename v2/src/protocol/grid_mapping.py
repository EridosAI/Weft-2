"""Weft Inner PAM v2 — protocol-to-map mapping (spec §6.3).

The protocol (§6.2) produces a per-trajectory value on each axis; aggregated
over a trajectory collection this is a distribution. Mapping to the map's
discrete grid points requires summarisation:

  * Multi-modal detection — fit a two-component Gaussian mixture and compare its
    BIC to a single Gaussian. If the BIC improvement exceeds a calibrated
    threshold (SCAFFOLDING; calibrated at PRE-E), the axis is flagged multi-modal
    and the two component medians are reported separately; otherwise median + IQR.
  * Nearest-grid-point lookup — per axis, the nearest grid value with the
    interpolation distance reported (spec §6.3 commits to option (i)).

A dependency-free 1-D two-component GMM (EM) is used so the module does not
require scikit-learn.
"""

from __future__ import annotations

import numpy as np

_LOG_2PI = float(np.log(2.0 * np.pi))


def _gaussian_loglik(values: np.ndarray, mu: float, var: float) -> np.ndarray:
    var = max(var, 1e-12)
    return -0.5 * (_LOG_2PI + np.log(var) + (values - mu) ** 2 / var)


def _bic(loglik: float, n: int, n_params: int) -> float:
    return n_params * np.log(max(n, 1)) - 2.0 * loglik


def fit_two_component_gmm(
    values: np.ndarray, n_iter: int = 200, tol: float = 1e-6, seed: int = 0
) -> dict:
    """Fit a 1-D two-component GMM by EM. Returns params + total log-likelihood."""
    v = np.asarray(values, dtype=np.float64).ravel()
    n = v.size
    rng = np.random.default_rng(seed)
    # Init: split at the median; means = sub-sample means.
    med = np.median(v)
    mu = np.array([v[v <= med].mean() if np.any(v <= med) else med - 1e-3,
                   v[v > med].mean() if np.any(v > med) else med + 1e-3])
    var = np.full(2, max(v.var(), 1e-6))
    w = np.array([0.5, 0.5])

    prev_ll = -np.inf
    for _ in range(n_iter):
        # E-step: responsibilities.
        log_resp = np.stack(
            [np.log(w[k] + 1e-12) + _gaussian_loglik(v, mu[k], var[k]) for k in range(2)],
            axis=1,
        )
        log_norm = np.logaddexp(log_resp[:, 0], log_resp[:, 1])
        ll = float(log_norm.sum())
        resp = np.exp(log_resp - log_norm[:, None])
        # M-step.
        nk = resp.sum(axis=0) + 1e-12
        w = nk / n
        mu = (resp * v[:, None]).sum(axis=0) / nk
        var = (resp * (v[:, None] - mu[None, :]) ** 2).sum(axis=0) / nk
        var = np.clip(var, 1e-9, None)
        if abs(ll - prev_ll) < tol:
            break
        prev_ll = ll

    return {"weights": w, "means": mu, "variances": var, "loglik": ll}


def detect_multimodal(values: np.ndarray, bic_improvement_threshold: float) -> dict:
    """Compare 2-component vs 1-component BIC (spec §6.3 multi-modal detection)."""
    v = np.asarray(values, dtype=np.float64).ravel()
    n = v.size
    if n < 4:
        # Too few samples to fit two components meaningfully.
        return {
            "multimodal": False,
            "bic_improvement": 0.0,
            "median": float(np.median(v)) if n else None,
            "iqr": float(np.subtract(*np.percentile(v, [75, 25]))) if n else None,
            "component_medians": None,
        }

    # Single Gaussian.
    mu1, var1 = float(v.mean()), float(max(v.var(), 1e-12))
    ll1 = float(_gaussian_loglik(v, mu1, var1).sum())
    bic1 = _bic(ll1, n, n_params=2)

    gmm = fit_two_component_gmm(v)
    bic2 = _bic(gmm["loglik"], n, n_params=5)
    improvement = float(bic1 - bic2)
    multimodal = improvement > bic_improvement_threshold

    result = {
        "multimodal": bool(multimodal),
        "bic_improvement": improvement,
        "median": float(np.median(v)),
        "iqr": float(np.subtract(*np.percentile(v, [75, 25]))),
        "component_medians": None,
    }
    if multimodal:
        # Hard-assign by responsibility and report per-component medians.
        order = np.argsort(gmm["means"])
        mu = gmm["means"][order]
        var = gmm["variances"][order]
        w = gmm["weights"][order]
        ll = np.stack(
            [np.log(w[k] + 1e-12) + _gaussian_loglik(v, mu[k], var[k]) for k in range(2)],
            axis=1,
        )
        assign = ll.argmax(axis=1)
        result["component_medians"] = [
            float(np.median(v[assign == k])) if np.any(assign == k) else float(mu[k])
            for k in range(2)
        ]
    return result


def nearest_grid_point(value: float, grid_values: np.ndarray) -> dict:
    """Nearest grid value to `value` and the interpolation distance (spec §6.3)."""
    grid = np.asarray(grid_values, dtype=np.float64)
    idx = int(np.argmin(np.abs(grid - value)))
    return {
        "nearest_value": float(grid[idx]),
        "nearest_index": idx,
        "interpolation_distance": float(abs(grid[idx] - value)),
    }

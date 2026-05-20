"""Shared utilities for v1 predictors and the path-prediction loss.

Spec §4.1 (loss); §7.1 (shared scaffolding, clamp constants); §7.5
(implementation discipline — three classes, one file each, shared
constants in this module).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from v1.src.config import LOG_VAR_CLAMP_MAX, LOG_VAR_CLAMP_MIN


def path_prediction_loss(
    mean: torch.Tensor,
    log_var: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Per-K Gaussian NLL averaged over K, mean over batch (spec §4.1).

    L_k(b) = 0.5 · (||target − mean||² / σ² + d · log σ²)
    L      = (Σ_k L_k) / K, then mean over batch

    Form 1 (per-K scalar isotropic) variance representation. Targets are
    detached at the call site (encoder is frozen anyway; explicit for
    clarity).

    Args:
        mean:    (B, K, d) predicted centreline.
        log_var: (B, K)    predicted per-K scalar log-variance.
        target:  (B, K, d) target embeddings.

    Returns:
        Scalar loss.
    """
    assert mean.ndim == 3, f"mean must be (B, K, d); got {tuple(mean.shape)}"
    assert log_var.ndim == 2, f"log_var must be (B, K); got {tuple(log_var.shape)}"
    assert target.shape == mean.shape, (
        f"target shape {tuple(target.shape)} != mean shape {tuple(mean.shape)}"
    )
    d = mean.shape[-1]
    sq_err = (target.detach() - mean).pow(2).sum(dim=-1)            # (B, K)
    inv_var = torch.exp(-log_var)                                    # (B, K)
    per_step = 0.5 * (sq_err * inv_var + d * log_var)                # (B, K)
    return per_step.mean(dim=1).mean(dim=0)


def clamp_log_var(log_var: torch.Tensor) -> torch.Tensor:
    """Apply LOG_VAR_CLAMP_MIN, LOG_VAR_CLAMP_MAX clamp from v1 config."""
    return log_var.clamp(LOG_VAR_CLAMP_MIN, LOG_VAR_CLAMP_MAX)


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def all_parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# Re-export so importers don't have to dual-import config and shared.
__all__ = [
    "LOG_VAR_CLAMP_MAX",
    "LOG_VAR_CLAMP_MIN",
    "all_parameter_count",
    "clamp_log_var",
    "path_prediction_loss",
    "trainable_parameter_count",
]

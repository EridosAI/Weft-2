"""Inner PAM predictor: window-of-W to K-step (mean, log-variance) path.

Architecture (instr §3.1, spec §3.3):
  input proj 1024 -> 512
  learned positional encoding over W=16 positions
  4-layer Transformer encoder (heads=8, dim_ff=2048, GELU, pre-LayerNorm)
  pool: last token (most recent frame)
  output proj 512 -> K*(d+1) -> reshape (K, d+1) -> split to (K, d), (K, 1)

Loss (spec §4.1):
  L_k = 0.5 * ( ||y_k - mu_k||^2 / sigma_k^2 + d * log sigma_k^2 )
  uniform across K, stop-gradient on targets, summed over K, mean over batch.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from v0.src.config import (
    EMBED_DIM,
    LOG_VAR_CLAMP_MAX,
    LOG_VAR_CLAMP_MIN,
    PRED_HEADS,
    PRED_HIDDEN,
    PRED_LAYERS,
    PRED_MLP_DIM,
    PREDICT_K,
    WINDOW_W,
)


class InnerPAM(nn.Module):
    def __init__(
        self,
        embed_dim: int = EMBED_DIM,
        window_w: int = WINDOW_W,
        predict_k: int = PREDICT_K,
        hidden: int = PRED_HIDDEN,
        n_layers: int = PRED_LAYERS,
        n_heads: int = PRED_HEADS,
        mlp_dim: int = PRED_MLP_DIM,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.window_w = window_w
        self.predict_k = predict_k
        self.hidden = hidden

        self.input_proj = nn.Linear(embed_dim, hidden)
        self.pos_emb = nn.Embedding(window_w, hidden)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=n_heads,
            dim_feedforward=mlp_dim,
            activation="gelu",
            norm_first=True,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_proj = nn.Linear(hidden, predict_k * (embed_dim + 1))

    def forward(self, window: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            window: (B, W, d) float tensor of W recent L2-normalised embeddings.
        Returns:
            mean: (B, K, d) predicted centreline.
            log_var: (B, K) predicted log-variance per step (clamped).
        """
        assert window.ndim == 3, f"window must be (B, W, d); got {tuple(window.shape)}"
        b, w, d = window.shape
        assert w == self.window_w, f"W mismatch: got {w}, expected {self.window_w}"
        assert d == self.embed_dim, f"d mismatch: got {d}, expected {self.embed_dim}"

        x = self.input_proj(window)                                 # (B, W, hidden)
        positions = torch.arange(self.window_w, device=window.device)
        x = x + self.pos_emb(positions).unsqueeze(0)                # (B, W, hidden)
        x = self.encoder(x)                                         # (B, W, hidden)
        last_token = x[:, -1, :]                                    # (B, hidden)
        flat = self.output_proj(last_token)                         # (B, K*(d+1))
        flat = flat.view(b, self.predict_k, self.embed_dim + 1)     # (B, K, d+1)
        mean = flat[..., : self.embed_dim]                          # (B, K, d)
        log_var = flat[..., self.embed_dim].clamp(
            LOG_VAR_CLAMP_MIN, LOG_VAR_CLAMP_MAX
        )                                                            # (B, K)

        assert mean.shape == (b, self.predict_k, self.embed_dim)
        assert log_var.shape == (b, self.predict_k)
        return mean, log_var


def gaussian_nll_loss(
    mean: torch.Tensor,
    log_var: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Per-step isotropic Gaussian NLL summed over K, mean over batch.

    L_k(b) = 0.5 * (||target - mean||^2 / sigma^2 + d * log sigma^2)
    Returns scalar loss with gradient flowing only into (mean, log_var); target is
    detached at the call site (encoder is frozen anyway; explicit for clarity).
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
    return per_step.sum(dim=1).mean(dim=0)


def confidence_from_log_var(
    log_var: torch.Tensor, m: int
) -> torch.Tensor:
    """Confidence = -mean log_var over the first M predicted steps (instr §5.1).

    Lower predicted variance -> higher confidence.
    """
    assert log_var.ndim == 2
    assert 1 <= m <= log_var.shape[1]
    return -log_var[:, :m].mean(dim=1)


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def all_parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

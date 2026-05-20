"""Weft Inner PAM v1 — Primary arm predictor.

Spec §7.2. K learnable output queries cross-attend into the encoder body's
W-positional output; per-K-position scalar log-variance.

Architecture (spec §7.2.2):
  input proj 1024 -> 512
  learned positional encoding over W=16 positions
  4-layer Transformer encoder (heads=8, dim_ff=2048, GELU, pre-LayerNorm)
  K learnable output queries (per-K parameters)
  TransformerDecoder with cross-attention into encoder output
  per-K output projection (mean: hidden -> d; log_var: hidden -> 1)

Loss (spec §4.1, predictor/inner_pam_v1_shared.path_prediction_loss):
  Form 1 (per-K scalar isotropic) Gaussian NLL.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from v1.src.config import (
    EMBED_DIM,
    OUTPUT_QUERY_INIT_STD,
    PRED_HEADS,
    PRED_HIDDEN,
    PRED_LAYERS,
    PRED_MLP_DIM,
    PREDICT_K,
    WINDOW_W,
)
from v1.src.predictor.inner_pam_v1_shared import clamp_log_var


class InnerPAM_v1_Primary(nn.Module):
    """Primary arm: K output queries + cross-attention decoder + per-K scalar log-var."""

    def __init__(
        self,
        embed_dim: int = EMBED_DIM,
        window_w: int = WINDOW_W,
        predict_k: int = PREDICT_K,
        hidden: int = PRED_HIDDEN,
        n_heads: int = PRED_HEADS,
        n_layers: int = PRED_LAYERS,
        mlp_dim: int = PRED_MLP_DIM,
        decoder_n_layers: int | None = None,
    ):
        super().__init__()
        # Spec §7.2.1: no silent default for decoder_n_layers.
        if decoder_n_layers is None:
            raise ValueError(
                "decoder_n_layers must be specified explicitly (SCAFFOLDING; "
                "calibrated by PRE-C per instr §6.3, written to config.py as "
                "V1_DECODER_N_LAYERS, not silently defaulted)."
            )

        self.embed_dim = embed_dim
        self.window_w = window_w
        self.predict_k = predict_k
        self.hidden = hidden
        self.decoder_n_layers = decoder_n_layers

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

        # K learnable output queries (v1 Primary-specific, per spec §7.2.2).
        # Per-K parameters: each row is independently learnable.
        self.output_queries = nn.Parameter(torch.empty(predict_k, hidden))
        nn.init.normal_(self.output_queries, std=OUTPUT_QUERY_INIT_STD)

        # Cross-attention decoder block (spec §7.2.2). Decoder layers each
        # contain self-attention + cross-attention into memory + FFN.
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden,
            nhead=n_heads,
            dim_feedforward=mlp_dim,
            activation="gelu",
            norm_first=True,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_n_layers)

        # Per-K output projection. Shared across K positions; per-K
        # differentiation is upstream in queries + cross-attention.
        self.output_proj_mean = nn.Linear(hidden, embed_dim)
        self.output_proj_log_var = nn.Linear(hidden, 1)

    def forward(self, window: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            window: (B, W, d) float tensor of W recent L2-normalised embeddings.

        Returns:
            mean:    (B, K, d) predicted centreline per K step.
            log_var: (B, K)    predicted scalar log-variance per K step (clamped).
        """
        assert window.ndim == 3, f"window must be (B, W, d); got {tuple(window.shape)}"
        b, w, d = window.shape
        assert w == self.window_w, f"W mismatch: got {w}, expected {self.window_w}"
        assert d == self.embed_dim, f"d mismatch: got {d}, expected {self.embed_dim}"

        # Input projection + positional embedding
        x = self.input_proj(window)                                # (B, W, hidden)
        positions = torch.arange(self.window_w, device=window.device)
        x = x + self.pos_emb(positions).unsqueeze(0)               # (B, W, hidden)

        # Encoder body produces W position-distinguished hidden vectors.
        memory = self.encoder(x)                                   # (B, W, hidden)

        # K learnable output queries expanded to batch.
        queries = self.output_queries.unsqueeze(0).expand(b, -1, -1)  # (B, K, hidden)

        # Cross-attention decoder: K queries attend into W memory positions.
        decoded = self.decoder(queries, memory)                    # (B, K, hidden)

        # Per-K output projection (parameters in queries + decoder give the
        # per-K differentiation; mean and log_var heads are shared linear layers).
        mean = self.output_proj_mean(decoded)                      # (B, K, d)
        log_var = self.output_proj_log_var(decoded).squeeze(-1)    # (B, K)
        log_var = clamp_log_var(log_var)

        assert mean.shape == (b, self.predict_k, self.embed_dim)
        assert log_var.shape == (b, self.predict_k)
        return mean, log_var

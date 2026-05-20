"""Weft Inner PAM v1 — Ablation 1 arm predictor (variance-head ablation).

Spec §7.3. Identical to Primary except the per-K log-variance head is
replaced with a single shared `nn.Parameter` scalar.

What this ablation isolates (spec §7.3.5): whether variance differentiation
requires *any* learned readout from per-K position-distinguished hidden
vectors (not merely per-K variance parameters). A learned linear from
`decoded[:, k, :]` to a scalar is the minimum machinery that could let
variance differentiate per K; Ablation 1 removes it entirely.
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


class InnerPAM_v1_Ablation1(nn.Module):
    """Ablation 1: K output queries + cross-attention decoder + shared scalar log-var."""

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

        self.output_queries = nn.Parameter(torch.empty(predict_k, hidden))
        nn.init.normal_(self.output_queries, std=OUTPUT_QUERY_INIT_STD)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden,
            nhead=n_heads,
            dim_feedforward=mlp_dim,
            activation="gelu",
            norm_first=True,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_n_layers)

        self.output_proj_mean = nn.Linear(hidden, embed_dim)
        # Spec §7.3.2: single shared log-variance parameter (instead of a
        # per-K linear head). Initialised to 0 (variance = 1) per instr §3.2.
        self.shared_log_var = nn.Parameter(torch.zeros(1))

    def forward(self, window: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        assert window.ndim == 3, f"window must be (B, W, d); got {tuple(window.shape)}"
        b, w, d = window.shape
        assert w == self.window_w, f"W mismatch: got {w}, expected {self.window_w}"
        assert d == self.embed_dim, f"d mismatch: got {d}, expected {self.embed_dim}"

        x = self.input_proj(window)
        positions = torch.arange(self.window_w, device=window.device)
        x = x + self.pos_emb(positions).unsqueeze(0)
        memory = self.encoder(x)

        queries = self.output_queries.unsqueeze(0).expand(b, -1, -1)
        decoded = self.decoder(queries, memory)                    # (B, K, hidden)

        mean = self.output_proj_mean(decoded)                      # (B, K, d)
        # Spec §7.3.3: variance is a single shared parameter, broadcast to (B, K).
        log_var = self.shared_log_var.expand(b, self.predict_k)    # (B, K)
        log_var = clamp_log_var(log_var)

        assert mean.shape == (b, self.predict_k, self.embed_dim)
        assert log_var.shape == (b, self.predict_k)
        return mean, log_var

"""PRE-C: Decoder layer count calibration (spec §7.2.1 SCAFFOLDING; instr §6.3).

Sweeps L_d ∈ {1, 2, 3, 4} on a 30k-frame Stage A subset, computes
calibration metrics, and selects the L_d with the lowest mean pairwise
output-query cosine among those satisfying the loss-curve-smoothness
stability criterion (instr §6.3.2).

Selected `decoder_n_layers` is written to `selected.json` at
`PATHS.results_pre_c / "selected.json"` with key `{"decoder_n_layers": int}`.

The calibration training itself runs through `OnlineTrainerV1` on a Primary
predictor with the candidate `decoder_n_layers`; this module provides the
post-training calibration-metric computation + selection helpers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from v1.src.config import (
    PRE_C_L_D_CANDIDATES,
    PRE_C_LOSS_SMOOTHNESS_RATIO_MAX,
    PRE_C_TIEBREAK_QUERY_COSINE_BAND,
    PREDICT_K,
)
from v1.src.predictor.inner_pam_v1_primary import InnerPAM_v1_Primary


@dataclass
class CalibrationMetrics:
    """Per-L_d calibration metrics (instr §6.3.1)."""

    decoder_n_layers: int
    loss_curve_smoothness_ratio: float    # std(Δloss) / |mean loss| over last 5k steps
    mean_pairwise_query_cosine: float     # output-query similarity, lower = more differentiated
    final_loss: float
    stable_dynamics: bool                  # loss_curve_smoothness_ratio < cutoff
    nan_inf_seen: bool


@dataclass
class CalibrationVerdict:
    selected_decoder_n_layers: Optional[int]
    reason: str
    per_l_d: list[CalibrationMetrics]


def compute_smoothness_ratio(loss_trace: np.ndarray, last_n: int = 5_000) -> float:
    """Std of step-to-step loss differences, divided by mean |loss|, over the
    last `last_n` steps. Lower = smoother."""
    if loss_trace.size < 2:
        return float("inf")
    tail = loss_trace[-last_n:] if loss_trace.size > last_n else loss_trace
    diffs = np.diff(tail)
    mean_abs = np.mean(np.abs(tail))
    if mean_abs == 0:
        return float("inf")
    return float(np.std(diffs) / mean_abs)


@torch.no_grad()
def compute_query_cosine(
    predictor: InnerPAM_v1_Primary,
    windows: torch.Tensor,
) -> float:
    """Mean pairwise cosine across K output-query post-decoder hidden vectors.

    Spec §6.3.1: "On a sample batch of 64 canonical windows, compute the
    K=16 output queries' hidden vectors post-decoder (decoded per spec
    §7.2.3). Compute pairwise cosine similarity across the K query outputs,
    averaged over the batch."

    `windows` shape: (B, W, d).
    """
    predictor.eval()
    b, w, d = windows.shape
    x = predictor.input_proj(windows)
    positions = torch.arange(w, device=windows.device)
    x = x + predictor.pos_emb(positions).unsqueeze(0)
    memory = predictor.encoder(x)
    queries = predictor.output_queries.unsqueeze(0).expand(b, -1, -1)
    decoded = predictor.decoder(queries, memory)              # (B, K, hidden)
    decoded_norm = decoded / (decoded.norm(dim=-1, keepdim=True) + 1e-12)
    # (B, K, K) cosine similarity, batch-mean.
    sim = torch.bmm(decoded_norm, decoded_norm.transpose(1, 2))
    # Off-diagonal mean (exclude self-similarities).
    mask = ~torch.eye(PREDICT_K, dtype=torch.bool, device=windows.device)
    off_diag = sim[:, mask].view(b, PREDICT_K, PREDICT_K - 1)
    return float(off_diag.mean().item())


def select_l_d(metrics: list[CalibrationMetrics]) -> CalibrationVerdict:
    """Selection rule (instr §6.3.2): among L_d values satisfying stable
    training dynamics, pick the lowest mean pairwise query cosine.

    Tie-break (within PRE_C_TIEBREAK_QUERY_COSINE_BAND) toward smaller L_d.
    """
    metrics_sorted = sorted(metrics, key=lambda m: m.decoder_n_layers)
    stable = [m for m in metrics_sorted if m.stable_dynamics and not m.nan_inf_seen]
    if not stable:
        return CalibrationVerdict(
            selected_decoder_n_layers=None,
            reason=(
                "No L_d candidate satisfies stable training dynamics "
                f"(loss_curve_smoothness_ratio < {PRE_C_LOSS_SMOOTHNESS_RATIO_MAX}). "
                "Escalate to v1 design chat per instr §6.3.2."
            ),
            per_l_d=metrics_sorted,
        )

    # Lowest mean pairwise cosine = most differentiated queries.
    best = min(stable, key=lambda m: m.mean_pairwise_query_cosine)
    # Tie-break: prefer smaller L_d if within band.
    candidates_in_band = [
        m
        for m in stable
        if m.mean_pairwise_query_cosine
        <= best.mean_pairwise_query_cosine + PRE_C_TIEBREAK_QUERY_COSINE_BAND
    ]
    selected = min(candidates_in_band, key=lambda m: m.decoder_n_layers)

    return CalibrationVerdict(
        selected_decoder_n_layers=selected.decoder_n_layers,
        reason=(
            f"Selected L_d={selected.decoder_n_layers}: "
            f"smoothness_ratio={selected.loss_curve_smoothness_ratio:.4f}, "
            f"mean_pairwise_query_cosine={selected.mean_pairwise_query_cosine:.4f}. "
            f"Stable candidates: {[m.decoder_n_layers for m in stable]}; "
            f"in-band: {[m.decoder_n_layers for m in candidates_in_band]}."
        ),
        per_l_d=metrics_sorted,
    )


def write_calibration_report(
    metrics: CalibrationMetrics,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "calibration_report.json").write_text(
        json.dumps(metrics.__dict__, indent=2)
    )


def write_summary(verdict: CalibrationVerdict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "selected_decoder_n_layers": verdict.selected_decoder_n_layers,
                "reason": verdict.reason,
                "per_l_d": [m.__dict__ for m in verdict.per_l_d],
            },
            indent=2,
        )
    )


def write_selection_lock(verdict: CalibrationVerdict, lock_path: Path) -> None:
    """Write the config.py-readable selection lock."""
    if verdict.selected_decoder_n_layers is None:
        raise ValueError("Cannot lock selection: no L_d satisfied stability criteria")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(
        json.dumps(
            {"decoder_n_layers": int(verdict.selected_decoder_n_layers)},
            indent=2,
        )
    )

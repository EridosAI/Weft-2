"""V2 Phase 1 — per-K eval + faithful training (pilot refinement 2).

The frozen PRE-D1a `compute_diff_metrics` aggregates Diff_μ / Diff_σ over the K
predicted positions. The pilot needs the per-K breakdown (methodology hypothesis:
K-aggregation may swamp longer-horizon signal with near-frame near-identity noise).
PRE-D1a is read-only, so this module provides:

  * `compute_diff_metrics_perk` — returns the SAME aggregates as PRE-D1a (the
    comparison anchor) PLUS un-aggregated per-K arrays for both heads.
  * `train_one_perk` — a FAITHFUL copy of PRE-D1a `train_one`'s training (same
    stream build via the imported `_stream`, same seed/optimizer/loop), evaluated
    with `compute_diff_metrics_perk`. The aggregate therefore reproduces `train_one`
    bit-identically (smoke-validated); only the per-K arrays are new.

K convention: index i in [0, K-1] is k = i+1 frames ahead. The pilot surfaces
k=1 (idx 0, next-frame), k=8 (idx 7), k=15 (idx 14); the full array is persisted.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from v1.src.config import (
    ADAM_BETAS, GRAD_CLIP_MAX_NORM, LR, PREDICT_K, WEIGHT_DECAY, WINDOW_W,
)
from v1.src.eval.per_item_ordinal_metrics import body_representation_at_window
from v1.src.predictor.inner_pam_v1_primary import InnerPAM_v1_Primary
from v1.src.predictor.inner_pam_v1_shared import path_prediction_loss
from v2.config import V2_SMOKE_CHECKPOINT_EVERY
# Reuse the EXACT frozen stream-builder + trajectory assessor + collapse eps so the
# training is identical to PRE-D1a train_one (only the eval is extended).
from v2.src.preflight.pre_d1a_endpoint_stability import (
    VARIANCE_COLLAPSE_EPS, _stream, assess_trajectory,
)
from v2.src.substrate.stream_builder import StreamParams

PER_K_REPORT_IDX = {"k1": 0, "k8": 7, "k15": 14}   # k = idx+1 (PREDICT_K=16)


@torch.no_grad()
def compute_diff_metrics_perk(predictor, eval_stream: torch.Tensor, device) -> dict:
    """PRE-D1a aggregates (anchor) + per-K Diff_μ / Diff_σ on a held-out stream."""
    L = eval_stream.shape[0]
    ts = list(range(WINDOW_W - 1, L - PREDICT_K))
    windows = torch.stack([eval_stream[t - WINDOW_W + 1: t + 1] for t in ts])     # (N,W,d)
    targets = torch.stack([eval_stream[t + 1: t + 1 + PREDICT_K] for t in ts])    # (N,K,d)
    means, log_vars = [], []
    for s in range(0, windows.shape[0], 512):
        m, lv = predictor(windows[s:s + 512])
        means.append(m); log_vars.append(lv)
    M = torch.cat(means); LV = torch.cat(log_vars)                                # (N,K,d),(N,K)
    var_pred = torch.exp(LV)                                                       # (N,K)

    m_var = M.var(dim=0)                       # (K,d)
    t_var = targets.var(dim=0)                 # (K,d)
    # Aggregate — IDENTICAL to PRE-D1a compute_diff_metrics (ratio of K,d means).
    diff_mu_agg = float((m_var.mean() / (t_var.mean() + 1e-12)).item())
    per_k_var_i = var_pred.var(dim=0)          # (K,)
    diff_sigma_agg = float(per_k_var_i.mean().item())
    # Per-K — mean over d at each k.
    diff_mu_per_k = (m_var.mean(dim=-1) / (t_var.mean(dim=-1) + 1e-12)).cpu().tolist()
    diff_sigma_per_k = per_k_var_i.cpu().tolist()

    return {
        "diff_mu_aggregate": diff_mu_agg,
        "diff_sigma_aggregate": diff_sigma_agg,
        "diff_mu_per_k": diff_mu_per_k,
        "diff_sigma_per_k": diff_sigma_per_k,
        "variance_collapse": bool(diff_sigma_agg < VARIANCE_COLLAPSE_EPS),
        "k_report_idx": PER_K_REPORT_IDX,
    }


def train_one_perk(params: StreamParams, L_d_main: int, seed: int, U, device,
                   training_steps: int, label: str = "") -> dict:
    """Faithful copy of PRE-D1a train_one training loop; per-K eval. Returns a dict.

    Training is byte-for-byte the PRE-D1a recipe (same `_stream`, seed, AdamW config,
    loop, grad-clip) so `diff_mu_aggregate` reproduces `train_one`'s `diff_mu`.
    body_representation tracking is dropped (it is no-grad/detached and does not affect
    training); everything that touches the optimizer is identical.
    """
    train_np = _stream(params, training_steps + WINDOW_W + PREDICT_K + 64, seed, U)
    eval_np = _stream(params, 2048, seed + 1000, U)
    emb = torch.from_numpy(train_np).to(device)
    eval_t = torch.from_numpy(eval_np).to(device)
    N = emb.shape[0]

    torch.manual_seed(seed)
    pred = InnerPAM_v1_Primary(decoder_n_layers=L_d_main).to(device)
    opt = torch.optim.AdamW(pred.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, betas=ADAM_BETAS)
    pred.train()
    fixed_win = eval_t[:WINDOW_W].unsqueeze(0)
    body_early_done = False                    # mirror train_one's first-checkpoint forward

    interval_means, run_sum, run_cnt, steps = [], 0.0, 0, 0
    nan_inf = False
    t0 = time.time()
    last_t = min(N - PREDICT_K - 1, WINDOW_W - 1 + training_steps)
    for t in range(WINDOW_W - 1, last_t):
        window = emb[t - WINDOW_W + 1: t + 1].unsqueeze(0)
        target = emb[t + 1: t + 1 + PREDICT_K].unsqueeze(0)
        mean, log_var = pred(window)
        loss = path_prediction_loss(mean, log_var, target)
        if not torch.isfinite(loss):
            nan_inf = True; break
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(pred.parameters(), GRAD_CLIP_MAX_NORM)
        opt.step()
        run_sum += float(loss.item()); run_cnt += 1; steps += 1
        if run_cnt >= V2_SMOKE_CHECKPOINT_EVERY:
            interval_means.append(run_sum / run_cnt); run_sum, run_cnt = 0.0, 0
            if not body_early_done:
                # train_one calls this (predictor in train() mode) at the first
                # checkpoint; the forward pass consumes RNG (dropout). Replicate it
                # exactly so the training trajectory stays bit-identical. Output unused.
                body_representation_at_window(pred, fixed_win)
                body_early_done = True
    wall = time.time() - t0
    if run_cnt > 0:
        interval_means.append(run_sum / run_cnt)

    rel_imp, still_descending, flag = assess_trajectory(interval_means, nan_inf)

    pred.eval()
    if nan_inf:
        metrics = {"diff_mu_aggregate": float("nan"), "diff_sigma_aggregate": float("nan"),
                   "diff_mu_per_k": None, "diff_sigma_per_k": None, "variance_collapse": True,
                   "k_report_idx": PER_K_REPORT_IDX}
    else:
        metrics = compute_diff_metrics_perk(pred, eval_t, device)

    return {
        "label": label, "L_d_main": L_d_main, "seed": seed,
        "wall_clock_s": round(wall, 2), "steps_trained": steps,
        "final_interval_loss": round(interval_means[-1], 6) if interval_means else float("nan"),
        "nan_inf": nan_inf, "still_descending": still_descending, "stability_flag": flag,
        # back-compat scalar names (== train_one) + per-K extension:
        "diff_mu": metrics["diff_mu_aggregate"], "diff_sigma": metrics["diff_sigma_aggregate"],
        **metrics,
    }

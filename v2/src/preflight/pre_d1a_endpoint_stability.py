"""V2-PRE-D1a — endpoint stability + bit-identical baseline + per-arm cost (instr §7.5).

40 arm-runs (Primary only):
  (a) 20 endpoint-stability runs: 5 axes x 2 endpoints (per-axis min/max) x
      2 L_d_main {1,4}, other 4 axes at midpoint (§11.2 CC interpretation).
  (b) 20 bit-identical baseline runs: all-midpoint, magnitude=0, L_d_main=1,
      n=20 seeds (§11.7) — the empirical baseline distribution PRE-E calibrates
      tau_W against.

Each run trains the v1 Primary arm (v1 predictor + path_prediction_loss + the
trainer's per-step contract) to V2_TRAINING_STEPS (lock-file, PRE-A=10000) and
records §7.5 stability indicators, wall-clock (Input 1), final loss, and
per-stream-point Diff_mu / Diff_sigma on a held-out eval stream.

PLATEAU CAVEAT (user-flagged): V2_TRAINING_STEPS was calibrated on a mid-param /
L_d=1 smoke run. Endpoints (esp. L_d=4) may need longer. Each run flags
`still_descending` if the final interval's relative loss improvement exceeds the
plateau threshold; the runner STOPs and surfaces if a non-trivial fraction of
endpoint runs are not-plateaued (design-chat decision: revisit V2_TRAINING_STEPS
before PRE-E / PRE-D2).
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch

from v1.src.config import (
    ADAM_BETAS, GRAD_CLIP_MAX_NORM, LR, PREDICT_K, WEIGHT_DECAY, WINDOW_W,
)
from v1.src.eval.per_item_ordinal_metrics import body_representation_at_window
from v1.src.predictor.inner_pam_v1_primary import InnerPAM_v1_Primary
from v1.src.predictor.inner_pam_v1_shared import path_prediction_loss
from v2.config import (
    V2_PLATEAU_REL_IMPROVEMENT, V2_SMOKE_CHECKPOINT_EVERY, get_v2_training_steps,
)
from v2.src.substrate.base_manifold_trajectory import load_or_create_U
from v2.src.substrate.stream_builder import StreamParams, build_stream

# Per-axis sanity grids (min / mid / max). Midpoint = held value (§11.2).
AXIS_GRID = {
    "magnitude":  {"min": 0.1, "mid": 0.5, "max": 0.9},
    "locality":   {"min": 0.1, "mid": 0.5, "max": 0.9},
    "continuity": {"min": 8,   "mid": 24,  "max": 60},   # continuity_center
    "period":     {"min": 64,  "mid": 128, "max": 256},  # period_P
    "dim":        {"min": 4,   "mid": 16,  "max": 64},   # manifold_dim_D
}
MID = {"magnitude_M": 0.5, "locality_L": 0.5, "continuity_center": 24,
       "period_P": 128, "manifold_dim_D": 16, "fidelity_F": 0.97}
VARIANCE_COLLAPSE_EPS = 1e-8
BODY_REPR_DRIFT_MAX = 0.5   # cosine drift threshold for an "unstable" flag


@dataclass
class ArmRunResult:
    label: str
    axis: str
    endpoint: str
    L_d_main: int
    seed: int
    stream_construction: dict
    wall_clock_s: float
    steps_trained: int
    final_interval_loss: float
    loss_trajectory: list
    diff_mu: float
    diff_sigma: float
    nan_inf: bool
    variance_collapse: bool
    body_repr_cosine: float
    plateau_rel_improvement: float
    still_descending: bool
    stability_flag: str   # stable | unstable | divergent


def _params_for(axis: str, endpoint: str) -> StreamParams:
    """Stream params with `axis` at `endpoint`, other axes at midpoint."""
    kw = dict(period_P=MID["period_P"], manifold_dim_D=MID["manifold_dim_D"],
              continuity_center=MID["continuity_center"], fidelity_F=MID["fidelity_F"],
              magnitude_M=MID["magnitude_M"], locality_L=MID["locality_L"])
    v = AXIS_GRID[axis][endpoint]
    if axis == "magnitude":
        kw["magnitude_M"] = v
    elif axis == "locality":
        kw["locality_L"] = v
    elif axis == "continuity":
        kw["continuity_center"] = v
    elif axis == "dim":
        kw["manifold_dim_D"] = v
    elif axis == "period":
        kw["period_P"] = v
        kw["continuity_center"] = max(1, round(MID["continuity_center"] * v / MID["period_P"]))
    return StreamParams(**kw)


def _stream(params: StreamParams, n_frames: int, seed: int, U) -> np.ndarray:
    p = StreamParams(**{**params.__dict__, "seed": seed,
                        "n_repetitions": math.ceil(n_frames / params.period_P) + 2})
    return build_stream(p, U).stream.astype(np.float32)


@torch.no_grad()
def compute_diff_metrics(predictor, eval_stream: torch.Tensor, device) -> tuple[float, float, bool]:
    """Per-stream-point Diff_mu (§7.1) and Diff_sigma (§7.2) on a held-out stream."""
    L = eval_stream.shape[0]
    ts = list(range(WINDOW_W - 1, L - PREDICT_K))
    windows = torch.stack([eval_stream[t - WINDOW_W + 1: t + 1] for t in ts])      # (N,W,d)
    targets = torch.stack([eval_stream[t + 1: t + 1 + PREDICT_K] for t in ts])     # (N,K,d)
    means, log_vars = [], []
    for s in range(0, windows.shape[0], 512):
        m, lv = predictor(windows[s:s + 512])
        means.append(m); log_vars.append(lv)
    M = torch.cat(means); LV = torch.cat(log_vars)                                 # (N,K,d),(N,K)
    var_pred = torch.exp(LV)                                                        # σ̂² (N,K)
    diff_mu = float((M.var(dim=0).mean() / (targets.var(dim=0).mean() + 1e-12)).item())
    per_k_var_i = var_pred.var(dim=0)                                               # (K,)
    diff_sigma = float(per_k_var_i.mean().item())
    return diff_mu, diff_sigma, bool(diff_sigma < VARIANCE_COLLAPSE_EPS)


def assess_trajectory(interval_means: list[float], nan_inf: bool,
                      plateau_threshold: float = V2_PLATEAU_REL_IMPROVEMENT
                      ) -> tuple[float, bool, str]:
    """Return (rel_improvement, still_descending, stability_flag) from a loss trajectory.

    path-prediction NLL goes NEGATIVE (small predicted variance), so:
      - plateau uses abs() in the denominator (a ">0" guard would misfire), and
      - loss-increase uses an ADDITIVE tolerance (a multiplicative *1.1 inverts on
        negative losses).
    stability_flag is training-stability only (§8.1 F-stability): divergent (NaN/Inf)
    / unstable (loss diverged upward) / stable.
    """
    if len(interval_means) >= 2 and interval_means[-2] != 0:
        rel_imp = (interval_means[-2] - interval_means[-1]) / abs(interval_means[-2])
    else:
        rel_imp = 0.0
    still_descending = (not nan_inf) and rel_imp > plateau_threshold
    if nan_inf:
        flag = "divergent"
    elif len(interval_means) >= 2 and \
            interval_means[-1] > interval_means[0] + 0.05 * abs(interval_means[0]):
        flag = "unstable"
    else:
        flag = "stable"
    return rel_imp, still_descending, flag


def train_one(params: StreamParams, L_d_main: int, seed: int, U, device,
              training_steps: int, label: str, axis: str, endpoint: str) -> ArmRunResult:
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
    body_early = None

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
            if body_early is None:
                body_early = body_representation_at_window(pred, fixed_win)[0].detach().cpu().numpy()
    wall = time.time() - t0
    if run_cnt > 0:
        interval_means.append(run_sum / run_cnt)

    # Plateau + training-stability flag (handles negative NLL correctly).
    # variance_collapse and body_repr_cosine are recorded as §7.5 indicators
    # alongside (low Diff_sigma is the F-var working-region signal, and body-repr
    # change across checkpoints is expected learning, not divergence — neither is
    # a training-stability failure).
    rel_imp, still_descending, flag = assess_trajectory(interval_means, nan_inf)

    pred.eval()
    if nan_inf:
        diff_mu = diff_sigma = float("nan"); var_collapse = True; body_cos = float("nan")
    else:
        diff_mu, diff_sigma, var_collapse = compute_diff_metrics(pred, eval_t, device)
        body_final = body_representation_at_window(pred, fixed_win)[0].detach().cpu().numpy()
        if body_early is not None:
            body_cos = float(np.dot(body_early, body_final) /
                             (np.linalg.norm(body_early) * np.linalg.norm(body_final) + 1e-12))
        else:
            body_cos = 1.0

    return ArmRunResult(
        label=label, axis=axis, endpoint=endpoint, L_d_main=L_d_main, seed=seed,
        stream_construction=build_stream(StreamParams(**{**params.__dict__, "seed": seed,
                                         "n_repetitions": 4}), U).construction,
        wall_clock_s=round(wall, 2), steps_trained=steps,
        final_interval_loss=round(interval_means[-1], 6) if interval_means else float("nan"),
        loss_trajectory=[round(x, 6) for x in interval_means],
        diff_mu=diff_mu, diff_sigma=diff_sigma, nan_inf=nan_inf,
        variance_collapse=var_collapse, body_repr_cosine=round(body_cos, 6) if body_cos == body_cos else None,
        plateau_rel_improvement=round(rel_imp, 5), still_descending=still_descending,
        stability_flag=flag,
    )


def endpoint_configs() -> list[tuple]:
    """20 endpoint configs: (axis, endpoint, L_d_main, seed)."""
    out, seed = [], 0
    for axis in ("magnitude", "locality", "continuity", "period", "dim"):
        for endpoint in ("min", "max"):
            for L_d_main in (1, 4):
                out.append((axis, endpoint, L_d_main, seed)); seed += 1
    return out


def baseline_configs() -> list[int]:
    """20 bit-identical baseline seeds (all-midpoint, magnitude=0, L_d_main=1)."""
    return list(range(100, 120))

"""V2 Phase 1 — confirmatory training-length test (root cause of the mean-head defect).

The sanity battery showed the predictor's mean head never learns at V2_TRAINING_STEPS=10000
(cos(mean,target)≈0, worse than trivial predict-last at 0.56). This trains ONE predictor
(mid mag=0 L_d=1, the cleanest learnable target — a smooth period-256 loop) far longer,
evaluating cos(mean,target) at step checkpoints, to decisively separate:
  * TRAINING-LENGTH insufficient  -> cos climbs toward / above the trivial 0.56 with steps.
  * NOT length alone (batching/recipe/deeper) -> cos stays ≈0 even at 100k steps.

Faithful to the v2 recipe (same _stream, seed, AdamW, batch=1 online loop); only adds
checkpoint evals (eval-mode, no-grad -> no training-RNG perturbation). ~24 min.
"""

from __future__ import annotations

import json
import time

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from v1.src.config import (
    ADAM_BETAS, GRAD_CLIP_MAX_NORM, LR, PREDICT_K, WEIGHT_DECAY, WINDOW_W,
)
from v1.src.predictor.inner_pam_v1_primary import InnerPAM_v1_Primary
from v1.src.predictor.inner_pam_v1_shared import path_prediction_loss
from v2.config import RESULTS_ROOT
from v2.src.phase1.sweep_grid import CONTINUITY_CENTER
from v2.src.preflight.pre_d1a_endpoint_stability import _stream
from v2.src.substrate.base_manifold_trajectory import load_or_create_U
from v2.src.substrate.stream_builder import StreamParams

MAX_STEPS = 200000
CKPTS = {2000, 10000, 25000, 50000, 100000, 150000, 200000}
PLATEAU_EPS = 0.02   # if cos_100k - cos_50k <= this, stop at 100k (plateaued); else push to 200k
MID = StreamParams(period_P=256, manifold_dim_D=16,
                   continuity_center=CONTINUITY_CENTER[256]["mid"],
                   magnitude_M=0.0, locality_L=0.5, fidelity_F=0.97)


def _cos(a, b):
    an = a / a.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    bn = b / b.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return (an * bn).sum(-1)


@torch.no_grad()
def eval_cos(pred, eval_t, device, nwin: int = 1000) -> dict:
    pred.eval()
    ts = list(range(WINDOW_W - 1, eval_t.shape[0] - PREDICT_K))[:nwin]
    W = torch.stack([eval_t[t - WINDOW_W + 1: t + 1] for t in ts])
    T = torch.stack([eval_t[t + 1: t + 1 + PREDICT_K] for t in ts])
    means = []
    for s in range(0, W.shape[0], 512):
        m, _ = pred(W[s:s + 512]); means.append(m)
    M = torch.cat(means)
    out = {"cos_k1": float(_cos(M[:, 0], T[:, 0]).mean()),
           "cos_allK": float(_cos(M, T).mean()),
           "mean_norm_k1": float(M[:, 0, :].norm(dim=-1).mean())}
    pred.train()
    return out


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    U = load_or_create_U()
    train_np = _stream(MID, MAX_STEPS + WINDOW_W + PREDICT_K + 64, 0, U)
    eval_np = _stream(MID, 2048, 1000, U)
    emb = torch.from_numpy(train_np).to(device)
    eval_t = torch.from_numpy(eval_np).to(device)
    N = emb.shape[0]

    # Trivial baseline: predict next frame = last window frame.
    ts = list(range(WINDOW_W - 1, eval_t.shape[0] - PREDICT_K))[:1000]
    triv = float(_cos(torch.stack([eval_t[t] for t in ts]),
                      torch.stack([eval_t[t + 1] for t in ts])).mean())
    print(f"[len] trivial predict-last baseline cos_k1 = {triv:.4f}; period-256 reps={N//256}")

    torch.manual_seed(0)
    pred = InnerPAM_v1_Primary(decoder_n_layers=1).to(device)
    opt = torch.optim.AdamW(pred.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, betas=ADAM_BETAS)
    pred.train()

    curve, steps, t0 = [], 0, time.time()
    cos_at = {}
    stopped_early = False
    last_t = min(N - PREDICT_K - 1, WINDOW_W - 1 + MAX_STEPS)
    for t in range(WINDOW_W - 1, last_t):
        window = emb[t - WINDOW_W + 1: t + 1].unsqueeze(0)
        target = emb[t + 1: t + 1 + PREDICT_K].unsqueeze(0)
        mean, log_var = pred(window)
        loss = path_prediction_loss(mean, log_var, target)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(pred.parameters(), GRAD_CLIP_MAX_NORM)
        opt.step()
        steps += 1
        if steps in CKPTS:
            ev = eval_cos(pred, eval_t, device)
            ev.update(steps=steps, loss=float(loss.item()), wall_min=round((time.time() - t0) / 60, 1))
            curve.append(ev)
            cos_at[steps] = ev["cos_k1"]
            print(f"[len] steps={steps:>6} cos_k1={ev['cos_k1']:+.4f} cos_allK={ev['cos_allK']:+.4f} "
                  f"mean_norm={ev['mean_norm_k1']:.3f} (trivial={triv:.3f}) loss={ev['loss']:.0f} "
                  f"[{ev['wall_min']}min]", flush=True)
            # Push past 100k only if still improving 50k->100k (else plateaued -> stop).
            if steps == 100000:
                gain = cos_at[100000] - cos_at.get(50000, cos_at[100000])
                if gain <= PLATEAU_EPS:
                    stopped_early = True
                    print(f"[len] plateaued at 100k (50k->100k gain {gain:+.4f} <= {PLATEAU_EPS}); "
                          f"not pushing to 200k.", flush=True)
                    break
                print(f"[len] still improving (50k->100k gain {gain:+.4f}); pushing to 200k ...", flush=True)

    final = curve[-1]["cos_k1"] if curve else float("nan")
    verdict = ("TRAINING-LENGTH: cos rises above the trivial baseline with more steps"
               if final > triv else
               "rising-but-below-trivial: partial learning, needs even more / better recipe"
               if final > 0.1 else
               "NOT training-length alone: cos stays ~0 at 100k -> batching/recipe/deeper defect")
    out = {"config": "mid mag=0 L_d=1", "max_steps": MAX_STEPS,
           "stopped_early_at_100k_plateau": stopped_early,
           "trivial_baseline_cos_k1": triv, "curve": curve, "verdict": verdict}
    (RESULTS_ROOT / "phase1" / "training_length_test.json").write_text(json.dumps(out, indent=2))
    print(f"[len] FINAL cos_k1={final:+.4f} vs trivial {triv:.4f} -> {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""V2 recalibration Stage 1 — multi-cell grokking detection (instr §3).

Generalises `scripts/run_phase1_training_length_test.py` (commit 313efa5) from a
single L_d=1 cell / single seed to the §3.1 grid of 7 cells x 3 predictor seeds
(21 grok-curve runs). Each run trains one Primary predictor to MAX_STEPS with
checkpoint evals of cos(mean, target) — the mean head's actual fit — and the
trivial last-frame baseline it must beat. The lock value is computed by the
mean-head-aware plateau detector (instr §3.3-§3.4), NOT NLL loss flatness.

Design decisions (recalibration instr §3.2):
  * Substrate is FIXED across the 3 predictor seeds of a cell (construction is
    deterministic; only predictor init + dropout RNG vary). Stream seeds are the
    fixed STREAM_SEED_TRAIN / STREAM_SEED_EVAL below.
  * cos is evaluated on the TRAINED-ON stream (§3.2 step 4; matches Stage 2 S1).
    Held-out cos (fresh-seed stream, same construction) is recorded as a
    supplementary field for the experiment chat.

Modes:
  --cell C1 --seed 0     run one (cell, seed); write grok_curve_C1_seed0.json
  --smoke                fast correctness pass: all 7 cells, seed 0, 600 steps
  --vram-probe           measure full-stream peak VRAM for L_d=1 and L_d=4
  --aggregate            per-cell aggregates + lock_decision.json + STOP checks
  --write-lock           (deliberate) update config V2_TRAINING_STEPS lock file
                         from lock_decision.json (only if no STOP fired)

Run from the repo root: `python3 v2/scripts/run_recalibration_stage1.py ...`
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

# Ensure repo root is on sys.path when run from any cwd (v1/scripts convention).
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from v1.src.config import (
    ADAM_BETAS, GRAD_CLIP_MAX_NORM, LR, PREDICT_K, WEIGHT_DECAY, WINDOW_W,
)
from v1.src.predictor.inner_pam_v1_primary import InnerPAM_v1_Primary
from v1.src.predictor.inner_pam_v1_shared import path_prediction_loss
from v2.config import RESULTS_ROOT, write_v2_training_steps
from v2.src.phase1.sweep_grid import CONTINUITY_CENTER, FIDELITY_F, MANIFOLD_D, PERIOD_P
from v2.src.preflight.mean_head_plateau import (
    mean_head_plateau_step, not_grokked_within_budget,
)
from v2.src.preflight.pre_d1a_endpoint_stability import _stream
from v2.src.substrate.base_manifold_trajectory import load_or_create_U
from v2.src.substrate.stream_builder import StreamParams, build_stream

# --------------------------------------------------------------------------
# Fixed protocol constants (instr §3.2 / §3.3).
# --------------------------------------------------------------------------

MAX_STEPS = 200000
CKPTS = (2000, 5000, 10000, 25000, 50000, 75000,
         100000, 125000, 150000, 175000, 200000)
STREAM_SEED_TRAIN = 0      # fixed substrate seed; predictor seed varies separately
STREAM_SEED_EVAL = 1000    # held-out stream: same construction, different noise
N_EVAL_WINDOWS = 1000
CLEAN_LOCK_LEVELS = (50000, 75000, 100000, 125000, 150000, 175000, 200000)
RESIDUAL_MARGIN = 1.1      # instr §3.3 (seed variation now measured, not assumed)
TRIVIAL_CLEARANCE = 0.10   # instr §3.3 not-grokked threshold
SPREAD_STOP_RATIO = 4.0    # instr §3.5 within-cell and inter-cell spread STOP
LOCK_STEP_STOP = 175000    # instr §3.5: candidate > this would push lock past 200k

STAGE1_DIR = RESULTS_ROOT / "recalibration" / "stage1"

# --------------------------------------------------------------------------
# The 7 cells (instr §3.1), built from the locked sweep_grid tables (no new
# substrate code). Magnitude=0 cells use the baseline magnitude (outside the
# grid's MAGNITUDE_M); C5 uses 0.9 (the strong-perturbation probe).
# --------------------------------------------------------------------------


def _sp(period_P, manifold_dim_D, continuity_center, magnitude_M):
    return StreamParams(
        period_P=period_P, manifold_dim_D=manifold_dim_D,
        continuity_center=continuity_center, magnitude_M=magnitude_M,
        locality_L=0.5, fidelity_F=FIDELITY_F,
    )


CELLS: dict[str, tuple[int, StreamParams]] = {
    # id: (L_d_main, StreamParams)   # note  per instr §3.1
    "C1": (1, _sp(256, 16, CONTINUITY_CENTER[256]["mid"], 0.0)),   # anchor (== template MID)
    "C2": (2, _sp(256, 16, CONTINUITY_CENTER[256]["mid"], 0.0)),   # capacity probe L_d=2
    "C3": (4, _sp(256, 16, CONTINUITY_CENTER[256]["mid"], 0.0)),   # capacity probe L_d=4
    "C4": (1, _sp(256, MANIFOLD_D["high"], CONTINUITY_CENTER[256]["mid"], 0.0)),  # D=128 boundary; forced C~1.0
    "C5": (1, _sp(256, 16, CONTINUITY_CENTER[256]["mid"], 0.9)),   # strong-perturbation probe
    "C6": (1, _sp(PERIOD_P["low"], MANIFOLD_D["low"], CONTINUITY_CENTER[32]["mid"], 0.0)),  # P=32, D=4 low-period
    "C7": (1, _sp(256, 16, CONTINUITY_CENTER[256]["high"], 0.0)),  # high-continuity probe
}
SEEDS = (0, 1, 2)


def _cos(a, b):
    an = a / a.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    bn = b / b.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return (an * bn).sum(-1)


def _trivial_cos(stream: torch.Tensor, nwin: int = N_EVAL_WINDOWS) -> float:
    """cos between consecutive frames (predict-last baseline) over the eval positions."""
    ts = list(range(WINDOW_W - 1, stream.shape[0] - PREDICT_K))[:nwin]
    a = torch.stack([stream[t] for t in ts])
    b = torch.stack([stream[t + 1] for t in ts])
    return float(_cos(a, b).mean())


@torch.no_grad()
def _eval_cos(pred, stream: torch.Tensor, nwin: int = N_EVAL_WINDOWS) -> dict:
    pred.eval()
    ts = list(range(WINDOW_W - 1, stream.shape[0] - PREDICT_K))[:nwin]
    W = torch.stack([stream[t - WINDOW_W + 1: t + 1] for t in ts])
    T = torch.stack([stream[t + 1: t + 1 + PREDICT_K] for t in ts])
    means = []
    for s in range(0, W.shape[0], 512):
        m, _ = pred(W[s:s + 512]); means.append(m)
    M = torch.cat(means)
    out = {"cos_k1": float(_cos(M[:, 0], T[:, 0]).mean()),
           "cos_allK": float(_cos(M, T).mean()),
           "mean_norm_k1": float(M[:, 0, :].norm(dim=-1).mean())}
    pred.train()
    return out


def _construction(params: StreamParams, U) -> dict:
    return build_stream(
        StreamParams(**{**params.__dict__, "seed": STREAM_SEED_TRAIN, "n_repetitions": 4}), U
    ).construction


def run_one(cell_id: str, seed: int, max_steps: int, ckpts: tuple,
            out_dir: Path, U, device) -> dict:
    """Train one (cell, seed) grok curve and write its JSON. Returns the result dict."""
    L_d, params = CELLS[cell_id]
    train_np = _stream(params, max_steps + WINDOW_W + PREDICT_K + 64, STREAM_SEED_TRAIN, U)
    held_np = _stream(params, 2048, STREAM_SEED_EVAL, U)
    emb = torch.from_numpy(train_np).to(device)
    held = torch.from_numpy(held_np).to(device)
    N = emb.shape[0]

    constr = _construction(params, U)
    measured = {"measured_C": constr["continuity_C"], "magnitude": constr["magnitude_M"],
                "locality": constr["locality_L"], "period": constr["period_P"],
                "D_global": constr["manifold_dim_D"]}
    triv_trained = _trivial_cos(emb)
    triv_heldout = _trivial_cos(held)

    torch.manual_seed(seed)
    pred = InnerPAM_v1_Primary(decoder_n_layers=L_d).to(device)
    opt = torch.optim.AdamW(pred.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, betas=ADAM_BETAS)
    pred.train()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    curve, steps, t0 = [], 0, time.time()
    last_t = min(N - PREDICT_K - 1, WINDOW_W - 1 + max_steps)
    print(f"[stage1 {cell_id} seed{seed}] L_d={L_d} P={params.period_P} D={params.manifold_dim_D} "
          f"c={params.continuity_center} mag={params.magnitude_M} | trivial(trained)={triv_trained:.4f} "
          f"reps={N // params.period_P}", flush=True)
    for t in range(WINDOW_W - 1, last_t):
        window = emb[t - WINDOW_W + 1: t + 1].unsqueeze(0)
        target = emb[t + 1: t + 1 + PREDICT_K].unsqueeze(0)
        mean, log_var = pred(window)
        loss = path_prediction_loss(mean, log_var, target)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(pred.parameters(), GRAD_CLIP_MAX_NORM)
        opt.step()
        steps += 1
        if steps in ckpts:
            ev = _eval_cos(pred, emb)
            ho = _eval_cos(pred, held)
            row = {"steps": steps, "cos_k1": ev["cos_k1"], "cos_allK": ev["cos_allK"],
                   "mean_norm_k1": ev["mean_norm_k1"],
                   "cos_k1_heldout": ho["cos_k1"], "cos_allK_heldout": ho["cos_allK"],
                   "loss": float(loss.item()), "wall_min": round((time.time() - t0) / 60, 2)}
            curve.append(row)
            print(f"[stage1 {cell_id} seed{seed}] steps={steps:>6} cos_k1={ev['cos_k1']:+.4f} "
                  f"(triv {triv_trained:+.3f}) cos_allK={ev['cos_allK']:+.4f} "
                  f"ho_cos_k1={ho['cos_k1']:+.4f} mean_norm={ev['mean_norm_k1']:.3f} "
                  f"loss={row['loss']:.0f} [{row['wall_min']}min]", flush=True)

    lock_candidate = mean_head_plateau_step(curve)
    not_grokked = not_grokked_within_budget(curve, triv_trained, TRIVIAL_CLEARANCE)
    peak_gb = (torch.cuda.max_memory_allocated() / 1e9) if device.type == "cuda" else None
    result = {
        "cell_id": cell_id, "seed": seed,
        "construction": {"L_d": L_d, "period_P": params.period_P,
                         "manifold_dim_D": params.manifold_dim_D,
                         "continuity_center": params.continuity_center,
                         "magnitude_M": params.magnitude_M, "locality_L": params.locality_L},
        "measured_properties": measured,
        "trivial_baseline_cos_k1": triv_trained,
        "trivial_baseline_cos_k1_heldout": triv_heldout,
        "curve": curve,
        "max_cos_k1": max((e["cos_k1"] for e in curve), default=float("nan")),
        "lock_step_candidate": lock_candidate,
        "not_grokked_within_budget": not_grokked,
        "peak_vram_gb": round(peak_gb, 2) if peak_gb is not None else None,
        "wall_min_total": round((time.time() - t0) / 60, 2),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"grok_curve_{cell_id}_seed{seed}.json"
    out_path.write_text(json.dumps(result, indent=2))
    flag = " NOT-GROKKED" if not_grokked else ""
    print(f"[stage1 {cell_id} seed{seed}] DONE lock_candidate={lock_candidate} "
          f"max_cos={result['max_cos_k1']:+.4f}{flag} peak_vram={result['peak_vram_gb']}GB "
          f"-> {out_path.name}", flush=True)
    return result


# --------------------------------------------------------------------------
# Aggregation + lock decision (instr §3.3 / §3.6)
# --------------------------------------------------------------------------


def _round_to_clean(x: float):
    """Smallest clean level >= x (instr §3.3 round-up), or None if > 200000."""
    for level in CLEAN_LOCK_LEVELS:
        if x <= level:
            return level
    return None


def aggregate(out_dir: Path) -> dict:
    """Read all 21 per-(cell,seed) files, write per-cell aggregates + lock_decision."""
    per_cell, missing = {}, []
    for cell_id in CELLS:
        seeds_data = []
        for seed in SEEDS:
            p = out_dir / f"grok_curve_{cell_id}_seed{seed}.json"
            if not p.exists():
                missing.append(p.name); continue
            seeds_data.append(json.loads(p.read_text()))
        if not seeds_data:
            continue
        L_d, params = CELLS[cell_id]
        candidates = [d["lock_step_candidate"] for d in seeds_data if d["lock_step_candidate"] is not None]
        any_not_grokked = any(d["not_grokked_within_budget"] for d in seeds_data)
        cand_max = max(candidates) if candidates else None
        cand_min = min(candidates) if candidates else None
        agg = {
            "cell_id": cell_id,
            "construction": seeds_data[0]["construction"],
            "measured_properties": seeds_data[0]["measured_properties"],
            "seeds": [d["seed"] for d in seeds_data],
            "trivial_baseline_cos_k1": seeds_data[0]["trivial_baseline_cos_k1"],
            "lock_step_candidate_per_seed": [d["lock_step_candidate"] for d in seeds_data],
            "max_cos_k1_per_seed": [round(d["max_cos_k1"], 4) for d in seeds_data],
            "not_grokked_per_seed": [d["not_grokked_within_budget"] for d in seeds_data],
            "lock_step_candidate_max": cand_max,
            "lock_step_candidate_median": (statistics.median(candidates) if candidates else None),
            "within_cell_spread_ratio": (cand_max / cand_min if cand_min else None),
            "any_not_grokked": any_not_grokked,
        }
        (out_dir / f"grok_curve_{cell_id}_aggregate.json").write_text(json.dumps(agg, indent=2))
        per_cell[cell_id] = agg

    # Lock decision + STOP conditions (instr §3.3 / §3.5).
    stops = []
    if missing:
        stops.append(f"missing runs: {sorted(set(missing))}")
    cell_maxes = {c: a["lock_step_candidate_max"] for c, a in per_cell.items()}
    valid_maxes = {c: v for c, v in cell_maxes.items() if v is not None}

    for c, a in per_cell.items():
        for seed_cand in a["lock_step_candidate_per_seed"]:
            if seed_cand is not None and seed_cand > LOCK_STEP_STOP:
                stops.append(f"{c}: a seed lock_step_candidate {seed_cand} > {LOCK_STEP_STOP}")
        if a["any_not_grokked"]:
            stops.append(f"{c}: not-grokked-within-budget (a seed never cleared trivial+{TRIVIAL_CLEARANCE})")
        r = a["within_cell_spread_ratio"]
        if r is not None and r > SPREAD_STOP_RATIO:
            stops.append(f"{c}: within-cell across-seed spread {r:.2f}x > {SPREAD_STOP_RATIO}x")

    inter_cell_ratio = None
    if len(valid_maxes) >= 2:
        inter_cell_ratio = max(valid_maxes.values()) / min(valid_maxes.values())
        if inter_cell_ratio > SPREAD_STOP_RATIO:
            stops.append(f"inter-cell lock_step_candidate_max spread {inter_cell_ratio:.2f}x > {SPREAD_STOP_RATIO}x")

    proposed = None
    overflow = False
    if valid_maxes and not missing:
        worst = max(valid_maxes.values())
        target = worst * RESIDUAL_MARGIN
        proposed = _round_to_clean(target)
        if proposed is None:
            overflow = True
            stops.append(f"max*{RESIDUAL_MARGIN} = {target:.0f} exceeds 200000 (budget ceiling)")

    decision = {
        "lock_steps_proposed": proposed,
        "residual_margin": RESIDUAL_MARGIN,
        "round_target_pre_clean": (max(valid_maxes.values()) * RESIDUAL_MARGIN if valid_maxes else None),
        "cell_lock_step_candidate_max": cell_maxes,
        "within_cell_spread_ratio": {c: a["within_cell_spread_ratio"] for c, a in per_cell.items()},
        "inter_cell_spread_ratio": inter_cell_ratio,
        "trivial_baselines": {c: a["trivial_baseline_cos_k1"] for c, a in per_cell.items()},
        "lock_step_candidate_per_seed": {c: a["lock_step_candidate_per_seed"] for c, a in per_cell.items()},
        "max_cos_k1_per_seed": {c: a["max_cos_k1_per_seed"] for c, a in per_cell.items()},
        "stop_conditions_fired": stops,
        "any_stop": bool(stops),
        "overflow_past_budget": overflow,
    }
    (out_dir / "lock_decision.json").write_text(json.dumps(decision, indent=2))
    print(json.dumps(decision, indent=2))
    print(f"\n[stage1 aggregate] proposed V2_TRAINING_STEPS = {proposed} | "
          f"STOP fired = {bool(stops)}", flush=True)
    return decision


def write_lock(out_dir: Path) -> int:
    """Deliberately update the config V2_TRAINING_STEPS lock from lock_decision.json.

    Refuses if a STOP fired or the lock decision is absent. Backs up the prior
    lock file (the invalidated PRE-A=10000 value) before overwriting.
    """
    decision_path = out_dir / "lock_decision.json"
    if not decision_path.exists():
        print("[stage1 write-lock] no lock_decision.json; run --aggregate first.", flush=True)
        return 1
    decision = json.loads(decision_path.read_text())
    if decision["any_stop"]:
        print(f"[stage1 write-lock] REFUSING: STOP fired -> {decision['stop_conditions_fired']}", flush=True)
        return 2
    proposed = decision["lock_steps_proposed"]
    if proposed is None:
        print("[stage1 write-lock] REFUSING: no proposed value.", flush=True)
        return 2
    from v2.config import _V2_TRAINING_STEPS_LOCK  # noqa: only for backup of prior value
    prior = None
    if _V2_TRAINING_STEPS_LOCK.exists():
        prior = json.loads(_V2_TRAINING_STEPS_LOCK.read_text())
        backup = _V2_TRAINING_STEPS_LOCK.parent / "v2_training_steps_pre_a_invalidated.json"
        backup.write_text(json.dumps(prior, indent=2))
        print(f"[stage1 write-lock] backed up prior lock ({prior}) -> {backup.name}", flush=True)
    write_v2_training_steps(proposed, {
        "source": "recalibration_stage1",
        "rationale": "multi-cell n=3 grok detection, mean-head-aware plateau (instr §3.3)",
        "lock_decision": str(decision_path),
        "prior_invalidated_value": (prior.get("v2_training_steps") if prior else None),
    })
    print(f"[stage1 write-lock] V2_TRAINING_STEPS lock updated -> {proposed}", flush=True)
    return 0


def vram_probe(U, device, probe_steps: int = 100) -> None:
    """Build full-size streams + models for L_d=1 and L_d=4; report peak VRAM."""
    if device.type != "cuda":
        print("[vram-probe] no CUDA; skipping."); return
    for cell_id in ("C1", "C3"):
        L_d, params = CELLS[cell_id]
        train_np = _stream(params, MAX_STEPS + WINDOW_W + PREDICT_K + 64, STREAM_SEED_TRAIN, U)
        emb = torch.from_numpy(train_np).to(device)
        torch.cuda.reset_peak_memory_stats()
        torch.manual_seed(0)
        pred = InnerPAM_v1_Primary(decoder_n_layers=L_d).to(device)
        opt = torch.optim.AdamW(pred.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, betas=ADAM_BETAS)
        pred.train()
        for t in range(WINDOW_W - 1, WINDOW_W - 1 + probe_steps):
            window = emb[t - WINDOW_W + 1: t + 1].unsqueeze(0)
            target = emb[t + 1: t + 1 + PREDICT_K].unsqueeze(0)
            loss = path_prediction_loss(*pred(window), target)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(pred.parameters(), GRAD_CLIP_MAX_NORM)
            opt.step()
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"[vram-probe] {cell_id} L_d={L_d}: stream={emb.shape} peak_alloc={peak:.2f}GB", flush=True)
        del emb, pred, opt
        torch.cuda.empty_cache()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", choices=list(CELLS))
    ap.add_argument("--seed", type=int)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--vram-probe", action="store_true")
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--write-lock", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.aggregate:
        aggregate(STAGE1_DIR); return 0
    if args.write_lock:
        return write_lock(STAGE1_DIR)

    U = load_or_create_U()

    if args.vram_probe:
        vram_probe(U, device); return 0

    if args.smoke:
        smoke_dir = STAGE1_DIR / "smoke"
        for cell_id in CELLS:
            run_one(cell_id, 0, max_steps=600, ckpts=(200, 400, 600),
                    out_dir=smoke_dir, U=U, device=device)
        print("[stage1 smoke] all 7 cells built + trained 600 steps OK.", flush=True)
        return 0

    if args.cell is None or args.seed is None:
        ap.error("provide --cell and --seed (or --smoke / --vram-probe / --aggregate / --write-lock)")
    run_one(args.cell, args.seed, MAX_STEPS, CKPTS, STAGE1_DIR, U, device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

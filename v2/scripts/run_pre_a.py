"""V2-PRE-A — construction-primitive sanity + smoke-test arm-run (instr §7.1).

Runs:
  1. Sanity sweep — 3 values per axis (magnitude, locality, continuity,
     repetition period, manifold dimensionality; + fidelity supplementary),
     building one stream per point and verifying the measured §4 property
     matches the construction value within tolerance. STOP trigger 1: any
     spec axis off-tolerance on > 1 of its 3 sweep values.
  2. Stream-contract smoke — instantiate the v1 OnlineTrainerV1 on a mid-
     parameter synthetic stream; its constructor enforces the L2-norm contract
     (norms 1 ± 1e-5). Failure => STOP trigger 2.
  3. Arch-forward smoke — forward one synthetic window through each of the
     three v1 predictor arms; non-finite output => STOP trigger 3 (light check;
     full assertions are V2-PRE-C).
  4. V2_TRAINING_STEPS calibration — train the Primary arm on a mid-parameter
     stream (v1 predictor + v1 path_prediction_loss + the trainer's exact
     per-step contract), observe the loss trajectory, and lock V2_TRAINING_STEPS
     at the plateau (instr §7.1 — derived, not guessed).

Outputs:
  results/pre_a/construction_sanity_report.json
  results/pre_a/stream_contract_smoke.json
  results/pre_a/v2_training_steps.json   (lock file; via config.write_v2_training_steps)
  data/embedding_U.npy
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from v1.src.config import (
    ADAM_BETAS,
    GRAD_CLIP_MAX_NORM,
    LR,
    PREDICT_K,
    WEIGHT_DECAY,
    WINDOW_W,
)
from v1.src.predictor.inner_pam_v1_ablation1 import InnerPAM_v1_Ablation1
from v1.src.predictor.inner_pam_v1_ablation2 import InnerPAM_v1_Ablation2
from v1.src.predictor.inner_pam_v1_primary import InnerPAM_v1_Primary
from v1.src.predictor.inner_pam_v1_shared import path_prediction_loss
from v1.src.trainer.online_trainer_v1 import (
    OnlineTrainerV1,
    TrainerConfig,
    TrainingStopped,
)
from v2.config import (
    CONSTRUCTION_MEASUREMENT_TOLERANCE as TOL,
    RESULTS_PRE_A,
    V2_PLATEAU_REL_IMPROVEMENT,
    V2_SMOKE_CHECKPOINT_EVERY,
    V2_SMOKE_MAX_STEPS,
    load_calibrated_thresholds,
    write_v2_training_steps,
)
from v2.src.protocol.protocol import apply_protocol
from v2.src.substrate.base_manifold_trajectory import load_or_create_U
from v2.src.substrate.stream_builder import StreamParams, build_stream

PLACEHOLDER_PARAMS = {  # held reference values for the swept axes (documented)
    "fidelity_F": 0.999, "magnitude_M": 0.0, "locality_L": 0.5,
    "continuity_center": 24, "period_P": 256, "manifold_dim_D": 16,
}


def _scaled_center(P: int) -> int:
    return max(1, round(24 * P / 256))


def _sweep_grid() -> dict:
    """Per-axis: (construction-key, measured-key, [StreamParams,...]) for 3 values."""
    SP = StreamParams
    return {
        "magnitude": ("magnitude_M", "magnitude_M", [
            SP(magnitude_M=v, fidelity_F=0.999, locality_L=0.5) for v in (0.1, 0.5, 0.9)]),
        "locality": ("locality_L", "locality_L", [
            SP(locality_L=v, magnitude_M=0.5, fidelity_F=0.999) for v in (0.1, 0.5, 0.9)]),
        "continuity": ("continuity_C", "continuity_C", [
            SP(continuity_center=v, magnitude_M=0.0, fidelity_F=0.999) for v in (8, 24, 60)]),
        "repetition_period": ("period_P", "period_P", [
            SP(period_P=P, continuity_center=_scaled_center(P), magnitude_M=0.0,
               fidelity_F=0.999) for P in (64, 128, 256)]),
        "manifold_dim": ("manifold_dim_D", "manifold_D_global", [
            SP(manifold_dim_D=v, continuity_center=48, magnitude_M=0.0,
               fidelity_F=0.999) for v in (4, 16, 64)]),
        "fidelity_supplementary": ("fidelity_F", "fidelity_F", [
            SP(fidelity_F=v, magnitude_M=0.0) for v in (0.8, 0.9, 0.97)]),
    }


SPEC_AXES = {"magnitude", "locality", "continuity", "repetition_period", "manifold_dim"}


def run_sanity_sweep(U: np.ndarray, th: dict) -> dict:
    grid = _sweep_grid()
    axes_report = {}
    any_spec_axis_failed = False
    for axis, (con_key, meas_key, params_list) in grid.items():
        values = []
        n_pass = 0
        for p in params_list:
            bs = build_stream(p, U)
            rec = apply_protocol(bs.stream, th)
            con = bs.construction[con_key]
            meas = rec[meas_key]
            reldev = (None if meas is None
                      else abs(meas - con) / max(abs(con), 1e-9))
            ok = reldev is not None and reldev <= TOL
            n_pass += int(ok)
            values.append({
                "construction": con, "measured": meas,
                "rel_deviation": reldev, "within_tolerance": bool(ok),
                "all_measured": {k: rec[k] for k in (
                    "period_P", "fidelity_F", "repetition_coverage", "magnitude_M",
                    "locality_L", "continuity_C", "manifold_D_local",
                    "manifold_D_global", "n_perturbed_detected")},
            })
        axis_ok = n_pass >= 2
        if axis in SPEC_AXES and not axis_ok:
            any_spec_axis_failed = True
        axes_report[axis] = {
            "construction_key": con_key, "measured_key": meas_key,
            "n_within_tolerance": n_pass, "n_values": len(params_list),
            "axis_ok": axis_ok, "is_spec_axis": axis in SPEC_AXES,
            "values": values,
        }
    return {
        "tolerance_relative": TOL,
        "held_reference_values": PLACEHOLDER_PARAMS,
        "manifold_dim_note": (
            "axis verified via D_global (participation ratio of full-trajectory "
            "covariance), which tracks the construction D; D_local underestimates "
            "for a 1-D trajectory because a local window of a curve does not span "
            "all manifold directions equally — local-PCA window calibrated at PRE-E."),
        "locality_note": (
            "construction locality is the ground-truth §4.2 value (measured with the "
            "true pre-perturbation reference + true perturbed set); the sweep verifies "
            "the protocol's estimated reference/detection recovers it."),
        "axes": axes_report,
        "stop_trigger_1_construction_mismatch": any_spec_axis_failed,
    }


def smoke_stream_contract(U: np.ndarray) -> dict:
    """Instantiate the v1 trainer on a mid stream; its ctor enforces the L2 contract."""
    bs = build_stream(StreamParams(), U)
    emb = bs.stream.astype(np.float32)
    norms = np.linalg.norm(emb, axis=1)
    try:
        OnlineTrainerV1(
            predictor=InnerPAM_v1_Primary(decoder_n_layers=2),
            embeddings=emb,
            config=TrainerConfig(arm_name="primary", stage="A",
                                 output_dir=RESULTS_PRE_A / "_contract_tmp",
                                 checkpoint_steps=(), final_step=0),
            device=torch.device("cpu"),
        )
        passed = True
        detail = "OnlineTrainerV1 accepted the synthetic stream (L2 contract satisfied)"
    except TrainingStopped as exc:
        passed = False
        detail = f"stream-contract failure: {exc}"
    return {
        "passed": bool(passed), "detail": detail,
        "norm_min": float(norms.min()), "norm_max": float(norms.max()),
        "stream_shape": list(emb.shape),
    }


def smoke_arch_forward(U: np.ndarray, device: torch.device) -> dict:
    """Forward one synthetic window through each arm; non-finite => STOP trigger 3."""
    bs = build_stream(StreamParams(), U)
    window = torch.from_numpy(bs.stream[:WINDOW_W][None, :, :]).float().to(device)
    arms = {
        "primary": InnerPAM_v1_Primary(decoder_n_layers=2),
        "ablation1": InnerPAM_v1_Ablation1(decoder_n_layers=2),
        "ablation2": InnerPAM_v1_Ablation2(),
    }
    results, all_ok = {}, True
    for name, model in arms.items():
        model.to(device).eval()
        with torch.no_grad():
            out = model(window)
        tensors = out if isinstance(out, (tuple, list)) else (out,)
        finite = all(bool(torch.isfinite(t).all()) for t in tensors)
        all_ok = all_ok and finite
        results[name] = {"output_finite": finite,
                         "output_shapes": [list(t.shape) for t in tensors]}
    return {"all_finite": all_ok, "arms": results}


def calibrate_training_steps(U: np.ndarray, device: torch.device) -> dict:
    """Train Primary on a mid stream; lock V2_TRAINING_STEPS at the loss plateau."""
    n_reps = (V2_SMOKE_MAX_STEPS + WINDOW_W + PREDICT_K) // 256 + 2
    bs = build_stream(StreamParams(period_P=256, n_repetitions=n_reps,
                                   magnitude_M=0.0, fidelity_F=0.999), U)
    emb = torch.from_numpy(bs.stream.astype(np.float32)).to(device)
    N = emb.shape[0]

    torch.manual_seed(0)
    pred = InnerPAM_v1_Primary(decoder_n_layers=2).to(device)
    opt = torch.optim.AdamW(pred.parameters(), lr=LR, weight_decay=WEIGHT_DECAY,
                            betas=ADAM_BETAS)
    pred.train()

    interval_means, steps_axis = [], []
    run_sum, run_cnt, steps_done = 0.0, 0, 0
    plateau_step = None
    t0 = time.time()
    last_t = min(N - PREDICT_K - 1, WINDOW_W - 1 + V2_SMOKE_MAX_STEPS)
    for t in range(WINDOW_W - 1, last_t):
        window = emb[t - WINDOW_W + 1 : t + 1].unsqueeze(0)        # (1, W, d)
        target = emb[t + 1 : t + 1 + PREDICT_K].unsqueeze(0)       # (1, K, d)
        mean, log_var = pred(window)
        loss = path_prediction_loss(mean, log_var, target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(pred.parameters(), GRAD_CLIP_MAX_NORM)
        opt.step()
        run_sum += float(loss.item()); run_cnt += 1; steps_done += 1

        if run_cnt >= V2_SMOKE_CHECKPOINT_EVERY:
            interval_means.append(run_sum / run_cnt)
            steps_axis.append(steps_done)
            run_sum, run_cnt = 0.0, 0
            # Plateau: 2 consecutive intervals with rel improvement below threshold.
            if len(interval_means) >= 3:
                imp1 = (interval_means[-2] - interval_means[-1]) / abs(interval_means[-2] + 1e-12)
                imp0 = (interval_means[-3] - interval_means[-2]) / abs(interval_means[-3] + 1e-12)
                if imp1 < V2_PLATEAU_REL_IMPROVEMENT and imp0 < V2_PLATEAU_REL_IMPROVEMENT:
                    plateau_step = steps_axis[-2]
                    break
    wall = time.time() - t0

    if plateau_step is None:
        plateau_step = steps_done  # never plateaued within cap
        plateau_detected = False
    else:
        plateau_detected = True
    # Round to a clean multiple of the checkpoint cadence.
    v2_training_steps = int(round(plateau_step / V2_SMOKE_CHECKPOINT_EVERY)
                            * V2_SMOKE_CHECKPOINT_EVERY)
    v2_training_steps = max(V2_SMOKE_CHECKPOINT_EVERY, v2_training_steps)
    return {
        "v2_training_steps": v2_training_steps,
        "plateau_detected": plateau_detected,
        "plateau_step_raw": int(plateau_step),
        "interval_means": interval_means,
        "interval_step_axis": steps_axis,
        "checkpoint_every": V2_SMOKE_CHECKPOINT_EVERY,
        "rel_improvement_threshold": V2_PLATEAU_REL_IMPROVEMENT,
        "smoke_max_steps": V2_SMOKE_MAX_STEPS,
        "steps_trained": steps_done,
        "wall_clock_seconds": wall,
        "ms_per_step": 1000.0 * wall / max(steps_done, 1),
        "device": str(device),
        "stream_shape": list(bs.stream.shape),
    }


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    RESULTS_PRE_A.mkdir(parents=True, exist_ok=True)
    print(f"[pre_a] device={device}")
    U = load_or_create_U()
    th = load_calibrated_thresholds()

    print("[pre_a] running sanity sweep ...")
    sweep = run_sanity_sweep(U, th)

    print("[pre_a] stream-contract smoke ...")
    contract = smoke_stream_contract(U)

    print("[pre_a] arch-forward smoke (3 arms) ...")
    arch = smoke_arch_forward(U, device)

    # STOP triggers (instr §7.1).
    stops = []
    if sweep["stop_trigger_1_construction_mismatch"]:
        stops.append("TRIGGER 1: construction-vs-measurement mismatch on >1 value of a spec axis")
    if not contract["passed"]:
        stops.append("TRIGGER 2: stream contract failed on synthetic stream")
    if not arch["all_finite"]:
        stops.append("TRIGGER 3: v1 predictor produced non-finite output on synthetic input")

    calib = None
    if not stops:
        print(f"[pre_a] calibrating V2_TRAINING_STEPS (cap {V2_SMOKE_MAX_STEPS}) ...")
        calib = calibrate_training_steps(U, device)
        write_v2_training_steps(calib["v2_training_steps"], {
            "source": "V2-PRE-A smoke run",
            "plateau_detected": calib["plateau_detected"],
            "plateau_step_raw": calib["plateau_step_raw"],
            "rel_improvement_threshold": calib["rel_improvement_threshold"],
        })
    else:
        print("[pre_a] STOP triggered — skipping V2_TRAINING_STEPS calibration")

    # Write outputs.
    (RESULTS_PRE_A / "construction_sanity_report.json").write_text(
        json.dumps(sweep, indent=2))
    (RESULTS_PRE_A / "stream_contract_smoke.json").write_text(json.dumps({
        "stream_contract": contract,
        "arch_forward_smoke": arch,
        "v2_training_steps_calibration": calib,
    }, indent=2))

    # Summary.
    print("\n[pre_a] === SANITY SWEEP ===")
    for axis, rep in sweep["axes"].items():
        tag = "SPEC" if rep["is_spec_axis"] else "supp"
        print(f"  [{tag}] {axis:<22} {rep['n_within_tolerance']}/{rep['n_values']} "
              f"within tol  -> {'OK' if rep['axis_ok'] else 'FAIL'}")
    print(f"[pre_a] stream contract: {'PASS' if contract['passed'] else 'FAIL'} "
          f"(norms [{contract['norm_min']:.6f}, {contract['norm_max']:.6f}])")
    print(f"[pre_a] arch-forward smoke: {'PASS' if arch['all_finite'] else 'FAIL'}")
    if calib:
        print(f"[pre_a] V2_TRAINING_STEPS = {calib['v2_training_steps']} "
              f"(plateau_detected={calib['plateau_detected']}, "
              f"{calib['ms_per_step']:.1f} ms/step on {calib['device']})")
    if stops:
        print("\n[pre_a] STOP TRIGGERS FIRED:")
        for s in stops:
            print("   -", s)
        return 1
    print("\n[pre_a] PRE-A complete: no STOP triggers.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

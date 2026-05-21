"""V2 Phase 1 sub-phase 1.1 — harness validation smoke (instr §7.1).

Validates the Phase 1 machinery before the controls/main-effects sweep:
  step 3  reproducibility   — re-run PRE-D2 magnitude@0.3 L_d=2 seed0; report the
                              CONCRETE Diff_μ delta vs the stored 0.020366676151752472
                              alongside the tolerance (n=10 CI half-width, or 0.02,
                              whichever is wider). PyTorch+CUDA is not bit-deterministic;
                              this checks the expected stochastic neighbourhood.
  step 6  eval semantics    — compute_diff_metrics returns per-stream-point scalars
                              (Diff_μ, Diff_σ), not per-(item,ordinal) arrays.
  step 5  classification    — classify_cell replicates PRE-D2's per-point n10/n20
                              classifications exactly (same thresholds + bootstrap).
  step 4  parallelism       — 2x/3x concurrency wall-clock + VRAM (§9.1/§9.2) -> lock.

Writes results/phase1/smoke_validation.json. STOP triggers per §7.1.
"""

from __future__ import annotations

import json
import os
import pathlib

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from v1.src.predictor.inner_pam_v1_primary import InnerPAM_v1_Primary
from v2.config import RESULTS_PRE_D2, RESULTS_ROOT, get_v2_training_steps
from v2.src.phase1 import parallel_harness as ph
from v2.src.phase1.arm_runner import run_arm
from v2.src.phase1.classification import classify_cell, load_thresholds
from v2.src.phase1.sweep_grid import cell_params
from v2.src.preflight.pre_d1a_endpoint_stability import compute_diff_metrics
from v2.src.preflight.pre_d2_n_validation import params_for
from v2.src.substrate.base_manifold_trajectory import load_or_create_U
from v2.src.substrate.stream_builder import StreamParams, build_stream

REPO_ROOT = str(pathlib.Path(__file__).resolve().parents[2])
PRE_D2_MAG03_SEED0 = 0.020366676151752472   # PRE-D2 points[0].diff_mu_values[0]


def step3_reproducibility(U, device, steps) -> dict:
    pt0 = json.loads((RESULTS_PRE_D2 / "n_validation_report.json").read_text())["points"][0]
    ci = pt0["classification_n10"]["head_mu"]
    half_ci = (ci["ci_high"] - ci["ci_low"]) / 2.0
    tolerance = max(half_ci, 0.02)

    params = params_for("magnitude", 0.3)        # exact PRE-D2 magnitude@0.3 construction
    r = run_arm(params, L_d_main=2, seed=0, training_steps=steps, U=U, device=device,
                label="smoke_repro_mag@0.3", axis="magnitude", endpoint="0.3")
    delta = abs(r.diff_mu - PRE_D2_MAG03_SEED0)
    return {
        "ref_diff_mu_seed0": PRE_D2_MAG03_SEED0,
        "reproduced_diff_mu": r.diff_mu,
        "delta": delta,
        "tolerance": tolerance,
        "tolerance_basis": (f"n=10 CI half-width (CI=[{ci['ci_low']:.5f},{ci['ci_high']:.5f}] "
                            f"-> {half_ci:.5f}) vs 0.02, whichever wider. NOTE: instructions' "
                            f"parenthetical '~0.063' mis-derives this half-width (=0.113); "
                            f"0.063 ~ tau_W_mu, a conflation."),
        "passed": bool(delta <= tolerance),
        "stability_flag": r.stability_flag,
        "nan_inf": r.nan_inf,
    }


def step6_eval_semantics(U, device) -> dict:
    bs = build_stream(StreamParams(period_P=256, n_repetitions=4, magnitude_M=0.3), U)
    eval_t = torch.from_numpy(bs.stream[:512].astype(np.float32)).to(device)
    pred = InnerPAM_v1_Primary(decoder_n_layers=2).to(device).eval()
    diff_mu, diff_sigma, var_collapse = compute_diff_metrics(pred, eval_t, device)
    is_scalar = (np.isscalar(diff_mu) or isinstance(diff_mu, float)) and \
                (np.isscalar(diff_sigma) or isinstance(diff_sigma, float))
    finite = bool(np.isfinite(diff_mu) and np.isfinite(diff_sigma))
    return {
        "diff_mu": float(diff_mu), "diff_sigma": float(diff_sigma),
        "per_stream_point_scalars": bool(is_scalar), "finite": finite,
        "passed": bool(is_scalar and finite),
        "note": "untrained net (semantics/shape check, not value calibration)",
    }


def step5_classification_replication(th) -> dict:
    report = json.loads((RESULTS_PRE_D2 / "n_validation_report.json").read_text())
    per_point, all_match = [], True
    for pt in report["points"]:
        mu, sg = pt["diff_mu_values"], pt["diff_sigma_values"]
        rec = {"label": pt["label"]}
        for n, key in ((10, "classification_n10"), (20, "classification_n20")):
            mine = classify_cell(mu, sg, th, n)
            stored = pt[key]
            match = (mine["head_mu"]["category"] == stored["head_mu"]["category"]
                     and mine["head_sigma"]["category"] == stored["head_sigma"]["category"]
                     and mine["band_resident"] == stored["band_resident"])
            all_match = all_match and match
            rec[f"n{n}"] = {"mine_overall": mine["overall"],
                            "mine_mu": mine["head_mu"]["category"],
                            "mine_sigma": mine["head_sigma"]["category"],
                            "stored_mu": stored["head_mu"]["category"],
                            "stored_sigma": stored["head_sigma"]["category"],
                            "match": bool(match)}
        per_point.append(rec)
    return {"all_match": bool(all_match), "n_points": len(per_point),
            "thresholds_used": th, "per_point": per_point}


def step4_parallelism(steps, out_dir) -> dict:
    mid = cell_params("mid", "mid", "mid", "mid", "mid")     # P=256,D=16,center39,mag0.3,loc0.5
    base = {"params": {k: getattr(mid, k) for k in
                       ("period_P", "manifold_dim_D", "continuity_center",
                        "fidelity_F", "magnitude_M", "locality_L")},
            "L_d_main": 2, "seed": 0, "training_steps": steps}

    m1 = ph.measure_concurrency(base, 1, REPO_ROOT, out_dir)
    m2 = ph.measure_concurrency(base, 2, REPO_ROOT, out_dir)
    ratio2 = m2["wall_s"] / max(m1["wall_s"], 1e-9)
    vram_ok2 = (m2["vram_peak_mb"] and m2["vram_total_mb"]
                and m2["vram_peak_mb"] * 1.5 <= m2["vram_total_mb"])

    result = {"baseline_n1": m1, "n2": m2, "ratio_2x": round(ratio2, 3),
              "vram_ok_2x": bool(vram_ok2)}

    stop = None
    if m2["oom"]:
        stop = "2x OOM"
        locked = 1
    elif ratio2 > 1.5:
        stop = f"2x wall-clock {ratio2:.2f}x baseline (>1.5x)"
        locked = 1
    elif ratio2 <= 1.2 and vram_ok2:
        # escalate to 3x
        m3 = ph.measure_concurrency(base, 3, REPO_ROOT, out_dir)
        ratio3 = m3["wall_s"] / max(m1["wall_s"], 1e-9)
        vram_ok3 = (m3["vram_peak_mb"] and m3["vram_total_mb"]
                    and m3["vram_peak_mb"] * 1.5 <= m3["vram_total_mb"])
        result.update({"n3": m3, "ratio_3x": round(ratio3, 3), "vram_ok_3x": bool(vram_ok3)})
        if m3["oom"]:
            stop = "3x OOM"
            locked = 2
        else:
            locked = 3 if (ratio3 <= 1.4 and vram_ok3) else 2
    else:
        locked = 2   # 2x acceptable (1.2 < ratio <= 1.5) but don't escalate

    result["parallelism_locked"] = locked
    result["stop"] = stop
    return result


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = RESULTS_ROOT / "phase1"
    out_dir.mkdir(parents=True, exist_ok=True)
    steps = get_v2_training_steps()
    th = load_thresholds()
    U = load_or_create_U()
    print(f"[smoke] device={device} training_steps={steps}")

    print("[smoke] step 3: PRE-D2 magnitude@0.3 reproducibility ...")
    s3 = step3_reproducibility(U, device, steps)
    print(f"[smoke]   delta={s3['delta']:.5f}  tolerance={s3['tolerance']:.5f}  "
          f"-> {'PASS' if s3['passed'] else 'FAIL'}")

    print("[smoke] step 6: eval-metric semantics ...")
    s6 = step6_eval_semantics(U, device)
    print(f"[smoke]   per-stream-point scalars={s6['per_stream_point_scalars']} "
          f"diff_mu={s6['diff_mu']:.4g} -> {'PASS' if s6['passed'] else 'FAIL'}")

    print("[smoke] step 5: classification replication vs PRE-D2 ...")
    s5 = step5_classification_replication(th)
    print(f"[smoke]   all_match={s5['all_match']} ({s5['n_points']} points x n10/n20) "
          f"-> {'PASS' if s5['all_match'] else 'FAIL'}")

    print("[smoke] step 4: parallelism (2x/3x) ...")
    s4 = step4_parallelism(min(3000, steps), str(out_dir / "_concurrency_tmp"))
    print(f"[smoke]   ratio_2x={s4['ratio_2x']} "
          + (f"ratio_3x={s4.get('ratio_3x')} " if "ratio_3x" in s4 else "")
          + f"-> locked {s4['parallelism_locked']}x")

    stops = []
    if not s3["passed"]:
        stops.append("STEP3: reproducibility delta exceeds tolerance")
    if not s6["passed"]:
        stops.append("STEP6: eval metric not per-stream-point scalar")
    if not s5["all_match"]:
        stops.append("STEP5: classification disagrees with PRE-D2")
    if s4["stop"]:
        stops.append(f"STEP4: {s4['stop']}")

    payload = {"training_steps": steps, "device": str(device),
               "step3_reproducibility": s3, "step6_eval_semantics": s6,
               "step5_classification_replication": {k: v for k, v in s5.items() if k != "per_point"},
               "step5_per_point": s5["per_point"],
               "step4_parallelism": s4,
               "stop_triggers": stops, "all_passed": not stops}
    (out_dir / "smoke_validation.json").write_text(json.dumps(payload, indent=2, default=str))
    print(f"[smoke] wrote {out_dir / 'smoke_validation.json'}")
    if stops:
        print("[smoke] STOP TRIGGERS:")
        for s in stops:
            print("   -", s)
        return 1
    print(f"[smoke] ALL PASS. parallelism locked = {s4['parallelism_locked']}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

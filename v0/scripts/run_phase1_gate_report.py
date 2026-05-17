"""Phase 1 gate report — disaggregated diagnostics for G1.1-G1.5 + S1-S4.

Per the session-2 protocol agreed with the reviewer:
  - G1.1 (no NaN/Inf) and G1.2 (loss decreased) are MECHANICAL — auto pass/fail
    against the documented criteria.
  - G1.3, G1.4, G1.5 carry SCAFFOLDING thresholds and are NOT auto-passed.
    The script produces the disaggregated numbers (per-shape, per-position,
    per-step k); the experiment chat decides pass/fail with optional
    recalibration justified in writing.

Outputs:
  results/inner_pam_v0/phase1_main/m3_trajectory.json
  results/inner_pam_v0/phase1_main/gate_report.json
  results/inner_pam_v0/phase1_main/gate_report.md
  results/inner_pam_v0/phase1_shuffle/sanity_check.md
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy import stats

REPO_ROOT = Path("/mnt/c/Users/Jason/Desktop/Eridos/Weft 2")
sys.path.insert(0, str(REPO_ROOT))

from v0.src.config import (  # noqa: E402
    G1_3_LOG_VAR_SEPARATION,
    G1_4_DIAGNOSTIC_HORIZONS,
    G1_4_GATE_HORIZON_K,
    G1_5_M3_FLOOR,
    GATE_ALPHA,
    HELD_OUT_LOOPS,
    NORMALITY_ALPHA,
    PATHS,
    PHASE1,
    PREDICT_K,
    SEED_PROBE_SAMPLING,
)
from v0.src.eval.controls import (  # noqa: E402
    s1_log_var_distribution,
    s2_m1_distribution,
    s3_cluster_sharpness,
    s4_quantitative_collapse_check,
    shuffle_sanity_check,
)
from v0.src.eval.metrics import (  # noqa: E402
    ProbeResult,
    m1_centreline_accuracy,
    m1_per_probe_centreline_at_k,
    m3_cluster_sharpness,
    run_probes_through_predictor,
)
from v0.src.eval.probes import build_probes, compute_held_out_boundary  # noqa: E402
from v0.src.predictor.inner_pam import InnerPAM  # noqa: E402


MAIN_DIR = PHASE1.results_main
SHUFFLE_DIR = PHASE1.results_shuffle


def _load_annotations(path: Path) -> list[dict]:
    out: list[dict] = []
    with path.open("r") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _find_checkpoints(dir_: Path) -> list[tuple[int, Path]]:
    out: list[tuple[int, Path]] = []
    for p in dir_.glob("ckpt_*.pt"):
        step = int(p.stem.split("_")[1])
        out.append((step, p))
    return sorted(out)


def _run_predictor_on_probes(
    ckpt_path: Path, embeddings: np.ndarray, probes: list, device: torch.device,
) -> list[ProbeResult]:
    predictor = InnerPAM().to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    predictor.load_state_dict(ckpt["predictor_state"])
    predictor.eval()
    return run_probes_through_predictor(probes, embeddings, predictor, device)


def _per_probe_log_var_summary(results: list[ProbeResult]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    by_type: dict[str, list[float]] = {"steady": [], "cue": []}
    for r in results:
        by_type[r.probe.probe_type].append(float(r.pred_log_var.mean()))
    for t, vals in by_type.items():
        arr = np.asarray(vals, dtype=np.float64)
        out[t] = {
            "n": int(arr.size),
            "mean": float(arr.mean()) if arr.size else float("nan"),
            "std": float(arr.std(ddof=0)) if arr.size else float("nan"),
            "median": float(np.median(arr)) if arr.size else float("nan"),
        }
    return out


def _paired_test_per_probe(
    main_cos: np.ndarray, shuffle_cos: np.ndarray
) -> dict[str, Any]:
    """One-sided paired test of main > shuffle, with Shapiro-Wilk fallback."""
    diff = main_cos - shuffle_cos
    if diff.size < 3:
        return {
            "n": int(diff.size),
            "mean_diff": float(diff.mean()) if diff.size else float("nan"),
            "test": "insufficient_n",
            "p_value": float("nan"),
            "pass_at_p_lt_01": False,
        }
    shapiro_stat, shapiro_p = stats.shapiro(diff)
    if shapiro_p < NORMALITY_ALPHA:
        test_name = "wilcoxon"
        try:
            w_stat, p_value = stats.wilcoxon(
                main_cos, shuffle_cos, alternative="greater"
            )
        except ValueError as e:
            return {
                "n": int(diff.size),
                "mean_diff": float(diff.mean()),
                "shapiro_stat": float(shapiro_stat),
                "shapiro_p": float(shapiro_p),
                "test": test_name,
                "test_error": str(e),
                "p_value": float("nan"),
                "pass_at_p_lt_01": False,
            }
    else:
        test_name = "paired_t"
        t_stat, p_value = stats.ttest_rel(
            main_cos, shuffle_cos, alternative="greater"
        )
    return {
        "n": int(diff.size),
        "mean_diff": float(diff.mean()),
        "std_diff": float(diff.std(ddof=0)),
        "shapiro_stat": float(shapiro_stat),
        "shapiro_p": float(shapiro_p),
        "test": test_name,
        "p_value": float(p_value),
        "pass_at_p_lt_01": bool(float(p_value) < GATE_ALPHA),
    }


def _check_nan_inf_in_state_dict(state_dict: dict) -> dict[str, Any]:
    bad: list[str] = []
    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor) and not torch.isfinite(v).all().item():
            bad.append(k)
    return {"any_non_finite": len(bad) > 0, "non_finite_keys": bad[:20]}


def _trajectory_monotonic_count(values: list[float]) -> dict[str, Any]:
    """Count non-decreasing transitions over the last 9."""
    if len(values) < 2:
        return {"n_transitions": 0, "n_non_decreasing": 0, "n_dips_allowed": 2}
    transitions = list(zip(values[:-1], values[1:]))
    last_9 = transitions[-9:] if len(transitions) >= 9 else transitions
    n_non_dec = sum(1 for a, b in last_9 if b >= a)
    return {
        "n_transitions_considered": len(last_9),
        "n_non_decreasing": int(n_non_dec),
        "n_dips_allowed": 2,
        "criterion_pass": bool(n_non_dec >= len(last_9) - 2),
    }


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[gate] device {device}", flush=True)

    annotations = _load_annotations(PHASE1.annotations)
    embeddings = np.load(PHASE1.embeddings)
    held_out_start, held_out_end = compute_held_out_boundary(annotations, HELD_OUT_LOOPS)
    probes_by_type = build_probes(
        annotations, held_out_start, held_out_end, seed=SEED_PROBE_SAMPLING
    )
    all_probes = probes_by_type["steady"] + probes_by_type["cue"]
    print(f"[gate] probes: steady={len(probes_by_type['steady'])} "
          f"cue={len(probes_by_type['cue'])} total={len(all_probes)}", flush=True)

    main_ckpts = _find_checkpoints(MAIN_DIR)
    shuffle_ckpts = _find_checkpoints(SHUFFLE_DIR)
    if not main_ckpts:
        print("[gate] FAIL: no main checkpoints found", file=sys.stderr)
        return 1
    print(f"[gate] main ckpts: {[s for s, _ in main_ckpts]}", flush=True)
    print(f"[gate] shuffle ckpts: {[s for s, _ in shuffle_ckpts]}", flush=True)

    # ---- Per-checkpoint analyses ------------------------------------------
    m3_trajectory: list[dict[str, Any]] = []
    log_var_trajectory: list[dict[str, Any]] = []
    for step, ckpt_path in main_ckpts:
        results = _run_predictor_on_probes(ckpt_path, embeddings, all_probes, device)
        m3 = m3_cluster_sharpness(results)
        lv = _per_probe_log_var_summary(results)
        m3_trajectory.append({
            "step": step,
            "cluster_sharpness": m3["cluster_sharpness"],
            "within_cluster_cosine_mean": m3["within_cluster_cosine_mean"],
            "cross_cluster_cosine_mean": m3["cross_cluster_cosine_mean"],
            "within_per_cluster": m3["within_per_cluster"],
            "n_clusters": m3["n_clusters"],
            "steady_state_mean_log_var": m3["steady_state_mean_log_var"],
        })
        log_var_trajectory.append({
            "step": step,
            "steady": lv["steady"],
            "cue": lv["cue"],
        })
        print(f"[gate] main ckpt {step}: M3 sharpness "
              f"{m3['cluster_sharpness']:.4f}", flush=True)

    (MAIN_DIR / "m3_trajectory.json").write_text(json.dumps(m3_trajectory, indent=2))
    (MAIN_DIR / "log_var_trajectory.json").write_text(
        json.dumps(log_var_trajectory, indent=2)
    )

    # ---- G1.1 — NaN/Inf check --------------------------------------------
    final_step, final_ckpt = main_ckpts[-1]
    state = torch.load(final_ckpt, map_location="cpu", weights_only=False)
    nan_check = _check_nan_inf_in_state_dict(state["predictor_state"])
    final_summary = json.loads(
        (MAIN_DIR / f"checkpoint_{final_step}.json").read_text()
    )
    g1_1 = {
        "predictor_state_any_non_finite": nan_check["any_non_finite"],
        "predictor_state_non_finite_keys": nan_check["non_finite_keys"],
        "final_loss_finite": bool(np.isfinite(final_summary["mean_loss_last_1k"])),
        "verdict": "PASS" if (
            not nan_check["any_non_finite"]
            and bool(np.isfinite(final_summary["mean_loss_last_1k"]))
        ) else "FAIL",
    }

    # ---- G1.2 — loss decreased --------------------------------------------
    first_summary = json.loads(
        (MAIN_DIR / f"checkpoint_{main_ckpts[0][0]}.json").read_text()
    )
    g1_2 = {
        "first_ckpt": main_ckpts[0][0],
        "first_ckpt_mean_loss_last_1k": float(first_summary["mean_loss_last_1k"]),
        "final_ckpt": final_step,
        "final_ckpt_mean_loss_last_1k": float(final_summary["mean_loss_last_1k"]),
        "decrease": float(
            first_summary["mean_loss_last_1k"]
            - final_summary["mean_loss_last_1k"]
        ),
        "verdict": (
            "PASS"
            if final_summary["mean_loss_last_1k"]
            < first_summary["mean_loss_last_1k"]
            else "FAIL"
        ),
    }

    # ---- G1.3 — variance learned structure --------------------------------
    final_lv = log_var_trajectory[-1]
    sep_main = final_lv["cue"]["mean"] - final_lv["steady"]["mean"]
    g1_3 = {
        "final_step": final_step,
        "main": {
            "steady_mean_log_var": final_lv["steady"]["mean"],
            "cue_mean_log_var": final_lv["cue"]["mean"],
            "separation_cue_minus_steady": sep_main,
        },
        "scaffolding_threshold_absolute": G1_3_LOG_VAR_SEPARATION,
        "verdict_against_scaffolding_threshold_absolute": (
            "PASS" if sep_main > G1_3_LOG_VAR_SEPARATION else "FAIL"
        ),
        "note": (
            "SCAFFOLDING threshold (+0.3 absolute separation, cue > steady). "
            "Pause for experiment-chat review per session-2 protocol."
        ),
    }

    # ---- G1.4 — multi-step centreline beats shuffle, paired ---------------
    # Need same probe set on main and shuffle at final ckpt.
    main_results = _run_predictor_on_probes(
        final_ckpt, embeddings, all_probes, device
    )
    g1_4: dict[str, Any] = {
        "final_step": final_step,
        "n_steady_probes": len([r for r in main_results if r.probe.probe_type == "steady"]),
        "n_cue_probes": len([r for r in main_results if r.probe.probe_type == "cue"]),
        "per_k": {},
    }
    shuffle_sanity = None
    shuffle_results: list[ProbeResult] = []
    if shuffle_ckpts:
        sh_step, sh_ckpt = shuffle_ckpts[-1]
        shuffle_results = _run_predictor_on_probes(
            sh_ckpt, embeddings, all_probes, device
        )
        # Augment G1.3 with main-vs-shuffle log_var comparison: does the main
        # predictor's variance structure differ from a temporally-destroyed
        # shuffle's? This is more interpretable than the absolute cue-steady
        # separation when the substrate (un-jittered dwell) is degenerate.
        sh_lv = _per_probe_log_var_summary(shuffle_results)
        g1_3["shuffle"] = {
            "steady_mean_log_var": sh_lv["steady"]["mean"],
            "cue_mean_log_var": sh_lv["cue"]["mean"],
            "separation_cue_minus_steady": (
                sh_lv["cue"]["mean"] - sh_lv["steady"]["mean"]
            ),
        }
        g1_3["main_minus_shuffle_log_var"] = {
            "steady": (
                final_lv["steady"]["mean"] - sh_lv["steady"]["mean"]
            ),
            "cue": (
                final_lv["cue"]["mean"] - sh_lv["cue"]["mean"]
            ),
        }
        # All horizons we report (k=1, 8, 16 in 1-indexed; convert to 0-indexed).
        for k1 in (1, 8, 16):
            k0 = k1 - 1
            # Gate is on STEADY probes only per instructions §7.7 ("across the 250
            # steady-state probes"); cue probes reported separately for context.
            main_steady = [r for r in main_results if r.probe.probe_type == "steady"]
            sh_steady = [r for r in shuffle_results if r.probe.probe_type == "steady"]
            main_cos_steady = np.asarray([
                float(
                    np.dot(
                        r.pred_mean[k0] / max(np.linalg.norm(r.pred_mean[k0]), 1e-12),
                        r.target[k0] / max(np.linalg.norm(r.target[k0]), 1e-12),
                    )
                )
                for r in main_steady
            ])
            sh_cos_steady = np.asarray([
                float(
                    np.dot(
                        r.pred_mean[k0] / max(np.linalg.norm(r.pred_mean[k0]), 1e-12),
                        r.target[k0] / max(np.linalg.norm(r.target[k0]), 1e-12),
                    )
                )
                for r in sh_steady
            ])
            g1_4["per_k"][f"k_{k1}_steady"] = _paired_test_per_probe(
                main_cos_steady, sh_cos_steady
            )
        g1_4["gated_horizon_k"] = G1_4_GATE_HORIZON_K
        g1_4["diagnostic_horizons"] = list(G1_4_DIAGNOSTIC_HORIZONS)
        gated = g1_4["per_k"].get(f"k_{G1_4_GATE_HORIZON_K}_steady", {})
        g1_4["verdict_at_gated_horizon"] = (
            "PASS" if gated.get("pass_at_p_lt_01") else "FAIL"
        )
    else:
        g1_4["verdict_at_gated_horizon"] = "NO_SHUFFLE_CHECKPOINTS"

    # ---- G1.5 — M3 trajectory + floor ------------------------------------
    sharpness_series = [m["cluster_sharpness"] for m in m3_trajectory]
    g1_5 = {
        "sharpness_per_checkpoint": [
            {"step": m["step"], "sharpness": m["cluster_sharpness"]}
            for m in m3_trajectory
        ],
        "sharpness_first": sharpness_series[0] if sharpness_series else float("nan"),
        "sharpness_final": sharpness_series[-1] if sharpness_series else float("nan"),
        "sharpness_grew": (
            bool(sharpness_series[-1] > sharpness_series[0])
            if len(sharpness_series) >= 2 else None
        ),
        "trajectory_monotonic_check": _trajectory_monotonic_count(sharpness_series),
        "scaffolding_floor": G1_5_M3_FLOOR,
        "floor_pass": (
            bool(sharpness_series[-1] > G1_5_M3_FLOOR) if sharpness_series else False
        ),
        "verdict_against_scaffolding_threshold": (
            "PASS"
            if (
                sharpness_series
                and sharpness_series[-1] > G1_5_M3_FLOOR
                and len(sharpness_series) >= 2
                and sharpness_series[-1] > sharpness_series[0]
                and _trajectory_monotonic_count(sharpness_series)["criterion_pass"]
            )
            else "FAIL"
        ),
        "note": (
            "SCAFFOLDING floor (>0.10). Trajectory criterion: monotonically "
            "non-decreasing on >=7 of last 9 transitions. Pause for "
            "experiment-chat review per session-2 protocol."
        ),
    }

    # ---- S1-S4 shuffle sanity check --------------------------------------
    if shuffle_results:
        shuffle_sanity = shuffle_sanity_check(main_results, shuffle_results)
        (SHUFFLE_DIR / "sanity_check.json").write_text(
            json.dumps(shuffle_sanity, indent=2)
        )

    # ---- Final gate report ------------------------------------------------
    gate_report = {
        "phase": "phase1",
        "final_step": final_step,
        "n_main_ckpts": len(main_ckpts),
        "n_shuffle_ckpts": len(shuffle_ckpts),
        "probes": {
            "n_steady": len(probes_by_type["steady"]),
            "n_cue": len(probes_by_type["cue"]),
            "held_out_region": [held_out_start, held_out_end],
        },
        "G1_1": g1_1,
        "G1_2": g1_2,
        "G1_3": g1_3,
        "G1_4": g1_4,
        "G1_5": g1_5,
        "shuffle_sanity_check": shuffle_sanity,
        "session2_protocol_note": (
            "G1.1 and G1.2 are MECHANICAL verdicts (PASS/FAIL against the "
            "documented criteria). G1.3, G1.4, G1.5 carry SCAFFOLDING thresholds "
            "and are NOT autonomously declared pass by this script — the verdict "
            "fields against the scaffolding thresholds are reported for the "
            "experiment chat's review with optional recalibration justified "
            "in writing."
        ),
    }
    (MAIN_DIR / "gate_report.json").write_text(json.dumps(gate_report, indent=2))
    print(f"[gate] wrote {MAIN_DIR / 'gate_report.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""V2-PRE-E runner — SCAFFOLDING calibration (instr §7.6; analytical).

STOP triggers:
  1. tau_W calibration target unreachable (no margin yields per-head p<0.05).
  2. PRE-B worked-example coverage below the calibrated coverage threshold
     (contradicts spec §2.5 by-construction claim).
"""

from __future__ import annotations

import json

from v2.config import BLOCK_SIZE_LONG, REF_NEIGHBOURHOOD_WINDOW, RESULTS_PRE_E
from v2.src.preflight import pre_e_scaffolding_calibration as pe
from v2.src.substrate.base_manifold_trajectory import load_or_create_U


def main() -> int:
    RESULTS_PRE_E.mkdir(parents=True, exist_ok=True)
    U = load_or_create_U()
    print("[pre_e] calibrating thresholds (analytical; PRE-A primitives, no training) ...")

    resid = pe.calibrate_residual_thresholds(U)
    rep = pe.calibrate_repetition_thresholds(U)
    cov = pe.calibrate_coverage_threshold(U)
    bic = pe.calibrate_bic_threshold()
    win = pe.calibrate_local_pca_window(U)
    tau_w = pe.calibrate_tau_w()
    cov_check = pe.post_coverage_check()

    thresholds = {
        "TAU_R": rep["TAU_R"], "REPETITION_NOISE_FLOOR": rep["REPETITION_NOISE_FLOOR"],
        "TAU_L": resid["TAU_L"], "TAU_PERT": resid["TAU_PERT"],
        "REPETITION_COVERAGE_THRESHOLD": cov["REPETITION_COVERAGE_THRESHOLD"],
        "BIC_IMPROVEMENT_THRESHOLD": bic["BIC_IMPROVEMENT_THRESHOLD"],
        "LOCAL_PCA_WINDOW": win["LOCAL_PCA_WINDOW"],
        "MANIFOLD_SUBSAMPLE_RATE": win["MANIFOLD_SUBSAMPLE_RATE"],
        "REF_NEIGHBOURHOOD_WINDOW": REF_NEIGHBOURHOOD_WINDOW,   # from PRE-A locality design
        "BLOCK_SIZE_LONG": BLOCK_SIZE_LONG,                     # carried (not exercised)
    }

    stops = []
    if tau_w["stop"]:
        stops.append(tau_w["stop"])
    cov_thr = thresholds["REPETITION_COVERAGE_THRESHOLD"]
    pre_b_cov = cov_check["pre_b_worked_example_coverage"]
    if pre_b_cov < cov_thr:
        stops.append(f"PRE-B worked-example coverage {pre_b_cov} < calibrated threshold "
                     f"{cov_thr} (contradicts spec §2.5 by-construction claim).")

    payload = {
        "thresholds": thresholds,
        "tau_W": tau_w,
        "post_coverage_check": {"calibrated_threshold": cov_thr,
                                "pre_b_coverage": pre_b_cov, "clears": pre_b_cov >= cov_thr},
        "provenance": {
            "residual_thresholds": resid, "repetition_thresholds": rep,
            "coverage_threshold": cov, "bic_threshold": bic, "local_pca_window": win,
            "tau_W_rule": "smallest margin where one-sample Wilcoxon signed-rank finds the "
                          "PRE-D1a n=20 baseline significantly below (median + tau_W) at p<0.05",
            "note": "REF_NEIGHBOURHOOD_WINDOW carried from PRE-A locality design; "
                    "BLOCK_SIZE_LONG carried (not exercised at this scale).",
        },
        "stop_triggers": stops,
    }
    (RESULTS_PRE_E / "scaffolding_calibration.json").write_text(json.dumps(payload, indent=2))

    print("\n[pre_e] === CALIBRATED THRESHOLDS ===")
    for k, v in thresholds.items():
        print(f"  {k:<32} {v}")
    print(f"  tau_W (mu)   {tau_w['per_head']['mu']['tau_W']}  (p={tau_w['per_head']['mu']['p_value']})")
    print(f"  tau_W (sigma){tau_w['per_head']['sigma']['tau_W']}  (p={tau_w['per_head']['sigma']['p_value']})")
    print(f"  coverage check: PRE-B {pre_b_cov} vs threshold {cov_thr} -> "
          f"{'clears' if pre_b_cov >= cov_thr else 'BELOW (STOP)'}")
    if stops:
        print("\n[pre_e] STOP TRIGGERS:")
        for s in stops:
            print("   -", s)
        return 1
    print("\n[pre_e] PRE-E complete: thresholds locked, no STOP.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

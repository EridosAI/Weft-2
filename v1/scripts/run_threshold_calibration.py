#!/usr/bin/env python3
"""Threshold calibration + verdict-pattern recommendation (spec §10.4.3, instr §5.4 / §9.2-9.3).

Reads the per-(item, ordinal) × arm matrix produced by
`run_per_item_ordinal_eval.py` and:

  1. Builds stability + differentiation reference distributions
     (instr §5.4 steps 1-2).
  2. Anchors percentile thresholds (instr §5.4 step 3).
  3. Classifies each cell into stable / differentiated / coupling /
     indeterminate (instr §5.4 step 4).
  4. Writes thresholds.json + distributions.json under
     `results/inner_pam_v1/threshold_calibration/`.
  5. Writes a verdict-pattern recommendation to
     `results/inner_pam_v1/verdict_recommendation/recommendation.json` per
     instr §9.3. CC produces a *recommendation*; final verdict assignment
     is done by the verdict-assignment chat per spec §11.1.4.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from v1.src.config import PATHS
from v1.src.eval.threshold_calibration import (
    METRICS_FOR_CALIBRATION,
    calibrate_thresholds,
    classify_cells,
    write_distributions,
    write_thresholds,
)


def _summarise_arm_classifications(
    matrix_classified: dict, arm: str
) -> dict:
    """Per-arm summary: for each metric, count classifications across
    bit-identical and input-varying rows."""
    out: dict = {"metric": {}, "bit_identical_summary": {}}
    for metric in METRICS_FOR_CALIBRATION:
        bit_id_counter: Counter = Counter()
        inv_counter: Counter = Counter()
        for row in matrix_classified["rows"]:
            cell = row["cells"].get(arm, {})
            classes = cell.get("classifications", {})
            label = classes.get(metric)
            if label is None:
                continue
            if row["bit_identical"]:
                bit_id_counter[label] += 1
            else:
                inv_counter[label] += 1
        out["metric"][metric] = {
            "bit_identical": dict(bit_id_counter),
            "input_varying": dict(inv_counter),
        }
    return out


def _recommend_verdict(matrix_classified: dict) -> dict:
    """Match the classified matrix against the V1A-V1P patterns (instr §9.3).

    Heuristic per spec §10.4.2 / instr §9.3:
      - V1A: primary input-varying mean & variance mostly 'differentiated';
             primary bit-identical mostly 'stable';
             ablation 2 input-varying mostly 'coupling' (or stable on bit-id).
             Ablation 1 mean differentiated, variance weakened.
      - V1B: primary + ablation1 both co-primary differentiated; ablation 2 coupling.
      - V1C: all three arms show coupling on input-varying rows.
      - V1D: heterogeneous pattern; specific sub-pattern detected.
      - V1E: all three arms show co-primary differentiation.
      - V1P: substrate failure (flagged elsewhere; we only recognise it if
             the matrix is empty / malformed).

    This is a coarse recommendation. The verdict-assignment chat (spec
    §11.1.4) reviews the recommendation alongside the matrix and the
    distributions.json data to issue the actual verdict.
    """
    arms = matrix_classified.get("arms", [])
    if not matrix_classified.get("rows"):
        return {"verdict": "V1P", "confidence": "low", "rationale": "empty matrix"}

    per_arm = {arm: _summarise_arm_classifications(matrix_classified, arm) for arm in arms}

    def frac(arm: str, metric: str, bit_id: bool, label: str) -> float:
        key = "bit_identical" if bit_id else "input_varying"
        counts = per_arm.get(arm, {}).get("metric", {}).get(metric, {}).get(key, {})
        total = sum(counts.values())
        return counts.get(label, 0) / max(1, total)

    # Differentiation: input-varying pairs classed 'differentiated'.
    # Stability: bit-identical pairs classed 'stable'.
    # Coupling: input-varying pairs classed 'coupling' (i.e., stable values
    # at input-varying rows -- the v0 BCDD pattern).
    primary_mean_diff = frac("primary", "mean_drift", False, "differentiated")
    primary_var_diff = frac("primary", "variance_drift", False, "differentiated")
    primary_mean_stab = frac("primary", "mean_drift", True, "stable")
    primary_var_stab = frac("primary", "variance_drift", True, "stable")
    abl1_var_diff = frac("ablation1", "variance_drift", False, "differentiated")
    abl1_mean_diff = frac("ablation1", "mean_drift", False, "differentiated")
    abl2_mean_coupling = frac("ablation2", "mean_drift", False, "coupling")
    abl2_var_coupling = frac("ablation2", "variance_drift", False, "coupling")
    abl2_mean_diff = frac("ablation2", "mean_drift", False, "differentiated")
    abl2_var_diff = frac("ablation2", "variance_drift", False, "differentiated")

    # Threshold for "mostly" pattern: SCAFFOLDING heuristic — 0.6.
    M = 0.6

    # V1A check
    v1a = (
        primary_mean_diff > M
        and primary_var_diff > M
        and primary_mean_stab > M
        and primary_var_stab > M
        and abl1_mean_diff > M
        and abl1_var_diff < M
        and (abl2_mean_coupling > M or abl2_var_coupling > M)
    )
    # V1B check
    v1b = (
        primary_mean_diff > M
        and primary_var_diff > M
        and abl1_mean_diff > M
        and abl1_var_diff > M
        and (abl2_mean_coupling > M or abl2_var_coupling > M)
    )
    # V1E check
    v1e = (
        primary_mean_diff > M and primary_var_diff > M
        and abl1_mean_diff > M and abl1_var_diff > M
        and abl2_mean_diff > M and abl2_var_diff > M
    )
    # V1C check
    v1c = (
        primary_mean_diff < (1 - M)
        and primary_var_diff < (1 - M)
        and (abl2_mean_coupling > M or abl2_var_coupling > M)
    )

    if v1a:
        verdict, confidence = "V1A", "high"
    elif v1b:
        verdict, confidence = "V1B", "high"
    elif v1e:
        verdict, confidence = "V1E", "high"
    elif v1c:
        verdict, confidence = "V1C", "high"
    elif primary_mean_diff > M and primary_var_diff < (1 - M):
        verdict, confidence = "V1D-mean-only", "medium"
    elif primary_var_diff > M and primary_mean_diff < (1 - M):
        verdict, confidence = "V1D-variance-only", "medium"
    else:
        verdict, confidence = "V1D-heterogeneous", "low"

    return {
        "verdict": verdict,
        "confidence": confidence,
        "fractions": {
            "primary_mean_diff_input_varying": primary_mean_diff,
            "primary_var_diff_input_varying": primary_var_diff,
            "primary_mean_stab_bit_identical": primary_mean_stab,
            "primary_var_stab_bit_identical": primary_var_stab,
            "abl1_mean_diff_input_varying": abl1_mean_diff,
            "abl1_var_diff_input_varying": abl1_var_diff,
            "abl2_mean_coupling_input_varying": abl2_mean_coupling,
            "abl2_var_coupling_input_varying": abl2_var_coupling,
            "abl2_mean_diff_input_varying": abl2_mean_diff,
            "abl2_var_diff_input_varying": abl2_var_diff,
        },
        "per_arm_summary": per_arm,
        "pattern_threshold_M": M,
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Threshold calibration + verdict recommendation")
    p.add_argument(
        "--matrix",
        type=Path,
        default=PATHS.results_matrix / "matrix.json",
    )
    args = p.parse_args()

    if not args.matrix.exists():
        raise SystemExit(f"[verdict] matrix not found: {args.matrix}")
    matrix = json.loads(args.matrix.read_text())

    thresholds = calibrate_thresholds(matrix)
    write_thresholds(thresholds, PATHS.results_thresholds / "thresholds.json")
    write_distributions(matrix, PATHS.results_thresholds / "distributions.json")

    classified = classify_cells(matrix, thresholds)
    (PATHS.results_thresholds / "matrix_classified.json").write_text(
        json.dumps(classified, indent=2)
    )

    recommendation = _recommend_verdict(classified)
    PATHS.results_verdict.mkdir(parents=True, exist_ok=True)
    (PATHS.results_verdict / "recommendation.json").write_text(
        json.dumps(recommendation, indent=2)
    )

    print(f"[verdict] verdict: {recommendation['verdict']}")
    print(f"[verdict] confidence: {recommendation['confidence']}")
    print(f"[verdict] thresholds: {PATHS.results_thresholds / 'thresholds.json'}")
    print(f"[verdict] recommendation: {PATHS.results_verdict / 'recommendation.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

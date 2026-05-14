"""Post-hoc analysis of the §8.7a in-flight transition diagnostic against the
restructured G2.T2 three-part criterion (session-7 instr §8.7a):

  (a) Trajectory direction (gated): perturbed-item mean log_var monotonically
      non-decreasing across loops 30 → 100, with up to 3 stochastic dips
      tolerated.
  (b) Trajectory shape (descriptive): characterise the curve at loops
      {30, 35, 50, 75, 100} as accelerating / decelerating / linear / flat.
  (c) Differential (gated): perturbed_widening_at_100 ≥ 2.0 × mean(|control
      drift| at 100), or mean(|control drift| at 100) ≤ 1e-3.

Reads `transition_diagnostic.json`, writes
`transition_g2t2_restructured.json` next to it, and prints a summary table.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path("/mnt/c/Users/Jason/Desktop/Eridos/Weft 2")

PERTURBED_VPS = (3, 4)             # Dresser, Sofa
CONTROL_VPS = (1, 2, 5)            # Bed, DiningTable, Television (§8.7a)
SHAPE_LOOPS = (30, 35, 50, 75, 100)
BASELINE_LOOP = 30
TARGET_LOOP = 100
DIPS_ALLOWED_MAX = 3
RATIO_THRESHOLD_MIN = 2.0
CONTROL_DRIFT_NEAR_ZERO = 1e-3


def _agg(record: dict[str, Any], vps: tuple[int, ...]) -> float:
    d = record.get("mean_log_var_by_viewing_position_id", {})
    vals = [d[str(int(v))] for v in vps if str(int(v)) in d]
    if not vals:
        return float("nan")
    return float(sum(vals) / len(vals))


def _shape_from_deltas(deltas: list[float]) -> str:
    if all(abs(d) < 0.01 for d in deltas):
        return "flat"
    incrementally_growing = all(
        deltas[i + 1] >= deltas[i] - 0.01 for i in range(len(deltas) - 1)
    )
    incrementally_shrinking = all(
        deltas[i + 1] <= deltas[i] + 0.01 for i in range(len(deltas) - 1)
    )
    if incrementally_growing and not incrementally_shrinking:
        return "accelerating"
    if incrementally_shrinking and not incrementally_growing:
        return "decelerating"
    # Roughly equal positive deltas => linear; otherwise mixed.
    if (
        all(d > 0 for d in deltas)
        and max(deltas) / max(min(deltas), 1e-12) < 2.0
    ):
        return "linear"
    return "mixed"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--diagnostic",
        type=Path,
        default=REPO_ROOT / "results/inner_pam_v0/phase2_main/transition_diagnostic.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "results/inner_pam_v0/phase2_main/transition_g2t2_restructured.json",
    )
    args = parser.parse_args()

    if not args.diagnostic.is_file():
        print(f"[t2-analysis] FAIL: diagnostic not found: {args.diagnostic}", file=sys.stderr)
        return 1

    d = json.loads(args.diagnostic.read_text())
    per_loop = d.get("per_loop", [])
    records = {int(r["loop_index"]): r for r in per_loop}
    loops_present = sorted(records.keys())
    print(f"[t2-analysis] loops present: {len(loops_present)} "
          f"[{min(loops_present)}..{max(loops_present)}]")

    needed = list(range(BASELINE_LOOP, TARGET_LOOP + 1))
    missing = [k for k in needed if k not in records]
    if missing:
        print(f"[t2-analysis] WARN: missing loops in 30..100 window: {missing}")

    # (a) Trajectory direction
    perturbed_series = [
        (k, _agg(records[k], PERTURBED_VPS))
        for k in needed
        if k in records and not math.isnan(_agg(records[k], PERTURBED_VPS))
    ]
    if len(perturbed_series) < 2:
        print("[t2-analysis] FAIL: too few perturbed-loop data points")
        return 2

    n_dips = 0
    dip_loops: list[tuple[int, float, int, float]] = []
    for i in range(1, len(perturbed_series)):
        loop_prev, lv_prev = perturbed_series[i - 1]
        loop_cur, lv_cur = perturbed_series[i]
        if lv_cur < lv_prev:
            n_dips += 1
            dip_loops.append((loop_prev, lv_prev, loop_cur, lv_cur))
    direction_pass = n_dips <= DIPS_ALLOWED_MAX
    print(f"[t2-analysis] (a) trajectory direction: n_dips={n_dips} "
          f"vs allowed_max={DIPS_ALLOWED_MAX} -> "
          f"{'PASS' if direction_pass else 'FAIL'}")

    # (b) Trajectory shape
    shape_points = []
    for k in SHAPE_LOOPS:
        if k in records:
            shape_points.append((k, _agg(records[k], PERTURBED_VPS)))
    deltas: list[float] = []
    for i in range(1, len(shape_points)):
        deltas.append(shape_points[i][1] - shape_points[i - 1][1])
    shape_label = _shape_from_deltas(deltas) if deltas else "insufficient_data"
    print(f"[t2-analysis] (b) trajectory shape: points={shape_points} "
          f"deltas={[round(x, 4) for x in deltas]} -> {shape_label}")

    # (c) Differential at loop 100
    if BASELINE_LOOP in records and TARGET_LOOP in records:
        base_record = records[BASELINE_LOOP]
        target_record = records[TARGET_LOOP]
        perturbed_widening = (
            _agg(target_record, PERTURBED_VPS) - _agg(base_record, PERTURBED_VPS)
        )
        per_control_drifts = {}
        for vp in CONTROL_VPS:
            lv_base = base_record["mean_log_var_by_viewing_position_id"].get(str(vp))
            lv_target = target_record["mean_log_var_by_viewing_position_id"].get(str(vp))
            if lv_base is None or lv_target is None:
                per_control_drifts[str(vp)] = float("nan")
            else:
                per_control_drifts[str(vp)] = float(lv_target - lv_base)
        control_widening_mean = float(
            sum(abs(v) for v in per_control_drifts.values()
                if not math.isnan(v)) / max(1, sum(1 for v in per_control_drifts.values() if not math.isnan(v)))
        )

        if perturbed_widening <= 0:
            ratio = 0.0
            differential_pass = False
            differential_reason = (
                f"perturbed_widening = {perturbed_widening:+.4f} <= 0 — "
                "perturbed-item variance is not widening (wrong direction); "
                "gate fails regardless of control magnitude"
            )
        elif control_widening_mean <= CONTROL_DRIFT_NEAR_ZERO:
            ratio = float("inf")
            differential_pass = True
            differential_reason = (
                f"control_widening_mean = {control_widening_mean:.6f} <= "
                f"{CONTROL_DRIFT_NEAR_ZERO}; locality clean by construction"
            )
        else:
            ratio = perturbed_widening / control_widening_mean
            differential_pass = bool(ratio >= RATIO_THRESHOLD_MIN)
            differential_reason = (
                f"ratio = perturbed_widening / control_widening_mean = "
                f"{perturbed_widening:.4f} / {control_widening_mean:.4f} = "
                f"{ratio:.3f} vs threshold {RATIO_THRESHOLD_MIN}"
            )
        print(f"[t2-analysis] (c) differential: "
              f"perturbed_widening={perturbed_widening:+.4f} "
              f"control_widening_mean={control_widening_mean:.4f} "
              f"ratio={ratio:.3f} -> {'PASS' if differential_pass else 'FAIL'}")
    else:
        perturbed_widening = float("nan")
        per_control_drifts = {}
        control_widening_mean = float("nan")
        ratio = float("nan")
        differential_pass = False
        differential_reason = (
            f"baseline (loop {BASELINE_LOOP}) or target (loop {TARGET_LOOP}) "
            f"missing from diagnostic JSON"
        )
        print(f"[t2-analysis] (c) differential: FAIL ({differential_reason})")

    overall_pass = direction_pass and differential_pass
    print(f"[t2-analysis] G2.T2 restructured overall: "
          f"{'PASS' if overall_pass else 'FAIL'} "
          f"(direction={direction_pass}, differential={differential_pass}, "
          f"shape={shape_label})")

    report = {
        "criterion": "G2.T2 restructured (session-7, instr §8.7a)",
        "diagnostic_source": str(args.diagnostic),
        "loops_in_window": [min(loops_present), max(loops_present)],
        "missing_loops_30_to_100": missing,
        "trajectory_direction": {
            "n_dips_observed": n_dips,
            "n_dips_allowed_max": DIPS_ALLOWED_MAX,
            "dip_loops": [
                {"prev_loop": p[0], "prev_lv": p[1], "cur_loop": p[2], "cur_lv": p[3]}
                for p in dip_loops
            ],
            "pass": direction_pass,
        },
        "trajectory_shape": {
            "perturbed_log_var_at_loops": dict(shape_points),
            "successive_deltas": deltas,
            "label": shape_label,
            "note": (
                "Descriptive, not gated. Flat fails the architectural claim "
                "(but would also trigger direction failure via no widening)."
            ),
        },
        "differential": {
            "baseline_loop": BASELINE_LOOP,
            "target_loop": TARGET_LOOP,
            "perturbed_widening": float(perturbed_widening)
                if not (isinstance(perturbed_widening, float) and math.isnan(perturbed_widening))
                else None,
            "per_control_drifts_loop_baseline_to_target": per_control_drifts,
            "control_widening_mean_abs": float(control_widening_mean)
                if not (isinstance(control_widening_mean, float) and math.isnan(control_widening_mean))
                else None,
            "ratio_perturbed_over_control": float(ratio)
                if math.isfinite(ratio) else (None if math.isnan(ratio) else "inf"),
            "ratio_threshold_min": RATIO_THRESHOLD_MIN,
            "control_drift_near_zero": CONTROL_DRIFT_NEAR_ZERO,
            "reason": differential_reason,
            "pass": differential_pass,
        },
        "overall_pass": overall_pass,
    }
    args.out.write_text(json.dumps(report, indent=2))
    print(f"[t2-analysis] wrote report: {args.out}")
    return 0 if overall_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())

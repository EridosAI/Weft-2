"""V2-PRE-D2 runner — n=10 vs n=20 CI validation (instr §7.7).

Usage:
  python3 v2/scripts/run_pre_d2.py            # full: 10 points x n=20 (~7 hr)
  python3 v2/scripts/run_pre_d2.py --smoke    # validation: 1 point x n=2 @ 500 steps
"""

from __future__ import annotations

import json
import sys

import numpy as np
import torch

from v2.config import RESULTS_PRE_D2, get_v2_training_steps
from v2.src.preflight import pre_d2_n_validation as d2
from v2.src.preflight.pre_d1a_endpoint_stability import train_one
from v2.src.substrate.base_manifold_trajectory import load_or_create_U


def main() -> int:
    smoke = "--smoke" in sys.argv
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    RESULTS_PRE_D2.mkdir(parents=True, exist_ok=True)
    U = load_or_create_U()
    steps = 500 if smoke else get_v2_training_steps()
    th = d2.load_thresholds()
    points = d2.sweep_points()
    n_reps = 2 if smoke else 20
    if smoke:
        points = points[:1]
    print(f"[pre_d2] device={device} steps={steps} L_d={d2.L_D_INTERMEDIATE} "
          f"points={len(points)} n_reps={n_reps} smoke={smoke}")
    print(f"[pre_d2] thresholds: mu={th['mu']:.5g} sigma={th['sigma']:.5g} "
          f"(baseline+τ_W)")

    point_results, stops = [], []
    for pi, (label, axis, value) in enumerate(points):
        params = d2.params_for(axis, value)
        mus, sigmas, nan_count = [], [], 0
        for seed in range(n_reps):
            r = train_one_safe(params, seed, U, device, steps, label, axis, value)
            if r["nan_inf"]:
                nan_count += 1
            else:
                mus.append(r["diff_mu"]); sigmas.append(r["diff_sigma"])
            print(f"[pre_d2] ({pi+1}/{len(points)}) {label} seed={seed} "
                  f"diff_mu={r['diff_mu']:.4g} diff_sigma={r['diff_sigma']:.3g}", flush=True)
        if nan_count == n_reps:
            stops.append(f"TRIGGER: sweep point {label} fully diverged ({nan_count}/{n_reps})")
            point_results.append({"label": label, "axis": axis, "value": value,
                                  "all_diverged": True})
            continue
        rec = {"label": label, "axis": axis, "value": value, "n_valid": len(mus),
               "diff_mu_values": mus, "diff_sigma_values": sigmas}
        for n in (10, 20):
            if len(mus) >= min(n, n_reps):
                rec[f"classification_n{n}"] = d2.classify_point(mus, sigmas, th, min(n, len(mus)))
        point_results.append(rec)

    # Aggregate (full only; smoke skips meaningful aggregation).
    agg = {}
    if not smoke:
        valid = [p for p in point_results if not p.get("all_diverged")]
        for n in (10, 20):
            disc = sum(1 for p in valid if p.get(f"classification_n{n}", {}).get("discriminable"))
            band = sum(1 for p in valid if p.get(f"classification_n{n}", {}).get("band_resident"))
            agg[f"n{n}"] = {"discriminable": disc, "band_resident": band, "total": len(valid)}
        gain = (agg["n10"]["band_resident"] - agg["n20"]["band_resident"]) / max(len(valid), 1)
        if agg["n10"]["band_resident"] <= max(1, int(0.2 * len(valid))):
            framing = "n=10 sufficient (low band-residence at n=10)"
        elif gain >= 0.1:
            framing = "n=10 marginal (band-residence drops materially n=10->n=20)"
        else:
            framing = "n=20 needed" if agg["n10"]["band_resident"] > 0.5 * len(valid) else "n=10 sufficient"
        agg["n10_to_n20_band_resident_reduction_fraction"] = round(gain, 3)
        agg["recommendation_framing"] = framing

    report = {
        "config": f"10 sweep points x n=20 x Primary @ L_d={d2.L_D_INTERMEDIATE} (intermediate, §11.6)",
        "training_steps": steps, "thresholds": th,
        "l_d_extrapolation_caveat": "L_d=2 intermediate; CI extrapolation to L_d_main {1,4} is approximate (§11.6).",
        "baseline_caveat": "threshold uses the PRE-D1a L_d_main=1 bit-identical baseline (§11.7); "
                           "L_d-dependence of the baseline is a closing item.",
        "aggregate": agg, "points": point_results, "stop_triggers": stops,
    }
    suffix = "_smoke" if smoke else ""
    (RESULTS_PRE_D2 / f"n_validation_report{suffix}.json").write_text(json.dumps(report, indent=2))

    print("\n[pre_d2] === SUMMARY ===")
    if not smoke:
        for n in (10, 20):
            a = agg[f"n{n}"]
            print(f"  n={n}: discriminable {a['discriminable']}/{a['total']}, "
                  f"band-resident {a['band_resident']}/{a['total']}")
        print(f"  band-residence reduction n10->n20: {agg['n10_to_n20_band_resident_reduction_fraction']}")
        print(f"  recommendation: {agg['recommendation_framing']}")
    if stops:
        print("[pre_d2] STOP TRIGGERS:"); [print("   -", s) for s in stops]
        return 1
    print("\n[pre_d2] complete.")
    return 0


def train_one_safe(params, seed, U, device, steps, label, axis, value) -> dict:
    r = train_one(params, d2.L_D_INTERMEDIATE, seed, U, device, steps, label, axis, str(value))
    return {"diff_mu": r.diff_mu, "diff_sigma": r.diff_sigma, "nan_inf": r.nan_inf}


if __name__ == "__main__":
    raise SystemExit(main())

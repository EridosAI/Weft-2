"""V2-PRE-D1a runner — 40 arm-runs (instr §7.5). See module docstring.

Usage:
  python3 v2/scripts/run_pre_d1a.py            # full: 40 arm-runs (~80 min)
  python3 v2/scripts/run_pre_d1a.py --smoke    # validation: 3 runs @ 500 steps
"""

from __future__ import annotations

import json
import sys

import numpy as np
import torch

from v2.config import RESULTS_PRE_D1A, get_v2_training_steps
from v2.src.preflight import pre_d1a_endpoint_stability as d1a
from v2.src.substrate.base_manifold_trajectory import load_or_create_U
from v2.src.substrate.stream_builder import StreamParams

# A non-trivial fraction of not-plateaued endpoints triggers a design-chat
# checkpoint on V2_TRAINING_STEPS (user-flagged caveat).
NOT_PLATEAUED_SURFACE_FRACTION = 0.25


def _baseline_params() -> StreamParams:
    return StreamParams(period_P=d1a.MID["period_P"], manifold_dim_D=d1a.MID["manifold_dim_D"],
                        continuity_center=d1a.MID["continuity_center"], fidelity_F=d1a.MID["fidelity_F"],
                        magnitude_M=0.0, locality_L=d1a.MID["locality_L"])


def main() -> int:
    smoke = "--smoke" in sys.argv
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    RESULTS_PRE_D1A.mkdir(parents=True, exist_ok=True)
    U = load_or_create_U()
    steps = 500 if smoke else get_v2_training_steps()
    print(f"[pre_d1a] device={device} training_steps={steps} smoke={smoke}")

    # --- (a) endpoint stability ---
    ep_cfgs = d1a.endpoint_configs()
    if smoke:
        ep_cfgs = ep_cfgs[:2]
    endpoint_results = []
    for i, (axis, endpoint, L_d_main, seed) in enumerate(ep_cfgs):
        label = f"ep_{axis}_{endpoint}_Ld{L_d_main}"
        print(f"[pre_d1a] ({i+1}/{len(ep_cfgs)}) {label} seed={seed} ...", flush=True)
        r = d1a.train_one(d1a._params_for(axis, endpoint), L_d_main, seed, U, device,
                          steps, label, axis, endpoint)
        print(f"    -> {r.stability_flag} wall={r.wall_clock_s}s diff_mu={r.diff_mu:.4g} "
              f"diff_sigma={r.diff_sigma:.4g} still_descending={r.still_descending}", flush=True)
        endpoint_results.append(r)

    # --- (b) bit-identical baseline ---
    base_seeds = d1a.baseline_configs()
    if smoke:
        base_seeds = base_seeds[:1]
    baseline_results = []
    for j, seed in enumerate(base_seeds):
        print(f"[pre_d1a] baseline ({j+1}/{len(base_seeds)}) seed={seed} ...", flush=True)
        r = d1a.train_one(_baseline_params(), 1, seed, U, device, steps,
                          f"baseline_m0_seed{seed}", "baseline", "m0")
        print(f"    -> {r.stability_flag} diff_mu={r.diff_mu:.4g} diff_sigma={r.diff_sigma:.4g}", flush=True)
        baseline_results.append(r)

    # --- aggregate ---
    ep = [vars(r) for r in endpoint_results]
    base = [vars(r) for r in baseline_results]
    ep_divergent = [r for r in endpoint_results if r.stability_flag == "divergent"]
    base_divergent = [r for r in baseline_results if r.stability_flag == "divergent"]
    not_plateaued = [r.label for r in endpoint_results if r.still_descending]

    walls = np.array([r.wall_clock_s for r in endpoint_results])
    by_ld = {ld: [r.wall_clock_s for r in endpoint_results if r.L_d_main == ld] for ld in (1, 4)}
    cost = {
        "mean_s": float(walls.mean()), "std_s": float(walls.std()),
        "per_L_d_main_mean_s": {str(k): (float(np.mean(v)) if v else None) for k, v in by_ld.items()},
        "per_axis_endpoint_s": {r.label: r.wall_clock_s for r in endpoint_results},
        "device": str(device),
    }

    base_mu = np.array([r.diff_mu for r in baseline_results if r.diff_mu == r.diff_mu])
    base_sigma = np.array([r.diff_sigma for r in baseline_results if r.diff_sigma == r.diff_sigma])
    def _dist(a):
        return {"n": int(a.size), "median": float(np.median(a)) if a.size else None,
                "iqr": float(np.subtract(*np.percentile(a, [75, 25]))) if a.size else None,
                "min": float(a.min()) if a.size else None, "max": float(a.max()) if a.size else None,
                "values_sorted": sorted(float(x) for x in a.tolist())}

    # L_d_main-dependence cross-check: magnitude=min endpoint Diff at L_d 1 vs 4 (§11.7).
    xcheck = {r.label: {"diff_mu": r.diff_mu, "diff_sigma": r.diff_sigma}
              for r in endpoint_results if r.axis == "magnitude" and r.endpoint == "min"}

    frac_not_plateaued = len(not_plateaued) / max(len(endpoint_results), 1)
    stops = []
    if endpoint_results and len(ep_divergent) == len(endpoint_results):
        stops.append("TRIGGER: all endpoint arm-runs diverged")
    if baseline_results and len(base_divergent) == len(baseline_results):
        stops.append("TRIGGER: all bit-identical baseline arm-runs diverged")
    plateau_surface = (not smoke) and frac_not_plateaued > NOT_PLATEAUED_SURFACE_FRACTION

    endpoint_report = {
        "training_steps": steps, "n_endpoint_runs": len(endpoint_results),
        "stability_counts": {f: sum(1 for r in endpoint_results if r.stability_flag == f)
                             for f in ("stable", "unstable", "divergent")},
        "not_plateaued_labels": not_plateaued,
        "fraction_not_plateaued": round(frac_not_plateaued, 3),
        "plateau_caveat_needs_design_chat": plateau_surface,
        "per_arm_cost_input1": cost,
        "runs": ep,
    }
    baseline_report = {
        "config": "all-midpoint, magnitude=0, L_d_main=1 (§11.7)", "n": len(baseline_results),
        "diff_mu_distribution": _dist(base_mu), "diff_sigma_distribution": _dist(base_sigma),
        "L_d_main_dependence_crosscheck_magnitude_min": xcheck,
        "runs": base,
    }
    suffix = "_smoke" if smoke else ""
    (RESULTS_PRE_D1A / f"endpoint_stability_report{suffix}.json").write_text(json.dumps(endpoint_report, indent=2))
    (RESULTS_PRE_D1A / f"bit_identical_baseline{suffix}.json").write_text(json.dumps(baseline_report, indent=2))

    print("\n[pre_d1a] === SUMMARY ===")
    print(f"  endpoint stability: {endpoint_report['stability_counts']}")
    print(f"  not-plateaued: {len(not_plateaued)}/{len(endpoint_results)} "
          f"({frac_not_plateaued:.0%}) {not_plateaued}")
    print(f"  per-arm cost: mean {cost['mean_s']:.1f}s, L_d=1 {cost['per_L_d_main_mean_s'].get('1')}, "
          f"L_d=4 {cost['per_L_d_main_mean_s'].get('4')}")
    print(f"  baseline Diff_mu median={baseline_report['diff_mu_distribution']['median']} "
          f"Diff_sigma median={baseline_report['diff_sigma_distribution']['median']}")
    if plateau_surface:
        print(f"\n[pre_d1a] *** PLATEAU CAVEAT: {frac_not_plateaued:.0%} of endpoints still descending "
              f"at {steps} steps -> STOP & surface for design-chat (revisit V2_TRAINING_STEPS).")
    if stops:
        print("[pre_d1a] STOP TRIGGERS:"); [print("   -", s) for s in stops]
        return 1
    print("\n[pre_d1a] complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

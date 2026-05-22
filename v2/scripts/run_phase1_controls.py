"""V2 Phase 1 sub-phase 1.2 — controls C1 + C2 (instr §7.2).

Fail-fast gate before the 1350-run main effects. Dispatches the 120 control arm-runs
at the smoke-locked parallelism, classifies each (control, cell, L_d) group at n=10,
writes c1_report.json / c2_report.json, and applies the §7.2 C1 STOP triggers.

Use --dry-run to print the spec inventory (no training).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

from v2.config import RESULTS_ROOT
from v2.src.phase1 import parallel_harness as ph
from v2.src.phase1.classification import load_thresholds
from v2.src.phase1.controls import c1_stop_check, classify_groups, control_specs

REPO_ROOT = str(pathlib.Path(__file__).resolve().parents[2])


def _locked_parallelism(default: int = 2) -> int:
    f = RESULTS_ROOT / "phase1" / "smoke_validation.json"
    try:
        return int(json.loads(f.read_text())["step4_parallelism"]["parallelism_locked"])
    except Exception:  # noqa: BLE001
        return default


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    out_dir = RESULTS_ROOT / "phase1" / "controls"
    out_dir.mkdir(parents=True, exist_ok=True)
    specs = control_specs(str(out_dir / "_runs"))

    n_c1 = sum(1 for s in specs if s["control"] == "C1")
    n_c2 = sum(1 for s in specs if s["control"] == "C2")
    print(f"[controls] specs: {len(specs)} total (C1={n_c1}, C2={n_c2})")
    if args.dry_run:
        for s in specs[:3] + specs[-3:]:
            print(f"  {s['label']}: {s['params']}")
        return 0

    par = _locked_parallelism()
    thresholds = load_thresholds()
    print(f"[controls] dispatching at parallelism={par}x; thresholds mu={thresholds['mu']:.5f} "
          f"sigma={thresholds['sigma']:.3g}")
    t0 = time.time()
    results = ph.run_batch(specs, n_concurrent=par, repo_root=REPO_ROOT)
    wall = time.time() - t0
    print(f"[controls] {len(results)} arm-runs done in {wall/60:.1f} min")

    grouped = classify_groups(results, thresholds)
    stop = c1_stop_check(grouped)

    c1 = {k: v for k, v in grouped.items() if v["control"] == "C1"}
    c2 = {k: v for k, v in grouped.items() if v["control"] == "C2"}

    c1_report = {
        "control": "C1 (bit-identical, magnitude=0, others midpoint)",
        "expected": "discriminably_non_working / band_resident (no working-region signal)",
        "thresholds": thresholds, "n_per_cell": 10,
        "stop_check": stop, "cells": c1,
        "wall_minutes": round(wall / 60, 1),
    }
    c2_report = {
        "control": "C2 (magnitude-only: mag swept, locality=0.9 grid-max, others midpoint)",
        "note": ("comparison vs main-effects magnitude axis is deferred to sub-phase 1.7 "
                 "aggregate (main effects not yet run)"),
        "thresholds": thresholds, "n_per_cell": 10, "cells": c2,
    }
    (out_dir / "c1_report.json").write_text(json.dumps(c1_report, indent=2, default=str))
    (out_dir / "c2_report.json").write_text(json.dumps(c2_report, indent=2, default=str))

    print("\n[controls] === C1 (bit-identical) ===")
    for k, v in sorted(c1.items()):
        print(f"  {k}: overall={v['overall']} (mu={v.get('head_mu')}, sigma={v.get('head_sigma')}) "
              f"diff_mu_median={v.get('diff_mu_median')}")
    print(f"[controls] C1 stop_check: working_frac={stop['discriminably_working_fraction']} "
          f"cross_L_d_working={stop['cross_L_d_consistent_working']} -> "
          f"{'STOP: ' + stop['stop'] if stop['stop'] else 'PASS'}")
    print("\n[controls] === C2 (magnitude-only) ===")
    for k, v in sorted(c2.items()):
        print(f"  {k}: overall={v['overall']} (mu={v.get('head_mu')}, sigma={v.get('head_sigma')}) "
              f"diff_mu_median={v.get('diff_mu_median')}")

    if stop["stop"]:
        print(f"\n[controls] STOP TRIGGER: {stop['stop']}")
        return 1
    print("\n[controls] C1/C2 complete; C1 fail-fast gate PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

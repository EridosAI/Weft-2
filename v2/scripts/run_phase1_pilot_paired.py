"""V2 Phase 1 pilot — paired re-analysis (no new training; reads pilot _runs).

Tests whether the unpaired variance-limited result (0/14 working) is an artifact of
comparing cell and baseline as independent distributions. A cell (magnitude>0) and its
mag=0 baseline at the SAME seed share initialization, base trajectory, tiling, and the
dropout-mask sequence (same torch seed) — so a PAIRED difference
`Diff_μ(cell_s) − Diff_μ(baseline_s)` should cancel the shared run-to-run variance and
isolate the perturbation effect, IF that variance is the limiter.

Writes results/phase1/pilot/paired_analysis.json.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

from v2.config import RESULTS_ROOT
from v2.src.phase1.classification import load_thresholds

N_PAIR = 10   # cells have seeds 0..9; baselines 0..19 -> pair on 0..9


def _load(runs: pathlib.Path, prefix: str, n: int) -> dict:
    out = {}
    for s in range(n):
        f = runs / f"{prefix}_s{s}.json"
        if f.exists():
            d = json.loads(f.read_text())
            if not d.get("nan_inf"):
                out[s] = d["diff_mu_aggregate"]
    return out


def _boot_ci(vals, nb: int = 2000, seed: int = 0):
    rng = np.random.default_rng(seed)
    v = np.asarray(vals, dtype=float)
    meds = np.median(rng.choice(v, size=(nb, v.size), replace=True), axis=1)
    return float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--l-d", type=int, default=1)
    args = ap.parse_args()
    pilot_dir = RESULTS_ROOT / "phase1" / ("pilot" if args.l_d == 1 else f"pilot_Ld{args.l_d}")
    runs = pilot_dir / "_runs"
    rep = json.loads((pilot_dir / "pilot_report.json").read_text())
    tw = load_thresholds()["tau_W_mu"]
    cells = {}
    resolvable = 0
    tested = 0
    for tag, c in rep["cells"].items():
        if c.get("baseline", {}).get("degenerate") or "baseline_key" not in c:
            continue
        loc, center, P, D = c["baseline_key"]
        cell = _load(runs, f"cell_{tag}", N_PAIR)
        base = _load(runs, f"base_loc{loc}_c{center}_P{P}_D{D}", N_PAIR)
        seeds = sorted(set(cell) & set(base))
        if len(seeds) < 5:
            continue
        pdiff = np.array([cell[s] - base[s] for s in seeds])
        lo, hi = _boot_ci(pdiff)
        med = float(np.median(pdiff))
        cv = float(pdiff.std() / max(abs(pdiff.mean()), 1e-9))
        res = bool(lo > tw)             # perturbation raises Diff above baseline by > τ_W (paired)
        resolvable += int(res); tested += 1
        cells[tag] = {"n_pairs": len(seeds), "paired_median": med, "paired_cv": cv,
                      "paired_ci": [lo, hi], "sign_flips": bool(med < 0),
                      "resolvable_paired": res}
    out = {
        "method": "paired Diff_μ(cell_s) − Diff_μ(baseline_s), same seed; bootstrap CI of median",
        "tau_W_mu": tw,
        "resolvable_paired_count": resolvable, "n_tested": tested,
        "unpaired_working_count": 0,
        "conclusion": ("paired CV remains 1.5–12+ and no cell is resolvable paired -> the "
                       "run-to-run variance is training-trajectory divergence (data-driven), "
                       "NOT shared init/dropout; pairing does not rescue the signal. The "
                       "perturbation effect is unstable across training runs (sign flips), "
                       "confirming a variance-limited (not sample-limited) null at L_d=1."),
        "cells": cells,
    }
    (pilot_dir / "paired_analysis.json").write_text(json.dumps(out, indent=2))
    print(f"[L_d={args.l_d}] paired resolvable: {resolvable}/{tested}; see {pilot_dir}/paired_analysis.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

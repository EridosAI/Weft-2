"""V2 Phase 1 — Option-1 PILOT (de-risk before the ~3-day full collection).

Per design-chat: pilot-first with per-config bit-identical baselines (Option 1),
plus three refinements:
  R1 high-magnitude probe: magnitude=0.9 added at all 3 crosses (beyond grid max 0.7).
  R2 per-K Diff: persist un-aggregated per-K Diff_μ/Diff_σ (eval_perk); classify at
     K-aggregated (current methodology) vs k=15 (long-horizon-only).
  R3 continuity/dim depth at mid cross: cont-low + D=4 (degenerate high-cont/D=128 skipped).

Inventory (deduped — cont-mid & D=16 at mid cross == the mid-cross/mag=0.3 cell):
  14 unique cells × n=10 (L_d=1)  + 5 mag=0 baseline configs × n=20  = 240 arm-runs.

Classification (Option 1): cell vs its OWN mag=0 baseline (same loc/continuity/P/D),
threshold = baseline median + per-head τ_W (existing PRE-E τ_W; k=15 reuses it as an
approximation — flagged). Baselines with σ variance-collapse -> baseline_degenerate.

Reporting: signal density at K-agg vs k=15; high-mag (0.9 vs 0.7) probe. Closing framing:
characterises inner-PAM-in-isolation (outer associative memory is out of v2 scope).

--dry-run prints the inventory (no training).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time
from collections import defaultdict

import numpy as np

from v2.config import RESULTS_ROOT
from v2.src.phase1 import parallel_harness as ph
from v2.src.phase1.classification import classify_cell, load_thresholds
from v2.src.phase1.sweep_grid import (
    CONTINUITY_CENTER, FIDELITY_F, LOCALITY_L, MAGNITUDE_M, MANIFOLD_D, PERIOD_P,
)
from v2.src.preflight.pre_d1a_endpoint_stability import VARIANCE_COLLAPSE_EPS

REPO_ROOT = str(pathlib.Path(__file__).resolve().parents[2])
PILOT_MAGS = [0.1, 0.3, 0.7, 0.9]      # incl. R1 high-mag probe 0.9 (grid max is 0.7)
N_CELL, N_BASE = 10, 20
K15_IDX = 14                            # k=15 (k = idx+1, PREDICT_K=16)


def _params(mag, loc, center, P, D):
    return {"period_P": P, "manifold_dim_D": D, "continuity_center": center,
            "fidelity_F": FIDELITY_F, "magnitude_M": mag, "locality_L": loc}


def build_specs(runs: pathlib.Path, l_d: int):
    runs.mkdir(parents=True, exist_ok=True)
    cells = []   # (tag, cross, mag, params)
    for cross in ("low", "mid", "high"):
        loc, P, D = LOCALITY_L[cross], PERIOD_P[cross], MANIFOLD_D[cross]
        center = CONTINUITY_CENTER[P][cross]
        for mag in PILOT_MAGS:
            cells.append((f"mag{mag}_{cross}", cross, mag, _params(mag, loc, center, P, D)))
    # R3: continuity/dim depth at mid cross (cont-mid & D=16 == mid/mag0.3 cell -> skip).
    midL, midP, midD = LOCALITY_L["mid"], PERIOD_P["mid"], MANIFOLD_D["mid"]
    midmag = MAGNITUDE_M["mid"]
    cells.append(("contLow_mid", "mid", midmag,
                  _params(midmag, midL, CONTINUITY_CENTER[midP]["low"], midP, midD)))
    cells.append(("D4_mid", "mid", midmag,
                  _params(midmag, midL, CONTINUITY_CENTER[midP]["mid"], midP, MANIFOLD_D["low"])))

    # Per-config mag=0 baselines (unique loc/center/P/D across cells).
    base_keys = {}
    for _, _, _, p in cells:
        base_keys[(p["locality_L"], p["continuity_center"], p["period_P"], p["manifold_dim_D"])] = True

    specs = []
    for (tag, cross, mag, p) in cells:
        for s in range(N_CELL):
            specs.append({"kind": "cell", "tag": tag, "cross": cross, "magnitude": mag,
                          "baseline_key": [p["locality_L"], p["continuity_center"],
                                           p["period_P"], p["manifold_dim_D"]],
                          "params": p, "L_d_main": l_d, "seed": s, "perk": True,
                          "label": f"cell_{tag}_s{s}",
                          "out_file": str(runs / f"cell_{tag}_s{s}.json")})
    for (loc, center, P, D) in base_keys:                       # L_d-SPECIFIC baselines
        btag = f"base_loc{loc}_c{center}_P{P}_D{D}"
        for s in range(N_BASE):
            specs.append({"kind": "baseline",
                          "baseline_key": [loc, center, P, D],
                          "params": _params(0.0, loc, center, P, D),
                          "L_d_main": l_d, "seed": s, "perk": True,
                          "label": f"{btag}_s{s}", "out_file": str(runs / f"{btag}_s{s}.json")})
    return specs, len(cells), len(base_keys)


def _outcome(working_fraction: float) -> str:
    """Pre-registered (design-chat) reading of an L_d arm's working density."""
    if working_fraction < 0.05:
        return "robust_null (<5% working)"
    if working_fraction > 0.25:
        return "capacity_signal (>25% working — capacity matters)"
    return "marginal (5-25% working — design-chat decides)"


def _collect(results):
    """Group valid results by (kind, identifier); keep agg + per-k arrays."""
    cells, bases = defaultdict(lambda: defaultdict(list)), defaultdict(lambda: defaultdict(list))
    for r in results:
        if r.get("nan_inf") or r.get("error") or r.get("diff_mu_per_k") is None:
            continue
        sp = r["spec"]
        bk = tuple(sp["baseline_key"])
        tgt = cells[sp["tag"]] if sp["kind"] == "cell" else bases[bk]
        tgt["mu_agg"].append(r["diff_mu_aggregate"])
        tgt["sigma_agg"].append(r["diff_sigma_aggregate"])
        tgt["mu_k15"].append(r["diff_mu_per_k"][K15_IDX])
        tgt["sigma_k15"].append(r["diff_sigma_per_k"][K15_IDX])
        if sp["kind"] == "cell":
            tgt["_meta"] = {"cross": sp["cross"], "magnitude": sp["magnitude"], "baseline_key": bk}
    return cells, bases


def classify_pilot(cells, bases, th):
    tw_mu, tw_sigma = th["tau_W_mu"], th["tau_W_sigma"]
    out = {}
    for tag, c in cells.items():
        bk = c["_meta"]["baseline_key"]
        b = bases.get(bk)
        rec = {"cross": c["_meta"]["cross"], "magnitude": c["_meta"]["magnitude"],
               "baseline_key": list(bk), "n_cell": len(c["mu_agg"])}
        if not b:
            rec["error"] = "no baseline"; out[tag] = rec; continue
        b_mu_agg, b_sig_agg = float(np.median(b["mu_agg"])), float(np.median(b["sigma_agg"]))
        b_mu_k15, b_sig_k15 = float(np.median(b["mu_k15"])), float(np.median(b["sigma_k15"]))
        rec["baseline"] = {"n": len(b["mu_agg"]), "mu_agg_median": b_mu_agg,
                           "sigma_agg_median": b_sig_agg, "mu_k15_median": b_mu_k15,
                           "degenerate": bool(b_sig_agg < VARIANCE_COLLAPSE_EPS)}
        # K-aggregated (current methodology) vs k=15 (long-horizon), each vs own baseline+τ_W.
        th_agg = {"mu": b_mu_agg + tw_mu, "sigma": b_sig_agg + tw_sigma}
        th_k15 = {"mu": b_mu_k15 + tw_mu, "sigma": b_sig_k15 + tw_sigma}
        rec["cell_mu_agg_median"] = float(np.median(c["mu_agg"]))
        rec["cell_mu_k15_median"] = float(np.median(c["mu_k15"]))
        rec["class_Kagg"] = classify_cell(c["mu_agg"], c["sigma_agg"], th_agg, len(c["mu_agg"]))["overall"]
        rec["class_k15"] = classify_cell(c["mu_k15"], c["sigma_k15"], th_k15, len(c["mu_k15"]))["overall"]
        if rec["baseline"]["degenerate"]:
            rec["class_Kagg"] = rec["class_k15"] = "baseline_degenerate"
        out[tag] = rec
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--l-d", type=int, default=1, help="decoder capacity L_d_main")
    args = ap.parse_args()
    l_d = args.l_d
    out_dir = RESULTS_ROOT / "phase1" / ("pilot" if l_d == 1 else f"pilot_Ld{l_d}")
    specs, n_cells, n_bases = build_specs(out_dir / "_runs", l_d)
    n_cell_runs = sum(1 for s in specs if s["kind"] == "cell")
    n_base_runs = sum(1 for s in specs if s["kind"] == "baseline")
    print(f"[pilot] L_d={l_d}: {n_cells} cells x n={N_CELL} = {n_cell_runs} + {n_bases} "
          f"L_d-specific baselines x n={N_BASE} = {n_base_runs}  => {len(specs)} runs (perk)")
    if args.dry_run:
        for s in specs[:2] + [s for s in specs if s["kind"] == "baseline"][:1]:
            print(f"  {s['label']}: kind={s['kind']} params={s['params']}")
        return 0

    par = 2
    try:
        par = int(json.loads((RESULTS_ROOT / "phase1" / "smoke_validation.json").read_text())
                  ["step4_parallelism"]["parallelism_locked"])
    except Exception:  # noqa: BLE001
        pass
    th = load_thresholds()
    t0 = time.time()
    results = ph.run_batch(specs, n_concurrent=par, repo_root=REPO_ROOT)
    print(f"[pilot] {len(results)} runs done in {(time.time()-t0)/60:.1f} min")

    cells, bases = _collect(results)
    classed = classify_pilot(cells, bases, th)

    valid = [c for c in classed.values() if "class_Kagg" in c]
    def density(key):
        n = len(valid)
        w = sum(1 for c in valid if c[key] == "discriminably_working")
        return {"working": w, "total": n, "fraction": round(w / max(n, 1), 3)}

    # R1 high-mag probe: 0.9 vs 0.7 per cross.
    probe = {}
    for cross in ("low", "mid", "high"):
        c07 = classed.get(f"mag0.7_{cross}", {}); c09 = classed.get(f"mag0.9_{cross}", {})
        probe[cross] = {"mag0.7": {"class_Kagg": c07.get("class_Kagg"),
                                   "mu_agg_median": c07.get("cell_mu_agg_median")},
                        "mag0.9": {"class_Kagg": c09.get("class_Kagg"),
                                   "mu_agg_median": c09.get("cell_mu_agg_median")}}

    report = {
        "design": "Option-1 pilot (per-config baseline); R1 mag=0.9, R2 per-K, R3 cont/dim depth",
        "L_d_main": l_d, "baselines": "L_d-specific (n=20)",
        "n_cells": n_cells, "n_baseline_configs": n_bases,
        "outcome_Kagg": _outcome(density("class_Kagg")["fraction"]),
        "outcome_k15": _outcome(density("class_k15")["fraction"]),
        "thresholds_tau_W": {"mu": th["tau_W_mu"], "sigma": th["tau_W_sigma"]},
        "classification_definition": ("cell Diff CI-low > own mag=0 baseline median + τ_W; "
                                      "k=15 reuses aggregate-calibrated τ_W (APPROX — flagged); "
                                      "σ variance-collapse baselines -> baseline_degenerate"),
        "signal_density_Kagg": density("class_Kagg"),
        "signal_density_k15": density("class_k15"),
        "high_mag_probe_0.9_vs_0.7": probe,
        "cells": classed,
        "closing_framing": "inner-PAM-in-isolation; outer associative memory out of v2 scope",
    }
    (out_dir / "pilot_report.json").write_text(json.dumps(report, indent=2, default=str))

    print(f"\n[pilot] signal density: K-agg {report['signal_density_Kagg']} | "
          f"k=15 {report['signal_density_k15']}")
    print("[pilot] per-cell (cross/mag: Kagg | k15 | cell_mu_agg vs baseline):")
    for tag, c in sorted(classed.items()):
        if "class_Kagg" in c:
            print(f"  {tag:<16} {c['class_Kagg']:<24} | {c['class_k15']:<24} "
                  f"| {c.get('cell_mu_agg_median', float('nan')):.4f} vs {c['baseline']['mu_agg_median']:.4f}"
                  + ("  [DEGEN]" if c['baseline']['degenerate'] else ""))
    print("[pilot] high-mag probe (0.9 vs 0.7):")
    for cross, p in probe.items():
        print(f"  {cross}: 0.7 {p['mag0.7']['class_Kagg']} ({p['mag0.7']['mu_agg_median']}) | "
              f"0.9 {p['mag0.9']['class_Kagg']} ({p['mag0.9']['mu_agg_median']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

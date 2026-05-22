"""V2 Phase 1 sub-phase 1.2.5 — within-cross-position baseline-variance diagnostic.

Decides between resolution Option 1 (per-config bit-identical baseline) and Option 2
(baseline grid keyed by config+L_d) for the threshold-non-transfer finding (§7 / 1.2):
characterise how much the magnitude=0 Diff_μ baseline varies across swept-axis values
*within* one cross-position × one L_d (C1 only varied L_d at a single config).

Design (design-chat 1.2.5 spec): mid cross-position, L_d=1, magnitude=0, n=10.
  D axis:          D in {4, 16, 128}, others mid.
  continuity axis: continuity_center in {19(0.077), 39(0.4), 59(0.8)}, others mid (D=16).
The shared midpoint (D=16, center=39, mid cross, L_d=1) == C1-L_d1 (already measured) ->
reuse its 10 diff_mu; only 4 new configs × n=10 = 40 new arm-runs (~1.3 hr at 2x).

Decision rule: relative spread = (max−min)/median over the 3 position-medians, per axis.
  max axis spread <= 0.25  -> Option 2 (decisively tight)  -> proceed
  max axis spread >= 0.35  -> Option 1 (decisively wide)   -> proceed
  0.25 < spread < 0.35     -> SURFACE (borderline; decide with design chat)
"""

from __future__ import annotations

import glob
import json
import pathlib
import time

import numpy as np

from v2.config import RESULTS_ROOT
from v2.src.phase1 import parallel_harness as ph
from v2.src.phase1.sweep_grid import (
    CONTINUITY_CENTER, FIDELITY_F, LOCALITY_L, MANIFOLD_D, PERIOD_P,
)

REPO_ROOT = str(pathlib.Path(__file__).resolve().parents[2])
MID_P = PERIOD_P["mid"]          # 256
MID_LOC = LOCALITY_L["mid"]      # 0.5
MID_CENTER = CONTINUITY_CENTER[MID_P]["mid"]   # 39
MID_D = MANIFOLD_D["mid"]        # 16
N = 10


def _params(D: int, center: int) -> dict:
    return {"period_P": MID_P, "manifold_dim_D": D, "continuity_center": center,
            "fidelity_F": FIDELITY_F, "magnitude_M": 0.0, "locality_L": MID_LOC}


def _spec(tag: str, D: int, center: int, seed: int, runs: pathlib.Path) -> dict:
    return {"tag": tag, "params": _params(D, center), "L_d_main": 1, "seed": seed,
            "label": f"basevar_{tag}_s{seed}",
            "out_file": str(runs / f"basevar_{tag}_s{seed}.json")}


def _c1_ld1_diff_mu() -> list[float]:
    """The shared midpoint (D=16, center=39, mid cross, L_d=1) = C1-L_d1; reuse its 10 reps."""
    vals = []
    for f in sorted(glob.glob(str(RESULTS_ROOT / "phase1" / "controls" / "_runs" /
                                  "C1_bit_identical_Ld1_s*.json"))):
        d = json.load(open(f))
        if not d.get("nan_inf"):
            vals.append(d["diff_mu"])
    return vals


def _spread(medians: list[float]) -> float:
    a = np.array(medians, dtype=float)
    return float((a.max() - a.min()) / np.median(a)) if a.size else float("nan")


def main() -> int:
    out_dir = RESULTS_ROOT / "phase1"
    runs = out_dir / "baseline_variance" / "_runs"
    runs.mkdir(parents=True, exist_ok=True)

    # 4 new configs (D=16/center=39 midpoint reused from C1-L_d1).
    new = [("D4", 4, MID_CENTER), ("D128", 128, MID_CENTER),
           ("contLow", MID_D, CONTINUITY_CENTER[MID_P]["low"]),
           ("contHigh", MID_D, CONTINUITY_CENTER[MID_P]["high"])]
    specs = [_spec(tag, D, c, s, runs) for (tag, D, c) in new for s in range(N)]
    print(f"[basevar] {len(specs)} new arm-runs (4 configs x n={N}); reusing C1-L_d1 midpoint")

    par = 2
    try:
        par = int(json.loads((out_dir / "smoke_validation.json").read_text())
                  ["step4_parallelism"]["parallelism_locked"])
    except Exception:  # noqa: BLE001
        pass

    t0 = time.time()
    results = ph.run_batch(specs, n_concurrent=par, repo_root=REPO_ROOT)
    print(f"[basevar] {len(results)} runs done in {(time.time()-t0)/60:.1f} min")

    by_tag = {}
    for res in results:
        by_tag.setdefault(res["spec"]["tag"], []).append(res.get("diff_mu"))
    mid_vals = _c1_ld1_diff_mu()

    def summ(vals):
        v = np.array([x for x in vals if x is not None], dtype=float)
        return {"n": int(v.size), "median": float(np.median(v)),
                "iqr": float(np.subtract(*np.percentile(v, [75, 25]))),
                "min": float(v.min()), "max": float(v.max())}

    mid = summ(mid_vals)
    d_axis = {"D4": summ(by_tag["D4"]), "D16_midpoint(=C1_Ld1)": mid, "D128": summ(by_tag["D128"])}
    c_axis = {"contLow_0.077": summ(by_tag["contLow"]), "contMid_0.4(=C1_Ld1)": mid,
              "contHigh_0.8": summ(by_tag["contHigh"])}

    d_spread = _spread([d_axis["D4"]["median"], mid["median"], d_axis["D128"]["median"]])
    c_spread = _spread([c_axis["contLow_0.077"]["median"], mid["median"], c_axis["contHigh_0.8"]["median"]])
    max_spread = max(d_spread, c_spread)

    if max_spread <= 0.25:
        decision = "Option 2 (baseline grid keyed by config+L_d) — decisively tight"
    elif max_spread >= 0.35:
        decision = "Option 1 (per-config bit-identical baseline) — decisively wide"
    else:
        decision = "BORDERLINE (0.25–0.35) — surface to design chat"

    payload = {
        "design": "mid cross, L_d=1, magnitude=0, n=10; D-axis {4,16,128} + continuity {0.077,0.4,0.8}",
        "midpoint_reused_from": "C1-L_d1 (all-midpoint config)",
        "d_axis": d_axis, "continuity_axis": c_axis,
        "d_axis_relative_spread": round(d_spread, 4),
        "continuity_axis_relative_spread": round(c_spread, 4),
        "max_relative_spread": round(max_spread, 4),
        "spread_metric": "(max-min)/median over the 3 position-medians",
        "decision_rule": "<=0.25 -> Option 2; >=0.35 -> Option 1; (0.25,0.35) -> surface",
        "decision": decision,
        "caveat_D128": ("(D=128,P=256) is the D==P//2 boundary: harmonics fill the whole band "
                        "so its continuity is forced ~1.0 (not mid 0.4); its baseline conflates "
                        "D with that forced continuity."),
    }
    (out_dir / "baseline_variance_diagnostic.json").write_text(json.dumps(payload, indent=2))

    print("\n[basevar] === D axis (mag=0, L_d=1, mid cross) ===")
    for k, v in d_axis.items():
        print(f"  {k}: median={v['median']:.4f} IQR={v['iqr']:.4f} (n={v['n']})")
    print(f"  D-axis relative spread = {d_spread:.3f}")
    print("[basevar] === continuity axis ===")
    for k, v in c_axis.items():
        print(f"  {k}: median={v['median']:.4f} IQR={v['iqr']:.4f} (n={v['n']})")
    print(f"  continuity-axis relative spread = {c_spread:.3f}")
    print(f"\n[basevar] max relative spread = {max_spread:.3f} -> {decision}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

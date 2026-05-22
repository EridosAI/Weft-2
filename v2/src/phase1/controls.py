"""V2 Phase 1 — C1 / C2 control conditions (spec §8; instr §6, §7.2).

C1 — bit-identical: magnitude=0 (no perturbation), other axes at midpoint. Validates
     the architecture does not register *absence* of perturbation as working-region signal.
C2 — magnitude-only: magnitude swept at the 3 grid values, other axes at midpoint but
     locality=0.9 (grid maximum; on-grid per adversarial-review Finding 8). Validates the
     magnitude axis is independently characterised.

Both at L_d_main {1,2,4}, n=10 (seeds 0..9). C1 = 30 arm-runs, C2 = 90 -> 120 total.
Construction params come from the locked sweep grid (sweep_grid.py).
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from v2.src.phase1.classification import classify_cell
from v2.src.phase1.sweep_grid import (
    CONTINUITY_CENTER, FIDELITY_F, L_D_MAIN, LOCALITY_L, MAGNITUDE_M, MANIFOLD_D, PERIOD_P,
)

C_N = 10                       # seeds 0..9 per cell (Phase-0.5 n=10)
_MID_P = PERIOD_P["mid"]       # 256
_MID_D = MANIFOLD_D["mid"]     # 16
_MID_CENTER = CONTINUITY_CENTER[_MID_P]["mid"]   # 39


def _params(magnitude_M: float, locality_L: float) -> dict:
    """Midpoint-axis StreamParams (period/dim/continuity at mid) with given mag/loc."""
    return {"period_P": _MID_P, "manifold_dim_D": _MID_D, "continuity_center": _MID_CENTER,
            "fidelity_F": FIDELITY_F, "magnitude_M": magnitude_M, "locality_L": locality_L}


def _spec(control: str, cell: str, params: dict, L_d: int, seed: int, runs: Path) -> dict:
    cell_s = cell.replace("@", "").replace(".", "")
    return {"control": control, "cell": cell, "params": params, "L_d_main": L_d, "seed": seed,
            "label": f"{control}_{cell}_Ld{L_d}_s{seed}",
            "out_file": str(runs / f"{control}_{cell_s}_Ld{L_d}_s{seed}.json")}


def control_specs(runs_dir: str) -> list[dict]:
    """All 120 control arm-run specs (C1: 30, C2: 90)."""
    runs = Path(runs_dir)
    runs.mkdir(parents=True, exist_ok=True)
    specs: list[dict] = []
    # C1 — bit-identical (magnitude=0, others at midpoint incl. locality mid).
    for L_d in L_D_MAIN:
        for seed in range(C_N):
            specs.append(_spec("C1", "bit_identical", _params(0.0, LOCALITY_L["mid"]),
                               L_d, seed, runs))
    # C2 — magnitude-only (mag in {0.1,0.3,0.7}, locality=0.9 grid-max, others midpoint).
    for mag in (MAGNITUDE_M["low"], MAGNITUDE_M["mid"], MAGNITUDE_M["high"]):
        for L_d in L_D_MAIN:
            for seed in range(C_N):
                specs.append(_spec("C2", f"mag@{mag}", _params(mag, LOCALITY_L["high"]),
                                   L_d, seed, runs))
    return specs


def classify_groups(results: list[dict], thresholds: dict) -> dict:
    """Group results by (control, cell, L_d_main); classify each group's n reps."""
    groups: dict = defaultdict(lambda: {"diff_mu": [], "diff_sigma": [],
                                         "nan": 0, "seeds": []})
    for res in results:
        sp = res.get("spec", {})
        key = (sp.get("control"), sp.get("cell"), sp.get("L_d_main"))
        g = groups[key]
        g["seeds"].append(sp.get("seed"))
        if res.get("nan_inf") or res.get("error"):
            g["nan"] += 1
        else:
            g["diff_mu"].append(res["diff_mu"])
            g["diff_sigma"].append(res["diff_sigma"])

    out = {}
    for (control, cell, L_d), g in groups.items():
        n_valid = len(g["diff_mu"])
        rec = {"control": control, "cell": cell, "L_d_main": L_d,
               "n_valid": n_valid, "n_nan": g["nan"]}
        if n_valid >= 1:
            cls = classify_cell(g["diff_mu"], g["diff_sigma"], thresholds, n_valid)
            rec.update({
                "overall": cls["overall"],
                "head_mu": cls["head_mu"]["category"],
                "head_sigma": cls["head_sigma"]["category"],
                "conflicting_heads": cls["conflicting_heads"],
                "diff_mu_median": cls["head_mu"]["median"],
                "diff_sigma_median": cls["head_sigma"]["median"],
                "diff_mu_ci": [cls["head_mu"]["ci_low"], cls["head_mu"]["ci_high"]],
            })
        else:
            rec.update({"overall": "all_diverged"})
        out[f"{control}|{cell}|Ld{L_d}"] = rec
    return out


def c1_stop_check(grouped: dict) -> dict:
    """§7.2 C1 STOP triggers: >15% discriminably-working over C1 cell-classifications,
    OR any C1 cell discriminably-working across all 3 L_d_main, OR a fully-diverged cell."""
    c1 = [v for v in grouped.values() if v["control"] == "C1"]
    working = [v for v in c1 if v["overall"] == "discriminably_working"]
    frac = len(working) / max(len(c1), 1)
    # cross-L_d consistency: is the (single) C1 cell working at all 3 L_d?
    by_cell = defaultdict(list)
    for v in c1:
        by_cell[v["cell"]].append(v["overall"])
    cross_consistent_working = any(
        all(o == "discriminably_working" for o in os) and len(os) == len(L_D_MAIN)
        for os in by_cell.values())
    diverged = [v for v in c1 if v["overall"] == "all_diverged"]

    stop = None
    if diverged:
        stop = f"C1 fully-diverged cell(s): {[ (v['cell'], v['L_d_main']) for v in diverged ]}"
    elif frac > 0.15:
        stop = f"C1 discriminably-working fraction {frac:.2f} > 0.15"
    elif cross_consistent_working:
        stop = "C1 cell discriminably-working across all 3 L_d_main"
    return {"n_c1_classifications": len(c1),
            "discriminably_working_fraction": round(frac, 3),
            "cross_L_d_consistent_working": cross_consistent_working,
            "fully_diverged": [f"{v['cell']}|Ld{v['L_d_main']}" for v in diverged],
            "stop": stop}

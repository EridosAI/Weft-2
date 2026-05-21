"""V2 Phase 1 — pre-build grid calibration + range/stability validation.

Authorised by the Phase-0.5 design-chat follow-up (2026-05-21/22):
  * Resolution: sweep CONSTRUCTION parameters, measure properties post-hoc; the
    map keeps measured-property axis labels (grid_mapping.nearest_grid_point).
  * D grid re-anchored to {4, 16, 128} (Path 1) to reduce infeasible (D, P) cells.
  * "Validate ranges first" before committing the out-of-PRE-validated-range cells
    (D=128; P in {32, 2048}).
  * Continuity needs a construction grid (continuity_center); PRE-A only calibrated
    centres {8,24,60} at one period (P=256, D=16) — insufficient for three periods.
    This script is the focused calibration sweep added to sub-phase 1.1.

This does NOT build Phase 1 modules and does NOT start main effects. It produces
`results/phase1/grid_calibration.json` so the design chat can commit the §5 rewrite
(specific continuity_center values per period + confirm magnitude/locality).

Three sections:
  A. Feasibility table — every main-effects (D, P) combo: P>=2D check + a real
     build_stream attempt; records measured C / D_global / D_local / norms.
  B. Continuity calibration — at each continuity-axis sweep CONTEXT (the cross
     position's (P, D)): sweep continuity_center across its feasible band, measure C
     (magnitude=0 reference), and report the centre nearest each §5 measured-C bin
     {0.077, 0.4, 0.8} with realised C + deviation + the achievable C range.
  C. Stability smokes — short (2000-step) Primary training at out-of-validated-range
     feasible extremes; checks no NaN/Inf and a finite plateau-ish trajectory.

SCAFFOLDING: all grid values below are tagged with their Phase-0.5 derivation.
"""

from __future__ import annotations

import json
import os
import time

# CUDA determinism (instr §1) — set before importing torch-using modules.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from v2.config import RESULTS_ROOT, load_calibrated_thresholds
from v2.src.property_measure.continuity import measure_continuity
from v2.src.property_measure.manifold_dim import measure_manifold_dim
from v2.src.substrate.base_manifold_trajectory import load_or_create_U
from v2.src.substrate.stream_builder import StreamParams, build_stream
from v2.src.preflight.pre_d1a_endpoint_stability import train_one

# ---- SCAFFOLDING — re-anchored §5 grids (Phase-0.5 follow-up) -----------------
D_GRID = [4, 16, 128]                 # SCAFFOLDING — D shift {4,32,256}->{4,16,128} (Path 1)
P_GRID = [32, 256, 2048]              # SCAFFOLDING — period_P (log), Phase-0.5 D4
MAG_GRID = [0.1, 0.3, 0.7]            # SCAFFOLDING — magnitude_M dial, Phase-0.5
LOC_GRID = [0.3, 0.5, 0.9]            # SCAFFOLDING — locality_L dial, Phase-0.5
C_TARGETS = [0.077, 0.4, 0.8]         # SCAFFOLDING — §5 measured-C bins (0.077 = PRE-B worked-example)

# Cross positions (low/mid/high) per Phase-0.5: index into the 3-value grids.
CROSS = {
    "low":  {"P": 32,   "D": 4},
    "mid":  {"P": 256,  "D": 16},
    "high": {"P": 2048, "D": 128},
}
# PRE-validated construction ranges (PRE-A/D1a/D2): D in [4,64], P in [64,256].
PRE_D_RANGE = (4, 64)
PRE_P_RANGE = (64, 256)

# Flag 1 (design-chat 2026-05-22): record the D-grid revision as a Phase-0.5
# commitment UPDATE via surface-and-confirm (not spec drift). Fact-checked.
GRID_REVISION_PROVENANCE = {
    "summary": ("manifold_dim_D grid revised {4,32,256} -> {4,16,128} via "
                "surface-and-confirm (design-chat-within-Phase-1, 2026-05-22); "
                "not CC drifting from spec."),
    "original_phase05_commitment_D": [4, 32, 256],
    "revised_D": [4, 16, 128],
    "drivers": [
        ("D=256 is construction-infeasible at P in {32,256} (centered_harmonics "
         "needs P>=2D, i.e. P>=512); in the original {4,32,256}x{32,256,2048} grid "
         "this is 3 infeasible (D,P) pairs (32,32),(256,32),(256,256) = 6 "
         "cross-structure cell-measurements per L_d_main."),
        "D=256 is also outside the PRE-A/PRE-D-validated D range (PRE swept D in {4,16,64}; max 64).",
    ],
    "log_spacing": "x4 (4->16) then x8 (16->128); original was uniform x8 (4->32->256).",
    "infeasible_remaining": ("1 unique pair (D=128,P=32) = 2 cross-structure "
                             "cell-measurements per L_d_main; dropped -> not_characterised."),
    "pre_a_range_fact_check": ("CORRECTION to the drafted note 'stays within PRE-A's "
        "validated range': D=128 is NOT within PRE-A's validated D range (PRE max D=64). "
        "Only D=4 and D=16 are. D=128 is out-of-range and is the reason sub-phase 1.0 "
        "ran the stability smokes at D=128 (and P in {32,2048}) — all trained stably "
        "(no NaN/Inf), justifying the extension."),
    "decision_reference": "Phase 1 first-session design chat (2026-05-21/22); see v2/PHASE1_PROGRESS.md",
}


def _feasible(D: int, P: int) -> bool:
    """centered_harmonics requires D distinct harmonics in [1, P//2] -> P >= 2D."""
    return D <= P // 2


def _safe_center(D: int, P: int) -> int:
    """A continuity_center guaranteed valid for (D, P) (mid of the feasible band)."""
    return max(1, min(P // 4, P // 2 - D // 2))


def _measure_stream(stream: np.ndarray, th: dict, cap: int = 4096) -> dict:
    s = stream[: min(stream.shape[0], cap)]
    out = {"n_frames_measured": int(s.shape[0])}
    try:
        out["C"] = measure_continuity(s)["C"]
    except Exception as e:  # noqa: BLE001
        out["C"] = None; out["C_error"] = repr(e)
    try:
        md = measure_manifold_dim(s, window=int(th["LOCAL_PCA_WINDOW"]),
                                  subsample_rate=max(8, int(th["MANIFOLD_SUBSAMPLE_RATE"])))
        out["D_global"] = md["D_global"]; out["D_local"] = md["D_local"]
    except Exception as e:  # noqa: BLE001
        out["D_global"] = None; out["D_error"] = repr(e)
    return out


# ---- A. Feasibility table ----------------------------------------------------

def feasibility_table(U, th) -> dict:
    cells = []
    n_reps_for = lambda P: int(np.clip(round(4096 / P), 4, 128))  # noqa: E731
    for D in D_GRID:
        for P in P_GRID:
            feasible = _feasible(D, P)
            rec = {"manifold_dim_D": D, "period_P": P, "P_ge_2D": feasible,
                   "min_period_for_D": 2 * D,
                   "out_of_pre_range": (D > PRE_D_RANGE[1] or P < PRE_P_RANGE[0]
                                        or P > PRE_P_RANGE[1])}
            if feasible:
                try:
                    sp = StreamParams(manifold_dim_D=D, period_P=P,
                                      continuity_center=_safe_center(D, P),
                                      magnitude_M=0.5, locality_L=0.5,
                                      n_repetitions=n_reps_for(P))
                    bs = build_stream(sp, U)
                    norms = np.linalg.norm(bs.stream, axis=1)
                    rec["built"] = True
                    rec["norm_min"] = float(norms.min()); rec["norm_max"] = float(norms.max())
                    rec["construction_continuity_C"] = bs.construction["continuity_C"]
                    rec["measured"] = _measure_stream(bs.stream, th)
                except Exception as e:  # noqa: BLE001
                    rec["built"] = False; rec["build_error"] = repr(e)
            else:
                rec["built"] = False
                rec["note"] = f"infeasible: needs P>={2*D}, got P={P}"
            cells.append(rec)
    infeasible = [(c["manifold_dim_D"], c["period_P"]) for c in cells if not c["P_ge_2D"]]
    return {"cells": cells, "infeasible_DP_pairs": infeasible,
            "n_infeasible": len(infeasible)}


# ---- B. Continuity calibration ----------------------------------------------

def continuity_calibration(U, th) -> dict:
    """For each continuity-axis sweep context (cross (P,D)), find continuity_center
    landing nearest each §5 measured-C bin."""
    contexts = {}
    for cross, pos in CROSS.items():
        P, D = pos["P"], pos["D"]
        if not _feasible(D, P):
            contexts[cross] = {"P": P, "D": D, "feasible": False,
                               "note": "context infeasible (P<2D); continuity sweep impossible here"}
            continue
        # Feasible continuity_center band: enough margin for D harmonics in [1,P//2].
        c_lo, c_hi = max(1, D // 2 + 1), max(1, P // 2 - D // 2)
        if c_hi <= c_lo:
            # Boundary D==P//2: whole band forced; continuity not controllable here.
            sp = StreamParams(manifold_dim_D=D, period_P=P, continuity_center=c_lo,
                              magnitude_M=0.0, fidelity_F=0.999,
                              n_repetitions=int(np.clip(round(4096 / P), 4, 128)))
            C = measure_continuity(build_stream(sp, U).stream)["C"]
            contexts[cross] = {"P": P, "D": D, "feasible": True,
                               "controllable": False, "fixed_C": C,
                               "note": "D==P//2: harmonics fill the band; continuity is fixed, not swept"}
            continue
        n_steps = min(12, c_hi - c_lo + 1)
        centers = sorted(set(int(round(x)) for x in np.linspace(c_lo, c_hi, n_steps)))
        sweep = []
        for cen in centers:
            sp = StreamParams(manifold_dim_D=D, period_P=P, continuity_center=cen,
                              magnitude_M=0.0, fidelity_F=0.999,
                              n_repetitions=int(np.clip(round(4096 / P), 4, 128)))
            bs = build_stream(sp, U)
            sweep.append({"continuity_center": cen,
                          "measured_C": measure_continuity(bs.stream)["C"],
                          "analytic_C": bs.construction["continuity_C"]})
        Cs = np.array([s["measured_C"] for s in sweep])
        picks = {}
        for tgt in C_TARGETS:
            i = int(np.argmin(np.abs(Cs - tgt)))
            picks[str(tgt)] = {"continuity_center": sweep[i]["continuity_center"],
                               "realised_C": sweep[i]["measured_C"],
                               "deviation": float(abs(sweep[i]["measured_C"] - tgt)),
                               "reachable": bool(Cs.min() <= tgt <= Cs.max())}
        contexts[cross] = {"P": P, "D": D, "feasible": True, "controllable": True,
                           "achievable_C_range": [float(Cs.min()), float(Cs.max())],
                           "center_band": [c_lo, c_hi], "sweep": sweep, "picks": picks}
    return {"targets": C_TARGETS, "contexts": contexts,
            "note": ("centre per (target-C, period) since C ~ 1-cos(2pi*center/P) is "
                     "period-dependent; magnitude=0 reference (perturbation adds a small "
                     "C offset in live cells, recorded post-hoc per cell).")}


# ---- C. Stability smokes -----------------------------------------------------

def stability_smokes(U, device, th) -> dict:
    # Out-of-validated-range feasible extremes (D=128 / P in {32,2048}).
    points = [(4, 32), (16, 32), (128, 256), (4, 2048), (128, 2048)]
    steps = 2000
    out = []
    for (D, P) in points:
        sp = StreamParams(manifold_dim_D=D, period_P=P,
                          continuity_center=_safe_center(D, P),
                          magnitude_M=0.5, locality_L=0.5, fidelity_F=0.97)
        t0 = time.time()
        r = train_one(sp, L_d_main=2, seed=0, U=U, device=device,
                      training_steps=steps, label=f"smoke_D{D}_P{P}",
                      axis="stability", endpoint="oor")
        out.append({"manifold_dim_D": D, "period_P": P, "steps": steps,
                    "wall_s": round(time.time() - t0, 1),
                    "nan_inf": r.nan_inf, "stability_flag": r.stability_flag,
                    "still_descending": r.still_descending,
                    "final_interval_loss": r.final_interval_loss,
                    "diff_mu": r.diff_mu, "diff_sigma": r.diff_sigma})
        print(f"[smoke] D={D} P={P}: flag={r.stability_flag} nan_inf={r.nan_inf} "
              f"loss={r.final_interval_loss} diff_mu={r.diff_mu}")
    return {"training_steps": steps, "points": out,
            "all_finite": all(not p["nan_inf"] for p in out)}


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = RESULTS_ROOT / "phase1"
    out_dir.mkdir(parents=True, exist_ok=True)
    th = load_calibrated_thresholds()
    U = load_or_create_U()
    print(f"[grid_cal] device={device}")

    print("[grid_cal] A. feasibility table ...")
    feas = feasibility_table(U, th)
    print(f"[grid_cal]   infeasible (D,P) pairs: {feas['infeasible_DP_pairs']}")

    print("[grid_cal] B. continuity calibration ...")
    cont = continuity_calibration(U, th)
    for cross, c in cont["contexts"].items():
        if c.get("controllable"):
            print(f"[grid_cal]   {cross} (P={c['P']},D={c['D']}) C-range {c['achievable_C_range']}; "
                  f"picks " + ", ".join(f"{k}->c{v['continuity_center']}(C={v['realised_C']:.3f})"
                                        for k, v in c["picks"].items()))
        else:
            print(f"[grid_cal]   {cross} (P={c['P']},D={c['D']}): {c.get('note')}")

    print("[grid_cal] C. stability smokes (2000-step) ...")
    smk = stability_smokes(U, device, th)

    payload = {
        "purpose": "Phase-1 pre-build grid calibration + range/stability validation",
        "d_grid": D_GRID, "p_grid": P_GRID, "mag_grid": MAG_GRID, "loc_grid": LOC_GRID,
        "c_targets": C_TARGETS, "cross_positions": CROSS,
        "pre_validated_ranges": {"D": PRE_D_RANGE, "P": PRE_P_RANGE},
        "grid_revision_provenance": GRID_REVISION_PROVENANCE,
        "A_feasibility": feas,
        "B_continuity_calibration": cont,
        "C_stability_smokes": smk,
    }
    (out_dir / "grid_calibration.json").write_text(json.dumps(payload, indent=2))
    print(f"[grid_cal] wrote {out_dir / 'grid_calibration.json'}")
    print(f"[grid_cal] feasibility: {feas['n_infeasible']} infeasible pair(s); "
          f"stability all-finite={smk['all_finite']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

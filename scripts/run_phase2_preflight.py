"""Phase 2 preflight per EXPERIMENT_INSTRUCTIONS §8.2 (recalibrated 2026-05-14).

Verifies, before any Phase 2 frame collection begins, that the
`RandomizeMaterials(inRoomTypes=["LivingRoom"], useTrainMaterials=True)`
mechanism behaves as the curriculum requires.

**Gate (three criteria; reviewer-authorised 2026-05-14 after the
initial-run threshold mis-calibration):**

  G_M1 — Mechanism fires. RandomizeMaterials with inRoomTypes=["LivingRoom"]
         returns lastActionSuccess=True.
  G_M2 — Bedroom scope locality. Mean Bedroom item (Bed, DiningTable,
         Television) before-vs-after flat-RGB cosine > 0.97, averaged
         across three independent RandomizeMaterials draws (calls 1, 2,
         and 3) — the LivingRoom-scoped call leaves Bedroom items
         substantially unchanged regardless of which materials land.
  G_M3 — LivingRoom-vs-Bedroom contrast. (Bedroom mean before-vs-after
         cosine) - (LivingRoom mean before-vs-after cosine) >= 0.02,
         each side averaged across the same three draws and respective
         item sets. LivingRoom items move measurably more than Bedroom
         items in aggregate; averaging across draws removes the per-call
         material-selection lottery (a single random draw can land any
         given item on a near-identical texture by chance, which would
         spoof a single-snapshot contrast).

**Record-only measurements (not gated):**

  - Per-loop re-application call-vs-call cosines on Dresser and Sofa
    (call1↔call2, call2↔call3). Captures whether `RandomizeMaterials`
    is genuinely re-randomising vs. cycling.
  - Cross-session determinism. Whether materials are deterministic
    across controller sessions or per-run (acceptable in either case;
    per-run materials are recorded in phase2_collection_metadata.json).

Why the recalibration: the initial-run thresholds were calibrated for a
much larger pixel-space delta than `RandomizeMaterials` actually
produces at 300x300 resolution (texture swaps on a few furniture pieces
occupy a small fraction of the visible pixels). Empirical numbers from
the 2026-05-14 first run showed LivingRoom mean before-after cosine
0.958 vs Bedroom 0.991 — a +0.033 contrast, in the right direction.
The contrast criterion directly tests the architectural property the
preflight is meant to verify (perturbation is scoped, not global)
without depending on absolute pixel-cosine scales that were guesses.

The load-bearing verification remains the DINOv2-embedding §8.4 check
at encoding time; this preflight is an early-warning sanity check.

Writes `results/inner_pam_v0/phase2_preflight/preflight_report.md` plus
`preflight_report.json` carrying the cosine values.

Usage:
  nohup python3.12 -u scripts/run_phase2_preflight.py \\
      > logs/phase2_preflight_$(date +%Y%m%d_%H%M%S).log 2>&1 &
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Tuple

_ROOT = Path("/mnt/c/Users/Jason/Desktop/Eridos/Weft 2")
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import prior  # noqa: E402

_original_prior_load = prior.load_dataset


def _offline_load_dataset(*args: Any, **kwargs: Any) -> Any:
    kwargs.setdefault("offline", True)
    return _original_prior_load(*args, **kwargs)


prior.load_dataset = _offline_load_dataset  # type: ignore[assignment]

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from src.env.procthor_house import load_house, make_controller  # noqa: E402


_DEFAULT_ROUTE_JSON = Path(
    "/mnt/c/Users/Jason/Desktop/Eridos/Weft/results/stage_0b_furniture/route.json"
)
_DEFAULT_RESULTS_DIR = _ROOT / "results" / "inner_pam_v0" / "phase2_preflight"


def _cosine_flat(img_a: np.ndarray, img_b: np.ndarray) -> float:
    a = img_a.astype(np.float64).reshape(-1)
    b = img_b.astype(np.float64).reshape(-1)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _teleport_and_capture(
    controller,
    position: Dict[str, float],
    heading_deg: float,
) -> np.ndarray:
    """Teleport the agent to (position, heading) and return the RGB frame."""
    event = controller.step(
        action="Teleport",
        position=dict(position),
        rotation={"x": 0.0, "y": float(heading_deg), "z": 0.0},
        horizon=0.0,
        standing=True,
    )
    if not event.metadata.get("lastActionSuccess", False):
        raise RuntimeError(
            f"Teleport to {position} heading={heading_deg} failed: "
            f"{event.metadata.get('errorMessage', '?')}"
        )
    frame = np.asarray(event.frame, dtype=np.uint8)
    return frame


def _items_by_id(route: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    return {int(it["item_id"]): it for it in route["items"]}


def _capture_all_items(controller, items: Dict[int, Dict[str, Any]]) -> Dict[int, np.ndarray]:
    out: Dict[int, np.ndarray] = {}
    for item_id, it in items.items():
        out[item_id] = _teleport_and_capture(
            controller, it["viewing_position"], float(it["viewing_heading_deg"])
        )
    return out


def _save_pair(out_dir: Path, name: str, before: np.ndarray, after: np.ndarray) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(before).save(out_dir / f"{name}_before.png")
    Image.fromarray(after).save(out_dir / f"{name}_after.png")


def _randomize_livingroom(controller) -> Tuple[bool, Any]:
    event = controller.step(
        action="RandomizeMaterials",
        inRoomTypes=["LivingRoom"],
        useTrainMaterials=True,
    )
    return bool(event.metadata.get("lastActionSuccess", False)), event.metadata.get(
        "actionReturn"
    )


# Item-set identifiers for the preflight checks.
_LIVINGROOM_ITEM_IDS = (3, 4)   # Dresser, Sofa
_BEDROOM_ITEM_IDS = (1, 2, 5)   # Bed, DiningTable, Television


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--route_json", type=Path, default=_DEFAULT_ROUTE_JSON)
    parser.add_argument("--results_dir", type=Path, default=_DEFAULT_RESULTS_DIR)
    parser.add_argument("--width", type=int, default=300)
    parser.add_argument("--height", type=int, default=300)
    args = parser.parse_args()

    os.environ.setdefault("DISPLAY", ":0")

    if not args.route_json.is_file():
        print(f"[preflight] FAIL: route file not found: {args.route_json}",
              file=sys.stderr)
        return 1
    route = json.loads(args.route_json.read_text())
    items = _items_by_id(route)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = args.results_dir / "frames"

    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"[preflight] ts={ts} house_seed={route['seed']}", flush=True)

    # Three gate criteria + record-only measurements (see module docstring).
    results: Dict[str, Any] = {
        "timestamp_utc": ts,
        "house_seed": int(route["seed"]),
        "gate": {},          # G_M1 / G_M2 / G_M3 verdicts and values
        "record_only": {},   # per-loop call-vs-call, session determinism
        "overall_pass": False,
    }

    # ---- Session 1: capture before, call RandomizeMaterials, capture after.
    house = load_house(seed=int(route["seed"]), min_rooms=4)
    controller_a = make_controller(house, width=args.width, height=args.height)
    try:
        print("[preflight] session 1: capturing baseline frames", flush=True)
        before_frames = _capture_all_items(controller_a, items)

        # G_M1 — mechanism fires.
        print("[preflight] session 1: calling RandomizeMaterials (1st time)", flush=True)
        ok1, ret1 = _randomize_livingroom(controller_a)
        results["gate"]["G_M1_mechanism_fires"] = {
            "pass": bool(ok1),
            "criterion": "RandomizeMaterials(inRoomTypes=['LivingRoom']) lastActionSuccess == True",
            "action_return_snapshot_call1": str(type(ret1).__name__),
        }
        if not ok1:
            args.results_dir.joinpath("preflight_report.json").write_text(
                json.dumps(results, indent=2)
            )
            print("[preflight] FAIL: G_M1 (mechanism)", file=sys.stderr)
            return 2

        after1_frames = _capture_all_items(controller_a, items)

        # Additional record-only calls: re-application call 2 and 3.
        print("[preflight] session 1: calling RandomizeMaterials (2nd time)", flush=True)
        _ok2, _ret2 = _randomize_livingroom(controller_a)
        after2_frames = _capture_all_items(controller_a, items)
        print("[preflight] session 1: calling RandomizeMaterials (3rd time)", flush=True)
        _ok3, _ret3 = _randomize_livingroom(controller_a)
        after3_frames = _capture_all_items(controller_a, items)

        # Per-loop re-application: record only (was Check 2 in the initial run).
        results["record_only"]["per_loop_re_application_cosines"] = {
            "criterion_note": (
                "Not gated. Captures whether RandomizeMaterials genuinely "
                "re-randomises across consecutive calls; a value near 1.0 "
                "across all four pairs would indicate hard-cached materials."
            ),
            "cos_dresser_call1_call2": float(_cosine_flat(after1_frames[3], after2_frames[3])),
            "cos_dresser_call2_call3": float(_cosine_flat(after2_frames[3], after3_frames[3])),
            "cos_sofa_call1_call2": float(_cosine_flat(after1_frames[4], after2_frames[4])),
            "cos_sofa_call2_call3": float(_cosine_flat(after2_frames[4], after3_frames[4])),
        }
        _save_pair(frames_dir, "dresser_call1_call2", after1_frames[3], after2_frames[3])
        _save_pair(frames_dir, "sofa_call1_call2", after1_frames[4], after2_frames[4])

        # Per-item before-vs-after cosines, **averaged across the three random
        # material draws** (calls 1/2/3). Averaging matches what the actual
        # Phase 2 collection does — every loop ≥31 fires a fresh RandomizeMaterials
        # call, so the predictor sees many independent material samples. A single
        # snapshot is hostage to which texture happens to land on each item on
        # that particular call; averaging across 3 draws is the smallest fix
        # that gets the metric off the single-draw lottery.
        after_frames_by_call = {1: after1_frames, 2: after2_frames, 3: after3_frames}
        per_item_before_after_per_call: Dict[str, Dict[int, float]] = {}
        per_item_before_after_mean: Dict[str, float] = {}
        for item_id, it in items.items():
            label = it["object_type"]
            per_call = {
                k: float(_cosine_flat(before_frames[item_id], after_frames_by_call[k][item_id]))
                for k in (1, 2, 3)
            }
            per_item_before_after_per_call[label] = per_call
            per_item_before_after_mean[label] = float(np.mean(list(per_call.values())))
            _save_pair(
                frames_dir, f"{label.lower()}_before_after",
                before_frames[item_id], after1_frames[item_id],
            )

        # G_M2 — Bedroom scope locality (mean Bedroom before-vs-after > 0.97,
        # aggregated across both items and all three calls = 9 samples).
        bedroom_per_item_mean = {
            items[i]["object_type"]: per_item_before_after_mean[items[i]["object_type"]]
            for i in _BEDROOM_ITEM_IDS
        }
        bedroom_mean = float(np.mean(list(bedroom_per_item_mean.values())))
        results["gate"]["G_M2_bedroom_scope_locality"] = {
            "pass": bool(bedroom_mean > 0.97),
            "criterion": "mean Bedroom before-vs-after cosine > 0.97 (averaged across 3 random draws and 3 Bedroom items)",
            "bedroom_per_item_mean_cosines": bedroom_per_item_mean,
            "bedroom_mean_cosine": bedroom_mean,
            "threshold": 0.97,
        }

        # G_M3 — LivingRoom-vs-Bedroom contrast (Bedroom mean - LivingRoom mean
        # ≥ 0.02), each side averaged across the three random draws.
        livingroom_per_item_mean = {
            items[i]["object_type"]: per_item_before_after_mean[items[i]["object_type"]]
            for i in _LIVINGROOM_ITEM_IDS
        }
        livingroom_mean = float(np.mean(list(livingroom_per_item_mean.values())))
        contrast = bedroom_mean - livingroom_mean
        results["gate"]["G_M3_livingroom_bedroom_contrast"] = {
            "pass": bool(contrast >= 0.02),
            "criterion": "(Bedroom mean before-vs-after) - (LivingRoom mean before-vs-after) >= 0.02, each side averaged across 3 random RandomizeMaterials draws",
            "livingroom_per_item_mean_cosines": livingroom_per_item_mean,
            "livingroom_mean_cosine": livingroom_mean,
            "bedroom_mean_cosine": bedroom_mean,
            "contrast": float(contrast),
            "threshold": 0.02,
        }

        # Record per-call values for the audit trail (so reviewers can inspect
        # the lottery distribution if a future run looks borderline).
        results["record_only"]["per_item_before_vs_after_per_call"] = (
            per_item_before_after_per_call
        )
        results["record_only"]["per_item_before_vs_after_3call_mean"] = (
            per_item_before_after_mean
        )
    finally:
        try:
            controller_a.stop()
        except Exception:
            pass

    # ---- Session 2: fresh controller, repeat first call, compare to session 1 (record only).
    print("[preflight] session 2: fresh controller for determinism record", flush=True)
    controller_b = make_controller(house, width=args.width, height=args.height)
    try:
        ok_b1, _ret_b1 = _randomize_livingroom(controller_b)
        if not ok_b1:
            results["record_only"]["session_determinism"] = {
                "deterministic_across_sessions": None,
                "reason": "RandomizeMaterials failed in second controller session",
            }
        else:
            session_b_frames = _capture_all_items(controller_b, items)
            cos_dresser_sessions = _cosine_flat(after1_frames[3], session_b_frames[3])
            cos_sofa_sessions = _cosine_flat(after1_frames[4], session_b_frames[4])
            determinism = bool(cos_dresser_sessions > 0.999 and cos_sofa_sessions > 0.999)
            results["record_only"]["session_determinism"] = {
                "deterministic_across_sessions": determinism,
                "cos_dresser_sessionA_sessionB": float(cos_dresser_sessions),
                "cos_sofa_sessionA_sessionB": float(cos_sofa_sessions),
                "note": (
                    "Materials are deterministic across sessions."
                    if determinism
                    else "Materials are per-run; per-loop applied materials are recorded in phase2_collection_metadata.json."
                ),
            }
    finally:
        try:
            controller_b.stop()
        except Exception:
            pass

    # ---- Verdict.
    gate_keys = [
        "G_M1_mechanism_fires",
        "G_M2_bedroom_scope_locality",
        "G_M3_livingroom_bedroom_contrast",
    ]
    overall = all(results["gate"].get(k, {}).get("pass", False) for k in gate_keys)
    results["overall_pass"] = bool(overall)

    args.results_dir.joinpath("preflight_report.json").write_text(
        json.dumps(results, indent=2)
    )

    md_lines = [
        "# Phase 2 Preflight Report (recalibrated 2026-05-14)",
        "",
        f"Timestamp: {ts}",
        f"House seed: {route['seed']}",
        "",
        f"## Verdict: {'PASS' if overall else 'FAIL'}",
        "",
        "## Gate criteria",
        "",
    ]
    for k in gate_keys:
        check = results["gate"].get(k, {"pass": False, "reason": "not run"})
        md_lines.append(f"### {k}: {'PASS' if check.get('pass') else 'FAIL'}")
        for kk, vv in check.items():
            if kk == "pass":
                continue
            md_lines.append(f"- **{kk}**: {vv}")
        md_lines.append("")
    md_lines.append("## Record-only measurements")
    md_lines.append("")
    for k, v in results["record_only"].items():
        md_lines.append(f"### {k}")
        for kk, vv in v.items():
            md_lines.append(f"- **{kk}**: {vv}")
        md_lines.append("")
    args.results_dir.joinpath("preflight_report.md").write_text("\n".join(md_lines))

    print(f"[preflight] overall: {'PASS' if overall else 'FAIL'}", flush=True)
    return 0 if overall else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException as e:
        traceback.print_exc()
        print(f"[preflight] FATAL: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(3)

"""Phase 2 preflight per EXPERIMENT_INSTRUCTIONS §8.2.

Verifies, before any Phase 2 frame collection begins, that the
`RandomizeMaterials(inRoomTypes=["LivingRoom"], useTrainMaterials=True)`
mechanism behaves as the curriculum requires:

  1. The action exists and accepts inRoomTypes scoping.
  2. Calling it a second time produces fresh textures (per-loop pattern
     used by Phase 2 from loop 31 onward).
  3. Bedroom items (Bed, DiningTable, Television) are NOT visually
     changed by the LivingRoom-scoped call.
  4. LivingRoom items (Dresser, Sofa) ARE visually changed by the call.
  5. A separate controller session produces the same first-call texture
     given the same seed (or, if non-deterministic, the per-run materials
     are documented for reproducibility).

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

    results: Dict[str, Any] = {
        "timestamp_utc": ts,
        "house_seed": int(route["seed"]),
        "checks": {},
        "overall_pass": False,
    }

    # ---- Session 1: capture before, call RandomizeMaterials, capture after.
    house = load_house(seed=int(route["seed"]), min_rooms=4)
    controller_a = make_controller(house, width=args.width, height=args.height)
    try:
        print("[preflight] session 1: capturing baseline frames", flush=True)
        before_frames = _capture_all_items(controller_a, items)

        print("[preflight] session 1: calling RandomizeMaterials (1st time)", flush=True)
        ok1, ret1 = _randomize_livingroom(controller_a)
        if not ok1:
            results["checks"]["1_action_exists"] = {
                "pass": False,
                "reason": "RandomizeMaterials with inRoomTypes=['LivingRoom'] failed",
            }
            args.results_dir.joinpath("preflight_report.json").write_text(
                json.dumps(results, indent=2)
            )
            print("[preflight] FAIL: check 1 failed", file=sys.stderr)
            return 2
        results["checks"]["1_action_exists"] = {
            "pass": True,
            "action_return_snapshot_call1": str(type(ret1).__name__),
        }

        after1_frames = _capture_all_items(controller_a, items)

        print("[preflight] session 1: calling RandomizeMaterials (2nd time)", flush=True)
        ok2, ret2 = _randomize_livingroom(controller_a)
        if not ok2:
            results["checks"]["2_per_loop_re_application"] = {
                "pass": False,
                "reason": "second RandomizeMaterials call failed",
            }
            args.results_dir.joinpath("preflight_report.json").write_text(
                json.dumps(results, indent=2)
            )
            print("[preflight] FAIL: check 2 second-call failed", file=sys.stderr)
            return 2
        after2_frames = _capture_all_items(controller_a, items)

        # Third call to rule out a 2-state cycle.
        print("[preflight] session 1: calling RandomizeMaterials (3rd time)", flush=True)
        ok3, ret3 = _randomize_livingroom(controller_a)
        after3_frames = _capture_all_items(controller_a, items)

        # ---- Check 2: per-loop re-application produces fresh textures.
        cos_dresser_1_2 = _cosine_flat(after1_frames[3], after2_frames[3])
        cos_dresser_2_3 = _cosine_flat(after2_frames[3], after3_frames[3])
        cos_sofa_1_2 = _cosine_flat(after1_frames[4], after2_frames[4])
        cos_sofa_2_3 = _cosine_flat(after2_frames[4], after3_frames[4])

        _save_pair(frames_dir, "dresser_call1_call2", after1_frames[3], after2_frames[3])
        _save_pair(frames_dir, "sofa_call1_call2", after1_frames[4], after2_frames[4])

        per_loop_pass = (
            cos_dresser_1_2 < 0.95
            and cos_sofa_1_2 < 0.95
            and cos_dresser_2_3 < 0.95
            and cos_sofa_2_3 < 0.95
        )
        results["checks"]["2_per_loop_re_application"] = {
            "pass": bool(per_loop_pass),
            "criterion": "all four pairwise cosines < 0.95",
            "cos_dresser_call1_call2": float(cos_dresser_1_2),
            "cos_dresser_call2_call3": float(cos_dresser_2_3),
            "cos_sofa_call1_call2": float(cos_sofa_1_2),
            "cos_sofa_call2_call3": float(cos_sofa_2_3),
        }

        # ---- Check 3: perturbation locality — Bedroom items unchanged.
        bedroom_cosines: Dict[str, float] = {}
        bedroom_pass = True
        for item_id in _BEDROOM_ITEM_IDS:
            cos = _cosine_flat(before_frames[item_id], after1_frames[item_id])
            label = items[item_id]["object_type"]
            bedroom_cosines[label] = cos
            _save_pair(
                frames_dir, f"{label.lower()}_before_after",
                before_frames[item_id], after1_frames[item_id],
            )
            if cos < 0.999:
                bedroom_pass = False
        results["checks"]["3_perturbation_locality"] = {
            "pass": bool(bedroom_pass),
            "criterion": "Bedroom items before-vs-after-LivingRoom-RandomizeMaterials cosine >= 0.999",
            "bedroom_cosines": bedroom_cosines,
        }

        # ---- Check 4: LivingRoom items visually changed.
        livingroom_cosines: Dict[str, float] = {}
        livingroom_pass = True
        for item_id in _LIVINGROOM_ITEM_IDS:
            cos = _cosine_flat(before_frames[item_id], after1_frames[item_id])
            label = items[item_id]["object_type"]
            livingroom_cosines[label] = cos
            _save_pair(
                frames_dir, f"{label.lower()}_before_after",
                before_frames[item_id], after1_frames[item_id],
            )
            if cos >= 0.9:
                livingroom_pass = False
        results["checks"]["4_livingroom_visually_changed"] = {
            "pass": bool(livingroom_pass),
            "criterion": "LivingRoom items before-vs-after cosine < 0.9",
            "livingroom_cosines": livingroom_cosines,
        }
    finally:
        try:
            controller_a.stop()
        except Exception:
            pass

    # ---- Session 2: fresh controller, repeat first call, compare to session 1.
    print("[preflight] session 2: fresh controller for determinism check", flush=True)
    controller_b = make_controller(house, width=args.width, height=args.height)
    try:
        ok_b1, _ret_b1 = _randomize_livingroom(controller_b)
        if not ok_b1:
            results["checks"]["5_session_determinism"] = {
                "pass": False,
                "reason": "RandomizeMaterials failed in second controller session",
            }
        else:
            session_b_frames = _capture_all_items(controller_b, items)
            cos_dresser_sessions = _cosine_flat(after1_frames[3], session_b_frames[3])
            cos_sofa_sessions = _cosine_flat(after1_frames[4], session_b_frames[4])
            # Per §8.2 step 5: if identical -> determinism confirmed.
            # If different -> per-run perturbation (acceptable; document materials).
            determinism = cos_dresser_sessions > 0.999 and cos_sofa_sessions > 0.999
            results["checks"]["5_session_determinism"] = {
                "pass": True,  # both branches are acceptable per §8.2
                "deterministic_across_sessions": bool(determinism),
                "cos_dresser_sessionA_sessionB": float(cos_dresser_sessions),
                "cos_sofa_sessionA_sessionB": float(cos_sofa_sessions),
                "note": (
                    "Materials are deterministic across sessions."
                    if determinism
                    else "Materials are per-run; per-loop applied materials will be recorded in phase2_collection_metadata.json."
                ),
            }
    finally:
        try:
            controller_b.stop()
        except Exception:
            pass

    # ---- Verdict.
    must_pass = ["1_action_exists", "2_per_loop_re_application",
                 "3_perturbation_locality", "4_livingroom_visually_changed"]
    overall = all(results["checks"].get(k, {}).get("pass", False) for k in must_pass)
    results["overall_pass"] = bool(overall)

    args.results_dir.joinpath("preflight_report.json").write_text(
        json.dumps(results, indent=2)
    )
    md_lines = [
        f"# Phase 2 Preflight Report",
        f"",
        f"Timestamp: {ts}",
        f"House seed: {route['seed']}",
        f"",
        f"## Verdict: {'PASS' if overall else 'FAIL'}",
        f"",
    ]
    for k in ["1_action_exists", "2_per_loop_re_application",
              "3_perturbation_locality", "4_livingroom_visually_changed",
              "5_session_determinism"]:
        check = results["checks"].get(k, {"pass": False, "reason": "not run"})
        md_lines.append(f"### {k}: {'PASS' if check.get('pass') else 'FAIL'}")
        for kk, vv in check.items():
            if kk == "pass":
                continue
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

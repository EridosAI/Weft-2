#!/usr/bin/env python3
"""Asset-replacement smoke test, single-item (Dresser).

Authorised by design chat (2026-05-19) as pre-investment characterisation,
not a substrate commitment (instr §1.5 unchanged). Single item, single path,
single substitute. No selection lock, no band re-anchoring, no PRE-C launch.

Bounds:
  - Single item: Dresser (item_id 3, LivingRoom).
  - Single mechanism path. The design-chat-spec'd "RemoveFromScene +
    SpawnTargetCircle" path is empirically unviable: SpawnTargetCircle is
    a nav-target indicator, not an asset spawner; `RemoveFromScene` hangs
    indefinitely on this AI2-THOR build (verified 2026-05-19, 90s timeout
    with no return). The design-chat-spec'd fallback "CreateObject with a
    built-in asset type" is also unviable: `CreateObject(objectType=...)`
    returns Unity-side `ArgumentOutOfRangeException` on this ProcTHOR
    seed-7 scene. The simplest *viable* path that achieves the
    design-chat-intent (visually-distinct content at the Dresser's
    canonical viewing pose) is:
      1. `DisableObject(objectId=Dresser.objectId)` — removes the Dresser
         from rendering. AI2-THOR action confirmed available.
      2. `PlaceObjectAtPoint(objectId=Chair.objectId, position=Dresser.position)`
         — relocates an existing Chair to the Dresser's coordinates.
  - Substitute: first Chair (objectType='Chair') in the scene. Arbitrary.
  - 2-hour budget. If unexpected substrate properties surface, stop+report.

Measurements (at viewing position 1 of each item, frames encoded via
DINOv2-Large frozen fp16):
  - cross-stage cosine drop at Dresser (perturbed)
  - cross-stage cosine drops at Bed, DiningTable, Sofa, Television (locality)
  - api_success at each AI2-THOR call (per run)
  - reproducibility: re-run with same seed, verify cross-stage cosines
    within 0.005

Output:
  v1/results/inner_pam_v1/pre_b_perturbation/smoke_test_asset_replacement_dresser/smoke_report.json
"""

from __future__ import annotations

import datetime as dt
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_ROOT = Path("/mnt/c/Users/Jason/Desktop/Eridos/Weft 2")
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import prior  # noqa: E402
_orig = prior.load_dataset
prior.load_dataset = (lambda f: lambda *a, **kw: f(*a, **{**kw, "offline": True}))(_orig)  # type: ignore[assignment]

import numpy as np  # noqa: E402
import torch  # noqa: E402

from shared.encoder.dinov2_encoder import load_frozen_dinov2  # noqa: E402
from shared.substrate.procthor_house import load_house, make_controller  # noqa: E402
from v0.src.env.material_perturbation_probe import (  # noqa: E402
    dinov2_encode_batch,
    items_by_id,
    teleport_and_capture,
)
from v1.src.config import PATHS  # noqa: E402


OUTPUT_DIR = (
    PATHS.results_pre_b / "smoke_test_asset_replacement_dresser"
)
ROUTE_JSON = _ROOT / "v0" / "data" / "route_phase2.json"

PERTURBED_ITEM_ID = 3   # Dresser
ALL_ITEM_IDS: Tuple[int, ...] = (1, 2, 3, 4, 5)


def _capture_frames(
    controller, items: Dict[int, Dict[str, Any]]
) -> Dict[int, np.ndarray]:
    out: Dict[int, np.ndarray] = {}
    for item_id in ALL_ITEM_IDS:
        it = items[item_id]
        out[item_id] = teleport_and_capture(
            controller, it["viewing_position"], float(it["viewing_heading_deg"])
        )
    return out


def _apply_replacement(
    controller, dresser_objectId: str, dresser_position: dict, substitute_objectId: str
) -> Dict[str, Any]:
    """Run the asset-replacement sequence; return a per-step api-success log."""
    log: Dict[str, Any] = {"steps": []}

    # Step 1: DisableObject(Dresser)
    ev = controller.step(action="DisableObject", objectId=dresser_objectId)
    log["steps"].append({
        "step": "DisableObject",
        "objectId": dresser_objectId,
        "lastActionSuccess": bool(ev.metadata["lastActionSuccess"]),
        "errorMessage": ev.metadata.get("errorMessage", ""),
    })

    # Step 2: PlaceObjectAtPoint(Chair, Dresser.position)
    ev = controller.step(
        action="PlaceObjectAtPoint",
        objectId=substitute_objectId,
        position=dresser_position,
    )
    log["steps"].append({
        "step": "PlaceObjectAtPoint",
        "objectId": substitute_objectId,
        "position": dresser_position,
        "lastActionSuccess": bool(ev.metadata["lastActionSuccess"]),
        "errorMessage": ev.metadata.get("errorMessage", ""),
    })

    log["all_steps_ok"] = all(s["lastActionSuccess"] for s in log["steps"])
    return log


def _run_one_pass(
    controller,
    items: Dict[int, Dict[str, Any]],
    dinov2,
    device: torch.device,
) -> Dict[str, Any]:
    """One full pass: capture pre, apply, capture post, return measurements."""
    # Resolve Dresser + Chair object IDs from the live scene.
    ev = controller.step(action="Pass")
    objects = ev.metadata["objects"]
    dressers = [o for o in objects if o["objectType"] == "Dresser"]
    if not dressers:
        return {"abort_reason": "no Dresser found in scene"}
    chairs = [o for o in objects if o["objectType"] == "Chair"]
    if not chairs:
        return {"abort_reason": "no Chair found in scene"}
    dresser = dressers[0]
    chair = chairs[0]

    # Capture pre-perturbation frames
    pre_frames = _capture_frames(controller, items)

    # Apply replacement
    apply_log = _apply_replacement(
        controller, dresser["objectId"], dresser["position"], chair["objectId"]
    )

    # Capture post-perturbation frames
    post_frames = _capture_frames(controller, items)

    # Encode pre + post in one batch for efficiency
    all_frames = (
        [pre_frames[i] for i in ALL_ITEM_IDS] + [post_frames[i] for i in ALL_ITEM_IDS]
    )
    emb = dinov2_encode_batch(dinov2, all_frames, device=device)
    n = len(ALL_ITEM_IDS)
    pre_emb = {item_id: emb[i] for i, item_id in enumerate(ALL_ITEM_IDS)}
    post_emb = {item_id: emb[n + i] for i, item_id in enumerate(ALL_ITEM_IDS)}

    drops = {
        item_id: float(1.0 - float(np.dot(pre_emb[item_id], post_emb[item_id])))
        for item_id in ALL_ITEM_IDS
    }

    return {
        "dresser_objectId": dresser["objectId"],
        "dresser_position": dresser["position"],
        "substitute_objectType": chair["objectType"],
        "substitute_objectId": chair["objectId"],
        "apply_log": apply_log,
        "cross_stage_cosine_drops": drops,
    }


def main() -> int:
    os.environ.setdefault("DISPLAY", ":0")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[smoke] device={device}", flush=True)

    route = json.loads(ROUTE_JSON.read_text())
    items = items_by_id(route)
    print(
        f"[smoke] route: seed={route['seed']} "
        f"items={[it['object_type'] for it in route['items']]}",
        flush=True,
    )

    print("[smoke] loading DINOv2-Large frozen encoder...", flush=True)
    dinov2 = load_frozen_dinov2(device)

    runs: List[Dict[str, Any]] = []
    t0 = time.time()

    for run_idx in range(2):  # primary + reproducibility
        print(f"\n[smoke] === run {run_idx + 1}/2 ===", flush=True)
        # Fresh controller each run so the scene state resets cleanly
        # (DisableObject + PlaceObjectAtPoint are not easily reversible).
        print(f"[smoke] loading house + controller (run {run_idx + 1})...", flush=True)
        house = load_house(seed=int(route["seed"]), min_rooms=4)
        controller = make_controller(house, width=300, height=300, grid_size=0.25)
        t_run = time.time()
        try:
            result = _run_one_pass(controller, items, dinov2, device)
        except Exception as e:
            import traceback as tb
            tb.print_exc()
            result = {"abort_reason": f"exception: {type(e).__name__}: {e}"}
        finally:
            try:
                controller.stop()
            except Exception:
                pass

        result["run_idx"] = run_idx
        result["run_wall_seconds"] = time.time() - t_run
        runs.append(result)

        if "abort_reason" in result:
            print(f"[smoke] run {run_idx + 1} ABORTED: {result['abort_reason']}", flush=True)
            break

        print(
            f"[smoke] run {run_idx + 1} drops: "
            f"{ {k: round(v, 4) for k, v in result['cross_stage_cosine_drops'].items()} } "
            f"(t={result['run_wall_seconds']:.1f}s)",
            flush=True,
        )

    # Build verdict
    if len(runs) >= 2 and "cross_stage_cosine_drops" in runs[0] and "cross_stage_cosine_drops" in runs[1]:
        repro_diffs = {
            k: abs(runs[0]["cross_stage_cosine_drops"][k] - runs[1]["cross_stage_cosine_drops"][k])
            for k in ALL_ITEM_IDS
        }
        repro_pass = all(v < 0.005 for v in repro_diffs.values())
    else:
        repro_diffs = {}
        repro_pass = None

    perturbed_drop_run1 = (
        runs[0]["cross_stage_cosine_drops"].get(PERTURBED_ITEM_ID)
        if "cross_stage_cosine_drops" in runs[0]
        else None
    )
    locality_drops_run1 = (
        {k: v for k, v in runs[0]["cross_stage_cosine_drops"].items() if k != PERTURBED_ITEM_ID}
        if "cross_stage_cosine_drops" in runs[0]
        else {}
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "wall_clock_seconds": time.time() - t0,
        "authorisation": (
            "Design-chat 2026-05-19: asset-replacement smoke test, single item "
            "(Dresser), single path, no selection commitment, no band amendment."
        ),
        "ai2thor_path_used": {
            "primary_attempted": "RemoveFromScene + SpawnTargetCircle",
            "primary_status": (
                "unviable: SpawnTargetCircle is a nav-target indicator; "
                "RemoveFromScene hangs indefinitely on this AI2-THOR build"
            ),
            "fallback_attempted": "CreateObject with built-in asset type (ArmChair)",
            "fallback_status": (
                "unviable: Unity-side ArgumentOutOfRangeException — prefab "
                "pool does not expose the requested asset for fresh creation"
            ),
            "actual_path": (
                "DisableObject(Dresser) + PlaceObjectAtPoint(Chair, Dresser.position) "
                "— the simplest viable path that achieves the design-chat-intent of "
                "visually-distinct content at the Dresser's canonical viewing pose. "
                "No new asset spawn; relocates an existing Chair into the Dresser's "
                "coordinates after the Dresser is disabled."
            ),
        },
        "perturbed_item_id": PERTURBED_ITEM_ID,
        "perturbed_item_name": "Dresser",
        "all_item_ids": list(ALL_ITEM_IDS),
        "runs": runs,
        "summary": {
            "perturbed_drop_run1": perturbed_drop_run1,
            "locality_drops_run1": locality_drops_run1,
            "reproducibility_diffs": repro_diffs,
            "reproducibility_pass": repro_pass,
        },
    }
    report_path = OUTPUT_DIR / "smoke_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\n[smoke] report written: {report_path}", flush=True)
    if perturbed_drop_run1 is not None:
        print(
            f"[smoke] perturbed Dresser drop (run 1): {perturbed_drop_run1:.4f}", flush=True
        )
        print(f"[smoke] locality drops (run 1): {locality_drops_run1}", flush=True)
        print(f"[smoke] reproducibility: pass={repro_pass}, diffs={repro_diffs}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

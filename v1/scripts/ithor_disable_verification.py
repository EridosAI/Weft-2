#!/usr/bin/env python3
"""iTHOR DisableObject single-call substrate-property probe.

Authorised by design chat (2026-05-19) as v1 closing-document precision
only. Tests whether the AI2-THOR 5.0.0 DisableObject render-NO-OP
behaviour is API-fundamental or ProcTHOR-specific by running a single
DisableObject call on a default iTHOR scene (FloorPlan1, no ProcTHOR).

Bounds:
  - Single call. No second mechanism. No exploration.
  - 90 s timeout on the DisableObject call itself.
  - Three outcome categories:
      "hang"           → API-fundamental (timeout)
      "metadata-only"  → API-fundamental (ok=True but pixel-sum-diff=0)
      "renders"        → ProcTHOR-specific (ok=True and pixel-sum-diff > 0)
"""

from __future__ import annotations

import datetime as dt
import json
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

_ROOT = Path("/mnt/c/Users/Jason/Desktop/Eridos/Weft 2")
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

import ai2thor  # noqa: E402
from ai2thor.controller import Controller  # noqa: E402

from shared.encoder.dinov2_encoder import load_frozen_dinov2  # noqa: E402
from v0.src.env.material_perturbation_probe import dinov2_encode_batch  # noqa: E402
from v1.src.config import PATHS  # noqa: E402


OUTPUT_DIR = PATHS.results_pre_b / "ithor_disable_verification"
SCENE_NAME = "FloorPlan1"
PRE_PNG_PATH = OUTPUT_DIR / "pre_call.png"
POST_PNG_PATH = OUTPUT_DIR / "post_call.png"
REPORT_PATH = OUTPUT_DIR / "probe_report.json"


class _TimeoutError(Exception):
    pass


def _timeout_handler(signum, frame):
    raise _TimeoutError("DisableObject call exceeded timeout")


def _pick_target(controller) -> Optional[Dict[str, Any]]:
    """Find a clearly visible non-structural object in the current view.

    Returns the first object with `visible=True` and `pickupable=True`
    (excluding Wall / Floor / Ceiling). If none with pickupable, falls
    back to the first `visible=True` non-structural object.
    """
    event = controller.step(action="Pass")
    objects = event.metadata["objects"]
    structural = {"Wall", "Floor", "Ceiling", "Window", "Doorway"}
    visible_objs = [o for o in objects if o.get("visible") and o["objectType"] not in structural]
    if not visible_objs:
        return None
    # Prefer pickupable for clean visual distinction
    pickupable = [o for o in visible_objs if o.get("pickupable")]
    if pickupable:
        return pickupable[0]
    return visible_objs[0]


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("DISPLAY", ":0")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    report: Dict[str, Any] = {
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "authorisation": (
            "Design chat 2026-05-19: iTHOR DisableObject verification, single call, "
            "30-minute budget, v1 closing-document precision."
        ),
        "ai2thor_version": ai2thor.__version__,
        "scene": SCENE_NAME,
        "device": str(device),
    }

    print(f"[ithor] ai2thor={ai2thor.__version__}, scene={SCENE_NAME}", flush=True)
    print("[ithor] launching Controller (no ProcTHOR house, default iTHOR)...", flush=True)
    t0 = time.time()
    controller = Controller(scene=SCENE_NAME, width=300, height=300)
    print(f"[ithor] controller up in {time.time() - t0:.1f}s", flush=True)

    try:
        # Snapshot the scene + pick a target
        target = _pick_target(controller)
        if target is None:
            report["outcome"] = "abort"
            report["abort_reason"] = "no visible non-structural object in default view"
            REPORT_PATH.write_text(json.dumps(report, indent=2))
            print("[ithor] ABORT: no visible non-structural object", flush=True)
            return 1

        report["target_object_id"] = target["objectId"]
        report["target_object_type"] = target["objectType"]
        report["target_object_position"] = target["position"]
        report["target_object_pickupable"] = bool(target.get("pickupable"))
        report["agent_pose_at_capture"] = {
            "position": controller.last_event.metadata["agent"]["position"],
            "rotation": controller.last_event.metadata["agent"]["rotation"],
            "horizon": controller.last_event.metadata["agent"].get("cameraHorizon"),
        }
        print(
            f"[ithor] target: {target['objectId']} ({target['objectType']}) "
            f"pickupable={target.get('pickupable')} at {target['position']}",
            flush=True,
        )

        # Pre-call frame
        pre_event = controller.step(action="Pass")
        pre_frame = np.asarray(pre_event.frame, dtype=np.uint8)
        Image.fromarray(pre_frame).save(PRE_PNG_PATH)
        report["pre_png_path"] = str(PRE_PNG_PATH)
        print(f"[ithor] pre frame saved ({pre_frame.shape}); sum={pre_frame.sum()}", flush=True)

        # DisableObject call with 90s timeout
        print(f"[ithor] calling DisableObject(objectId={target['objectId']}) with 90s timeout...", flush=True)
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(90)
        disable_t0 = time.time()
        try:
            disable_event = controller.step(action="DisableObject", objectId=target["objectId"])
            disable_dt = time.time() - disable_t0
            signal.alarm(0)
            disable_ok = bool(disable_event.metadata.get("lastActionSuccess", False))
            disable_err = disable_event.metadata.get("errorMessage", "")
            report["disable_call"] = {
                "lastActionSuccess": disable_ok,
                "errorMessage": disable_err[:500],
                "wall_seconds": disable_dt,
            }
            print(
                f"[ithor] DisableObject returned in {disable_dt:.2f}s: ok={disable_ok}",
                flush=True,
            )
            if not disable_ok:
                print(f"[ithor]   errorMessage: {disable_err[:300]}", flush=True)
        except _TimeoutError:
            signal.alarm(0)
            report["disable_call"] = {
                "lastActionSuccess": "TIMEOUT",
                "wall_seconds": time.time() - disable_t0,
                "errorMessage": "Call exceeded 90s timeout (no return from controller.step)",
            }
            report["outcome"] = "hang"
            report["conclusion"] = (
                "API-fundamental limitation: DisableObject hangs on iTHOR FloorPlan1, "
                "mirroring the RemoveFromScene hang observed on ProcTHOR. The render-NO-OP "
                "and hang behaviours are properties of ai2thor==5.0.0 itself, independent "
                "of scene type."
            )
            REPORT_PATH.write_text(json.dumps(report, indent=2))
            print(f"[ithor] outcome=hang (timeout); report written", flush=True)
            return 0

        # Post-call frame
        post_event = controller.step(action="Pass")
        post_frame = np.asarray(post_event.frame, dtype=np.uint8)
        Image.fromarray(post_frame).save(POST_PNG_PATH)
        report["post_png_path"] = str(POST_PNG_PATH)
        print(f"[ithor] post frame saved; sum={post_frame.sum()}", flush=True)

        # Pixel-sum-diff
        pixel_sum_diff = int(np.abs(pre_frame.astype(np.int64) - post_frame.astype(np.int64)).sum())
        report["pixel_sum_diff"] = pixel_sum_diff
        print(f"[ithor] pixel_sum_diff = {pixel_sum_diff}", flush=True)

        # DINOv2 cosine drop (single forward pass)
        print("[ithor] loading DINOv2-Large frozen encoder...", flush=True)
        dinov2 = load_frozen_dinov2(device)
        emb = dinov2_encode_batch(dinov2, [pre_frame, post_frame], device=device)
        cos = float(np.dot(emb[0], emb[1]))
        cosine_drop = 1.0 - cos
        report["dinov2_cosine"] = cos
        report["dinov2_cosine_drop"] = cosine_drop
        print(f"[ithor] DINOv2 cosine drop = {cosine_drop:.6f}", flush=True)

        # Re-check target object's metadata to see if it became visible=False
        post_objs_for_target = [
            o for o in post_event.metadata["objects"] if o["objectId"] == target["objectId"]
        ]
        report["target_in_metadata_after_call"] = {
            "present": len(post_objs_for_target) > 0,
            "visible": (
                bool(post_objs_for_target[0]["visible"]) if post_objs_for_target else None
            ),
        }

        # Classify outcome
        if not disable_ok:
            report["outcome"] = "api_failure"
            report["conclusion"] = (
                f"DisableObject failed on iTHOR with lastActionSuccess=False (errorMessage: "
                f"{disable_err[:200]!r}). API-fundamental limitation on ai2thor==5.0.0."
            )
        elif pixel_sum_diff == 0:
            report["outcome"] = "metadata-only"
            report["conclusion"] = (
                "API-fundamental limitation: DisableObject on iTHOR FloorPlan1 succeeds at "
                "the metadata layer (lastActionSuccess=True) but produces zero pixel change "
                "in the rendered scene (pixel_sum_diff = 0). This mirrors the ProcTHOR-side "
                "observation exactly. The render-NO-OP behaviour is a property of "
                "ai2thor==5.0.0's DisableObject implementation regardless of scene type."
            )
        else:
            report["outcome"] = "renders"
            report["conclusion"] = (
                f"ProcTHOR-specific limitation: DisableObject on iTHOR FloorPlan1 produces a "
                f"visible render change (pixel_sum_diff = {pixel_sum_diff}, DINOv2 cosine "
                f"drop = {cosine_drop:.6f}). The render-NO-OP behaviour observed on ProcTHOR "
                f"is therefore tied to procedural-house geometry / loading, not to "
                f"DisableObject itself. The v1 substrate finding is ProcTHOR-scoped."
            )

        REPORT_PATH.write_text(json.dumps(report, indent=2))
        print(f"[ithor] report written: {REPORT_PATH}", flush=True)
        print(f"[ithor] OUTCOME: {report['outcome']}", flush=True)
        print(f"[ithor] CONCLUSION: {report['conclusion']}", flush=True)
        return 0

    finally:
        try:
            controller.stop()
        except Exception:
            pass
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    raise SystemExit(main())

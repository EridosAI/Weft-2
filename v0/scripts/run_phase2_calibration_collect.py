"""5-loop calibration collection for the new continuous-motion substrate.

Per session-4 substrate change (HANDOFF entry 2026-05-13): the 30-frame
static dwell was producing identity-prediction targets, not trajectories.
The new substrate has the agent moving continuously through each item
(item enters view -> fills view -> slides out of view); see
src/env/continuous_motion_explorer.py.

This script runs 5 loops of the new trajectory, saves raw 256x256 PNGs
and per-frame JSONL annotations, and writes a summary JSON. The downstream
DINOv2 encoding + motion-continuity check happens in a separate analysis
step.

No perturbation is applied (no RandomizeMaterials, no asset replacement).
This is the bare baseline trajectory; Phase 2's LivingRoom retexture and
Phase 3's TV replacement wrap this trajectory in later sessions.

Usage:
  nohup python3.12 -u scripts/run_phase2_calibration_collect.py \\
      > logs/phase2_calibration_$(date +%Y%m%d_%H%M%S).log 2>&1 &
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List

_ROOT = Path("/mnt/c/Users/Jason/Desktop/Eridos/Weft 2")
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Prior offline shim — same as the previous repo's collection scripts.
import prior  # noqa: E402

_original_prior_load = prior.load_dataset


def _offline_load_dataset(*args: Any, **kwargs: Any) -> Any:
    kwargs.setdefault("offline", True)
    return _original_prior_load(*args, **kwargs)


prior.load_dataset = _offline_load_dataset  # type: ignore[assignment]

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from shared.substrate.continuous_motion_env import ContinuousMotionEnv  # noqa: E402


_DEFAULT_ROUTE_JSON = Path(
    "/mnt/c/Users/Jason/Desktop/Eridos/Weft/results/stage_0b_furniture/route.json"
)
_DEFAULT_OUT_DIR = _ROOT / "data" / "phase2_calibration"
_DEFAULT_RESULTS_DIR = _ROOT / "results" / "phase2_calibration"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--route_json", type=Path, default=_DEFAULT_ROUTE_JSON)
    parser.add_argument("--out_frames_dir", type=Path,
                        default=_DEFAULT_OUT_DIR / "frames")
    parser.add_argument("--out_annotations", type=Path,
                        default=_DEFAULT_OUT_DIR / "annotations.jsonl")
    parser.add_argument("--out_summary", type=Path,
                        default=_DEFAULT_RESULTS_DIR / "calibration_summary.json")
    parser.add_argument("--num_loops", type=int, default=5)
    parser.add_argument("--max_frames", type=int, default=5000)
    parser.add_argument("--width", type=int, default=300)
    parser.add_argument("--height", type=int, default=300)
    parser.add_argument("--frame_size", type=int, default=256)
    parser.add_argument("--close_up_length_m", type=float, default=2.0)
    parser.add_argument("--close_up_step_m", type=float, default=0.20)
    parser.add_argument("--densify_step_m", type=float, default=0.20)
    args = parser.parse_args()

    os.environ.setdefault("DISPLAY", ":0")
    if not args.route_json.is_file():
        print(f"[calib] FAIL: route file not found: {args.route_json}",
              file=sys.stderr)
        return 1
    route = json.loads(args.route_json.read_text())

    args.out_frames_dir.mkdir(parents=True, exist_ok=True)
    args.out_annotations.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)

    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"[calib] ts={ts} house_seed={route['seed']} "
          f"close_up_length_m={args.close_up_length_m} "
          f"close_up_step_m={args.close_up_step_m} "
          f"densify_step_m={args.densify_step_m} "
          f"num_loops={args.num_loops}", flush=True)

    env = ContinuousMotionEnv(
        house_seed=int(route["seed"]),
        route_items=route["items"],
        width=args.width,
        height=args.height,
        frame_size=args.frame_size,
        close_up_length_m=float(args.close_up_length_m),
        close_up_step_m=float(args.close_up_step_m),
        densify_step_m=float(args.densify_step_m),
    )

    annotations_fh = args.out_annotations.open("w", buffering=1)
    per_item_close_up_counts: Dict[int, int] = {}
    transit_count = 0
    frame_idx = 0
    failed = False
    msg = ""
    t0 = time.time()
    try:
        while frame_idx < int(args.max_frames):
            frame = env.next_frame()
            obs = env.last_observation
            if int(obs.get("loop_index", -1)) >= int(args.num_loops):
                break
            Image.fromarray(frame).save(
                args.out_frames_dir / f"frame_{frame_idx:08d}.png"
            )
            rec = {
                "frame_idx": int(frame_idx),
                "current_room": str(env.current_room_name()),
                "viewing_position_id": int(obs.get("viewing_position_id", -1)),
                "furniture_object_id": obs.get("furniture_object_id"),
                "furniture_object_type": obs.get("furniture_object_type"),
                "phase": str(obs.get("phase", "?")),
                "loop_index": int(obs.get("loop_index", -1)),
                "close_up_apex_flag": bool(obs.get("close_up_apex_flag", False)),
                "loop_boundary_flag": bool(env.episode_boundary_flag),
                "position": obs.get("position"),
                "rotation_y": float(obs.get("rotation_y", 0.0)),
                "action_success": bool(obs.get("action_success", True)),
            }
            annotations_fh.write(json.dumps(rec) + "\n")
            if rec["phase"] == "close_up":
                vp = int(rec["viewing_position_id"])
                per_item_close_up_counts[vp] = per_item_close_up_counts.get(vp, 0) + 1
            else:
                transit_count += 1
            if frame_idx % 50 == 0 and frame_idx > 0:
                elapsed = time.time() - t0
                print(f"[calib] frame={frame_idx} loop={rec['loop_index']} "
                      f"phase={rec['phase']} vp={rec['viewing_position_id']} "
                      f"apex={rec['close_up_apex_flag']} "
                      f"elapsed={elapsed:.1f}s", flush=True)
            frame_idx += 1
    except BaseException as e:
        failed = True
        msg = f"{type(e).__name__}: {e}"
        traceback.print_exc()
    finally:
        annotations_fh.close()
        try:
            env.close()
        except Exception:
            pass

    wall_s = time.time() - t0
    summary = {
        "stage_tag": "phase2_calibration",
        "launch_timestamp_utc": ts,
        "wall_clock_seconds": wall_s,
        "frames_written": frame_idx,
        "num_loops_requested": int(args.num_loops),
        "close_up_length_m": float(args.close_up_length_m),
        "close_up_step_m": float(args.close_up_step_m),
        "densify_step_m": float(args.densify_step_m),
        "close_up_frames_per_item": per_item_close_up_counts,
        "transit_frames": transit_count,
        "explorer_stats": dict(env.explorer_stats),
        "house_seed": int(route["seed"]),
        "items": [it["object_type"] for it in route["items"]],
        "failed": failed,
        "fail_msg": msg,
    }
    args.out_summary.write_text(json.dumps(summary, indent=2))
    print(f"[calib] wrote summary to {args.out_summary}", flush=True)
    print(f"[calib] {frame_idx} frames in {wall_s:.1f}s "
          f"({frame_idx / wall_s:.1f} fps)", flush=True)
    return 0 if not failed else 2


if __name__ == "__main__":
    raise SystemExit(main())

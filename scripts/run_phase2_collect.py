"""Phase 2 frame collection per EXPERIMENT_INSTRUCTIONS §8.3.

Stage A (loops 1..perturbation_start_loop-1, default 1..30): pure
continuous-motion route, no `RandomizeMaterials` call. Frames carry
`perturbation_active = False`.

Stage B (loops perturbation_start_loop..end, default 31..205):
`RandomizeMaterials(inRoomTypes=["LivingRoom"], useTrainMaterials=True)`
is fired once at the start of every loop. Frames carry
`perturbation_active = True`. Per-loop applied materials are recorded
to `data/phase2_collection_metadata.json`.

Output:
  - PNG frames at `data/phase2_frames/frame_{idx:08d}.png` (gitignored)
  - JSONL annotations at `data/phase2_annotations.jsonl`
  - Summary at `results/inner_pam_v0/phase2_main/collection_summary.json`
  - Per-loop materials at `data/phase2_collection_metadata.json`

Usage:
  nohup python3.12 -u scripts/run_phase2_collect.py \\
      > logs/phase2_collect_$(date +%Y%m%d_%H%M%S).log 2>&1 &
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

import prior  # noqa: E402

_original_prior_load = prior.load_dataset


def _offline_load_dataset(*args: Any, **kwargs: Any) -> Any:
    kwargs.setdefault("offline", True)
    return _original_prior_load(*args, **kwargs)


prior.load_dataset = _offline_load_dataset  # type: ignore[assignment]

from PIL import Image  # noqa: E402

from src.config import (  # noqa: E402
    PHASE_2_FRAME_BUDGET,
    PHASE_2_PERTURBATION_START_LOOP,
)
from src.env.continuous_motion_env import ContinuousMotionEnv  # noqa: E402
from src.env.explorer_phase2 import Phase2RetextureEnv  # noqa: E402


_DEFAULT_ROUTE_JSON = Path(
    "/mnt/c/Users/Jason/Desktop/Eridos/Weft/results/stage_0b_furniture/route.json"
)
_DEFAULT_FRAMES_DIR = _ROOT / "data" / "phase2_frames"
_DEFAULT_ANNOTATIONS = _ROOT / "data" / "phase2_annotations.jsonl"
_DEFAULT_METADATA = _ROOT / "data" / "phase2_collection_metadata.json"
_DEFAULT_SUMMARY = _ROOT / "results" / "inner_pam_v0" / "phase2_main" / "collection_summary.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--route_json", type=Path, default=_DEFAULT_ROUTE_JSON)
    parser.add_argument("--out_frames_dir", type=Path, default=_DEFAULT_FRAMES_DIR)
    parser.add_argument("--out_annotations", type=Path, default=_DEFAULT_ANNOTATIONS)
    parser.add_argument("--out_metadata", type=Path, default=_DEFAULT_METADATA)
    parser.add_argument("--out_summary", type=Path, default=_DEFAULT_SUMMARY)
    parser.add_argument("--max_frames", type=int, default=PHASE_2_FRAME_BUDGET)
    parser.add_argument(
        "--perturbation_start_loop", type=int,
        default=PHASE_2_PERTURBATION_START_LOOP,
        help="loops with loop_index >= this value fire RandomizeMaterials per loop",
    )
    parser.add_argument("--width", type=int, default=300)
    parser.add_argument("--height", type=int, default=300)
    parser.add_argument("--frame_size", type=int, default=256)
    parser.add_argument("--close_up_length_m", type=float, default=2.0)
    parser.add_argument("--close_up_step_m", type=float, default=0.20)
    parser.add_argument("--densify_step_m", type=float, default=0.20)
    args = parser.parse_args()

    os.environ.setdefault("DISPLAY", ":0")
    if not args.route_json.is_file():
        print(f"[collect] FAIL: route not found: {args.route_json}", file=sys.stderr)
        return 1
    route = json.loads(args.route_json.read_text())

    args.out_frames_dir.mkdir(parents=True, exist_ok=True)
    args.out_annotations.parent.mkdir(parents=True, exist_ok=True)
    args.out_metadata.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)

    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(
        f"[collect] ts={ts} house_seed={route['seed']} "
        f"perturbation_start_loop={args.perturbation_start_loop} "
        f"max_frames={args.max_frames}",
        flush=True,
    )

    base_env = ContinuousMotionEnv(
        house_seed=int(route["seed"]),
        route_items=route["items"],
        width=args.width,
        height=args.height,
        frame_size=args.frame_size,
        close_up_length_m=float(args.close_up_length_m),
        close_up_step_m=float(args.close_up_step_m),
        densify_step_m=float(args.densify_step_m),
    )
    env = Phase2RetextureEnv(base_env, perturbation_start_loop=args.perturbation_start_loop)

    annotations_fh = args.out_annotations.open("w", buffering=1)
    stage_a_frames = 0
    stage_b_frames = 0
    per_item_close_up_counts: Dict[int, int] = {}
    transit_count = 0
    failed = False
    msg = ""
    frame_idx = 0
    t0 = time.time()
    try:
        while frame_idx < int(args.max_frames):
            frame = env.next_frame()
            obs = env.last_observation
            perturbation_active = bool(env.perturbation_active_for_current_frame())
            Image.fromarray(frame).save(
                args.out_frames_dir / f"frame_{frame_idx:08d}.png"
            )
            rec = {
                "frame_idx": int(frame_idx),
                "current_room": str(env.current_room_name()),
                "viewing_position_id": int(obs.get("viewing_position_id", -1)),
                "furniture_object_id": obs.get("furniture_object_id"),
                "furniture_object_type": obs.get("furniture_object_type"),
                "phase_segment": str(obs.get("phase", "?")),  # close_up | transit
                "loop_index": int(obs.get("loop_index", -1)),
                "close_up_apex_flag": bool(obs.get("close_up_apex_flag", False)),
                "loop_boundary_flag": bool(env.episode_boundary_flag),
                "position": obs.get("position"),
                "rotation_y": float(obs.get("rotation_y", 0.0)),
                "action_success": bool(obs.get("action_success", True)),
                # Phase 2 specific fields (instr §8.3):
                "phase": "phase2",
                "perturbation": "livingroom_retexture",
                "perturbation_active": perturbation_active,
            }
            annotations_fh.write(json.dumps(rec) + "\n")
            if rec["phase_segment"] == "close_up":
                vp = int(rec["viewing_position_id"])
                per_item_close_up_counts[vp] = per_item_close_up_counts.get(vp, 0) + 1
            else:
                transit_count += 1
            if perturbation_active:
                stage_b_frames += 1
            else:
                stage_a_frames += 1
            if frame_idx % 100 == 0 and frame_idx > 0:
                elapsed = time.time() - t0
                print(
                    f"[collect] frame={frame_idx} loop={rec['loop_index']} "
                    f"phase_seg={rec['phase_segment']} vp={rec['viewing_position_id']} "
                    f"pert={int(perturbation_active)} elapsed={elapsed:.1f}s "
                    f"fps={frame_idx / max(elapsed, 1e-3):.1f}",
                    flush=True,
                )
            frame_idx += 1
    except BaseException as e:
        failed = True
        msg = f"{type(e).__name__}: {e}"
        traceback.print_exc()
    finally:
        annotations_fh.close()
        try:
            args.out_metadata.write_text(
                json.dumps(
                    {
                        "house_seed": int(route["seed"]),
                        "perturbation_start_loop": int(args.perturbation_start_loop),
                        "materials_by_loop": {
                            str(k): v for k, v in env.materials_by_loop.items()
                        },
                    },
                    indent=2,
                )
            )
        except Exception as ee:
            print(f"[collect] WARN: failed to write metadata: {ee}", file=sys.stderr)
        try:
            env.close()
        except Exception:
            pass

    wall_s = time.time() - t0
    summary = {
        "stage_tag": "phase2_main_collection",
        "launch_timestamp_utc": ts,
        "wall_clock_seconds": float(wall_s),
        "frames_written": int(frame_idx),
        "max_frames_requested": int(args.max_frames),
        "perturbation_start_loop": int(args.perturbation_start_loop),
        "stage_a_frames": int(stage_a_frames),
        "stage_b_frames": int(stage_b_frames),
        "close_up_frames_per_item": {int(k): int(v) for k, v in per_item_close_up_counts.items()},
        "transit_frames": int(transit_count),
        "perturbed_loops_count": int(len(env.materials_by_loop)),
        "explorer_stats": dict(env.explorer_stats),
        "house_seed": int(route["seed"]),
        "items": [it["object_type"] for it in route["items"]],
        "failed": bool(failed),
        "fail_msg": str(msg),
    }
    args.out_summary.write_text(json.dumps(summary, indent=2))
    print(
        f"[collect] {frame_idx} frames in {wall_s:.1f}s "
        f"({frame_idx / max(wall_s, 1e-3):.1f} fps); "
        f"stageA={stage_a_frames} stageB={stage_b_frames}",
        flush=True,
    )
    return 0 if not failed else 2


if __name__ == "__main__":
    raise SystemExit(main())

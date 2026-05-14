"""Trajectory diagnostic for the v2 calibration substrate.

Per the 2026-05-14 sixth-STOP reviewer directive: the existing motion-
continuity check (run_phase2_calibration_analyse.py) gates on DINOv2
bit-identicity of consecutive frames, not on continuity of the 3D
agent trajectory. Reviewer reports visual observation of (1) camera-
height bobbing and (2) the agent appearing to jump through walls at
some transitions. Both would be invisible to the bit-identicity gate.

This script runs three diagnostics on the existing v2 calibration data
(no new collection needed):

  1. Extract per-frame agent pose (x, y, z, rotation_y) from the
     annotations file. Plot y across one full loop. If y varies by
     more than ~5 cm, the agent's height is changing during the
     trajectory — flag.

  2. Compute 3D Euclidean displacement between consecutive poses.
     Expected: near 0.20 m (the densified-Teleport step). Flag any
     displacement > 0.25 m. Large outliers indicate teleports across
     non-traversable gaps (the "jumping through walls" mechanism).

  3. For frame-pairs flagged in (2), render contact sheets of the
     frames immediately before, at, and after the discontinuity,
     alongside pose information. The frames are saved as side-by-side
     PNGs at `results/phase2_calibration_v2/discontinuity_frames/`.

Writes findings to `results/phase2_calibration_v2/trajectory_diagnostic.json`.

Usage:
  python3.12 -u scripts/run_phase2_trajectory_diagnostic.py \\
      > logs/phase2_trajectory_diagnostic_$(date +%Y%m%d_%H%M%S).log 2>&1
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path("/mnt/c/Users/Jason/Desktop/Eridos/Weft 2")

_DEFAULT_ANNOT = REPO_ROOT / "data" / "phase2_calibration_v2" / "annotations.jsonl"
_DEFAULT_FRAMES = REPO_ROOT / "data" / "phase2_calibration_v2" / "frames"
_DEFAULT_OUT_JSON = REPO_ROOT / "results" / "phase2_calibration_v2" / "trajectory_diagnostic.json"
_DEFAULT_OUT_FRAMES = REPO_ROOT / "results" / "phase2_calibration_v2" / "discontinuity_frames"

_FLAG_DISPLACEMENT_M = 0.25
_FLAG_Y_VARIATION_M = 0.05   # 5 cm; expected y is constant given the explorer's design
_MAX_CONTACT_SHEETS = 30


def _load_annotations(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with path.open("r") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _pose_array(annotations: List[Dict[str, Any]]) -> np.ndarray:
    """(N, 4) array of (x, y, z, rotation_y_deg) per frame."""
    rows = []
    for a in annotations:
        pos = a.get("position") or {}
        rows.append([
            float(pos.get("x", float("nan"))),
            float(pos.get("y", float("nan"))),
            float(pos.get("z", float("nan"))),
            float(a.get("rotation_y", float("nan"))),
        ])
    return np.asarray(rows, dtype=np.float64)


def _contact_sheet(
    frame_paths: List[Path],
    labels: List[str],
    out_path: Path,
    pad_px: int = 10,
    label_height_px: int = 30,
) -> None:
    """Concatenate frames horizontally with text labels under each."""
    imgs = [Image.open(p).convert("RGB") for p in frame_paths]
    w, h = imgs[0].size
    total_w = w * len(imgs) + pad_px * (len(imgs) - 1)
    total_h = h + label_height_px
    canvas = Image.new("RGB", (total_w, total_h), (32, 32, 32))
    x = 0
    for im, label in zip(imgs, labels):
        canvas.paste(im, (x, 0))
        try:
            draw = ImageDraw.Draw(canvas)
            try:
                font = ImageFont.truetype(
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14
                )
            except OSError:
                font = ImageFont.load_default()
            draw.text((x + 5, h + 5), label, fill=(255, 255, 255), font=font)
        except Exception:
            pass
        x += w + pad_px
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--annotations", type=Path, default=_DEFAULT_ANNOT)
    parser.add_argument("--frames_dir", type=Path, default=_DEFAULT_FRAMES)
    parser.add_argument("--out_json", type=Path, default=_DEFAULT_OUT_JSON)
    parser.add_argument("--out_frames", type=Path, default=_DEFAULT_OUT_FRAMES)
    parser.add_argument("--max_contact_sheets", type=int, default=_MAX_CONTACT_SHEETS)
    args = parser.parse_args()

    if not args.annotations.is_file():
        print(f"[traj] FAIL: annotations not found: {args.annotations}", file=sys.stderr)
        return 1
    if not args.frames_dir.is_dir():
        print(f"[traj] FAIL: frames_dir not found: {args.frames_dir}", file=sys.stderr)
        return 1

    annotations = _load_annotations(args.annotations)
    n = len(annotations)
    print(f"[traj] loaded {n} annotations from {args.annotations}", flush=True)

    poses = _pose_array(annotations)        # (N, 4)
    xs, ys, zs, rys = poses[:, 0], poses[:, 1], poses[:, 2], poses[:, 3]

    # ---- Diagnostic 1: y-coordinate variation -----------------------------
    y_overall = {
        "min": float(np.nanmin(ys)),
        "max": float(np.nanmax(ys)),
        "mean": float(np.nanmean(ys)),
        "std": float(np.nanstd(ys, ddof=0)),
        "range": float(np.nanmax(ys) - np.nanmin(ys)),
    }
    # Unique y values rounded to 4 decimal places (catches discrete jumps).
    unique_ys = np.unique(np.round(ys, 4))
    print(f"[traj] y stats: min={y_overall['min']:.4f} max={y_overall['max']:.4f} "
          f"mean={y_overall['mean']:.4f} std={y_overall['std']:.4f} "
          f"range={y_overall['range']:.4f} unique_values={len(unique_ys)}",
          flush=True)

    # Per-loop y series for the first complete loop (for plot-equivalent inspection).
    loop_indices = [int(a.get("loop_index", -1)) for a in annotations]
    first_loop = max(0, min((li for li in loop_indices if li >= 0), default=0))
    first_loop_y = ys[np.array(loop_indices) == first_loop].tolist()
    y_first_loop_stats = {
        "loop_index": int(first_loop),
        "n_frames": int(len(first_loop_y)),
        "min": float(min(first_loop_y)) if first_loop_y else None,
        "max": float(max(first_loop_y)) if first_loop_y else None,
        "range": float(max(first_loop_y) - min(first_loop_y)) if first_loop_y else None,
    }
    y_diagnostic_flag = bool(y_overall["range"] > _FLAG_Y_VARIATION_M)
    print(f"[traj] y range {y_overall['range']:.4f}m vs flag threshold "
          f"{_FLAG_Y_VARIATION_M}m: {'FLAG' if y_diagnostic_flag else 'ok'}",
          flush=True)

    # ---- Diagnostic 2: 3D displacement between consecutive frames ---------
    dx = np.diff(xs)
    dy = np.diff(ys)
    dz = np.diff(zs)
    displacement = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    n_pairs = int(len(displacement))

    bin_edges = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.50, 1.0, 2.0, float("inf")]
    bin_labels = [f"[{bin_edges[i]:.2f}, {bin_edges[i+1]:.2f})" for i in range(len(bin_edges) - 1)]
    bin_counts = np.histogram(displacement, bins=bin_edges)[0].tolist()
    displacement_distribution = {
        "n_pairs": n_pairs,
        "mean": float(np.mean(displacement)),
        "median": float(np.median(displacement)),
        "std": float(np.std(displacement, ddof=0)),
        "min": float(displacement.min()),
        "max": float(displacement.max()),
        "p99": float(np.percentile(displacement, 99)),
        "histogram_bin_labels": bin_labels,
        "histogram_counts": bin_counts,
    }
    flagged_idx = np.where(displacement > _FLAG_DISPLACEMENT_M)[0]
    print(f"[traj] displacement: mean={displacement_distribution['mean']:.4f}m "
          f"median={displacement_distribution['median']:.4f}m "
          f"max={displacement_distribution['max']:.4f}m "
          f"n_flagged_gt_{_FLAG_DISPLACEMENT_M}m={len(flagged_idx)}",
          flush=True)

    flagged_pairs: List[Dict[str, Any]] = []
    for k_idx in flagged_idx:
        i = int(k_idx)
        a_prev = annotations[i]
        a_next = annotations[i + 1]
        flagged_pairs.append({
            "consec_pair_idx": i,                 # pair is (frame i, frame i+1)
            "frame_a_idx": int(a_prev.get("frame_idx", i)),
            "frame_b_idx": int(a_next.get("frame_idx", i + 1)),
            "displacement_m": float(displacement[i]),
            "dx_m": float(dx[i]),
            "dy_m": float(dy[i]),
            "dz_m": float(dz[i]),
            "position_before": a_prev.get("position"),
            "position_after": a_next.get("position"),
            "rotation_y_before": float(a_prev.get("rotation_y", 0.0)),
            "rotation_y_after": float(a_next.get("rotation_y", 0.0)),
            "loop_index_before": int(a_prev.get("loop_index", -1)),
            "loop_index_after": int(a_next.get("loop_index", -1)),
            "phase_before": str(a_prev.get("phase", a_prev.get("phase_segment", "?"))),
            "phase_after": str(a_next.get("phase", a_next.get("phase_segment", "?"))),
            "loop_boundary_flag": bool(a_next.get("loop_boundary_flag", False)),
        })

    # ---- Diagnostic 3: render contact sheets for flagged frames -----------
    args.out_frames.mkdir(parents=True, exist_ok=True)
    contact_sheet_paths: List[Dict[str, Any]] = []
    # Sort flagged pairs by descending displacement; render the top-K.
    flagged_pairs_sorted = sorted(
        flagged_pairs, key=lambda p: -p["displacement_m"]
    )
    rendered = 0
    for fp in flagged_pairs_sorted:
        if rendered >= args.max_contact_sheets:
            break
        i = fp["consec_pair_idx"]
        prev_idx = max(0, i - 1)
        cur_idx = i
        nxt_idx = i + 1
        nxt2_idx = min(n - 1, i + 2)
        # Build a 4-frame sheet: i-1, i, i+1, i+2 (the discontinuity is i→i+1).
        idxs = [prev_idx, cur_idx, nxt_idx, nxt2_idx]
        paths: List[Path] = []
        labels: List[str] = []
        for k_idx in idxs:
            ann = annotations[k_idx]
            fi = int(ann.get("frame_idx", k_idx))
            p = args.frames_dir / f"frame_{fi:08d}.png"
            if not p.is_file():
                paths = []
                break
            paths.append(p)
            pos = ann.get("position") or {}
            labels.append(
                f"f={fi}  x={pos.get('x', 0):.2f}  "
                f"y={pos.get('y', 0):.3f}  z={pos.get('z', 0):.2f}  "
                f"h={ann.get('rotation_y', 0):.1f}  loop={ann.get('loop_index', -1)}"
            )
        if not paths:
            continue
        out_name = (
            f"disc_{rendered:03d}_pair{i:06d}_disp{fp['displacement_m']:.3f}m.png"
        )
        out_path = args.out_frames / out_name
        try:
            _contact_sheet(paths, labels, out_path)
        except Exception as e:
            print(f"[traj] WARN: contact sheet render failed for pair {i}: {e}",
                  file=sys.stderr)
            continue
        contact_sheet_paths.append({
            "consec_pair_idx": i,
            "displacement_m": fp["displacement_m"],
            "contact_sheet": str(out_path.relative_to(REPO_ROOT)),
            "frames_in_sheet": idxs,
        })
        rendered += 1
        print(f"[traj] wrote {out_path.relative_to(REPO_ROOT)}", flush=True)

    # ---- Compose report ---------------------------------------------------
    report = {
        "annotations_path": str(args.annotations),
        "frames_dir": str(args.frames_dir),
        "n_frames": int(n),
        "diagnostic_1_y_coordinate": {
            "y_overall": y_overall,
            "n_unique_y_values_rounded_4dp": int(len(unique_ys)),
            "unique_y_values_rounded_4dp": [
                float(v) for v in unique_ys[:20]
            ],
            "y_first_loop_stats": y_first_loop_stats,
            "flag_threshold_m": float(_FLAG_Y_VARIATION_M),
            "flagged": y_diagnostic_flag,
        },
        "diagnostic_2_displacement": {
            "flag_threshold_m": float(_FLAG_DISPLACEMENT_M),
            "distribution": displacement_distribution,
            "n_flagged": int(len(flagged_idx)),
            "flagged_pairs_count_by_loop": _count_by_loop(flagged_pairs),
            "flagged_pairs": flagged_pairs[:200],   # cap the JSON size
        },
        "diagnostic_3_contact_sheets": {
            "max_rendered": int(args.max_contact_sheets),
            "n_rendered": int(rendered),
            "out_dir": str(args.out_frames.relative_to(REPO_ROOT)),
            "sheets": contact_sheet_paths,
        },
        "forceAction_usage_note": (
            "ContinuousMotionExplorer._teleport calls "
            "controller.step(action='Teleport', ..., forceAction=True) at "
            "src/env/continuous_motion_explorer.py:426. forceAction bypasses "
            "AI2-THOR's navigability + collider checks: the agent can be "
            "placed at any position regardless of whether a NavMesh path "
            "exists between successive positions. This is the most likely "
            "cause of the visual observations the reviewer reported "
            "(camera-height bobbing if forceAction also relaxes floor "
            "snap; agent-through-walls if NavMesh checks are skipped at "
            "transit corner rotations or close-up sweep points)."
        ),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2))
    print(f"[traj] wrote {args.out_json.relative_to(REPO_ROOT)}", flush=True)

    return 0


def _count_by_loop(flagged: List[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[int, int] = {}
    for f in flagged:
        loop = int(f["loop_index_before"])
        counts[loop] = counts.get(loop, 0) + 1
    return {str(k): int(v) for k, v in sorted(counts.items())}


if __name__ == "__main__":
    raise SystemExit(main())

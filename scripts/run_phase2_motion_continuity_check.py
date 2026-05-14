"""Within-loop motion-continuity check for candidate viewing poses (Phase 2 substrate fix).

For each candidate pose surviving the pose search, sweeps the close-up
segment (2 m perpendicular to viewing_heading, densified at 0.20 m)
and verifies that consecutive-frame DINOv2 cosines are not bit-identical
(> 0.9999). This is the same check session 4's full calibration did,
but localised to a single item's close-up path so it can screen candidate
poses cheaply before re-running the full 5-loop calibration.

Per the 2026-05-14 reviewer directive: NavMesh navigability and
within-loop consecutive-frame DINOv2 cosine cleanliness are verified
per-candidate before investing in visual inspection or rationale.

Reads: results/inner_pam_v0/phase2_pose_search/pose_search_report.json
Writes: results/inner_pam_v0/phase2_motion_continuity/report.{md,json}

Usage:
  nohup python3.12 -u scripts/run_phase2_motion_continuity_check.py \\
      > logs/phase2_motion_continuity_$(date +%Y%m%d_%H%M%S).log 2>&1 &
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
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
import torch  # noqa: E402

from src.encoder.dinov2_encoder import load_frozen_dinov2  # noqa: E402
from src.env.material_perturbation_probe import (  # noqa: E402
    dinov2_encode_batch,
    items_by_id,
    teleport_and_capture,
)
from src.env.procthor_house import load_house, make_controller  # noqa: E402


_DEFAULT_ROUTE_JSON = Path(
    "/mnt/c/Users/Jason/Desktop/Eridos/Weft/results/stage_0b_furniture/route.json"
)
_DEFAULT_POSE_REPORT = (
    _ROOT / "results" / "inner_pam_v0" / "phase2_pose_search"
    / "pose_search_report.json"
)
_DEFAULT_RESULTS_DIR = _ROOT / "results" / "inner_pam_v0" / "phase2_motion_continuity"

_CLOSE_UP_LENGTH_M = 2.0
_CLOSE_UP_STEP_M = 0.20
_BIT_IDENTICAL_COSINE_THRESHOLD = 0.9999


def _perpendicular_unit_ccw(heading_deg: float) -> Tuple[float, float]:
    """Unit vector 90° CCW from the heading (top-down screen sense)."""
    rad = math.radians(heading_deg)
    fx, fz = math.sin(rad), math.cos(rad)
    return -fz, fx


def _close_up_path(
    viewing_position: Dict[str, float],
    heading_deg: float,
) -> List[Dict[str, float]]:
    """Compute the close-up path: 2 m perpendicular to heading, 0.20 m step."""
    px, pz = float(viewing_position["x"]), float(viewing_position["z"])
    py = float(viewing_position.get("y", 0.901))
    perp_x, perp_z = _perpendicular_unit_ccw(heading_deg)
    half = _CLOSE_UP_LENGTH_M / 2.0
    n_each_side = int(round(half / _CLOSE_UP_STEP_M))
    points: List[Dict[str, float]] = []
    for i in range(-n_each_side, n_each_side + 1):
        d = i * _CLOSE_UP_STEP_M
        points.append({
            "x": px + d * perp_x,
            "y": py,
            "z": pz + d * perp_z,
        })
    return points


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--route_json", type=Path, default=_DEFAULT_ROUTE_JSON)
    parser.add_argument("--pose_report", type=Path, default=_DEFAULT_POSE_REPORT)
    parser.add_argument("--results_dir", type=Path, default=_DEFAULT_RESULTS_DIR)
    parser.add_argument("--top_k", type=int, default=5,
                        help="check motion continuity for the top K candidates "
                             "per item (ranked by DINOv2 stability mean)")
    parser.add_argument("--width", type=int, default=300)
    parser.add_argument("--height", type=int, default=300)
    args = parser.parse_args()

    os.environ.setdefault("DISPLAY", ":0")
    if not args.route_json.is_file():
        print(f"[mc] FAIL: route file not found: {args.route_json}", file=sys.stderr)
        return 1
    if not args.pose_report.is_file():
        print(f"[mc] FAIL: pose report not found: {args.pose_report}", file=sys.stderr)
        return 1

    route = json.loads(args.route_json.read_text())
    items = items_by_id(route)
    pose_report = json.loads(args.pose_report.read_text())
    args.results_dir.mkdir(parents=True, exist_ok=True)

    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"[mc] ts={ts} house_seed={route['seed']} top_k={args.top_k}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("[mc] FAIL: CUDA not available", file=sys.stderr)
        return 1

    # Collect (label, candidate) pairs to check, restricted to the top-K per item.
    to_check: List[Tuple[str, Dict[str, Any]]] = []
    for label, item_block in pose_report.get("items", {}).items():
        ranked = item_block.get("ranked_candidates", [])
        for cand in ranked[: args.top_k]:
            if not cand.get("screened_in", False):
                continue
            to_check.append((label, cand))
    if not to_check:
        print("[mc] no candidates to check; nothing to do", flush=True)
        (args.results_dir / "report.json").write_text(
            json.dumps({"checked": [], "ts": ts}, indent=2)
        )
        return 0
    print(f"[mc] checking motion continuity for {len(to_check)} candidate(s)",
          flush=True)

    house = load_house(seed=int(route["seed"]), min_rooms=4)
    controller = make_controller(house, width=args.width, height=args.height)

    # Capture all close-up path frames for each candidate.
    all_frames: List[np.ndarray] = []
    capture_keys: List[Tuple[str, int]] = []   # (candidate_id, frame_idx_within_path)
    candidate_path_lengths: Dict[str, int] = {}
    try:
        for label, cand in to_check:
            cid = cand["candidate_id"]
            heading = float(cand["heading_deg"])
            pos = cand["viewing_position"]
            path = _close_up_path(pos, heading)
            print(f"[mc] {cid}: sweeping {len(path)} points along close-up path",
                  flush=True)
            captured = 0
            for k, p in enumerate(path):
                try:
                    # force_action=True: close-up path steps at 0.20 m don't
                    # align to AI2-THOR's 0.25 m grid, so Teleport's grid check
                    # would reject them. The off-grid sweep is the diagnostic.
                    frame = teleport_and_capture(controller, p, heading, force_action=True)
                except Exception as e:
                    print(f"[mc] {cid}: teleport failed at step {k} (pos={p}): {e}",
                          flush=True)
                    continue
                all_frames.append(frame)
                capture_keys.append((cid, k))
                captured += 1
            candidate_path_lengths[cid] = captured
            print(f"[mc] {cid}: captured {captured}/{len(path)} sweep frames",
                  flush=True)
    finally:
        try:
            controller.stop()
        except Exception:
            pass

    if not all_frames:
        print("[mc] FAIL: no frames captured", file=sys.stderr)
        return 2

    print(f"[mc] loading DINOv2-large + encoding {len(all_frames)} frames",
          flush=True)
    model = load_frozen_dinov2(device)
    embeddings = dinov2_encode_batch(model, all_frames, device)

    # Group embeddings by candidate.
    by_cid: Dict[str, List[Tuple[int, np.ndarray]]] = {}
    for i, (cid, k) in enumerate(capture_keys):
        by_cid.setdefault(cid, []).append((k, embeddings[i]))

    per_candidate: List[Dict[str, Any]] = []
    for label, cand in to_check:
        cid = cand["candidate_id"]
        if cid not in by_cid:
            continue
        seq = sorted(by_cid[cid], key=lambda t: t[0])
        embs = np.stack([emb for _, emb in seq])
        # Consecutive cosines (already L2-normed -> dot product).
        if embs.shape[0] < 2:
            consec_cosines: List[float] = []
        else:
            consec = np.einsum("ij,ij->i", embs[:-1], embs[1:])
            consec_cosines = [float(x) for x in consec]
        n_bit_identical = sum(
            1 for c in consec_cosines if c > _BIT_IDENTICAL_COSINE_THRESHOLD
        )
        passes = bool(n_bit_identical == 0 and len(consec_cosines) > 0)
        record = {
            "item_label": label,
            "candidate_id": cid,
            "heading_deg": float(cand["heading_deg"]),
            "viewing_position": cand["viewing_position"],
            "n_sweep_frames": int(embs.shape[0]),
            "n_consec_pairs": int(max(0, embs.shape[0] - 1)),
            "consec_cosines_min": float(min(consec_cosines)) if consec_cosines else None,
            "consec_cosines_max": float(max(consec_cosines)) if consec_cosines else None,
            "consec_cosines_mean": float(sum(consec_cosines) / len(consec_cosines))
            if consec_cosines else None,
            "n_bit_identical": n_bit_identical,
            "motion_continuity_pass": passes,
            "dinov2_stability_mean": float(cand.get("dinov2_mean", float("nan"))),
            "dinov2_stability_std": float(cand.get("dinov2_std", float("nan"))),
            "passes_stability_threshold": bool(cand.get("passes_stability_threshold", False)),
        }
        per_candidate.append(record)
        cmin = record["consec_cosines_min"]
        cmax = record["consec_cosines_max"]
        if cmin is None or cmax is None:
            print(f"[mc] {cid}: insufficient sweep frames ({record['n_sweep_frames']}) "
                  f"for consec analysis; pass={passes}", flush=True)
        else:
            print(f"[mc] {cid}: consec min={cmin:.4f} max={cmax:.4f} "
                  f"bit_identical={n_bit_identical} pass={passes}", flush=True)

    # Survivors = motion-continuity-pass AND stability-pass.
    survivors = [r for r in per_candidate
                 if r["motion_continuity_pass"] and r["passes_stability_threshold"]]
    survivors.sort(key=lambda r: r["dinov2_stability_mean"], reverse=True)

    report = {
        "timestamp_utc": ts,
        "house_seed": int(route["seed"]),
        "top_k_checked": int(args.top_k),
        "candidates_checked": per_candidate,
        "survivors_ranked": survivors,
        "best_candidate": survivors[0] if survivors else None,
    }
    (args.results_dir / "report.json").write_text(json.dumps(report, indent=2))

    md = ["# Phase 2 Motion-Continuity Check",
          "",
          f"Timestamp: {ts}",
          f"House seed: {route['seed']}",
          f"Top-K candidates checked per item: {args.top_k}",
          "",
          "## Per-candidate verdict",
          "",
          "| candidate | heading | consec_min | consec_max | bit_identical | "
          "stability_mean | motion_pass | stability_pass |",
          "|---|---:|---:|---:|---:|---:|---|---|"]
    for r in per_candidate:
        md.append(
            f"| {r['candidate_id']} | {r['heading_deg']:.1f}° | "
            f"{r['consec_cosines_min']:.4f} | {r['consec_cosines_max']:.4f} | "
            f"{r['n_bit_identical']} | {r['dinov2_stability_mean']:.4f} | "
            f"{'✓' if r['motion_continuity_pass'] else '✗'} | "
            f"{'✓' if r['passes_stability_threshold'] else '✗'} |"
        )
    md.append("")
    if report["best_candidate"]:
        b = report["best_candidate"]
        md.append("## Best surviving candidate")
        md.append("")
        md.append(f"- **{b['candidate_id']}**: heading {b['heading_deg']:.2f}°, "
                  f"position (x={b['viewing_position']['x']:.2f}, "
                  f"z={b['viewing_position']['z']:.2f})")
        md.append(f"- DINOv2 stability mean: {b['dinov2_stability_mean']:.4f}")
        md.append(f"- Motion-continuity consec cosine range: "
                  f"[{b['consec_cosines_min']:.4f}, {b['consec_cosines_max']:.4f}]")
    else:
        md.append("## No surviving candidate")
        md.append("")
        md.append("STOP per the reviewer directive: no feasible pose found.")
    (args.results_dir / "report.md").write_text("\n".join(md))
    print(f"[mc] wrote {args.results_dir / 'report.json'}", flush=True)
    if report["best_candidate"]:
        b = report["best_candidate"]
        print(f"[mc] best: {b['candidate_id']} "
              f"stability={b['dinov2_stability_mean']:.4f}", flush=True)
    return 0 if survivors else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException as e:
        traceback.print_exc()
        print(f"[mc] FATAL: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(3)

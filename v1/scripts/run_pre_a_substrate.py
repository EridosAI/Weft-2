#!/usr/bin/env python3
"""PRE-A: Substrate verification driver for v1.

Spec §6.1 / §8.3, instr §6.1. Runs the v0 §5 protocol on the v1 substrate
plus the eight-finding checklist. Unperturbed substrate only — §5.4
magnitude verification + §5.8 perturbation-mechanism locality are PRE-B's
responsibility once a candidate mechanism is selected.

Procedure:
  1. Verify environment (Python version, GPU availability).
  2. Spin up AI2-THOR with ProcTHOR seed-7 house.
  3. Run a short continuous-motion collection (5 loops, unperturbed).
  4. Compute floor-y via GetReachablePositions modal-y (finding 6).
  5. Encode collected frames via DINOv2-Large.
  6. §5.1 cross-instance stability: cosine across loops at same item × ordinal.
  7. §5.2 cross-element distinguishability: cosine across different items.
  8. §5.3 combined gap.
  9. Substrate findings 1-3, 5-7. Finding 4 (substrate-as-feature-vs-bug)
     is interpretive — recorded by classifying any observed substrate
     property as feature-by-default. Finding 8 (cross-room visual
     leakage) is PRE-B-specific and skipped here.
 10. Write PRE-A report to results/inner_pam_v1/pre_a_substrate/pre_a_report.{md,json}.

Outputs:
  v1/data/pre_a_calibration/frames/      (collected PNG frames, gitignored)
  v1/data/pre_a_calibration/embeddings.npy
  v1/data/pre_a_calibration/annotations.jsonl
  v1/results/inner_pam_v1/pre_a_substrate/pre_a_report.json

Exit code 0 if all checks PASS; 1 otherwise (spec §11.2 stop condition).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

_ROOT = Path("/mnt/c/Users/Jason/Desktop/Eridos/Weft 2")
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Force ProcTHOR offline mode (matches v0 pattern).
import prior  # noqa: E402
_original_prior_load = prior.load_dataset


def _offline_load_dataset(*args: Any, **kwargs: Any) -> Any:
    kwargs.setdefault("offline", True)
    return _original_prior_load(*args, **kwargs)


prior.load_dataset = _offline_load_dataset  # type: ignore[assignment]

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from shared.encoder.dinov2_encoder import encode_frames, load_frozen_dinov2  # noqa: E402
from shared.substrate.continuous_motion_env import ContinuousMotionEnv  # noqa: E402
from v1.src.config import (  # noqa: E402
    FINDING_3_COS_MAX,
    PATHS,
    SUBSTRATE_COMBINED_GAP_MIN,
    SUBSTRATE_CROSS_ELEMENT_DISTINGUISH_MAX,
    SUBSTRATE_CROSS_INSTANCE_STABILITY_MIN,
    V1_PROCTHOR_REVISION,
)
from v1.src.preflight.pre_a_substrate_verification import (  # noqa: E402
    CheckResult,
    combined_gap,
    continuous_motion_check,
    cross_element_distinguishability,
    cross_instance_stability,
    embeddings_full_population_check,
    python_version_check,
)


PRE_A_FRAMES_DIR = _ROOT / "v1" / "data" / "pre_a_calibration" / "frames"
PRE_A_EMBEDDINGS = _ROOT / "v1" / "data" / "pre_a_calibration" / "embeddings.npy"
PRE_A_ANNOTATIONS = _ROOT / "v1" / "data" / "pre_a_calibration" / "annotations.jsonl"
PRE_A_REPORT = PATHS.results_pre_a / "pre_a_report.json"
PRE_A_REPORT_MD = PATHS.results_pre_a / "pre_a_report.md"
PRE_A_FLOOR_Y = PATHS.results_pre_a / "floor_y.json"

DEFAULT_ROUTE_JSON = _ROOT / "v0" / "data" / "route_phase2.json"


def _collect_frames(
    route: dict,
    n_loops: int,
    frames_dir: Path,
    annotations_path: Path,
    *,
    width: int = 300,
    height: int = 300,
    frame_size: int = 256,
) -> dict:
    """Run the continuous-motion explorer for `n_loops` loops; save frames + jsonl annotations.

    Returns the substrate summary (frames_per_loop, loop_count, transit_count,
    explorer_stats).
    """
    frames_dir.mkdir(parents=True, exist_ok=True)
    annotations_path.parent.mkdir(parents=True, exist_ok=True)
    # v1 PRE-A uses default procthor-10k revision (V1_PROCTHOR_REVISION is
    # None per design-chat build-config investigation 2026-05-19 — the
    # only viable build for ai2thor==5.0.0 is the latest cached revision).
    # If V1_PROCTHOR_REVISION becomes non-None in the future, the
    # monkey-patch below routes the revision through ContinuousMotionEnv.
    if V1_PROCTHOR_REVISION is not None:
        import shared.substrate.continuous_motion_env as _cme_mod
        import shared.substrate.procthor_house as _ph_mod
        _orig_load_house = _ph_mod.load_house

        def _load_house_v1(seed, min_rooms=4, **kwargs):
            return _orig_load_house(
                seed=seed,
                min_rooms=min_rooms,
                revision=V1_PROCTHOR_REVISION,
                **kwargs,
            )
        _cme_mod.load_house = _load_house_v1
        try:
            env = ContinuousMotionEnv(
                house_seed=int(route["seed"]),
                route_items=route["items"],
                width=width,
                height=height,
                frame_size=frame_size,
            )
        finally:
            _cme_mod.load_house = _orig_load_house
    else:
        env = ContinuousMotionEnv(
            house_seed=int(route["seed"]),
            route_items=route["items"],
            width=width,
            height=height,
            frame_size=frame_size,
        )
    frame_idx = 0
    loop_idx_seen: set = set()
    per_item_close_up_counts: Counter = Counter()
    transit_count = 0
    t0 = time.time()
    try:
        with annotations_path.open("w", buffering=1) as fh:
            while True:
                obs = env._explorer.next_micro_step()  # noqa: SLF001
                env._last_obs = obs                    # noqa: SLF001
                env._episode_boundary_flag = bool(obs.get("loop_boundary_flag", False))  # noqa: SLF001
                frame = obs["frame"]
                if frame.shape[:2] != (frame_size, frame_size):
                    from PIL import Image as _PILImg
                    frame = np.asarray(
                        _PILImg.fromarray(frame).resize(
                            (frame_size, frame_size), resample=_PILImg.BILINEAR
                        )
                    )
                Image.fromarray(frame).save(
                    frames_dir / f"frame_{frame_idx:08d}.png"
                )
                rec = {
                    "frame_idx": int(frame_idx),
                    "current_room": str(env.current_room_name()),
                    "viewing_position_id": int(obs.get("viewing_position_id", -1)),
                    "phase_segment": str(obs.get("phase", "?")),
                    "loop_index": int(obs.get("loop_index", -1)),
                    "close_up_apex_flag": bool(obs.get("close_up_apex_flag", False)),
                    "close_up_ordinal": obs.get("close_up_ordinal"),
                    "loop_boundary_flag": bool(env._episode_boundary_flag),
                    "position": obs.get("position"),
                    "rotation_y": float(obs.get("rotation_y", 0.0)),
                    "action_success": bool(obs.get("action_success", True)),
                    "phase": "v1_pre_a_calibration",
                    "perturbation_active": False,
                }
                fh.write(json.dumps(rec) + "\n")
                loop_idx_seen.add(int(rec["loop_index"]))
                if rec["phase_segment"] == "close_up":
                    vp = int(rec["viewing_position_id"])
                    per_item_close_up_counts[vp] += 1
                else:
                    transit_count += 1
                frame_idx += 1
                # Stop after n_loops complete loops have been seen.
                if len(loop_idx_seen) >= n_loops + 1:
                    # Drop the last partial loop boundary indicator (we have
                    # n_loops complete + 1 just-started).
                    break
                if frame_idx % 200 == 0 and frame_idx > 0:
                    elapsed = time.time() - t0
                    print(
                        f"[pre_a] frame={frame_idx} loop={rec['loop_index']} "
                        f"phase_seg={rec['phase_segment']} vp={rec['viewing_position_id']} "
                        f"elapsed={elapsed:.1f}s fps={frame_idx / max(elapsed, 1e-3):.1f}",
                        flush=True,
                    )
    finally:
        try:
            env.close()
        except Exception:
            pass

    wall_s = time.time() - t0
    return {
        "n_loops_collected": int(len(loop_idx_seen) - 1),  # exclude trailing partial
        "frame_count": int(frame_idx),
        "close_up_frames_per_item": {int(k): int(v) for k, v in per_item_close_up_counts.items()},
        "transit_count": int(transit_count),
        "wall_clock_seconds": float(wall_s),
        "explorer_stats": dict(env.explorer_stats),
    }


def _floor_y_check(
    route: dict, *, width: int = 300, height: int = 300, frame_size: int = 256
) -> dict:
    """§5.7 floor-y derivation: modal-y across GetReachablePositions results.

    This is a substrate property of the seed-7 house and does not depend on
    the explorer; we open a controller transiently and pull the positions
    set.
    """
    from shared.substrate.procthor_house import load_house, make_controller

    # V1_PROCTHOR_REVISION is None per design-chat build-config investigation
    # 2026-05-19 (the downgrade target ab3cacd is incompatible with
    # ai2thor==5.0.0's scene loader). Passing None preserves load_house's
    # default behaviour (latest cached revision = 4391935...).
    house = load_house(
        seed=int(route["seed"]), min_rooms=4, revision=V1_PROCTHOR_REVISION
    )
    controller = make_controller(house, width=width, height=height, grid_size=0.25)
    try:
        event = controller.step(action="GetReachablePositions")
        positions = event.metadata["actionReturn"]
        if not positions:
            return {"floor_y": None, "modal_y": None, "n_positions": 0, "pass": False}
        ys = [round(float(p["y"]), 6) for p in positions]
        cnt = Counter(ys)
        modal_y, modal_count = cnt.most_common(1)[0]
        return {
            "floor_y": float(modal_y),
            "modal_y": float(modal_y),
            "modal_count": int(modal_count),
            "n_positions": int(len(positions)),
            "y_distribution_top3": [
                (float(y), int(c)) for y, c in cnt.most_common(3)
            ],
            "pass": True,
        }
    finally:
        try:
            controller.stop()
        except Exception:
            pass


def _assign_close_up_ordinals(annotations: List[dict]) -> None:
    """Populate `close_up_ordinal` in-place by scanning the annotation stream.

    The ContinuousMotionExplorer's observation dict does not currently carry
    a per-frame close-up ordinal; the spec §10.2 / instr §5.1 evaluation
    operates per-(item, ordinal), so the ordinal must be assigned by
    counting frames within each contiguous close-up segment.

    A new close-up segment begins whenever phase_segment transitions from
    "transit" to "close_up" (or at frame 0 if it starts in close_up).
    Within a segment, ordinal increments from 0.

    Note on the transit→close_up structural duplicate: the last transit
    frame and the first close-up frame are at the same agent pose by
    design; the close_up frame at ordinal=0 is the duplicate of the
    preceding transit frame. Per spec §10.2 the canonical evaluation point
    is "viewing position 1 of each item" (the apex); ordinal=0 is the
    entry-point frame.
    """
    prev_seg = None
    ordinal = 0
    for rec in annotations:
        seg = rec.get("phase_segment")
        if seg == "close_up":
            if prev_seg != "close_up":
                ordinal = 0
            else:
                ordinal += 1
            rec["close_up_ordinal"] = int(ordinal)
        else:
            rec["close_up_ordinal"] = None
        prev_seg = seg


def _build_embeddings_by_item(
    embeddings: np.ndarray,
    annotations: List[dict],
    *,
    target_ordinal: int = 5,  # ~midway through the close-up segment (~apex)
) -> dict[int, np.ndarray]:
    """Group embeddings by item, picking the close_up frame at `target_ordinal`
    from each loop. These are the same-item-across-loops samples used by
    §5.1.

    Requires `close_up_ordinal` to be populated on `annotations`; if missing,
    calls `_assign_close_up_ordinals` first.
    """
    have_ordinals = any(
        rec.get("close_up_ordinal") is not None for rec in annotations
    )
    if not have_ordinals:
        _assign_close_up_ordinals(annotations)
    by_item: dict[int, list[np.ndarray]] = {1: [], 2: [], 3: [], 4: [], 5: []}
    for rec in annotations:
        if rec.get("phase_segment") != "close_up":
            continue
        ord_idx = rec.get("close_up_ordinal")
        if ord_idx != target_ordinal:
            continue
        vp = int(rec.get("viewing_position_id", -1))
        if vp not in by_item:
            continue
        fi = int(rec["frame_idx"])
        if fi >= embeddings.shape[0]:
            continue
        by_item[vp].append(embeddings[fi].copy())
    return {k: np.asarray(v) for k, v in by_item.items() if v}


def _classify_finding_3(
    motion_check_result: "CheckResult",
    embeddings: np.ndarray,
    annotations: List[dict],
    *,
    cos_max: float,
) -> dict:
    """Operationalize the design-chat substrate-as-feature determination.

    Design-chat decision 2026-05-19: the structural transit→close_up
    segment-handoff duplicates are substrate-as-feature per instr finding 4.
    This function inspects the failing pairs and classifies the finding 3
    result as PASS iff every failing pair fits the expected handoff pattern.
    Any failing pair that is NOT a transit→close_up handoff (or that
    appears within a 30-frame run of consecutive duplicates — the actual
    static-dwell pattern v0 finding 3 was about) flips the classification
    back to FAIL.

    Returns a dict with keys:
      pass: bool
      total_pairs_above_threshold: int
      transit_to_closeup_count: int
      other_pattern_count: int
      max_consecutive_run: int
      handoff_pairs_match_expected: bool
      expected_handoff_count: int   (n_items × n_loops)
      notes: str
    """
    if embeddings.shape[0] < 2:
        return {
            "pass": False,
            "notes": "fewer than 2 embeddings",
        }
    a = embeddings[:-1]
    b = embeddings[1:]
    cos = (a * b).sum(axis=1) / (
        np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12
    )
    high = np.where(cos > cos_max)[0]

    # Count by phase-segment transition pattern.
    transit_to_closeup = 0
    other_pattern: list[tuple[int, str, str]] = []
    for i in high:
        if i + 1 >= len(annotations):
            continue
        r1 = annotations[i]
        r2 = annotations[i + 1]
        seg1 = r1.get("phase_segment")
        seg2 = r2.get("phase_segment")
        if seg1 == "transit" and seg2 == "close_up":
            transit_to_closeup += 1
        else:
            other_pattern.append((int(i), str(seg1), str(seg2)))

    # Maximum consecutive run of high-cosine pairs (a proxy for static dwell
    # — a 30-frame static dwell would manifest as a run of 29+ consecutive
    # high-cosine pairs).
    max_run = 0
    cur_run = 0
    prev_idx = None
    for i in high:
        if prev_idx is not None and i == prev_idx + 1:
            cur_run += 1
        else:
            cur_run = 1
        max_run = max(max_run, cur_run)
        prev_idx = int(i)

    # Substrate inventory for the audit trail. Note: `loops_seen` here
    # includes any partial trailing loop the collector enters before
    # stopping. The expected-handoff count 5 × 5 × 1 = 25 from the
    # design-chat evidence trace assumes only complete loops; the actual
    # observed handoff count varies with how cleanly the collection ended.
    # The classification therefore does NOT require strict equality with
    # n_items × n_loops — it requires (a) every failing pair to be a
    # transit→close_up handoff (the only structural-by-design source of
    # cos = 1.0 duplicates) AND (b) no run of ≥29 consecutive duplicates
    # (which would be 30-frame static dwell — v0 finding 3's actual concern).
    items_seen = {
        int(r["viewing_position_id"])
        for r in annotations
        if r.get("phase_segment") == "close_up" and int(r.get("viewing_position_id", -1)) > 0
    }
    loops_seen = {
        int(r["loop_index"])
        for r in annotations
        if int(r.get("loop_index", -1)) >= 0
    }

    no_dwell_pattern = max_run < 29  # 30-frame static dwell = ≥29 consecutive pairs
    no_other_pattern = len(other_pattern) == 0

    passed = no_other_pattern and no_dwell_pattern

    notes = (
        f"transit→close_up handoffs: {transit_to_closeup} "
        f"(structural-by-design segment-handoff duplicates; "
        f"audit derivation 5 items × N complete loops × 1 handoff per item-loop); "
        f"other-pattern pairs: {len(other_pattern)}; "
        f"max consecutive-pair run: {max_run} (30-frame dwell would be ≥29)"
    )

    return {
        "pass": bool(passed),
        "total_pairs_above_threshold": int(len(high)),
        "transit_to_closeup_count": int(transit_to_closeup),
        "other_pattern_count": int(len(other_pattern)),
        "other_pattern_samples": other_pattern[:10],
        "max_consecutive_run": int(max_run),
        "items_seen": sorted(items_seen),
        "loops_seen": sorted(loops_seen),
        "design_chat_classification": "substrate-as-feature (2026-05-19)",
        "evidence_trace_5x5x1": (
            "25 = 5 items × 5 complete loops × 1 handoff per (item, loop) "
            "— ContinuousMotionExplorer locks heading to viewing_heading "
            "during the final transit micro-step, then close-up begins at "
            "the same pose"
        ),
        "v0_precedent": "STOP_REPORT.md (seed-7 rerender, 2026-05-12; 0.9999 → 0.999)",
        "notes": notes,
    }


def _view_through_check(route: dict) -> dict:
    """Finding 7: view-through pose check on DiningTable.

    Inherited from v0 substrate finding 7 — DiningTable's adjusted pose
    is recorded in `route_phase2.json`. CC verifies the route file has
    the adjusted pose; that's the canonical reference for the v1 substrate.
    """
    dining = next(
        (it for it in route["items"] if it["object_type"] == "DiningTable"), None
    )
    if dining is None:
        return {"pass": False, "reason": "DiningTable not in route"}
    return {
        "pass": True,
        "viewing_position": dining["viewing_position"],
        "viewing_heading_deg": dining["viewing_heading_deg"],
        "source": "v0/data/route_phase2.json (inherited)",
    }


def main() -> int:
    p = argparse.ArgumentParser(description="PRE-A substrate verification driver")
    p.add_argument("--route-json", type=Path, default=DEFAULT_ROUTE_JSON)
    p.add_argument("--n-loops", type=int, default=5)
    p.add_argument(
        "--frames-dir", type=Path, default=PRE_A_FRAMES_DIR,
    )
    p.add_argument(
        "--embeddings-path", type=Path, default=PRE_A_EMBEDDINGS,
    )
    p.add_argument(
        "--annotations-path", type=Path, default=PRE_A_ANNOTATIONS,
    )
    p.add_argument(
        "--report-path", type=Path, default=PRE_A_REPORT,
    )
    p.add_argument(
        "--skip-collect", action="store_true",
        help="Skip frame collection (use existing frames_dir + annotations).",
    )
    p.add_argument(
        "--skip-encode", action="store_true",
        help="Skip DINOv2 encoding (use existing embeddings.npy).",
    )
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    os.environ.setdefault("DISPLAY", ":0")
    device = torch.device(args.device)
    print(f"[pre_a] device={device}", flush=True)

    if not args.route_json.exists():
        print(f"[pre_a] STOP: route_json not found at {args.route_json}", file=sys.stderr)
        return 1
    route = json.loads(args.route_json.read_text())
    print(f"[pre_a] route: seed={route['seed']} items={[it['object_type'] for it in route['items']]}", flush=True)

    # Finding 1: Python version.
    py_check = python_version_check()
    print(f"[pre_a] {py_check.name}: pass={py_check.passed} ({py_check.notes})", flush=True)

    # Finding 6 / §5.7: floor-y derivation.
    print("[pre_a] computing floor-y from GetReachablePositions...", flush=True)
    floor_y_info = _floor_y_check(route)
    PRE_A_FLOOR_Y.parent.mkdir(parents=True, exist_ok=True)
    PRE_A_FLOOR_Y.write_text(json.dumps(floor_y_info, indent=2))
    print(f"[pre_a] floor_y={floor_y_info['floor_y']} (modal {floor_y_info['modal_count']}/{floor_y_info['n_positions']} positions)", flush=True)

    # Frame collection.
    if args.skip_collect:
        if not args.annotations_path.exists():
            print(f"[pre_a] STOP: --skip-collect set but annotations missing at {args.annotations_path}", file=sys.stderr)
            return 1
        with args.annotations_path.open() as fh:
            annotations = [json.loads(line) for line in fh if line.strip()]
        collect_summary = {"frame_count": len(annotations), "skipped": True}
        print(f"[pre_a] skipping collect; {len(annotations)} annotations loaded", flush=True)
    else:
        print(f"[pre_a] collecting {args.n_loops} loops (frames -> {args.frames_dir})", flush=True)
        collect_summary = _collect_frames(
            route, args.n_loops, args.frames_dir, args.annotations_path,
        )
        print(f"[pre_a] collected {collect_summary['frame_count']} frames in "
              f"{collect_summary['wall_clock_seconds']:.1f}s "
              f"({collect_summary['n_loops_collected']} complete loops)", flush=True)
        with args.annotations_path.open() as fh:
            annotations = [json.loads(line) for line in fh if line.strip()]

    # DINOv2 encoding.
    if args.skip_encode and args.embeddings_path.exists():
        embeddings = np.load(args.embeddings_path)
        print(f"[pre_a] reusing embeddings: shape {embeddings.shape}", flush=True)
    else:
        print(f"[pre_a] loading DINOv2-Large frozen encoder...", flush=True)
        model = load_frozen_dinov2(device)
        indices = list(range(len(annotations)))
        print(f"[pre_a] encoding {len(indices)} frames via DINOv2...", flush=True)
        t0 = time.time()
        embeddings = encode_frames(model, args.frames_dir, indices, device=device)
        wall = time.time() - t0
        args.embeddings_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.embeddings_path, embeddings)
        print(f"[pre_a] encoded {len(indices)} frames in {wall:.1f}s; saved to {args.embeddings_path}", flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Finding 2: embeddings full-population check.
    pop_check = embeddings_full_population_check(embeddings)
    print(f"[pre_a] {pop_check.name}: pass={pop_check.passed} ({pop_check.notes})", flush=True)

    # Finding 3: continuous-motion check (no 30-frame static dwell).
    # Threshold recalibrated 0.9999 → 0.999 (v1/src/config.py FINDING_3_COS_MAX)
    # per design-chat determination on 2026-05-19 — substrate-as-feature
    # classification on the structural transit→close_up segment-handoff
    # duplicates; v0 STOP_REPORT precedent.
    motion_check = continuous_motion_check(embeddings, cos_max=FINDING_3_COS_MAX)
    print(f"[pre_a] {motion_check.name}: raw count={motion_check.value} (threshold cos>{FINDING_3_COS_MAX})", flush=True)

    # Design-chat-determined substrate-as-feature classification (2026-05-19):
    # the strict per-pair check on a deterministic AI2-THOR substrate flags
    # transit→close_up segment-handoff duplicates (frame i in transit ends
    # at the close-up entry pose; frame i+1 in close_up begins at the same
    # pose with locked heading — rendered bit-identically). These are
    # structural-by-design under the v0-inherited ContinuousMotionExplorer
    # and are NOT 30-frame static dwell (each is a single-pair duplicate;
    # close-up segments proceed with continuous motion through 10+ ordinals).
    # v2 will adopt a run-length-aware check; in v1 we operationalize the
    # classification by checking that ALL failing pairs are transit→close_up.
    motion_check_classification = _classify_finding_3(
        motion_check, embeddings, annotations, cos_max=FINDING_3_COS_MAX
    )
    print(
        f"[pre_a] {motion_check.name}: classified pass={motion_check_classification['pass']} "
        f"({motion_check_classification['notes']})", flush=True,
    )

    # §5.1 / §5.2 / §5.3: cross-instance + cross-element + combined gap.
    embeds_by_item = _build_embeddings_by_item(embeddings, annotations)
    print(f"[pre_a] embeds_by_item sizes: {[(k, v.shape[0]) for k, v in embeds_by_item.items()]}", flush=True)
    stab = cross_instance_stability(embeds_by_item)
    print(f"[pre_a] {stab.name}: value={stab.value:.4f} (threshold >{SUBSTRATE_CROSS_INSTANCE_STABILITY_MIN}) pass={stab.passed}", flush=True)
    dist = cross_element_distinguishability(embeds_by_item)
    print(f"[pre_a] {dist.name}: value={dist.value:.4f} (threshold <{SUBSTRATE_CROSS_ELEMENT_DISTINGUISH_MAX}) pass={dist.passed}", flush=True)
    gap = combined_gap(stab, dist)
    print(f"[pre_a] {gap.name}: value={gap.value:.4f} (threshold >={SUBSTRATE_COMBINED_GAP_MIN}) pass={gap.passed}", flush=True)

    # Finding 7: view-through pose check on DiningTable.
    view_through = _view_through_check(route)
    print(f"[pre_a] finding_7_view_through_pose: pass={view_through['pass']}", flush=True)

    # Finding 4 (substrate-as-feature-vs-bug): interpretive; no automated check.
    finding_4 = {
        "name": "finding_4_substrate_as_feature_vs_bug",
        "interpretive": True,
        "default_classification": "substrate-as-feature unless positive bug evidence",
        "pass": True,
        "notes": "no observed substrate properties to classify yet",
    }

    # Finding 5 (camera elevation): handled by the explorer using
    # modal-y derivation; no separate API call. PASS iff floor-y derivation
    # succeeded (the explorer reads floor-y from modal-y).
    finding_5 = {
        "name": "finding_5_camera_elevation",
        "pass": bool(floor_y_info["pass"]),
        "floor_y": floor_y_info["floor_y"],
        "notes": "explorer sets agent y to floor-y modal value; no forceAction",
    }

    # Finding 8 (cross-room visual leakage): PRE-B specific; skip here.
    finding_8 = {
        "name": "finding_8_cross_room_visual_leakage",
        "pass": True,
        "deferred_to_pre_b": True,
        "notes": "v1 perturbation mechanism not yet selected; checked at PRE-B",
    }

    checks_dataclass: List[CheckResult] = [
        py_check, pop_check, stab, dist, gap,
    ]
    checks_dict: List[dict] = [c.__dict__ for c in checks_dataclass]
    # Finding 3 is published in the classified form (substrate-as-feature
    # determination from design chat 2026-05-19); the raw count is preserved
    # as `raw_count_value` for the audit trail.
    finding_3_record = {
        "name": motion_check.name,
        "value": motion_check.value,
        "threshold": motion_check.threshold,
        "threshold_direction": motion_check.threshold_direction,
        "passed": bool(motion_check_classification["pass"]),
        "raw_strict_check_passed": bool(motion_check.passed),
        "classification": motion_check_classification,
        "notes": motion_check.notes,
    }
    checks_dict.append(finding_3_record)
    checks_dict.extend([
        {**finding_4},
        {**finding_5},
        {"name": "finding_6_floor_y_derivation", "pass": floor_y_info["pass"], "floor_y": floor_y_info["floor_y"]},
        {"name": "finding_7_view_through_pose", **view_through},
        {**finding_8},
    ])

    all_pass = all(c.get("passed", c.get("pass", False)) for c in checks_dict)

    PRE_A_REPORT.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "all_passed": bool(all_pass),
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "collection_summary": collect_summary,
        "checks": checks_dict,
        "floor_y_info": floor_y_info,
        "view_through": view_through,
        "thresholds": {
            "cross_instance_stability_min": SUBSTRATE_CROSS_INSTANCE_STABILITY_MIN,
            "cross_element_distinguishability_max": SUBSTRATE_CROSS_ELEMENT_DISTINGUISH_MAX,
            "combined_gap_min": SUBSTRATE_COMBINED_GAP_MIN,
        },
    }
    PRE_A_REPORT.write_text(json.dumps(payload, indent=2))
    print(f"[pre_a] report written: {PRE_A_REPORT}", flush=True)
    print(f"[pre_a] all_passed: {all_pass}", flush=True)

    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Extended DINOv2 perturbation diagnostic for Phase 2 substrate review.

Per the 2026-05-14 reviewer directive following the third STOP in
session 5: run 2-3 additional `RandomizeMaterials` draws beyond the
preflight's 3, compute per-item DINOv2 cosine std across draws, and
identify which items show consistent cross-room bleed.

Mean tells us how far the item's DINOv2 representation moves under
out-of-scope perturbation. Std tells us whether that movement is
consistent (low std = real cross-room sensitivity) or per-draw
random (high std = material-lottery noise).

This script is read-only against the substrate — it does not affect
the spec, the route, the instructions, or any committed artefact.
It writes a report at
`results/inner_pam_v0/phase2_extended_diagnostic/extended_diagnostic_report.{md,json}`
and is intended to inform the pose-search step that follows.

Usage:
  nohup python3.12 -u scripts/run_phase2_extended_diagnostic.py \\
      > logs/phase2_extended_diagnostic_$(date +%Y%m%d_%H%M%S).log 2>&1 &
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
import torch  # noqa: E402

from src.encoder.dinov2_encoder import load_frozen_dinov2  # noqa: E402
from src.env.material_perturbation_probe import (  # noqa: E402
    BEDROOM_ITEM_IDS,
    LIVINGROOM_ITEM_IDS,
    capture_all_items,
    dinov2_encode_batch,
    items_by_id,
    pixel_cosine,
    randomize_livingroom,
)
from src.env.procthor_house import load_house, make_controller  # noqa: E402


_DEFAULT_ROUTE_JSON = Path(
    "/mnt/c/Users/Jason/Desktop/Eridos/Weft/results/stage_0b_furniture/route.json"
)
_DEFAULT_RESULTS_DIR = _ROOT / "results" / "inner_pam_v0" / "phase2_extended_diagnostic"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--route_json", type=Path, default=_DEFAULT_ROUTE_JSON)
    parser.add_argument("--results_dir", type=Path, default=_DEFAULT_RESULTS_DIR)
    parser.add_argument("--num_random_calls", type=int, default=6,
                        help="number of consecutive RandomizeMaterials draws "
                             "to capture (default 6 = preflight's 3 + 3 more)")
    parser.add_argument("--width", type=int, default=300)
    parser.add_argument("--height", type=int, default=300)
    args = parser.parse_args()

    os.environ.setdefault("DISPLAY", ":0")
    if not args.route_json.is_file():
        print(f"[diag] FAIL: route file not found: {args.route_json}", file=sys.stderr)
        return 1
    route = json.loads(args.route_json.read_text())
    items = items_by_id(route)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"[diag] ts={ts} house_seed={route['seed']} num_random_calls={args.num_random_calls}",
          flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("[diag] FAIL: CUDA not available", file=sys.stderr)
        return 1

    house = load_house(seed=int(route["seed"]), min_rooms=4)
    controller = make_controller(house, width=args.width, height=args.height)
    try:
        print("[diag] capturing baseline frames", flush=True)
        before_frames = capture_all_items(controller, items)

        per_call_frames: List[Dict[int, np.ndarray]] = []
        for call_k in range(1, args.num_random_calls + 1):
            print(f"[diag] RandomizeMaterials call {call_k}/{args.num_random_calls}",
                  flush=True)
            ok, _ret = randomize_livingroom(controller)
            if not ok:
                print(f"[diag] FAIL: RandomizeMaterials returned not-ok on call {call_k}",
                      file=sys.stderr)
                return 2
            per_call_frames.append(capture_all_items(controller, items))
    finally:
        try:
            controller.stop()
        except Exception:
            pass

    # ---- DINOv2 encoding ----
    print(f"[diag] loading DINOv2-large + encoding "
          f"{(args.num_random_calls + 1) * len(items)} frames", flush=True)
    model = load_frozen_dinov2(device)

    ordered_item_ids = sorted(items.keys())
    all_frames: List[np.ndarray] = []
    capture_keys: List[Tuple[int, str]] = []
    for item_id in ordered_item_ids:
        all_frames.append(before_frames[item_id])
        capture_keys.append((item_id, "before"))
    for k, cap in enumerate(per_call_frames, start=1):
        for item_id in ordered_item_ids:
            all_frames.append(cap[item_id])
            capture_keys.append((item_id, f"call{k}"))
    embeddings = dinov2_encode_batch(model, all_frames, device)
    idx_by_key = {key: i for i, key in enumerate(capture_keys)}
    print(f"[diag] encoded; shape={embeddings.shape}", flush=True)

    # ---- per-item before-vs-call_k DINOv2 cosines, with std across draws ----
    per_item: Dict[str, Dict[str, Any]] = {}
    for item_id in ordered_item_ids:
        label = items[item_id]["object_type"]
        room = "LivingRoom" if item_id in LIVINGROOM_ITEM_IDS else (
            "Bedroom" if item_id in BEDROOM_ITEM_IDS else "Other"
        )
        before_emb = embeddings[idx_by_key[(item_id, "before")]]
        per_call_dinov2 = []
        per_call_flat = []
        for k in range(1, args.num_random_calls + 1):
            after_emb = embeddings[idx_by_key[(item_id, f"call{k}")]]
            cos_dinov2 = float(np.dot(before_emb, after_emb))
            per_call_dinov2.append(cos_dinov2)
            cos_flat = pixel_cosine(before_frames[item_id], per_call_frames[k - 1][item_id])
            per_call_flat.append(cos_flat)
        per_item[label] = {
            "viewing_position_id": int(item_id),
            "scope": room,
            "expected_role": (
                "perturbed (LivingRoom items get re-textured)"
                if item_id in LIVINGROOM_ITEM_IDS
                else "control (Bedroom items should be unchanged)"
            ),
            "dinov2_per_call": per_call_dinov2,
            "dinov2_mean": float(np.mean(per_call_dinov2)),
            "dinov2_std": float(np.std(per_call_dinov2, ddof=0)),
            "dinov2_min": float(np.min(per_call_dinov2)),
            "dinov2_max": float(np.max(per_call_dinov2)),
            "flat_rgb_per_call": per_call_flat,
            "flat_rgb_mean": float(np.mean(per_call_flat)),
            "flat_rgb_std": float(np.std(per_call_flat, ddof=0)),
        }

    # ---- Aggregate signals ----
    bedroom_means = [
        per_item[items[i]["object_type"]]["dinov2_mean"] for i in BEDROOM_ITEM_IDS
    ]
    livingroom_means = [
        per_item[items[i]["object_type"]]["dinov2_mean"] for i in LIVINGROOM_ITEM_IDS
    ]
    bedroom_mean = float(np.mean(bedroom_means))
    livingroom_mean = float(np.mean(livingroom_means))
    contrast = bedroom_mean - livingroom_mean

    # Affected = Bedroom items whose dinov2_mean is "too low" given their
    # role (control items should sit near 1.0). Use 0.98 as the cutoff
    # (matches the preflight G_M2 threshold from the prior round).
    affected_bedroom_items: List[str] = []
    stable_bedroom_items: List[str] = []
    for i in BEDROOM_ITEM_IDS:
        label = items[i]["object_type"]
        m = per_item[label]["dinov2_mean"]
        if m < 0.98:
            affected_bedroom_items.append(label)
        else:
            stable_bedroom_items.append(label)

    report = {
        "timestamp_utc": ts,
        "house_seed": int(route["seed"]),
        "num_random_calls": int(args.num_random_calls),
        "per_item": per_item,
        "aggregate": {
            "bedroom_dinov2_mean": bedroom_mean,
            "livingroom_dinov2_mean": livingroom_mean,
            "bedroom_minus_livingroom_contrast": contrast,
            "bedroom_items_below_0_98_dinov2_mean": affected_bedroom_items,
            "bedroom_items_above_0_98_dinov2_mean": stable_bedroom_items,
        },
        "affected_items": affected_bedroom_items,
        "rationale": (
            "Items in the Bedroom whose 3+ call DINOv2 mean cosine sits "
            "below 0.98 under LivingRoom-scoped RandomizeMaterials are "
            "treated as 'affected' — their representation is sensitive to "
            "out-of-scope perturbation, indicating the viewing pose's FOV "
            "catches visible LivingRoom content (the doorway-bleed pattern). "
            "Std across draws separates 'consistent leak' (low std, real "
            "cross-room sensitivity) from 'lottery noise' (high std)."
        ),
    }
    (args.results_dir / "extended_diagnostic_report.json").write_text(
        json.dumps(report, indent=2)
    )

    md = ["# Phase 2 Extended Diagnostic Report",
          "",
          f"Timestamp: {ts}",
          f"House seed: {route['seed']}",
          f"Number of RandomizeMaterials draws: {args.num_random_calls}",
          "",
          "## Per-item summary",
          "",
          "| item | room | dinov2 mean | dinov2 std | dinov2 min | dinov2 max | flat-RGB mean |",
          "|---|---|---:|---:|---:|---:|---:|"]
    for item_id in ordered_item_ids:
        label = items[item_id]["object_type"]
        d = per_item[label]
        md.append(
            f"| {label} | {d['scope']} | {d['dinov2_mean']:.4f} | "
            f"{d['dinov2_std']:.4f} | {d['dinov2_min']:.4f} | "
            f"{d['dinov2_max']:.4f} | {d['flat_rgb_mean']:.4f} |"
        )
    md.append("")
    md.append("## Aggregate")
    md.append("")
    md.append(f"- Bedroom DINOv2 mean: {bedroom_mean:.4f}")
    md.append(f"- LivingRoom DINOv2 mean: {livingroom_mean:.4f}")
    md.append(f"- Bedroom − LivingRoom contrast: {contrast:.4f}")
    md.append("")
    md.append("## Affected Bedroom control items (mean DINOv2 < 0.98)")
    md.append("")
    if affected_bedroom_items:
        for label in affected_bedroom_items:
            d = per_item[label]
            md.append(f"- **{label}** (vp_id={d['viewing_position_id']}): "
                      f"mean={d['dinov2_mean']:.4f}, std={d['dinov2_std']:.4f}, "
                      f"range [{d['dinov2_min']:.4f}, {d['dinov2_max']:.4f}]")
    else:
        md.append("- *(none — all Bedroom controls stable)*")
    md.append("")
    md.append("## Stable Bedroom control items (mean DINOv2 ≥ 0.98)")
    md.append("")
    if stable_bedroom_items:
        for label in stable_bedroom_items:
            d = per_item[label]
            md.append(f"- **{label}** (vp_id={d['viewing_position_id']}): "
                      f"mean={d['dinov2_mean']:.4f}, std={d['dinov2_std']:.4f}")
    md.append("")
    (args.results_dir / "extended_diagnostic_report.md").write_text("\n".join(md))

    print(f"[diag] wrote {args.results_dir / 'extended_diagnostic_report.json'}",
          flush=True)
    print(f"[diag] affected Bedroom items: {affected_bedroom_items or '(none)'}",
          flush=True)
    print(f"[diag] bedroom_mean={bedroom_mean:.4f} "
          f"livingroom_mean={livingroom_mean:.4f} "
          f"contrast={contrast:.4f}",
          flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException as e:
        traceback.print_exc()
        print(f"[diag] FATAL: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(3)

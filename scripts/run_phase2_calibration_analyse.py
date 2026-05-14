"""Encode the 5-loop calibration frames via DINOv2 and run motion-continuity diagnostics.

Per session-4 calibration step (HANDOFF entry 2026-05-13): the continuous-
motion substrate must produce non-bit-identical consecutive frames. This
script encodes the calibration frames via the verified DINOv2 protocol and
computes:

  - Per-frame embedding norms (sanity: all in [1-1e-5, 1+1e-5]).
  - Consecutive-frame cosines (one per (i, i+1) pair).
  - Disaggregations by phase: close_up vs transit; within-close-up vs at-phase-boundary.
  - Same-item cross-loop comparisons (apex frame at item N in loop L vs same in loop L'):
    detects whether the loop is bit-identically reproducible across loops, which
    would defeat the substrate change.

Outputs:
  results/phase2_calibration/continuity_report.json
  data/phase2_calibration/embeddings.npy  (gitignored)

Threshold notes:
  - Bit-identical = cos > 0.9999; we want to see ZERO bit-identical pairs.
  - Genuine continuous motion typically has cos ~ 0.92-0.99 between consecutive
    frames; rotation-heavy steps drop into 0.85-0.95.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path("/mnt/c/Users/Jason/Desktop/Eridos/Weft 2")
sys.path.insert(0, str(REPO_ROOT))

from src.encoder.dinov2_encoder import (  # noqa: E402
    encode_frames,
    load_frozen_dinov2,
)


def _load_annotations(path: Path) -> list[dict]:
    out = []
    with path.open("r") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--frames_dir", type=Path,
                        default=REPO_ROOT / "data/phase2_calibration/frames")
    parser.add_argument("--annotations", type=Path,
                        default=REPO_ROOT / "data/phase2_calibration/annotations.jsonl")
    parser.add_argument("--out_embeddings", type=Path,
                        default=REPO_ROOT / "data/phase2_calibration/embeddings.npy")
    parser.add_argument("--out_report", type=Path,
                        default=REPO_ROOT / "results/phase2_calibration/continuity_report.json")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    if not args.frames_dir.is_dir():
        print(f"FAIL: frames_dir not found: {args.frames_dir}", file=sys.stderr)
        return 1
    annotations = _load_annotations(args.annotations)
    n = len(annotations)
    print(f"[analyse] {n} annotations loaded", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[analyse] device: {device}", flush=True)
    model = load_frozen_dinov2(device=device)
    print("[analyse] encoding frames...", flush=True)
    indices = list(range(n))
    embeddings = encode_frames(
        model, args.frames_dir, indices, device,
        batch_size=args.batch_size, num_workers=args.num_workers,
    )
    args.out_embeddings.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out_embeddings, embeddings)
    print(f"[analyse] wrote {args.out_embeddings}", flush=True)

    # Sanity
    norms = np.linalg.norm(embeddings, axis=1)
    all_norm_ok = bool(((norms >= 1.0 - 1e-5) & (norms <= 1.0 + 1e-5)).all())

    # Consecutive-frame cosines
    consec_cos = np.einsum("ij,ij->i", embeddings[:-1], embeddings[1:]).astype(np.float64)
    bit_identical_mask = consec_cos > 0.9999

    # Disaggregate by phase: pair (i, i+1) tagged by annotations[i].phase + annotations[i+1].phase
    phase_pair: list[str] = []
    for i in range(n - 1):
        p0 = str(annotations[i].get("phase", "?"))
        p1 = str(annotations[i + 1].get("phase", "?"))
        phase_pair.append(f"{p0}->{p1}")

    pair_groups: dict[str, list[float]] = {}
    for k, c in enumerate(consec_cos):
        g = phase_pair[k]
        pair_groups.setdefault(g, []).append(float(c))

    pair_summary = {
        g: {
            "n": int(len(vals)),
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=0)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "median": float(np.median(vals)),
            "n_bit_identical_gt_0.9999": int(sum(1 for v in vals if v > 0.9999)),
        }
        for g, vals in pair_groups.items()
    }

    # Same-item cross-loop apex comparison.
    apex_by_item: dict[int, list[int]] = {}
    for i, a in enumerate(annotations):
        if a.get("close_up_apex_flag"):
            vp = int(a.get("viewing_position_id", 0))
            if 1 <= vp <= 5:
                apex_by_item.setdefault(vp, []).append(i)
    cross_loop_apex: dict[int, dict[str, float]] = {}
    for vp, idx in apex_by_item.items():
        if len(idx) < 2:
            continue
        arr = embeddings[idx]
        norms_sub = np.linalg.norm(arr, axis=1, keepdims=True)
        arr_n = arr / np.maximum(norms_sub, 1e-12)
        sim = arr_n @ arr_n.T
        iu = np.triu_indices(sim.shape[0], k=1)
        within = sim[iu]
        cross_loop_apex[vp] = {
            "n_apex_instances": int(len(idx)),
            "mean_cosine_across_loops": float(np.mean(within)),
            "std": float(np.std(within, ddof=0)),
            "min": float(np.min(within)),
            "max": float(np.max(within)),
            "n_bit_identical_gt_0.9999": int(sum(1 for v in within if v > 0.9999)),
        }

    n_consec_bit_identical = int(bit_identical_mask.sum())

    # Within-loop motion-continuity (spec §2.3) cares about IN-MOTION phase
    # pairs: close_up→close_up and transit→transit. Boundary pairs
    # (close_up→transit, transit→close_up) can have a 1-frame duplication at
    # the phase boundary if the last transit step lands at exactly
    # (close_up_start, viewing_heading); this is a cosmetic artefact, not a
    # motion-continuity failure (session-4 open decision 4, accepted as
    # cosmetic 2026-05-14). Verdict is computed only on the in-motion pairs.
    in_motion_pair_keys = ("close_up->close_up", "transit->transit")
    in_motion_bit_identical = sum(
        pair_summary[k]["n_bit_identical_gt_0.9999"]
        for k in in_motion_pair_keys
        if k in pair_summary
    )
    boundary_bit_identical = sum(
        v["n_bit_identical_gt_0.9999"] for k, v in pair_summary.items()
        if k not in in_motion_pair_keys
    )

    # Cross-loop apex bit-identicity is the expected Stage A baseline state
    # under the curriculum framing (no jitter; identical loops produce
    # identical embeddings). It's reported as informational, not a verdict
    # input.

    report = {
        "n_frames": int(n),
        "norm_check": {
            "min": float(norms.min()),
            "max": float(norms.max()),
            "all_in_[1-1e-5, 1+1e-5]": all_norm_ok,
        },
        "consecutive_frame_cosine": {
            "n_pairs": int(len(consec_cos)),
            "mean": float(consec_cos.mean()),
            "std": float(consec_cos.std(ddof=0)),
            "min": float(consec_cos.min()),
            "max": float(consec_cos.max()),
            "n_bit_identical_gt_0.9999_total": int(n_consec_bit_identical),
            "n_bit_identical_in_motion_pairs": int(in_motion_bit_identical),
            "n_bit_identical_boundary_pairs": int(boundary_bit_identical),
            "fraction_bit_identical_total": float(
                n_consec_bit_identical / len(consec_cos)
            ),
            "by_phase_pair": pair_summary,
        },
        "cross_loop_apex_comparison": cross_loop_apex,
        "motion_continuity_verdict": (
            "pass"
            if all_norm_ok and in_motion_bit_identical == 0
            else "fail"
        ),
        "verdict_criteria": (
            "PASS iff norms in [1-1e-5, 1+1e-5] AND zero bit-identical "
            "(cos>0.9999) pairs within in-motion phase pairs "
            "(close_up->close_up, transit->transit). Boundary pairs "
            "(close_up->transit, transit->close_up) reported separately "
            "as cosmetic; cross-loop apex bit-identicity is the curriculum's "
            "Stage A baseline (informational, not gated)."
        ),
        "items_with_bit_identical_cross_loop_apex": [
            vp for vp, stats in cross_loop_apex.items()
            if stats["n_bit_identical_gt_0.9999"] > 0
        ],
        "items_with_bit_identical_cross_loop_apex_note": (
            "Bit-identicity across Stage A loops at the same viewing pose is "
            "the curriculum's expected baseline (no jitter; deterministic "
            "rendering at fixed pose). This list is informational; the "
            "motion_continuity_verdict does not depend on it."
        ),
    }
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text(json.dumps(report, indent=2))
    print(f"[analyse] wrote {args.out_report}", flush=True)
    print(f"[analyse] verdict: {report['motion_continuity_verdict']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

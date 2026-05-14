"""Cross-loop invariance diagnostic on Stage A control-item close-up frames.

Question (reviewer): on a corrected substrate with a deterministic renderer and
no in-scope perturbation, Stage A close-up frames at a control item should be
bit-identical across loops. The §8.4 control gaps (Bed 0.0045, TV 0.0068) are
inconsistent with that assumption.

Method: pick K Stage A loops; for each item ∈ {Bed=1, Television=5}, gather
close-up frames; group frames by their within-close-up ordinal index (matching
the trajectory step inside the close-up segment); compute pairwise pixel-MD5
hashes (on the decoded RGB numpy array) and pairwise DINOv2 cosines (from the
existing data/phase2_embeddings/embeddings.npy) at each ordinal index. Also
do the same on the apex-flagged frame.

Writes results/inner_pam_v0/phase2_main/cross_loop_invariance_check.json with
per-item statistics suitable for the Case A / B / C interpretation in HANDOFF.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

REPO_ROOT = Path("/mnt/c/Users/Jason/Desktop/Eridos/Weft 2")
FRAMES_DIR = REPO_ROOT / "data/phase2_frames"
ANNOT = REPO_ROOT / "data/phase2_annotations.jsonl"
EMB = REPO_ROOT / "data/phase2_embeddings/embeddings.npy"
OUT = REPO_ROOT / "results/inner_pam_v0/phase2_main/cross_loop_invariance_check.json"

# Stage A loops to compare — spread across the 0..30 unperturbed window.
SAMPLE_LOOPS = (1, 5, 10, 15, 20, 25, 28)
ITEMS = {"Bed": 1, "Television": 5}
SKIP_NOISY_CONTROL = "DiningTable"  # reviewer note: known noisy control diagnostic; skip


def _load_annotations(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open("r") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _frame_md5(frame_idx: int) -> tuple[str, str]:
    """Return (decoded_pixel_md5, raw_file_md5) for frame `frame_idx`."""
    p = FRAMES_DIR / f"frame_{frame_idx:08d}.png"
    raw = p.read_bytes()
    raw_md5 = hashlib.md5(raw).hexdigest()
    arr = np.asarray(Image.open(p).convert("RGB"))
    pixel_md5 = hashlib.md5(arr.tobytes()).hexdigest()
    return pixel_md5, raw_md5


def _summary_stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"n": 0, "min": None, "max": None, "mean": None}
    return {
        "n": int(len(values)),
        "min": float(min(values)),
        "max": float(max(values)),
        "mean": float(sum(values) / len(values)),
    }


def main() -> int:
    if not ANNOT.is_file():
        print(f"[xloop] FAIL: annotations not found: {ANNOT}", file=sys.stderr)
        return 1
    if not EMB.is_file():
        print(f"[xloop] FAIL: embeddings not found: {EMB}", file=sys.stderr)
        return 1

    annotations = _load_annotations(ANNOT)
    print(f"[xloop] loaded {len(annotations)} annotations")

    embeddings = np.load(EMB)
    print(f"[xloop] loaded embeddings shape={embeddings.shape} dtype={embeddings.dtype}")

    # Build: per (item, loop) -> list of frame_idx in close_up order.
    by_item_loop: dict[tuple[int, int], list[int]] = {}
    apex_by_item_loop: dict[tuple[int, int], int] = {}
    for a in annotations:
        if a.get("phase_segment") != "close_up":
            continue
        if not bool(a.get("perturbation_active") is False):
            continue
        loop = int(a.get("loop_index", -1))
        if loop not in SAMPLE_LOOPS:
            continue
        vp = int(a.get("viewing_position_id", -1))
        if vp not in ITEMS.values():
            continue
        by_item_loop.setdefault((vp, loop), []).append(int(a["frame_idx"]))
        if bool(a.get("close_up_apex_flag", False)):
            apex_by_item_loop[(vp, loop)] = int(a["frame_idx"])

    # frame_idx lists are already in temporal order because the annotation
    # file is written in temporal order. Confirm by checking they are
    # monotonically increasing.
    for k, v in by_item_loop.items():
        assert v == sorted(v), f"frames out of order for {k}: {v}"

    report: dict[str, Any] = {
        "method": (
            "For each (item, loop) collect close-up frames in temporal order. "
            "Group by within-close-up ordinal index. At each ordinal i, compute "
            "pairwise pixel-MD5 matches and pairwise DINOv2 cosines across the "
            "sampled loops. Also report the apex frame's stats separately."
        ),
        "sample_loops": list(SAMPLE_LOOPS),
        "items": dict(ITEMS),
        "skipped": [SKIP_NOISY_CONTROL],
        "per_item": {},
    }

    for item_label, vp in ITEMS.items():
        present_loops = sorted(l for (v, l) in by_item_loop if v == vp)
        if len(present_loops) < 2:
            print(f"[xloop] SKIP {item_label}: insufficient loops "
                  f"({present_loops})", flush=True)
            continue

        # Pose match: for each loop, the list of (x, z, rotation_y) within the
        # close-up. The trajectory should be deterministic — same poses at the
        # same ordinal across loops.
        pose_match_summary: dict[str, Any] = {}
        loop_to_poses: dict[int, list[tuple[float, float, float]]] = {}
        for loop in present_loops:
            frames = by_item_loop[(vp, loop)]
            poses: list[tuple[float, float, float]] = []
            for fidx in frames:
                a = annotations[fidx]
                p = a["position"]
                poses.append((float(p["x"]), float(p["z"]), float(a["rotation_y"])))
            loop_to_poses[loop] = poses
        # Compare loops' pose sequences pairwise.
        ref_loop = present_loops[0]
        ref_poses = loop_to_poses[ref_loop]
        ref_len = len(ref_poses)
        all_lengths_equal = all(len(loop_to_poses[l]) == ref_len for l in present_loops)
        pose_max_dx_per_ordinal: list[float] = [0.0] * ref_len if all_lengths_equal else []
        pose_max_drot_per_ordinal: list[float] = [0.0] * ref_len if all_lengths_equal else []
        if all_lengths_equal:
            for i in range(ref_len):
                max_dx = 0.0
                max_drot = 0.0
                ref_x, ref_z, ref_r = ref_poses[i]
                for l in present_loops[1:]:
                    x, z, r = loop_to_poses[l][i]
                    d_xy = ((x - ref_x) ** 2 + (z - ref_z) ** 2) ** 0.5
                    d_rot = abs(((r - ref_r + 180) % 360) - 180)
                    if d_xy > max_dx:
                        max_dx = d_xy
                    if d_rot > max_drot:
                        max_drot = d_rot
                pose_max_dx_per_ordinal[i] = float(max_dx)
                pose_max_drot_per_ordinal[i] = float(max_drot)
        pose_match_summary = {
            "all_lengths_equal": bool(all_lengths_equal),
            "close_up_length_at_ref_loop": int(ref_len),
            "max_xz_displacement_across_loops_max_over_ordinals": (
                float(max(pose_max_dx_per_ordinal)) if pose_max_dx_per_ordinal else None
            ),
            "max_rotation_deviation_across_loops_max_over_ordinals_deg": (
                float(max(pose_max_drot_per_ordinal)) if pose_max_drot_per_ordinal else None
            ),
        }

        # Per-ordinal MD5 + cosine. Only meaningful if all lengths equal.
        per_ordinal: list[dict[str, Any]] = []
        if all_lengths_equal:
            for i in range(ref_len):
                frame_idxs_at_i = [by_item_loop[(vp, l)][i] for l in present_loops]
                pixel_hashes: list[str] = []
                raw_hashes: list[str] = []
                for fidx in frame_idxs_at_i:
                    ph, rh = _frame_md5(fidx)
                    pixel_hashes.append(ph)
                    raw_hashes.append(rh)
                n_unique_pixel_md5 = len(set(pixel_hashes))
                n_unique_raw_md5 = len(set(raw_hashes))
                # DINOv2 cosines pairwise (matrix and stats).
                emb_at_i = embeddings[frame_idxs_at_i].astype(np.float64)
                # Already L2-normed; cosine = dot.
                cos_mat = emb_at_i @ emb_at_i.T
                # Off-diagonal cosines.
                n = cos_mat.shape[0]
                off_diag: list[float] = []
                for a_idx in range(n):
                    for b_idx in range(a_idx + 1, n):
                        off_diag.append(float(cos_mat[a_idx, b_idx]))
                per_ordinal.append({
                    "ordinal": int(i),
                    "frame_idxs_per_loop": [
                        {"loop": int(present_loops[k]), "frame_idx": int(frame_idxs_at_i[k])}
                        for k in range(len(present_loops))
                    ],
                    "n_loops_sampled": int(len(present_loops)),
                    "n_unique_pixel_md5": int(n_unique_pixel_md5),
                    "n_unique_raw_md5": int(n_unique_raw_md5),
                    "pixel_md5_all_identical": bool(n_unique_pixel_md5 == 1),
                    "pixel_md5_per_loop": pixel_hashes,
                    "pairwise_cosines_off_diag": _summary_stats(off_diag),
                })

        # Apex-specific (single frame per loop).
        apex_frames = [
            apex_by_item_loop[(vp, l)]
            for l in present_loops
            if (vp, l) in apex_by_item_loop
        ]
        apex_block: dict[str, Any] = {}
        if len(apex_frames) >= 2:
            pixel_hashes_a: list[str] = []
            raw_hashes_a: list[str] = []
            for fidx in apex_frames:
                ph, rh = _frame_md5(fidx)
                pixel_hashes_a.append(ph)
                raw_hashes_a.append(rh)
            emb_a = embeddings[apex_frames].astype(np.float64)
            cos_mat_a = emb_a @ emb_a.T
            off_diag_a: list[float] = []
            for ai in range(cos_mat_a.shape[0]):
                for bi in range(ai + 1, cos_mat_a.shape[0]):
                    off_diag_a.append(float(cos_mat_a[ai, bi]))
            apex_block = {
                "apex_frame_idxs_per_loop": [
                    {"loop": int(present_loops[k]), "frame_idx": int(apex_frames[k])}
                    for k in range(len(apex_frames))
                ],
                "n_loops_sampled": int(len(apex_frames)),
                "n_unique_pixel_md5": int(len(set(pixel_hashes_a))),
                "n_unique_raw_md5": int(len(set(raw_hashes_a))),
                "pixel_md5_all_identical": bool(len(set(pixel_hashes_a)) == 1),
                "pixel_md5_per_loop": pixel_hashes_a,
                "pairwise_cosines_off_diag": _summary_stats(off_diag_a),
            }

        # Aggregate item-level summary.
        if all_lengths_equal and per_ordinal:
            n_ordinals = len(per_ordinal)
            all_pixel_identical = all(o["pixel_md5_all_identical"] for o in per_ordinal)
            per_ord_cos_min = min(
                o["pairwise_cosines_off_diag"].get("min", 1.0) or 1.0
                for o in per_ordinal
            )
            per_ord_cos_max = max(
                o["pairwise_cosines_off_diag"].get("max", 1.0) or 1.0
                for o in per_ordinal
            )
            per_ord_cos_mean_of_means = sum(
                o["pairwise_cosines_off_diag"].get("mean", 1.0) or 1.0
                for o in per_ordinal
            ) / n_ordinals
            item_summary = {
                "n_ordinals_compared": int(n_ordinals),
                "n_loops_compared": int(len(present_loops)),
                "all_ordinals_pixel_md5_identical_across_loops": bool(all_pixel_identical),
                "any_ordinal_pixel_md5_identical_across_loops": bool(
                    any(o["pixel_md5_all_identical"] for o in per_ordinal)
                ),
                "n_ordinals_with_all_pixel_md5_identical": int(
                    sum(1 for o in per_ordinal if o["pixel_md5_all_identical"])
                ),
                "cosine_per_ordinal_off_diag_min": float(per_ord_cos_min),
                "cosine_per_ordinal_off_diag_max": float(per_ord_cos_max),
                "cosine_per_ordinal_off_diag_mean_of_means": float(per_ord_cos_mean_of_means),
            }
        else:
            item_summary = {
                "n_ordinals_compared": 0,
                "n_loops_compared": int(len(present_loops)),
                "reason": "close-up segment lengths differ across loops or no data",
            }

        report["per_item"][item_label] = {
            "viewing_position_id": int(vp),
            "loops_sampled": present_loops,
            "pose_match": pose_match_summary,
            "summary": item_summary,
            "apex": apex_block,
            "per_ordinal": per_ordinal,
        }

        print(f"[xloop] {item_label}: loops={present_loops}", flush=True)
        print(f"  pose_match: {pose_match_summary}", flush=True)
        print(f"  summary: {item_summary}", flush=True)
        if apex_block:
            print(f"  apex: n_unique_pixel_md5={apex_block['n_unique_pixel_md5']} "
                  f"identical={apex_block['pixel_md5_all_identical']} "
                  f"cosine={apex_block['pairwise_cosines_off_diag']}",
                  flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2))
    print(f"[xloop] wrote report: {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

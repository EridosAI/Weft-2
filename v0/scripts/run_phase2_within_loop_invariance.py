"""Diagnostic: within-loop input invariance check.

For Bed (vp=1) at Stage B loops {50, 75, 100}, compute pairwise pixel-MD5
hashes and pairwise DINOv2 cosines across the 11 close-up ordinal
positions within each single loop.

Note on substrate: the 11 close-up ordinals correspond to 11 *different*
agent poses along a 2 m perpendicular pass through the viewing position,
not 11 frames at the same pose. So the pairwise cosines within a single
loop reflect *pose-driven view changes* (parallax, occlusion shift) at
fixed scene state (RandomizeMaterials fires once per loop). The
reviewer's "Expected: bit-identical or very close (cosine ~1.0)"
hypothesis is the diagnostic question — the data answers whether
pose-driven view changes produce small cosine drops (consistent with
within-loop input near-invariance) or large drops (within-loop input
varies substantially with pose).

Output: results/inner_pam_v0/phase2_main/within_loop_invariance.json.
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
FRAMES_DIR = REPO_ROOT / "v0/data/phase2_frames"
ANNOT = REPO_ROOT / "v0/data/phase2_annotations.jsonl"
EMB = REPO_ROOT / "v0/data/phase2_embeddings/embeddings.npy"
OUT = REPO_ROOT / "v0/results/inner_pam_v0/phase2_main/within_loop_invariance.json"

# Stage B loops to check on Bed (vp=1, Bedroom control item).
SAMPLE_LOOPS = (50, 75, 100)
ITEM_VP = 1  # Bed


def _load_annotations(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open("r") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _frame_pixel_md5(frame_idx: int) -> str:
    p = FRAMES_DIR / f"frame_{frame_idx:08d}.png"
    arr = np.asarray(Image.open(p).convert("RGB"))
    return hashlib.md5(arr.tobytes()).hexdigest()


def main() -> int:
    annotations = _load_annotations(ANNOT)
    embeddings = np.load(EMB)

    report: dict[str, Any] = {
        "method": (
            "For Bed (vp=1) at each Stage B loop in {50, 75, 100}, gather the "
            "close-up frames in temporal order; compute pairwise pixel-MD5 "
            "hashes and pairwise DINOv2 cosines across the within-loop "
            "ordinals. The 11 ordinals are at 11 different agent poses along "
            "the close-up sweep, so cosines reflect pose-driven view changes "
            "at fixed scene state (RandomizeMaterials fires once per loop)."
        ),
        "item": "Bed",
        "viewing_position_id": int(ITEM_VP),
        "stage_b_loops_sampled": list(SAMPLE_LOOPS),
        "per_loop": {},
    }

    for loop in SAMPLE_LOOPS:
        frames: list[int] = []
        poses: list[tuple[float, float, float]] = []
        for a in annotations:
            if (
                int(a.get("viewing_position_id", -1)) == ITEM_VP
                and int(a.get("loop_index", -1)) == loop
                and a.get("phase_segment") == "close_up"
            ):
                frames.append(int(a["frame_idx"]))
                p = a["position"]
                poses.append((float(p["x"]), float(p["z"]), float(a["rotation_y"])))

        if not frames:
            report["per_loop"][str(loop)] = {"reason": "no close-up frames"}
            continue

        # Pixel hashes
        pixel_md5s = [_frame_pixel_md5(f) for f in frames]
        n_unique = len(set(pixel_md5s))

        # Embeddings + pairwise cosines
        emb = embeddings[frames].astype(np.float64)
        cos_mat = emb @ emb.T  # already L2-normed
        n = len(frames)
        off_diag: list[float] = []
        for i in range(n):
            for j in range(i + 1, n):
                off_diag.append(float(cos_mat[i, j]))

        # Pose deltas across the close-up sweep (between consecutive ordinals)
        pose_deltas: list[float] = []
        for i in range(1, len(poses)):
            dx = poses[i][0] - poses[i - 1][0]
            dz = poses[i][1] - poses[i - 1][1]
            pose_deltas.append((dx * dx + dz * dz) ** 0.5)

        # Pose total span (ordinal 0 to ordinal 10)
        x0, z0, _ = poses[0]
        xN, zN, _ = poses[-1]
        pose_total_span = ((xN - x0) ** 2 + (zN - z0) ** 2) ** 0.5

        loop_report = {
            "n_ordinals": int(n),
            "frame_idxs": [int(f) for f in frames],
            "pixel_md5_per_ordinal": pixel_md5s,
            "n_unique_pixel_md5": int(n_unique),
            "all_identical_pixel_md5": bool(n_unique == 1),
            "pairwise_cosines_off_diag": {
                "n_pairs": int(len(off_diag)),
                "min": float(min(off_diag)),
                "max": float(max(off_diag)),
                "mean": float(sum(off_diag) / len(off_diag)),
            },
            "pose_step_distances_m": [float(x) for x in pose_deltas],
            "pose_total_span_m": float(pose_total_span),
            "rotation_y_min_max": [
                float(min(r for _, _, r in poses)),
                float(max(r for _, _, r in poses)),
            ],
        }
        report["per_loop"][str(loop)] = loop_report
        print(f"[wloop] loop {loop}: n_ordinals={n} "
              f"n_unique_pixel_md5={n_unique} "
              f"cos_min={loop_report['pairwise_cosines_off_diag']['min']:.6f} "
              f"cos_max={loop_report['pairwise_cosines_off_diag']['max']:.6f} "
              f"pose_total_span={pose_total_span:.3f}m",
              flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2))
    print(f"[wloop] wrote report: {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Diagnostic: variance trajectory disaggregated by close-up ordinal position.

For each item ∈ {Bed=1, Television=5, Dresser=3, Sofa=4} and each loop in
{30, 50, 75, 100}, load the predictor from the checkpoint nearest to (and
no earlier than) the end of that loop, then for each close-up ordinal
position i ∈ {0..10} run the predictor on the W-frame window ending at
that ordinal's training step and capture mean log_var (averaged over K=16
prediction steps and embedding dim).

Reading: if a control item's variance change across loops is driven by
input variation, per-ordinal trajectories should vary in proportion to
per-ordinal input variation. If driven by cross-item coupling independent
of input, per-ordinal trajectories should be uniform. Compare control
items (Bed, TV) to perturbed items (Dresser, Sofa).

Output: results/inner_pam_v0/phase2_main/variance_by_ordinal.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path("/mnt/c/Users/Jason/Desktop/Eridos/Weft 2")
sys.path.insert(0, str(REPO_ROOT))

from src.config import EMBED_DIM, PREDICT_K, WINDOW_W  # noqa: E402
from src.predictor.inner_pam import InnerPAM  # noqa: E402

ANNOT_PATH = REPO_ROOT / "data/phase2_annotations.jsonl"
EMB_PATH = REPO_ROOT / "data/phase2_embeddings/embeddings.npy"
PHASE2_DIR = REPO_ROOT / "results/inner_pam_v0/phase2_main"
OUT_PATH = PHASE2_DIR / "variance_by_ordinal.json"

ITEMS = {"Bed": 1, "Television": 5, "Dresser": 3, "Sofa": 4}

# (loop_index, checkpoint_step) — checkpoint nearest to but past the end of
# the target loop. Loops are 0-indexed; loop X covers frames [X*360, X*360+359].
# ckpt_S contains predictor weights at training step S (where step = ordered
# position in the n_train stream; equivalent to frame_idx in the temporal
# sweep).
SAMPLE_POINTS = [
    (30, "ckpt_12000.pt"),   # loop 30 ends at frame 11519; ckpt at step 12000 ≈ loop 33
    (50, "ckpt_20000.pt"),   # loop 50 ends at frame 18359; ckpt at step 20000 ≈ loop 55
    (75, "ckpt_30000.pt"),   # loop 75 ends at frame 27359; ckpt at step 30000 ≈ loop 83
    (100, "ckpt_36360.pt"),  # loop 100 ends at frame 36359; ckpt at step 36360 ≈ loop 101 start
]


def _load_annotations() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with ANNOT_PATH.open("r") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _close_up_frames(
    annotations: list[dict[str, Any]],
    viewing_position_id: int,
    loop_index: int,
) -> list[int]:
    """Return frame indices of close-up frames at the given (item, loop), in
    temporal order. There are typically 11 ordinals (positions 0..10).
    """
    out: list[int] = []
    for a in annotations:
        if (
            int(a.get("viewing_position_id", -1)) == viewing_position_id
            and int(a.get("loop_index", -1)) == loop_index
            and a.get("phase_segment") == "close_up"
        ):
            out.append(int(a["frame_idx"]))
    return out


def main() -> int:
    if not ANNOT_PATH.is_file():
        print(f"[var-ord] FAIL: annotations not found: {ANNOT_PATH}", file=sys.stderr)
        return 1
    if not EMB_PATH.is_file():
        print(f"[var-ord] FAIL: embeddings not found: {EMB_PATH}", file=sys.stderr)
        return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[var-ord] device: {device}", flush=True)

    annotations = _load_annotations()
    print(f"[var-ord] {len(annotations)} annotations", flush=True)

    embeddings_np = np.load(EMB_PATH)
    print(f"[var-ord] embeddings: shape={embeddings_np.shape} dtype={embeddings_np.dtype}",
          flush=True)
    embeddings = torch.from_numpy(embeddings_np).to(device)

    report: dict[str, Any] = {
        "method": (
            "For each (item, loop) pair, load the predictor from the checkpoint "
            "nearest to (and not earlier than) the end of that loop, then for "
            "each close-up ordinal i (0..10) build the W-frame input window "
            "ending at frame_idx[ordinal=i] - K and the K-frame target, run "
            "predictor.forward(window), and capture mean(log_var across K steps)."
        ),
        "items": dict(ITEMS),
        "sample_points": [
            {"loop": l, "checkpoint": c} for l, c in SAMPLE_POINTS
        ],
        "window_w": int(WINDOW_W),
        "predict_k": int(PREDICT_K),
        "embed_dim": int(EMBED_DIM),
        "per_item": {},
    }

    for item_label, vp in ITEMS.items():
        per_loop: dict[str, Any] = {}
        for loop_idx, ckpt_name in SAMPLE_POINTS:
            ckpt_path = PHASE2_DIR / ckpt_name
            if not ckpt_path.is_file():
                print(f"[var-ord] SKIP {item_label} loop {loop_idx}: "
                      f"checkpoint not found {ckpt_path}", flush=True)
                continue

            close_up_frames = _close_up_frames(annotations, vp, loop_idx)
            if not close_up_frames:
                print(f"[var-ord] SKIP {item_label} loop {loop_idx}: "
                      f"no close-up frames", flush=True)
                continue

            # Load predictor
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            predictor = InnerPAM().to(device).eval()
            predictor.load_state_dict(ckpt["predictor_state"])

            per_ordinal: list[dict[str, Any]] = []
            with torch.no_grad():
                for ordinal_i, target_end in enumerate(close_up_frames):
                    # Window ends at target_end - K; targets are [target_end - K + 1, target_end].
                    window_end = target_end - PREDICT_K
                    window_start = window_end - WINDOW_W + 1
                    if window_start < 0:
                        per_ordinal.append({
                            "ordinal": int(ordinal_i),
                            "target_frame_idx": int(target_end),
                            "skipped": "window_start < 0 (insufficient history)",
                        })
                        continue
                    window = embeddings[window_start : window_end + 1].unsqueeze(0)
                    assert window.shape == (1, WINDOW_W, EMBED_DIM)
                    mean, log_var = predictor(window)
                    mean_log_var = float(log_var.mean().item())
                    per_step_log_var = [float(x) for x in log_var[0].tolist()]
                    per_ordinal.append({
                        "ordinal": int(ordinal_i),
                        "target_frame_idx": int(target_end),
                        "mean_log_var_over_K": mean_log_var,
                        "per_step_log_var": per_step_log_var,
                    })

            # Per-loop summary statistics
            mean_logvars = [r["mean_log_var_over_K"] for r in per_ordinal if "mean_log_var_over_K" in r]
            summary = {
                "n_ordinals": len(mean_logvars),
                "mean_log_var_across_ordinals": float(np.mean(mean_logvars)) if mean_logvars else None,
                "std_log_var_across_ordinals": float(np.std(mean_logvars)) if mean_logvars else None,
                "min_log_var_across_ordinals": float(np.min(mean_logvars)) if mean_logvars else None,
                "max_log_var_across_ordinals": float(np.max(mean_logvars)) if mean_logvars else None,
                "range_log_var_across_ordinals": (
                    float(np.max(mean_logvars) - np.min(mean_logvars))
                    if mean_logvars else None
                ),
            }
            per_loop[str(loop_idx)] = {
                "checkpoint": ckpt_name,
                "summary": summary,
                "per_ordinal": per_ordinal,
            }
            print(f"[var-ord] {item_label} loop {loop_idx} (ckpt={ckpt_name}): "
                  f"n={summary['n_ordinals']} "
                  f"mean={summary['mean_log_var_across_ordinals']:.4f} "
                  f"std={summary['std_log_var_across_ordinals']:.4f} "
                  f"range={summary['range_log_var_across_ordinals']:.4f}",
                  flush=True)

        report["per_item"][item_label] = {
            "viewing_position_id": int(vp),
            "per_loop": per_loop,
        }

    # Cross-loop trajectory summary per item: how does mean(log_var across ordinals)
    # change loop 30 → 100?
    print("[var-ord] --- cross-loop trajectory summary ---", flush=True)
    for item_label in ITEMS:
        loops_data = report["per_item"][item_label]["per_loop"]
        sorted_loops = sorted(loops_data.keys(), key=int)
        traj = []
        for l in sorted_loops:
            mean_lv = loops_data[l]["summary"].get("mean_log_var_across_ordinals")
            traj.append((int(l), mean_lv))
        print(f"  {item_label}: " + ", ".join(
            f"loop{l}={lv:.4f}" if lv is not None else f"loop{l}=NA"
            for l, lv in traj
        ), flush=True)
        report["per_item"][item_label]["trajectory_mean_log_var"] = [
            {"loop": l, "mean_log_var": lv} for l, lv in traj
        ]

    PHASE2_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, indent=2))
    print(f"[var-ord] wrote report: {OUT_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

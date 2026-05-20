#!/usr/bin/env python3
"""Run training for a single v1 arm across Stage A (or Stage B).

Wires together the spec §9 / instr §4 training loop and instr §4.6
per-checkpoint per-(item, ordinal) logging. Stage B resumes from the
end-of-Stage-A checkpoint per spec §9.2.

Usage examples:
  # Stage A on Primary, freshly initialised:
  python v1/scripts/run_arm_train.py --arm primary --stage A

  # Stage B on Ablation 1, resuming from end-of-Stage-A:
  python v1/scripts/run_arm_train.py --arm ablation1 --stage B \\
    --resume v1/results/inner_pam_v1/arm_ablation1/ckpt_end_stage_a.pt

Prerequisites:
  - PRE-A / PRE-B / PRE-C / PRE-D all PASS.
  - data/v1_shared/embeddings_stage_a.npy (and ..._stage_b.npy for Stage B).
  - data/v1_shared/annotations_stage_a.jsonl (resp. _stage_b.jsonl).
"""

from __future__ import annotations

import argparse
import json
import sys
from functools import partial
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from v1.src.config import (
    PATHS,
    PREDICT_K,
    STAGE_A_FRAME_BUDGET,
    WINDOW_W,
    get_v1_decoder_n_layers,
    stage_a_checkpoint_steps,
)
from v1.src.eval.per_item_ordinal_metrics import (
    build_canonical_pairs,
    evaluate_per_item_ordinal,
    load_annotations,
    write_per_item_ordinal_json,
)
from v1.src.predictor.inner_pam_v1_ablation1 import InnerPAM_v1_Ablation1
from v1.src.predictor.inner_pam_v1_ablation2 import InnerPAM_v1_Ablation2
from v1.src.predictor.inner_pam_v1_primary import InnerPAM_v1_Primary
from v1.src.trainer.online_trainer_v1 import OnlineTrainerV1, TrainerConfig


ARM_CLASSES = {
    "primary": InnerPAM_v1_Primary,
    "ablation1": InnerPAM_v1_Ablation1,
    "ablation2": InnerPAM_v1_Ablation2,
}


def construct_predictor(arm: str) -> torch.nn.Module:
    cls = ARM_CLASSES[arm]
    if arm == "ablation2":
        return cls()
    return cls(decoder_n_layers=get_v1_decoder_n_layers())


def output_dir_for_arm(arm: str) -> Path:
    return {
        "primary": PATHS.results_arm_primary,
        "ablation1": PATHS.results_arm_ablation1,
        "ablation2": PATHS.results_arm_ablation2,
    }[arm]


def main() -> int:
    p = argparse.ArgumentParser(description="Train a single v1 arm × stage")
    p.add_argument("--arm", required=True, choices=list(ARM_CLASSES.keys()))
    p.add_argument("--stage", required=True, choices=["A", "B"])
    p.add_argument(
        "--embeddings",
        type=Path,
        default=None,
        help="Stage-specific embeddings.npy path; defaults to PATHS.embeddings_stage_{a,b}.",
    )
    p.add_argument(
        "--annotations",
        type=Path,
        default=None,
        help="Stage-specific annotations.jsonl path.",
    )
    p.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume from checkpoint (required for Stage B).",
    )
    p.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Stage A defaults to STAGE_A_FRAME_BUDGET; Stage B defaults to max-loop cap.",
    )
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    p.add_argument("--git-commit", default=None)
    args = p.parse_args()

    arm = args.arm
    stage = args.stage
    device = torch.device(args.device)

    # Path resolution.
    embeddings_path = args.embeddings or (
        PATHS.embeddings_stage_a if stage == "A" else PATHS.embeddings_stage_b
    )
    annotations_path = args.annotations or (
        PATHS.annotations_stage_a if stage == "A" else PATHS.annotations_stage_b
    )

    print(f"[train] arm={arm} stage={stage} device={device}")
    print(f"[train] embeddings: {embeddings_path}")
    print(f"[train] annotations: {annotations_path}")

    embeddings = np.load(embeddings_path)
    annotations = load_annotations(annotations_path)
    pairs = build_canonical_pairs(annotations)
    print(f"[train] embeddings shape: {embeddings.shape}; canonical pairs: {len(pairs)}")

    arm_output_dir = output_dir_for_arm(arm)
    arm_output_dir.mkdir(parents=True, exist_ok=True)

    # Construct + (optionally) resume predictor.
    torch.manual_seed(0)
    predictor = construct_predictor(arm)

    # Canonical checkpoint schedule.
    if stage == "A":
        checkpoint_steps = stage_a_checkpoint_steps()
        # max_frames defaults to STAGE_A_FRAME_BUDGET.
        n_frames = args.max_frames or STAGE_A_FRAME_BUDGET
    else:
        # Stage B: every 10k steps (instr §4.5 standard cadence). Maximum
        # bound is set by --max-frames (caller may compute the
        # signal-stability-calibrated end externally).
        n_frames = args.max_frames or embeddings.shape[0]
        # Approximate checkpoint steps: 10k thereafter starting from
        # end-of-Stage-A.
        if args.resume is None:
            raise SystemExit("Stage B requires --resume from end-of-Stage-A checkpoint")
        # Note: the canonical Stage B checkpoint set is approximate here; in
        # practice the verdict-evaluation only uses the end-of-Stage-B checkpoint.
        start = STAGE_A_FRAME_BUDGET
        end = start + n_frames
        checkpoint_steps = tuple(range(start, end, 10_000)) + (end - 1,)

    # Per-(item, ordinal) callback (instr §4.6).
    def callback(trainer: OnlineTrainerV1, step: int) -> None:
        records = evaluate_per_item_ordinal(
            trainer.predictor, trainer.embeddings, pairs, trainer.device
        )
        write_per_item_ordinal_json(
            records,
            arm_output_dir / f"per_item_ordinal_{step}.json",
            step=step,
            arm=arm,
            stage=stage,
        )

    config = TrainerConfig(
        arm_name=arm,
        stage=stage,
        output_dir=arm_output_dir,
        checkpoint_steps=tuple(int(s) for s in checkpoint_steps),
        final_step=int(checkpoint_steps[-1]),
        git_commit=args.git_commit,
        per_item_ordinal_callback=callback,
    )

    if args.resume is not None:
        print(f"[train] resuming from {args.resume}")
        trainer = OnlineTrainerV1.from_checkpoint(
            predictor=predictor,
            ckpt_path=args.resume,
            embeddings=embeddings,
            config=config,
            device=device,
        )
    else:
        trainer = OnlineTrainerV1(
            predictor=predictor,
            embeddings=embeddings,
            config=config,
            device=device,
        )

    print(f"[train] running n_frames={n_frames} from step {trainer.start_step}")
    trainer.run(n_frames=n_frames)
    print(f"[train] done; final step {trainer.current_step}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

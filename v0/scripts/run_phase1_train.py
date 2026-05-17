"""Phase 1 main training run.

Loads the precomputed DINOv2 embeddings at `data/dinov2_embeddings/embeddings.npy`
(100,000 frames; full stream including transit). Trains the Inner PAM predictor
online, single-pass, in temporal order. Reserves the last HELD_OUT_LOOPS for
evaluation. Writes checkpoints at the cadence in `config.PHASE1_CKPT_EVERY`.

CC-launched via nohup per CODING_STANDARDS.md §5.2:
  nohup python3.12 -u scripts/run_phase1_train.py \
      > logs/phase1_main_$(date +%Y%m%d_%H%M%S).log 2>&1 &
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

# Repo-root import path.
REPO_ROOT = Path("/mnt/c/Users/Jason/Desktop/Eridos/Weft 2")
sys.path.insert(0, str(REPO_ROOT))

from v0.src.config import (  # noqa: E402
    HELD_OUT_LOOPS,
    PATHS,
    PHASE1,
    PHASE1_CKPT_EVERY,
    SEED_PREDICTOR_INIT,
    TAU_CALIB_END_STEP,
    TAU_CALIB_START_STEP,
)
from v0.src.eval.probes import compute_held_out_boundary  # noqa: E402
from v0.src.memory.memory_bank import MemoryBank  # noqa: E402
from v0.src.mixing.recall_mixer import compute_tau_from_confidences  # noqa: E402
from v0.src.predictor.inner_pam import InnerPAM  # noqa: E402
from v0.src.trainer.online_trainer import OnlineTrainer, TrainerConfig  # noqa: E402


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def _load_annotations(path: Path) -> list[dict]:
    out = []
    with path.open("r") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path, default=PHASE1.results_main)
    parser.add_argument("--ckpt_every", type=int, default=PHASE1_CKPT_EVERY)
    parser.add_argument("--max_steps", type=int, default=-1,
                        help="optional upper bound; -1 == train through stream end")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    git = _git_commit()
    print(f"[phase1] git commit: {git}", flush=True)
    device = torch.device(args.device if torch.cuda.is_available() or args.device != "cuda" else "cpu")
    print(f"[phase1] device: {device}", flush=True)

    # ---- Load data --------------------------------------------------------
    embeddings = np.load(PATHS.embeddings)
    annotations = _load_annotations(PATHS.annotations_phase1)
    assert embeddings.shape[0] == len(annotations)
    print(f"[phase1] embeddings: shape={embeddings.shape}, dtype={embeddings.dtype}",
          flush=True)

    held_out_start, held_out_end = compute_held_out_boundary(
        annotations, HELD_OUT_LOOPS
    )
    n_train = held_out_start
    print(f"[phase1] held-out region: frames [{held_out_start}, {held_out_end}); "
          f"n_train={n_train}", flush=True)

    # ---- Construct predictor + bank --------------------------------------
    torch.manual_seed(SEED_PREDICTOR_INIT)
    predictor = InnerPAM()
    bank = MemoryBank()
    print(f"[phase1] predictor trainable params: "
          f"{sum(p.numel() for p in predictor.parameters() if p.requires_grad)}",
          flush=True)

    # Checkpoint schedule: every PHASE1_CKPT_EVERY steps up to either the
    # max-steps cap or the natural training horizon (n_train).
    final_step = (
        int(args.max_steps) if args.max_steps > 0 else n_train - 1
    )
    ckpt_schedule = tuple(
        s for s in range(args.ckpt_every, final_step, args.ckpt_every)
    )

    cfg = TrainerConfig(
        phase_name="phase1",
        perturbation_tag="none",
        output_dir=args.output_dir,
        checkpoint_steps=ckpt_schedule,
        final_step=final_step,
        git_commit=git,
    )

    trainer = OnlineTrainer(
        predictor=predictor,
        bank=bank,
        embeddings=embeddings,
        annotations=annotations,
        n_train=n_train,
        device=device,
        cfg=cfg,
    )

    # ---- Init-time checks (§4.7) ------------------------------------------
    init_report = trainer.assert_init_invariants()
    (args.output_dir / "init_report.json").write_text(json.dumps(init_report, indent=2))
    print(f"[phase1] init-time checks: trainable_params="
          f"{init_report['trainable_params']} "
          f"(within {int(init_report['param_count_tolerance_frac']*100)}% tolerance: "
          f"{init_report['param_count_within_tolerance']})", flush=True)
    if not init_report["param_count_within_tolerance"]:
        print(f"[phase1] FAIL: param count outside tolerance", file=sys.stderr)
        return 2

    # ---- Train ------------------------------------------------------------
    print(f"[phase1] training to step {final_step}; "
          f"checkpoint every {args.ckpt_every} steps", flush=True)
    summary = trainer.train()
    print(f"[phase1] training done: {summary['elapsed_seconds']:.1f}s, "
          f"{summary['n_gradient_steps_actual']} gradient steps", flush=True)

    # ---- τ calibration ---------------------------------------------------
    if summary["n_gradient_steps_actual"] >= TAU_CALIB_END_STEP:
        tau = compute_tau_from_confidences(
            trainer.confidences,
            start_step=TAU_CALIB_START_STEP,
            end_step=TAU_CALIB_END_STEP,
            step_indices=trainer.step_indices,
        )
        (args.output_dir / "tau_calibration.json").write_text(
            json.dumps(
                {
                    "tau": tau,
                    "calibration_window_steps": [
                        int(TAU_CALIB_START_STEP), int(TAU_CALIB_END_STEP)
                    ],
                    "n_confidence_samples_in_window": int(
                        ((trainer.step_indices >= TAU_CALIB_START_STEP)
                         & (trainer.step_indices < TAU_CALIB_END_STEP)).sum()
                    ),
                    "git_commit": git,
                },
                indent=2,
            )
        )
        print(f"[phase1] tau calibrated to {tau:.6f}", flush=True)
    else:
        print(f"[phase1] insufficient steps for tau calibration "
              f"({summary['n_gradient_steps_actual']} < {TAU_CALIB_END_STEP})",
              file=sys.stderr)

    # ---- Final summary ---------------------------------------------------
    (args.output_dir / "training_summary.json").write_text(
        json.dumps(
            {
                "phase": "phase1",
                "final_step": final_step,
                "n_train": n_train,
                "held_out_region": [held_out_start, held_out_end],
                "elapsed_seconds": summary["elapsed_seconds"],
                "n_gradient_steps_actual": summary["n_gradient_steps_actual"],
                "git_commit": git,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

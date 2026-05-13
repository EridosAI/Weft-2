"""Phase 2 main training run with in-flight transition diagnostic.

Loads Phase 2 embeddings + annotations, trains InnerPAM online single-pass
starting from a freshly-initialised predictor (Phase 1 discarded as
substrate-degenerate; session-4 disposition), and runs the §8.7a Stage A
→ Stage B in-flight transition diagnostic. If any of G2.T1 / G2.T2 / G2.T3
trips, the trainer stops at the end of the post-onset window and writes a
marker file for the launching session to detect.

Held-out: last HELD_OUT_LOOPS loops of the Phase 2 stream.
Checkpoint cadence: PHASE_2_3_CKPT_STEPS (phase-relative steps), plus end.

This script is committed for execution in a follow-on session once the
experiment-chat reviews and signs off on the Phase 2 collection +
encoding outputs (session-5 STOP point).

Usage:
  nohup python3.12 -u scripts/run_phase2_train.py \\
      > logs/phase2_main_$(date +%Y%m%d_%H%M%S).log 2>&1 &
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path("/mnt/c/Users/Jason/Desktop/Eridos/Weft 2")
sys.path.insert(0, str(REPO_ROOT))

from src.config import (  # noqa: E402
    HELD_OUT_LOOPS,
    PATHS,
    PHASE2,
    PHASE_2_3_CKPT_STEPS,
    SEED_PREDICTOR_INIT,
    TAU_CALIB_END_STEP,
    TAU_CALIB_START_STEP,
    TRANSITION_BASELINE_LOOPS,
    TRANSITION_CONTROL_DRIFT_MAX,
    TRANSITION_CONTROL_ITEMS,
    TRANSITION_LOG_VAR_WIDENING_MIN,
    TRANSITION_LOSS_SPIKE_RATIO,
    TRANSITION_PERTURBED_ITEMS,
    TRANSITION_POST_ONSET_LOOPS,
)
from src.eval.probes import compute_held_out_boundary  # noqa: E402
from src.memory.memory_bank import MemoryBank  # noqa: E402
from src.mixing.recall_mixer import compute_tau_from_confidences  # noqa: E402
from src.predictor.inner_pam import InnerPAM  # noqa: E402
from src.trainer.online_trainer import OnlineTrainer, TrainerConfig  # noqa: E402


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
    out: list[dict] = []
    with path.open("r") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path, default=PHASE2.results_main)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    git = _git_commit()
    print(f"[phase2] git commit: {git}", flush=True)
    device = torch.device(
        args.device if torch.cuda.is_available() or args.device != "cuda" else "cpu"
    )
    print(f"[phase2] device: {device}", flush=True)

    # ---- Load data --------------------------------------------------------
    if not PATHS.phase2_embeddings.is_file():
        print(f"[phase2] FAIL: embeddings not found at {PATHS.phase2_embeddings}",
              file=sys.stderr)
        return 1
    if not PATHS.phase2_annotations.is_file():
        print(f"[phase2] FAIL: annotations not found at {PATHS.phase2_annotations}",
              file=sys.stderr)
        return 1

    embeddings = np.load(PATHS.phase2_embeddings)
    annotations = _load_annotations(PATHS.phase2_annotations)
    assert embeddings.shape[0] == len(annotations), (
        f"phase2 embeddings rows {embeddings.shape[0]} != "
        f"annotations rows {len(annotations)}"
    )
    print(f"[phase2] embeddings: shape={embeddings.shape}, dtype={embeddings.dtype}",
          flush=True)

    held_out_start, held_out_end = compute_held_out_boundary(
        annotations, HELD_OUT_LOOPS
    )
    n_train = held_out_start
    print(f"[phase2] held-out region: frames [{held_out_start}, {held_out_end}); "
          f"n_train={n_train}", flush=True)

    # ---- Construct predictor + bank (fresh; Phase 1 discarded). -----------
    torch.manual_seed(SEED_PREDICTOR_INIT)
    predictor = InnerPAM()
    bank = MemoryBank()
    print(f"[phase2] predictor trainable params: "
          f"{sum(p.numel() for p in predictor.parameters() if p.requires_grad)}",
          flush=True)

    final_step = n_train - 1
    ckpt_schedule = tuple(
        s for s in PHASE_2_3_CKPT_STEPS if 0 < s < final_step
    )

    transition_path = args.output_dir / "transition_diagnostic.json"
    cfg = TrainerConfig(
        phase_name=PHASE2.name,
        perturbation_tag=PHASE2.perturbation_tag,
        output_dir=args.output_dir,
        checkpoint_steps=ckpt_schedule,
        final_step=final_step,
        git_commit=git,
        # In-flight transition diagnostic (instr §8.7a).
        transition_diagnostic_enabled=True,
        transition_diagnostic_path=transition_path,
        transition_perturbed_items=TRANSITION_PERTURBED_ITEMS,
        transition_control_items=TRANSITION_CONTROL_ITEMS,
        transition_baseline_loops=TRANSITION_BASELINE_LOOPS,
        transition_post_onset_loops=TRANSITION_POST_ONSET_LOOPS,
        transition_loss_spike_ratio=TRANSITION_LOSS_SPIKE_RATIO,
        transition_log_var_widening_min=TRANSITION_LOG_VAR_WIDENING_MIN,
        transition_control_drift_max=TRANSITION_CONTROL_DRIFT_MAX,
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

    # ---- Init-time checks ------------------------------------------------
    init_report = trainer.assert_init_invariants()
    (args.output_dir / "init_report.json").write_text(json.dumps(init_report, indent=2))
    if not init_report["param_count_within_tolerance"]:
        print(f"[phase2] FAIL: param count outside tolerance", file=sys.stderr)
        return 2

    # ---- Train ----------------------------------------------------------
    print(f"[phase2] training to step {final_step}; "
          f"ckpts at {ckpt_schedule}", flush=True)
    summary = trainer.train()
    print(f"[phase2] training done: {summary['elapsed_seconds']:.1f}s, "
          f"{summary['n_gradient_steps_actual']} gradient steps, "
          f"transition_gate_tripped={summary['transition_diagnostic_gate_tripped']}",
          flush=True)

    # ---- τ calibration (carries forward — Phase 2 uses Phase 2 window). --
    # Per instr §5.2 the τ calibration is on Phase 1 in the original design;
    # with Phase 1 discarded as substrate-degenerate, Phase 2 inherits the
    # role. The 5k-10k window applies to Phase 2's own step count.
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
                    "git_commit": git,
                },
                indent=2,
            )
        )
        print(f"[phase2] tau calibrated to {tau:.6f}", flush=True)

    (args.output_dir / "training_summary.json").write_text(
        json.dumps(
            {
                "phase": PHASE2.name,
                "final_step": final_step,
                "n_train": n_train,
                "held_out_region": [held_out_start, held_out_end],
                "elapsed_seconds": summary["elapsed_seconds"],
                "n_gradient_steps_actual": summary["n_gradient_steps_actual"],
                "transition_diagnostic_gate_tripped": bool(
                    summary["transition_diagnostic_gate_tripped"]
                ),
                "transition_diagnostic_trip_record": summary[
                    "transition_diagnostic_trip_record"
                ],
                "git_commit": git,
            },
            indent=2,
        )
    )

    # Non-zero exit if the in-flight diagnostic tripped.
    return 3 if summary["transition_diagnostic_gate_tripped"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

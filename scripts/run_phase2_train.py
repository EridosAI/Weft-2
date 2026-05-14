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
    parser.add_argument(
        "--max_loops", type=int, default=None,
        help=(
            "Halt training as soon as a step's loop_index exceeds this value. "
            "Used in session 6 to stop at loop 35 (transition-diagnostic review)."
        ),
    )
    parser.add_argument(
        "--resume_from", type=Path, default=None,
        help=(
            "Resume training from this `.pt` checkpoint (and the matching "
            "ckpt_<step>/ bank directory next to it). Disables the in-flight "
            "transition diagnostic on the assumption that its evaluation "
            "window has already passed."
        ),
    )
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

    # ---- Construct predictor + bank (fresh or resumed). ------------------
    torch.manual_seed(SEED_PREDICTOR_INIT)
    predictor = InnerPAM()
    resume_step: int | None = None
    resume_optimizer_state: dict | None = None
    if args.resume_from is not None:
        if not args.resume_from.is_file():
            print(f"[phase2] FAIL: resume_from not a file: {args.resume_from}",
                  file=sys.stderr)
            return 1
        ckpt = torch.load(args.resume_from, map_location=device, weights_only=False)
        predictor.load_state_dict(ckpt["predictor_state"])
        resume_step = int(ckpt["step"])
        resume_optimizer_state = ckpt["optimizer_state"]
        try:
            # torch.load(map_location=device) puts the RNG byte-tensor on the
            # GPU; torch.set_rng_state needs a ByteTensor on CPU.
            torch.set_rng_state(ckpt["rng_torch"].cpu().byte())
            np.random.set_state(ckpt["rng_numpy"])
        except Exception as exc:
            print(f"[phase2] WARN: failed to restore RNG state: {exc}",
                  file=sys.stderr)
        bank_dir = args.resume_from.parent / f"ckpt_{resume_step}"
        if not bank_dir.is_dir():
            print(f"[phase2] FAIL: bank dir not found alongside checkpoint: "
                  f"{bank_dir}", file=sys.stderr)
            return 1
        bank = MemoryBank.load(bank_dir)
        print(f"[phase2] resumed from step={resume_step}, "
              f"bank_size={bank.size()}, ckpt={args.resume_from}",
              flush=True)
    else:
        bank = MemoryBank()
    print(f"[phase2] predictor trainable params: "
          f"{sum(p.numel() for p in predictor.parameters() if p.requires_grad)}",
          flush=True)

    final_step = n_train - 1
    ckpt_schedule = tuple(
        s for s in PHASE_2_3_CKPT_STEPS if 0 < s < final_step
    )

    # Transition diagnostic:
    #   Fresh run: enabled in session-6 mode (auto-trips at first-step-of-
    #     loop-(post_end+1) per the original §8.7a).
    #   Resume:    enabled in extended mode (per session-7 restructuring) —
    #     loads the prior diagnostic JSON's per_loop into memory, skips the
    #     session-6 boundary auto-trip, and re-evaluates G2.T1/T3 at every
    #     checkpoint over the extended window. G2.T2 is evaluated post-hoc
    #     against the loop-100 trajectory per the three-part criterion.
    diag_enabled = True
    extended_mode = (resume_step is not None)
    transition_path = args.output_dir / "transition_diagnostic.json"
    cfg = TrainerConfig(
        phase_name=PHASE2.name,
        perturbation_tag=PHASE2.perturbation_tag,
        output_dir=args.output_dir,
        checkpoint_steps=ckpt_schedule,
        final_step=final_step,
        git_commit=git,
        max_loops=args.max_loops,
        resume_step=resume_step,
        # In-flight transition diagnostic (instr §8.7a).
        transition_diagnostic_enabled=diag_enabled,
        transition_diagnostic_extended_mode=extended_mode,
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
        resume_optimizer_state=resume_optimizer_state,
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
    # role. The 5k-10k window applies to Phase 2's own step count. On resume,
    # the calibration window has already been passed (the prior session
    # wrote tau_calibration.json at the checkpoint step ≥ 10k), so skip
    # re-calibration — the resumed trainer's confidences log starts at
    # resume_step+1 ≫ TAU_CALIB_END_STEP and would produce an empty window.
    if (
        resume_step is None
        and summary["n_gradient_steps_actual"] >= TAU_CALIB_END_STEP
    ):
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

    summary_path = args.output_dir / "training_summary.json"
    if summary_path.exists():
        # Don't clobber an earlier session's summary on resume — keep the
        # historical record by writing a numbered sidecar instead.
        i = 1
        while (args.output_dir / f"training_summary.session{i}.json").exists():
            i += 1
        summary_path = args.output_dir / f"training_summary.session{i}.json"
    summary_path.write_text(
        json.dumps(
            {
                "phase": PHASE2.name,
                "final_step": final_step,
                "n_train": n_train,
                "held_out_region": [held_out_start, held_out_end],
                "elapsed_seconds": summary["elapsed_seconds"],
                "n_gradient_steps_actual": summary["n_gradient_steps_actual"],
                "last_step_done": summary.get("last_step_done"),
                "last_loop_seen": summary.get("last_loop_seen"),
                "max_loops": summary.get("max_loops"),
                "stopped_at_max_loops": bool(summary.get("stopped_at_max_loops")),
                "resume_step": resume_step,
                "resume_from": str(args.resume_from) if args.resume_from else None,
                "transition_diagnostic_enabled": bool(diag_enabled),
                "transition_diagnostic_extended_mode": bool(extended_mode),
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
    print(f"[phase2] wrote summary: {summary_path}", flush=True)

    # Non-zero exit if the in-flight diagnostic tripped.
    return 3 if summary["transition_diagnostic_gate_tripped"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Phase 1 shuffle control (C2 — instr §6.3, §7.5).

Same predictor architecture, same loss, same optimizer; the only change is
that the *training-step order* is a seeded permutation of the unshuffled
stream. The bank still ingests in true temporal order so the shuffle
isolates training-signal shuffling, not bank-state shuffling. Held-out
evaluation uses the unshuffled held-out region.
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
    PHASE1,
    PHASE1_CKPT_EVERY,
    SEED_PREDICTOR_INIT,
    SEED_SHUFFLE_PERMUTATION,
)
from src.eval.probes import compute_held_out_boundary  # noqa: E402
from src.memory.memory_bank import MemoryBank  # noqa: E402
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
    out = []
    with path.open("r") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path, default=PHASE1.results_shuffle)
    parser.add_argument("--ckpt_every", type=int, default=PHASE1_CKPT_EVERY)
    parser.add_argument("--shuffle_seed", type=int, default=SEED_SHUFFLE_PERMUTATION)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    git = _git_commit()
    device = torch.device(args.device if torch.cuda.is_available() or args.device != "cuda" else "cpu")

    embeddings = np.load(PATHS.embeddings)
    annotations = _load_annotations(PATHS.annotations_phase1)
    held_out_start, held_out_end = compute_held_out_boundary(annotations, HELD_OUT_LOOPS)
    n_train = held_out_start

    torch.manual_seed(SEED_PREDICTOR_INIT)
    predictor = InnerPAM()
    bank = MemoryBank()

    final_step = int(args.max_steps) if args.max_steps > 0 else n_train - 1
    ckpt_schedule = tuple(
        s for s in range(args.ckpt_every, final_step, args.ckpt_every)
    )

    cfg = TrainerConfig(
        phase_name="phase1_shuffle",
        perturbation_tag="none",
        output_dir=args.output_dir,
        checkpoint_steps=ckpt_schedule,
        final_step=final_step,
        shuffle_seed=int(args.shuffle_seed),
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
    init_report = trainer.assert_init_invariants()
    (args.output_dir / "init_report.json").write_text(json.dumps(init_report, indent=2))
    if not init_report["param_count_within_tolerance"]:
        print(f"[phase1_shuffle] FAIL: param count outside tolerance", file=sys.stderr)
        return 2

    summary = trainer.train()
    (args.output_dir / "training_summary.json").write_text(
        json.dumps(
            {
                "phase": "phase1_shuffle",
                "final_step": final_step,
                "n_train": n_train,
                "held_out_region": [held_out_start, held_out_end],
                "elapsed_seconds": summary["elapsed_seconds"],
                "n_gradient_steps_actual": summary["n_gradient_steps_actual"],
                "shuffle_seed": int(args.shuffle_seed),
                "git_commit": git,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

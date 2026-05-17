"""Phase 1 shuffle control (C2 — instr §6.3, §7.5, spec §10.1).

Spec-correct version (replaces the prior visit-order-only permutation, which
matched the literal §7.5 wording but contradicted spec §10.1's rationale —
see HANDOFF session-2 entry).

The training portion of the embedding stream (frames [0, n_train)) is permuted
once at the start of training using `np.random.default_rng(SEED_SHUFFLE_PERMUTATION)`.
After permutation, the trainer builds windows from contiguous positions in the
permuted stream — but those contiguous positions now hold random unrelated
frames, so the window has no temporal coherence and the K-step target is
unrelated to the window. Temporal structure is destroyed at the source, as
spec §10.1 requires. The predictor cannot learn path structure because there
is no path.

Annotations are permuted in lockstep so each bank entry's metadata reflects
the original frame the embedding came from.

The held-out region [n_train, N) is NOT permuted; held-out evaluation uses
the unshuffled held-out region per instr §7.5.
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

from v0.src.config import (  # noqa: E402
    HELD_OUT_LOOPS,
    PATHS,
    PHASE1,
    PHASE1_CKPT_EVERY,
    SEED_PREDICTOR_INIT,
    SEED_SHUFFLE_PERMUTATION,
)
from v0.src.eval.probes import compute_held_out_boundary  # noqa: E402
from v0.src.memory.memory_bank import MemoryBank  # noqa: E402
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


def _apply_permutation(
    embeddings: np.ndarray, annotations: list[dict], n_train: int, seed: int,
) -> tuple[np.ndarray, list[dict], np.ndarray]:
    """Permute the first n_train rows in lockstep; held-out region untouched."""
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(n_train).astype(np.int64)
    permuted_emb = embeddings.copy()
    permuted_emb[:n_train] = embeddings[perm]
    permuted_anns = list(annotations)
    for new_pos, src_pos in enumerate(perm):
        permuted_anns[new_pos] = annotations[int(src_pos)]
    return permuted_emb, permuted_anns, perm


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

    # Spec-correct shuffle: permute the embedding+annotation stream at source.
    permuted_emb, permuted_anns, perm = _apply_permutation(
        embeddings, annotations, n_train, int(args.shuffle_seed),
    )
    print(f"[phase1_shuffle] permutation seed={args.shuffle_seed}, "
          f"first 5 perm indices: {perm[:5].tolist()}", flush=True)
    print(f"[phase1_shuffle] embeddings shape={permuted_emb.shape}; "
          f"held-out region [{held_out_start}, {held_out_end}) preserved",
          flush=True)

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
        git_commit=git,
    )

    trainer = OnlineTrainer(
        predictor=predictor,
        bank=bank,
        embeddings=permuted_emb,
        annotations=permuted_anns,
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
                "shuffle_kind": "embedding_stream_permutation_at_source",
                "spec_section": "§10.1, §6.3 (temporal structure destroyed)",
                "final_step": final_step,
                "n_train": n_train,
                "held_out_region": [held_out_start, held_out_end],
                "elapsed_seconds": summary["elapsed_seconds"],
                "n_gradient_steps_actual": summary["n_gradient_steps_actual"],
                "shuffle_seed": int(args.shuffle_seed),
                "first_5_perm_indices": perm[:5].tolist(),
                "git_commit": git,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Trainer tests for v1 (instr §4, spec §9)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from v1.src.config import EMBED_DIM, PREDICT_K, WINDOW_W
from v1.src.predictor.inner_pam_v1_primary import InnerPAM_v1_Primary
from v1.src.trainer.online_trainer_v1 import (
    OnlineTrainerV1,
    TrainerConfig,
    TrainingStopped,
)


DECODER_N_LAYERS = 2


def _make_l2_norm_embeddings(n: int, d: int = EMBED_DIM, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    arr = rng.standard_normal((n, d)).astype(np.float32)
    arr = arr / np.linalg.norm(arr, axis=1, keepdims=True)
    return arr


def test_trainer_rejects_non_normalised_embeddings(tmp_path: Path):
    n = 200
    bad = np.ones((n, EMBED_DIM), dtype=np.float32)  # norms = sqrt(d), not 1
    model = InnerPAM_v1_Primary(decoder_n_layers=DECODER_N_LAYERS)
    cfg = TrainerConfig(
        arm_name="primary",
        stage="A",
        output_dir=tmp_path,
        checkpoint_steps=(),
        final_step=n - 1,
    )
    with pytest.raises(TrainingStopped, match="L2 norm"):
        OnlineTrainerV1(predictor=model, embeddings=bad, config=cfg)


def test_trainer_skip_until_W_contract(tmp_path: Path):
    """First training step should be at t = W - 1; earlier steps return None."""
    n = 200
    embeds = _make_l2_norm_embeddings(n)
    model = InnerPAM_v1_Primary(decoder_n_layers=DECODER_N_LAYERS)
    cfg = TrainerConfig(
        arm_name="primary",
        stage="A",
        output_dir=tmp_path,
        checkpoint_steps=(),
        final_step=n - 1,
    )
    trainer = OnlineTrainerV1(
        predictor=model, embeddings=embeds, config=cfg, device=torch.device("cpu")
    )
    for t in range(WINDOW_W - 1):
        assert trainer._build_window_and_target(t) == (None, None)
    window, target = trainer._build_window_and_target(WINDOW_W - 1)
    assert window is not None and window.shape == (1, WINDOW_W, EMBED_DIM)
    assert target is not None and target.shape == (1, PREDICT_K, EMBED_DIM)


def test_trainer_runs_few_steps(tmp_path: Path):
    """Smoke test: a small training run completes without error."""
    n = 100
    embeds = _make_l2_norm_embeddings(n)
    model = InnerPAM_v1_Primary(decoder_n_layers=DECODER_N_LAYERS)
    cfg = TrainerConfig(
        arm_name="primary",
        stage="A",
        output_dir=tmp_path,
        checkpoint_steps=(WINDOW_W + 4,),  # one checkpoint mid-run
        final_step=n - 1,
    )
    trainer = OnlineTrainerV1(
        predictor=model, embeddings=embeds, config=cfg, device=torch.device("cpu")
    )
    trainer.run(n_frames=n)
    # Checkpoint should exist
    assert (tmp_path / f"ckpt_{WINDOW_W + 4}.pt").exists()
    assert (tmp_path / f"checkpoint_{WINDOW_W + 4}.json").exists()
    # End-of-stage canonical alias
    assert (tmp_path / "ckpt_end_stage_a.pt").exists()
    # JSON parses
    json.loads((tmp_path / f"checkpoint_{WINDOW_W + 4}.json").read_text())


def test_trainer_resume(tmp_path: Path):
    """Trainer can resume from a checkpoint and continue training."""
    n = 80
    embeds = _make_l2_norm_embeddings(n)
    model = InnerPAM_v1_Primary(decoder_n_layers=DECODER_N_LAYERS)
    cfg = TrainerConfig(
        arm_name="primary",
        stage="A",
        output_dir=tmp_path,
        checkpoint_steps=(WINDOW_W + 5,),
        final_step=WINDOW_W + 5,
    )
    trainer = OnlineTrainerV1(
        predictor=model, embeddings=embeds, config=cfg, device=torch.device("cpu")
    )
    trainer.run(n_frames=WINDOW_W + 6)
    ckpt_path = tmp_path / f"ckpt_{WINDOW_W + 5}.pt"
    assert ckpt_path.exists()

    model2 = InnerPAM_v1_Primary(decoder_n_layers=DECODER_N_LAYERS)
    cfg2 = TrainerConfig(
        arm_name="primary",
        stage="B",
        output_dir=tmp_path / "stage_b",
        checkpoint_steps=(),
        final_step=n - 1,
    )
    trainer2 = OnlineTrainerV1.from_checkpoint(
        predictor=model2,
        ckpt_path=ckpt_path,
        embeddings=embeds,
        config=cfg2,
        device=torch.device("cpu"),
    )
    assert trainer2.start_step == WINDOW_W + 5 + 1
    trainer2.run(n_frames=n - trainer2.start_step)
    assert (tmp_path / "stage_b" / "ckpt_end_stage_b.pt").exists()

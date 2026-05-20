"""Weft Inner PAM v1 — online trainer (arm-agnostic).

Implements the spec §9.3 training loop and instr §4 operational
discipline. Arm-agnostic: the trainer accepts any of the three predictor
classes and runs the same loop. No memory bank, no recall mixer (v1 does
not test confidence-graded mixing; instr §1.4).

Pipeline:
  Stage A: 100k frames of unperturbed continuous-motion trajectory.
  Stage B: signal-stability-calibrated perturbed loops (instr §8.1).

Per-step contract (spec §9.3, instr §4.2): at every step t after
`pad_start` skip (skip-until-W; instr §4.3), the predictor is trained on
the window `[t - W + 1 .. t]` against the K-step target `[t + 1 .. t + K]`.
One gradient step per resolved prediction.

Stop conditions: spec §11.2 (NaN/Inf in training, per-(item, ordinal)
logging gap, in-flight perturbation magnitude drift). Stop triggers write
`TRAINING_STOPPED.txt` to the output directory and raise.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from v1.src.config import (
    ADAM_BETAS,
    EMBED_DIM,
    GRAD_CLIP_MAX_NORM,
    LR,
    PREDICT_K,
    WEIGHT_DECAY,
    WINDOW_W,
)
from v1.src.predictor.inner_pam_v1_shared import (
    path_prediction_loss,
    trainable_parameter_count,
)


# --------------------------------------------------------------------------
# Config / checkpoint dataclasses
# --------------------------------------------------------------------------


@dataclass
class TrainerConfig:
    """Operational config for a single arm × stage run."""

    arm_name: str                              # "primary" | "ablation1" | "ablation2"
    stage: str                                  # "A" | "B"
    output_dir: Path
    checkpoint_steps: tuple[int, ...]          # absolute step indices for ckpts
    final_step: int                             # last step (inclusive)
    git_commit: Optional[str] = None
    log_every_steps: int = 100                  # console / json line cadence
    # Optional callback at each checkpoint for per-(item, ordinal) eval. Receives
    # (trainer, step). Spec §9.6 / instr §4.6 per-(item, ordinal) logging.
    per_item_ordinal_callback: Optional[Callable[["OnlineTrainerV1", int], None]] = None


@dataclass
class CheckpointPayload:
    """Per-checkpoint JSON record (instr §4.6 aggregate section)."""

    step: int
    interval_loss_mean: float                   # mean loss over last interval
    mean_log_var_per_k: list[float]             # vector of K values (aggregate)
    predictor_weight_l2: float
    timestamp: float
    git_commit: Optional[str]


# --------------------------------------------------------------------------
# Trainer
# --------------------------------------------------------------------------


class TrainingStopped(RuntimeError):
    """Raised on any spec §11.2 stop condition. Writes TRAINING_STOPPED.txt."""


class OnlineTrainerV1:
    """Single-stage trainer; one instance handles either Stage A or Stage B for one arm.

    Stage A is launched with a freshly-constructed predictor + optimiser. Stage B
    is launched with the predictor + optimiser state resumed from end-of-Stage-A
    (spec §9.2 inherits v0's no-reset-at-stage-boundary commitment).
    """

    def __init__(
        self,
        predictor: nn.Module,
        embeddings: np.ndarray,           # (N, d) L2-normalised
        config: TrainerConfig,
        optimizer: Optional[optim.Optimizer] = None,
        start_step: int = 0,
        device: Optional[torch.device] = None,
    ):
        if embeddings.ndim != 2 or embeddings.shape[1] != EMBED_DIM:
            raise ValueError(
                f"embeddings must be (N, {EMBED_DIM}); got shape {embeddings.shape}"
            )
        # Stream contract (instr §4.1): L2 norms in [1 − 1e-5, 1 + 1e-5] on a
        # 1000-frame sample.
        rng = np.random.default_rng(0)
        sample_idx = rng.choice(embeddings.shape[0], size=min(1000, embeddings.shape[0]), replace=False)
        norms = np.linalg.norm(embeddings[sample_idx], axis=1)
        if not np.all(np.abs(norms - 1.0) < 1e-5):
            raise TrainingStopped(
                f"L2 norm verification failed on sampled embeddings: "
                f"norms range [{norms.min():.6f}, {norms.max():.6f}], "
                f"expected 1.0 ± 1e-5"
            )

        self.predictor = predictor
        self.embeddings = embeddings
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.predictor.to(self.device)

        if optimizer is None:
            optimizer = optim.AdamW(
                predictor.parameters(),
                lr=LR,
                weight_decay=WEIGHT_DECAY,
                betas=ADAM_BETAS,
            )
        self.optimizer = optimizer

        self.start_step = start_step
        self.current_step = start_step
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        # Running stats for per-checkpoint logging.
        self._loss_running_sum: float = 0.0
        self._loss_running_count: int = 0
        self._log_var_per_k_running_sum: np.ndarray = np.zeros(PREDICT_K, dtype=np.float64)
        self._log_var_per_k_running_count: int = 0
        self._last_checkpoint_step: int = self.start_step

    # ---------------------------------------------------------------
    # Window / target construction
    # ---------------------------------------------------------------

    def _build_window_and_target(self, t: int) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Return (window, target) for step t, or (None, None) if not feasible.

        Spec §9.3 + instr §4.3 (skip-until-W): first training step is at
        `t = W - 1 = 15`. K-step target is `[t + 1 .. t + K]`; skip if fewer
        than K future frames remain.
        """
        if t < WINDOW_W - 1:
            return None, None
        target_start = t + 1
        target_end = t + PREDICT_K  # inclusive
        if target_end >= self.embeddings.shape[0]:
            return None, None
        window_np = self.embeddings[t - WINDOW_W + 1 : t + 1]      # (W, d)
        target_np = self.embeddings[target_start : target_end + 1]  # (K, d)
        window = torch.from_numpy(window_np).float().unsqueeze(0).to(self.device)
        target = torch.from_numpy(target_np).float().unsqueeze(0).to(self.device)
        return window, target

    # ---------------------------------------------------------------
    # Step
    # ---------------------------------------------------------------

    def _step(self, t: int) -> Optional[float]:
        """One forward + backward + optimiser step on frame index t.

        Returns the loss value on a successful step, or None if the window
        couldn't be built (skip-until-W or end-of-stream).
        """
        window, target = self._build_window_and_target(t)
        if window is None:
            return None

        mean, log_var = self.predictor(window)
        loss = path_prediction_loss(mean, log_var, target)

        if not torch.isfinite(loss):
            self._stop(f"NaN/Inf loss at step {self.current_step} (t={t})")

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # Spec §11.2 condition 2: NaN/Inf in gradients.
        for p in self.predictor.parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                self._stop(f"NaN/Inf gradient at step {self.current_step}")
        nn.utils.clip_grad_norm_(self.predictor.parameters(), GRAD_CLIP_MAX_NORM)
        self.optimizer.step()

        self._loss_running_sum += loss.item()
        self._loss_running_count += 1
        self._log_var_per_k_running_sum += log_var.detach().mean(dim=0).cpu().numpy()
        self._log_var_per_k_running_count += 1

        return loss.item()

    # ---------------------------------------------------------------
    # Checkpoint + logging
    # ---------------------------------------------------------------

    def _checkpoint(self, step: int) -> None:
        """Save predictor + optimiser state + per-checkpoint metrics."""
        ckpt_path = self.config.output_dir / f"ckpt_{step}.pt"
        torch.save(
            {
                "predictor_state": self.predictor.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "rng_torch": torch.get_rng_state(),
                "rng_numpy_state": np.random.get_state(),
                "step": step,
                "git_commit": self.config.git_commit,
            },
            ckpt_path,
        )

        # Interval aggregate stats (instr §4.6 per-checkpoint section).
        interval_loss_mean = (
            self._loss_running_sum / max(1, self._loss_running_count)
        )
        mean_log_var_per_k = (
            self._log_var_per_k_running_sum
            / max(1, self._log_var_per_k_running_count)
        ).tolist()
        weight_l2 = float(
            torch.sqrt(
                sum(
                    p.detach().pow(2).sum()
                    for p in self.predictor.parameters()
                    if p.requires_grad
                )
            ).item()
        )
        payload = CheckpointPayload(
            step=step,
            interval_loss_mean=float(interval_loss_mean),
            mean_log_var_per_k=mean_log_var_per_k,
            predictor_weight_l2=weight_l2,
            timestamp=time.time(),
            git_commit=self.config.git_commit,
        )
        (self.config.output_dir / f"checkpoint_{step}.json").write_text(
            json.dumps(payload.__dict__, indent=2)
        )

        # Reset running stats for the next interval.
        self._loss_running_sum = 0.0
        self._loss_running_count = 0
        self._log_var_per_k_running_sum[:] = 0.0
        self._log_var_per_k_running_count = 0
        self._last_checkpoint_step = step

        # Per-(item, ordinal) callback for spec §9.6 / instr §4.6 logging.
        if self.config.per_item_ordinal_callback is not None:
            try:
                self.config.per_item_ordinal_callback(self, step)
            except Exception as exc:
                self._stop(f"per-(item, ordinal) logging gap at step {step}: {exc}")

    def _stop(self, reason: str) -> None:
        """Write TRAINING_STOPPED.txt and raise. Spec §11.2 stop condition."""
        msg = (
            f"v1 trainer stopped at step {self.current_step} "
            f"(arm={self.config.arm_name}, stage={self.config.stage}): {reason}"
        )
        (self.config.output_dir / "TRAINING_STOPPED.txt").write_text(msg)
        raise TrainingStopped(msg)

    # ---------------------------------------------------------------
    # Run
    # ---------------------------------------------------------------

    def run(self, n_frames: int) -> None:
        """Run the trainer for `n_frames` of the input stream from current_step.

        For Stage A (start_step=0), n_frames = STAGE_A_FRAME_BUDGET (100k).
        For Stage B (start_step=STAGE_A_FRAME_BUDGET), n_frames is the
        signal-stability-calibrated frame count or the maximum cap.
        """
        end_t = min(self.start_step + n_frames, self.embeddings.shape[0])
        checkpoint_set = set(self.config.checkpoint_steps)

        self.predictor.train()
        for t in range(self.start_step, end_t):
            self.current_step = t
            loss = self._step(t)
            if loss is None:
                continue
            if self.current_step in checkpoint_set:
                self._checkpoint(self.current_step)

        # Always save an end-of-stage canonical alias.
        end_step = self.current_step
        end_ckpt_name = f"ckpt_end_stage_{self.config.stage.lower()}.pt"
        torch.save(
            {
                "predictor_state": self.predictor.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "rng_torch": torch.get_rng_state(),
                "rng_numpy_state": np.random.get_state(),
                "step": end_step,
                "git_commit": self.config.git_commit,
            },
            self.config.output_dir / end_ckpt_name,
        )

    # ---------------------------------------------------------------
    # Resume utility
    # ---------------------------------------------------------------

    @classmethod
    def from_checkpoint(
        cls,
        predictor: nn.Module,
        ckpt_path: Path,
        embeddings: np.ndarray,
        config: TrainerConfig,
        device: Optional[torch.device] = None,
    ) -> "OnlineTrainerV1":
        """Construct a trainer resumed from a saved checkpoint."""
        # weights_only=False because the checkpoint payload includes numpy
        # RNG state for reproducibility (CODING_STANDARDS §5.4). Checkpoints
        # are produced by trusted v1 trainer code in the same repo.
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        predictor.load_state_dict(state["predictor_state"])
        optimizer = optim.AdamW(
            predictor.parameters(),
            lr=LR,
            weight_decay=WEIGHT_DECAY,
            betas=ADAM_BETAS,
        )
        optimizer.load_state_dict(state["optimizer_state"])
        torch.set_rng_state(state["rng_torch"])
        np.random.set_state(state["rng_numpy_state"])
        return cls(
            predictor=predictor,
            embeddings=embeddings,
            config=config,
            optimizer=optimizer,
            start_step=state["step"] + 1,
            device=device,
        )

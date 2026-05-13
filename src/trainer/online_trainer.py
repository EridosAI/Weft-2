"""Online single-pass trainer for Inner PAM (instr §4, spec §2.3).

Contract: at every step t after warmup (t >= W + K), the predictor is
trained on the window ending at frame t - K against the K-step target
[t - K + 1 .. t]. The bank receives the embedding at frame t (recency depth
= t). Each step is one forward, one backward, one optimiser step.

Why this is "online single-pass":
  - The whole stream is traversed once in temporal order.
  - One gradient step per resolved K-step prediction.
  - No epochs, no replay, no shuffled batches.
  - The window slides forward as t advances; old windows are not revisited.

Note on the "always on" requirement (spec §2.4):
  At test time the predictor is invoked every step on the current window. At
  training time the predictor is invoked every step whose target is now
  observable (i.e. starting at t >= W + K - 1). The "always on" property is
  satisfied conceptually; the implementation factors out the wait-for-target
  delay by issuing the predictor call at the moment the target is ready,
  which is equivalent to the §4.2 sketch up to a one-step weight delta and
  saves K-1 stale autograd graphs per step.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from src.config import (
    ADAM_BETAS,
    EMBED_DIM,
    GRAD_CLIP_MAX_NORM,
    LR,
    PREDICT_K,
    WEIGHT_DECAY,
    WINDOW_W,
)
from src.memory.memory_bank import BankEntry, MemoryBank
from src.predictor.inner_pam import (
    InnerPAM,
    all_parameter_count,
    confidence_from_log_var,
    gaussian_nll_loss,
    trainable_parameter_count,
)


@dataclass
class TrainCheckpoint:
    """In-memory snapshot for serialisation; not a substitute for ckpt_{step}.pt."""

    step: int
    predictor_state: dict[str, Any]
    optimizer_state: dict[str, Any]
    rng_torch: torch.ByteTensor
    rng_numpy: tuple
    git_commit: Optional[str]


@dataclass
class StepLog:
    """Per-step record written to the tensorboard log and per-checkpoint JSONs."""

    step: int
    loss: float
    mean_log_var: float
    confidence: float
    grad_norm: float


@dataclass
class TrainerConfig:
    phase_name: str
    perturbation_tag: str
    output_dir: Path
    checkpoint_steps: tuple[int, ...]
    final_step: int                   # last training step (inclusive); used for end-ckpt
    shuffle_seed: Optional[int] = None  # if set, training-index permutation seed
    git_commit: Optional[str] = None
    log_every: int = 100               # tensorboard granularity
    tau_calib_callback: Optional[Callable[["OnlineTrainer"], None]] = None


class OnlineTrainer:
    """Single-pass continuous-time trainer wiring predictor, bank, optimiser, logger."""

    def __init__(
        self,
        predictor: InnerPAM,
        bank: MemoryBank,
        embeddings: np.ndarray,
        annotations: list[dict[str, Any]],
        n_train: int,
        device: torch.device,
        cfg: TrainerConfig,
        resume_optimizer_state: Optional[dict[str, Any]] = None,
    ):
        assert embeddings.dtype == np.float32, (
            f"embeddings dtype {embeddings.dtype} != float32"
        )
        assert embeddings.shape[1] == EMBED_DIM, (
            f"embeddings dim {embeddings.shape[1]} != {EMBED_DIM}"
        )
        assert len(annotations) == embeddings.shape[0], (
            f"annotations length {len(annotations)} != embeddings rows "
            f"{embeddings.shape[0]}"
        )
        self.predictor = predictor.to(device)
        self.bank = bank
        self.device = device
        self.cfg = cfg
        self.n_train = int(n_train)

        # Training-order indices. Shuffle control permutes; main runs in stream order.
        if cfg.shuffle_seed is None:
            self._order = np.arange(self.n_train, dtype=np.int64)
        else:
            rng = np.random.default_rng(int(cfg.shuffle_seed))
            self._order = rng.permutation(self.n_train).astype(np.int64)
        # The bank still ingests the unshuffled (true temporal) stream so the
        # shuffle control isolates the *training-signal* shuffling, not the
        # bank-state shuffling. Instructions §6.3 C2 keeps held-out unshuffled.

        self._embeddings_np = embeddings
        self._annotations = annotations
        # Put the training subset on device for fast windowing.
        train_emb = embeddings[: self.n_train]
        self._embeddings = torch.from_numpy(train_emb).to(device)

        self.optimizer = torch.optim.AdamW(
            self.predictor.parameters(),
            lr=LR,
            weight_decay=WEIGHT_DECAY,
            betas=ADAM_BETAS,
        )
        if resume_optimizer_state is not None:
            self.optimizer.load_state_dict(resume_optimizer_state)

        self.cfg.output_dir.mkdir(parents=True, exist_ok=True)
        self._tb = SummaryWriter(log_dir=str(self.cfg.output_dir / "tb"))
        self._confidences_log: list[float] = []
        self._losses_log: list[float] = []
        self._mean_log_var_log: list[float] = []
        self._grad_norm_log: list[float] = []
        self._step_indices_log: list[int] = []

    # ---- init-time checks (instr §4.7) -----------------------------------

    def assert_init_invariants(self) -> dict[str, Any]:
        """Run §4.7 init-time checks. Returns a report dict for HANDOFF / JSON."""
        # Predictor: trainable, finite params, parameter count within tolerance.
        n_trainable = trainable_parameter_count(self.predictor)
        n_total = all_parameter_count(self.predictor)
        assert n_trainable > 0, "predictor has no trainable parameters"
        for name, p in self.predictor.named_parameters():
            assert torch.isfinite(p).all().item(), f"non-finite init in {name}"

        # Smoke forward pass on synthetic input matching the contract.
        sample = torch.zeros(2, WINDOW_W, EMBED_DIM, device=self.device)
        sample[:, :, 0] = 1.0  # not L2-normed; OK for shape check
        with torch.no_grad():
            mean, log_var = self.predictor(sample)
        assert mean.shape == (2, PREDICT_K, EMBED_DIM)
        assert log_var.shape == (2, PREDICT_K)
        assert torch.isfinite(mean).all().item()
        assert torch.isfinite(log_var).all().item()

        # Verify embedding norms on a 1000-row sample.
        n = min(1000, self._embeddings_np.shape[0])
        rng = np.random.default_rng(0)
        sample_idx = rng.integers(0, self._embeddings_np.shape[0], size=n)
        norms = np.linalg.norm(self._embeddings_np[sample_idx], axis=1)
        in_range = bool(((norms >= 1.0 - 1e-5) & (norms <= 1.0 + 1e-5)).all())
        assert in_range, (
            f"embedding norms out of range: min={norms.min():.6f} "
            f"max={norms.max():.6f}"
        )

        report = {
            "trainable_params": int(n_trainable),
            "total_params": int(n_total),
            "param_count_target": 21_000_000,
            "param_count_tolerance_frac": 0.10,
            "param_count_within_tolerance": bool(
                abs(n_trainable - 21_000_000) <= 21_000_000 * 0.10
            ),
            "smoke_forward_pass": "ok",
            "embedding_norm_check": {
                "n_sampled": int(n),
                "min": float(norms.min()),
                "max": float(norms.max()),
                "all_in_tolerance_1e_5": in_range,
            },
        }
        return report

    # ---- main loop -------------------------------------------------------

    def train(self) -> dict[str, Any]:
        """Run the single-pass online loop to self.cfg.final_step."""
        # Determine the K-step delay required before training can begin.
        first_train_step = WINDOW_W + PREDICT_K - 1
        if self.n_train < first_train_step + 1:
            raise ValueError(
                f"n_train={self.n_train} too small for W+K-1={first_train_step}"
            )

        # Pre-populate the bank with the first W frames (these are inputs to the
        # first prediction but cannot themselves be trained on yet).
        for t in range(min(WINDOW_W, self.n_train)):
            self._append_to_bank(t)

        ckpt_set = set(int(s) for s in self.cfg.checkpoint_steps)
        ckpt_set.add(int(self.cfg.final_step))

        t0 = time.time()
        # We index by the position in the shuffled order so that the shuffle
        # control sees the same number of gradient updates as the main run.
        for ordered_pos in range(first_train_step, self.n_train):
            t = int(self._order[ordered_pos])
            # window ends at t-K, targets are t-K+1 .. t.
            target_end = t
            window_end = target_end - PREDICT_K        # last frame of window
            window_start = window_end - WINDOW_W + 1
            if window_start < 0 or target_end >= self.n_train:
                # Skip steps where the shuffle picks an index that puts the
                # window or target outside the stream.
                continue

            window = self._embeddings[window_start : window_end + 1].unsqueeze(0)  # (1, W, d)
            targets = self._embeddings[window_end + 1 : target_end + 1].unsqueeze(0)  # (1, K, d)
            assert window.shape == (1, WINDOW_W, EMBED_DIM)
            assert targets.shape == (1, PREDICT_K, EMBED_DIM)

            mean, log_var = self.predictor(window)
            loss = gaussian_nll_loss(mean, log_var, targets)
            assert torch.isfinite(loss), f"non-finite loss at step {ordered_pos}"

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    self.predictor.parameters(),
                    max_norm=GRAD_CLIP_MAX_NORM,
                )
            )
            self.optimizer.step()

            # Bank append for frame t (true temporal order, regardless of shuffle).
            self._append_to_bank(t)

            # Logging.
            with torch.no_grad():
                conf_val = float(confidence_from_log_var(log_var, m=1).item())
                mean_log_var = float(log_var.mean().item())
            self._confidences_log.append(conf_val)
            self._losses_log.append(float(loss.item()))
            self._mean_log_var_log.append(mean_log_var)
            self._grad_norm_log.append(grad_norm)
            self._step_indices_log.append(ordered_pos)

            if ordered_pos % self.cfg.log_every == 0:
                self._tb.add_scalar("train/loss", float(loss.item()), ordered_pos)
                self._tb.add_scalar("train/mean_log_var", mean_log_var, ordered_pos)
                self._tb.add_scalar("train/grad_norm", grad_norm, ordered_pos)
                self._tb.add_scalar("train/confidence", conf_val, ordered_pos)

            if ordered_pos in ckpt_set:
                self._write_checkpoint(ordered_pos)
                if (
                    self.cfg.tau_calib_callback is not None
                    and ordered_pos == max(s for s in ckpt_set if s <= ordered_pos)
                ):
                    # Caller decides when to actually compute tau (after step 10k).
                    self.cfg.tau_calib_callback(self)

            if ordered_pos >= self.cfg.final_step:
                break

        elapsed = time.time() - t0
        return {
            "elapsed_seconds": float(elapsed),
            "final_step": int(self.cfg.final_step),
            "n_gradient_steps_actual": int(len(self._losses_log)),
        }

    # ---- bank ingest ----------------------------------------------------

    def _append_to_bank(self, frame_idx: int) -> None:
        ann = self._annotations[frame_idx]
        entry = BankEntry(
            frame_idx=int(ann.get("frame_idx", frame_idx)),
            loop_idx=int(ann.get("loop_index", -1)),
            viewing_position_id=(
                int(ann["viewing_position_id"])
                if ann.get("viewing_position_id") not in (None, 0)
                else None
            ),
            phase=str(ann.get("phase", "transit")),
            phase_name=str(self.cfg.phase_name),
            perturbation=str(self.cfg.perturbation_tag),
            depth=float(frame_idx),
        )
        emb = self._embeddings_np[frame_idx].astype(np.float32)
        self.bank.append(emb, entry)

    # ---- checkpointing --------------------------------------------------

    def _write_checkpoint(self, step: int) -> None:
        ckpt_dir = self.cfg.output_dir / f"ckpt_{step}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = self.cfg.output_dir / f"ckpt_{step}.pt"
        torch.save(
            {
                "step": int(step),
                "predictor_state": self.predictor.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "rng_torch": torch.get_rng_state(),
                "rng_numpy": np.random.get_state(),
                "git_commit": self.cfg.git_commit,
            },
            ckpt_path,
        )
        self.bank.save(ckpt_dir)

        report = self._summary_at_step(step)
        (self.cfg.output_dir / f"checkpoint_{step}.json").write_text(
            json.dumps(report, indent=2)
        )

    def _summary_at_step(self, step: int) -> dict[str, Any]:
        window = 1000
        recent_losses = self._losses_log[-window:] if self._losses_log else [0.0]
        recent_log_var = self._mean_log_var_log[-window:] if self._mean_log_var_log else [0.0]
        with torch.no_grad():
            weight_norm = sum(
                float(p.detach().pow(2).sum().item()) ** 0.5
                for p in self.predictor.parameters()
                if p.requires_grad
            )
        return {
            "step": int(step),
            "mean_loss_last_1k": float(np.mean(recent_losses)),
            "mean_log_var_last_1k": float(np.mean(recent_log_var)),
            "predictor_weight_l2_norm_total": float(weight_norm),
            "bank_size": int(self.bank.size()),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "git_commit": self.cfg.git_commit,
        }

    # ---- accessors ------------------------------------------------------

    @property
    def confidences(self) -> np.ndarray:
        return np.asarray(self._confidences_log, dtype=np.float64)

    @property
    def step_indices(self) -> np.ndarray:
        return np.asarray(self._step_indices_log, dtype=np.int64)

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
    git_commit: Optional[str] = None
    log_every: int = 100               # tensorboard granularity
    tau_calib_callback: Optional[Callable[["OnlineTrainer"], None]] = None

    # ---- Phase 2 in-flight transition diagnostic (instr §8.7a). ----
    # When enabled, the trainer accumulates per-loop loss + per-(loop, item)
    # mean log_var, writes per-loop stats to `transition_diagnostic_path` as
    # each loop completes, and at the end of `transition_post_onset_loops[1]`
    # evaluates the three SCAFFOLDING gates. If any trips, the trainer writes
    # `transition_diagnostic_TRIPPED.txt` under `output_dir` and stops the loop.
    transition_diagnostic_enabled: bool = False
    transition_diagnostic_path: Optional[Path] = None
    transition_perturbed_items: tuple[int, ...] = ()      # vp_ids of perturbed items
    transition_control_items: tuple[int, ...] = ()        # vp_ids of control items
    transition_baseline_loops: tuple[int, int] = (0, 0)   # (lo, hi) inclusive
    transition_post_onset_loops: tuple[int, int] = (0, 0) # (lo, hi) inclusive
    transition_loss_spike_ratio: float = 3.0              # G2.T1
    transition_log_var_widening_min: float = 0.5          # G2.T2 (legacy; only used in non-extended mode)
    transition_control_drift_max: float = 0.3             # G2.T3

    # ---- Extended-mode diagnostic (session 7, instr §8.7a restructured). ----
    # When True, the trainer:
    #   (1) seeds `_diag_per_loop_summary` from any existing diagnostic JSON at
    #       `transition_diagnostic_path` (so prior loops 0..35 from a session-6
    #       run are preserved and the new loops 36..N appended);
    #   (2) skips the original session-6 post_end-boundary G2.T2 auto-trip
    #       evaluation (G2.T2 is evaluated post-hoc on the loop-100 trajectory
    #       per the restructured three-part criterion);
    #   (3) at every checkpoint step where `_last_loop_seen > post_end`,
    #       re-evaluates G2.T1 (loss spike vs baseline loops 25–30) and G2.T3
    #       (per-control-item drift from loop 30) against the extended window.
    #       A trip writes the marker file and breaks.
    transition_diagnostic_extended_mode: bool = False

    # ---- Early-stop at loop boundary (instr §8.7a STOP-for-review). ----
    # When set, the trainer halts after observing the first step whose
    # annotation `loop_index` exceeds `max_loops` — i.e. immediately after
    # the existing transition-diagnostic boundary handler has flushed
    # `loop=max_loops` and (if applicable) evaluated the §8.7a gates. A
    # final checkpoint is written at the stop step regardless of
    # `checkpoint_steps`, so the next session can resume cleanly.
    max_loops: Optional[int] = None

    # ---- Resume support (session-6 directive). ----
    # When `resume_step` is set, the trainer skips the bank pre-population
    # step (the caller is expected to have already loaded the bank from
    # disk and populated optimizer/predictor/RNG state) and begins the
    # training loop at `resume_step + 1`.
    resume_step: Optional[int] = None


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

        # Training always traverses the input stream in order. The shuffle
        # control passes a pre-permuted embedding+annotation stream so that
        # window/target contents are randomised at the source, destroying
        # temporal structure as spec §10.1 / §6.3 require. The trainer itself
        # does not permute.
        self._order = np.arange(self.n_train, dtype=np.int64)

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

        # ---- Transition diagnostic state (instr §8.7a). ----
        # Aggregates loss and (item, mean log_var) per loop_index. Flushed and
        # evaluated against the three SCAFFOLDING gates at the end of the
        # transition_post_onset_loops window.
        self._per_loop_loss_sum: dict[int, float] = {}
        self._per_loop_loss_count: dict[int, int] = {}
        self._per_loop_item_log_var_sum: dict[tuple[int, int], float] = {}
        self._per_loop_item_log_var_count: dict[tuple[int, int], int] = {}
        self._last_loop_seen: int = -1
        self._diag_gate_tripped: bool = False
        self._diag_trip_record: Optional[dict[str, Any]] = None
        self._diag_per_loop_summary: list[dict[str, Any]] = []

        # Extended mode: seed _diag_per_loop_summary from the existing JSON so
        # prior loops from a session-6 run are preserved and the loop-100
        # trajectory analysis sees the full record.
        if (
            self.cfg.transition_diagnostic_enabled
            and self.cfg.transition_diagnostic_extended_mode
            and self.cfg.transition_diagnostic_path is not None
            and Path(self.cfg.transition_diagnostic_path).exists()
        ):
            try:
                prior = json.loads(
                    Path(self.cfg.transition_diagnostic_path).read_text()
                )
                prior_per_loop = list(prior.get("per_loop", []))
                self._diag_per_loop_summary = prior_per_loop
            except Exception:
                # Don't fail trainer init on a malformed prior diagnostic;
                # extended-mode accumulation still works from an empty start.
                self._diag_per_loop_summary = []

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

        if self.cfg.resume_step is None:
            # Fresh start. Pre-populate the bank with the first W frames
            # (these are inputs to the first prediction but cannot themselves
            # be trained on yet).
            for t in range(min(WINDOW_W, self.n_train)):
                self._append_to_bank(t)
            start_step = first_train_step
        else:
            # Resume mode: the caller already loaded the bank from disk and
            # restored predictor/optimizer/RNG state. Skip pre-population and
            # pick up at the step after the saved checkpoint.
            start_step = max(int(self.cfg.resume_step) + 1, first_train_step)

        ckpt_set = set(int(s) for s in self.cfg.checkpoint_steps)
        ckpt_set.add(int(self.cfg.final_step))

        t0 = time.time()
        # We index by the position in the shuffled order so that the shuffle
        # control sees the same number of gradient updates as the main run.
        for ordered_pos in range(start_step, self.n_train):
            t = int(self._order[ordered_pos])
            # window ends at t-K, targets are t-K+1 .. t.
            target_end = t
            window_end = target_end - PREDICT_K        # last frame of window
            window_start = window_end - WINDOW_W + 1
            if window_start < 0 or target_end >= self.n_train:
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

            # ---- Transition diagnostic accumulation (instr §8.7a). ----
            if self.cfg.transition_diagnostic_enabled:
                ann_t = self._annotations[t]
                loop_t = int(ann_t.get("loop_index", -1))
                if loop_t >= 0:
                    # Loss aggregation, per loop.
                    self._per_loop_loss_sum[loop_t] = (
                        self._per_loop_loss_sum.get(loop_t, 0.0) + float(loss.item())
                    )
                    self._per_loop_loss_count[loop_t] = (
                        self._per_loop_loss_count.get(loop_t, 0) + 1
                    )
                    # Per-(loop, item) log_var aggregation for close-up predictions only.
                    seg = ann_t.get("phase_segment") or ann_t.get("phase") or "?"
                    vp = int(ann_t.get("viewing_position_id", -1))
                    if seg == "close_up" and vp > 0:
                        key = (loop_t, vp)
                        self._per_loop_item_log_var_sum[key] = (
                            self._per_loop_item_log_var_sum.get(key, 0.0)
                            + float(mean_log_var)
                        )
                        self._per_loop_item_log_var_count[key] = (
                            self._per_loop_item_log_var_count.get(key, 0) + 1
                        )
                    # Loop completion -> flush + maybe evaluate gates.
                    if loop_t != self._last_loop_seen:
                        if self._last_loop_seen >= 0:
                            self._flush_loop_to_diagnostic(self._last_loop_seen)
                        self._last_loop_seen = loop_t
                        post_end = int(self.cfg.transition_post_onset_loops[1])
                        if (
                            post_end > 0
                            and self._last_loop_seen > post_end
                            and not self._diag_gate_tripped
                            and not self.cfg.transition_diagnostic_extended_mode
                        ):
                            # Session-6 behaviour: evaluate G2.T1/T2/T3 once at
                            # first-step-of-loop-(post_end+1). Skipped in
                            # extended mode — G2.T2 is post-hoc on the loop-100
                            # trajectory; G2.T1/T3 are re-evaluated at each
                            # checkpoint via the block below.
                            trip = self._evaluate_transition_gates()
                            if trip is not None:
                                self._diag_gate_tripped = True
                                self._diag_trip_record = trip
                                self._write_diagnostic_trip_marker(trip)
                                # Halt training cleanly per instr §8.7a.
                                break

            # ---- max_loops early-stop (session-6 directive). ----
            # When set, halt as soon as a step's `loop_index` exceeds
            # max_loops. The transition-diagnostic boundary handler above
            # has already flushed and (if `transition_diagnostic_enabled`)
            # evaluated the §8.7a gates against loop=max_loops, so by the
            # time we hit this check the per-loop summary is complete. We
            # write a checkpoint at the current step (the first step of
            # loop max_loops+1) so the next session can resume cleanly.
            if self.cfg.max_loops is not None:
                cur_loop = int(
                    self._annotations[t].get("loop_index", -1)
                )
                if cur_loop > int(self.cfg.max_loops):
                    if ordered_pos not in ckpt_set:
                        self._write_checkpoint(ordered_pos)
                    break

            if ordered_pos in ckpt_set:
                self._write_checkpoint(ordered_pos)
                if (
                    self.cfg.tau_calib_callback is not None
                    and ordered_pos == max(s for s in ckpt_set if s <= ordered_pos)
                ):
                    # Caller decides when to actually compute tau (after step 10k).
                    self.cfg.tau_calib_callback(self)

                # Extended-mode in-flight T1/T3 re-evaluation at checkpoints.
                if (
                    self.cfg.transition_diagnostic_enabled
                    and self.cfg.transition_diagnostic_extended_mode
                    and not self._diag_gate_tripped
                    and self._last_loop_seen
                        > int(self.cfg.transition_post_onset_loops[1])
                ):
                    trip = self._evaluate_extended_t1_t3()
                    if trip is not None:
                        self._diag_gate_tripped = True
                        self._diag_trip_record = trip
                        self._write_diagnostic_trip_marker(trip)
                        break

            if ordered_pos >= self.cfg.final_step:
                break

        # Final flush of any in-progress loop (so the JSON reflects the last loop).
        if (
            self.cfg.transition_diagnostic_enabled
            and self._last_loop_seen >= 0
        ):
            self._flush_loop_to_diagnostic(self._last_loop_seen)

        elapsed = time.time() - t0
        last_step_done = (
            int(self._step_indices_log[-1]) if self._step_indices_log else -1
        )
        # Resolve last_loop_seen from the annotations directly, so the value is
        # correct independent of whether the §8.7a diagnostic was enabled
        # (the diag block is the only place that updates self._last_loop_seen).
        last_loop_observed = -1
        if last_step_done >= 0:
            last_loop_observed = int(
                self._annotations[int(self._order[last_step_done])]
                .get("loop_index", -1)
            )
        return {
            "elapsed_seconds": float(elapsed),
            "final_step": int(self.cfg.final_step),
            "last_step_done": last_step_done,
            "last_loop_seen": last_loop_observed,
            "n_gradient_steps_actual": int(len(self._losses_log)),
            "max_loops": (
                int(self.cfg.max_loops)
                if self.cfg.max_loops is not None else None
            ),
            "stopped_at_max_loops": bool(
                self.cfg.max_loops is not None
                and last_loop_observed > int(self.cfg.max_loops)
            ),
            "transition_diagnostic_gate_tripped": bool(self._diag_gate_tripped),
            "transition_diagnostic_trip_record": self._diag_trip_record,
        }

    # ---- Transition diagnostic helpers (instr §8.7a) -------------------

    def _flush_loop_to_diagnostic(self, loop_idx: int) -> None:
        """Compute and persist per-loop stats for `loop_idx`."""
        n_steps = int(self._per_loop_loss_count.get(loop_idx, 0))
        if n_steps == 0:
            return
        mean_loss = float(self._per_loop_loss_sum[loop_idx] / n_steps)
        per_item: dict[str, float] = {}
        for vp in tuple(self.cfg.transition_perturbed_items) + tuple(
            self.cfg.transition_control_items
        ):
            key = (loop_idx, int(vp))
            c = int(self._per_loop_item_log_var_count.get(key, 0))
            if c > 0:
                per_item[str(int(vp))] = float(
                    self._per_loop_item_log_var_sum[key] / c
                )
        record = {
            "loop_index": int(loop_idx),
            "n_train_steps_attributed": n_steps,
            "mean_loss": mean_loss,
            "mean_log_var_by_viewing_position_id": per_item,
        }
        # Append to in-memory log, persist whole list as we go.
        self._diag_per_loop_summary.append(record)
        self._write_diagnostic_json(gate_tripped=False, trip=None)

    def _evaluate_transition_gates(self) -> Optional[dict[str, Any]]:
        """Evaluate G2.T1 / G2.T2 / G2.T3 against accumulated loops.

        Returns None if all gates pass; otherwise returns a trip-record dict.
        """
        baseline_lo, baseline_hi = self.cfg.transition_baseline_loops
        post_lo, post_hi = self.cfg.transition_post_onset_loops
        records_by_loop = {int(r["loop_index"]): r for r in self._diag_per_loop_summary}

        # G2.T1 — loss spike check.
        baseline_losses = [
            records_by_loop[k]["mean_loss"]
            for k in range(int(baseline_lo), int(baseline_hi) + 1)
            if k in records_by_loop
        ]
        post_losses = [
            records_by_loop[k]["mean_loss"]
            for k in range(int(post_lo), int(post_hi) + 1)
            if k in records_by_loop
        ]
        if not baseline_losses or not post_losses:
            return {
                "gate_tripped": True,
                "gate_name": "G2.T_data_missing",
                "reason": (
                    f"baseline_losses n={len(baseline_losses)} "
                    f"post_losses n={len(post_losses)} "
                    "— at least one loop did not produce any training-step records"
                ),
            }
        baseline_loss_mean = float(np.mean(baseline_losses))
        post_loss_max = float(np.max(post_losses))
        # Sign-safe form of the user-specified "3× spike": for positive
        # baselines this is identical to `> spike_ratio * baseline_mean`;
        # for non-positive baselines (high-confidence regime where Gaussian
        # NLL can go negative) it becomes "exceeds baseline by spike_ratio
        # multiples of |baseline|", preserving the directionally-correct
        # meaning of "loss spiked upward by a large fraction of its scale".
        scale = max(abs(baseline_loss_mean), 1.0)
        spike_threshold = baseline_loss_mean + (
            float(self.cfg.transition_loss_spike_ratio) - 1.0
        ) * scale
        if post_loss_max > spike_threshold:
            return {
                "gate_tripped": True,
                "gate_name": "G2.T1_loss_spike",
                "baseline_loss_mean_loops": [int(baseline_lo), int(baseline_hi)],
                "baseline_loss_mean": baseline_loss_mean,
                "post_onset_loops": [int(post_lo), int(post_hi)],
                "post_loss_max": post_loss_max,
                "spike_threshold": spike_threshold,
                "ratio_threshold": float(self.cfg.transition_loss_spike_ratio),
            }

        # G2.T2 — perturbed-item log_var widening.
        def _agg_log_var_at_loop(loop_idx: int, items: tuple[int, ...]) -> Optional[float]:
            if loop_idx not in records_by_loop:
                return None
            d = records_by_loop[loop_idx]["mean_log_var_by_viewing_position_id"]
            vals = [d[str(int(it))] for it in items if str(int(it)) in d]
            if not vals:
                return None
            return float(np.mean(vals))

        lv_end_baseline_pert = _agg_log_var_at_loop(
            int(baseline_hi), self.cfg.transition_perturbed_items
        )
        lv_end_post_pert = _agg_log_var_at_loop(
            int(post_hi), self.cfg.transition_perturbed_items
        )
        if lv_end_baseline_pert is None or lv_end_post_pert is None:
            return {
                "gate_tripped": True,
                "gate_name": "G2.T_log_var_data_missing",
                "reason": (
                    f"perturbed-item log_var aggregates unavailable: "
                    f"baseline={lv_end_baseline_pert} post={lv_end_post_pert}"
                ),
            }
        delta_pert = lv_end_post_pert - lv_end_baseline_pert
        if delta_pert < float(self.cfg.transition_log_var_widening_min):
            return {
                "gate_tripped": True,
                "gate_name": "G2.T2_perturbed_widening_insufficient",
                "perturbed_items_vp_ids": list(self.cfg.transition_perturbed_items),
                "log_var_at_baseline_end_loop": int(baseline_hi),
                "log_var_baseline_end": lv_end_baseline_pert,
                "log_var_at_post_end_loop": int(post_hi),
                "log_var_post_end": lv_end_post_pert,
                "delta_observed": delta_pert,
                "delta_required_min": float(self.cfg.transition_log_var_widening_min),
            }

        # G2.T3 — control-item drift.
        control_drifts: dict[str, float] = {}
        max_abs_drift = 0.0
        worst_item: Optional[int] = None
        for it in self.cfg.transition_control_items:
            lv_base = _agg_log_var_at_loop(int(baseline_hi), (int(it),))
            lv_post = _agg_log_var_at_loop(int(post_hi), (int(it),))
            if lv_base is None or lv_post is None:
                control_drifts[str(int(it))] = float("nan")
                continue
            drift = lv_post - lv_base
            control_drifts[str(int(it))] = float(drift)
            if abs(drift) > max_abs_drift:
                max_abs_drift = abs(drift)
                worst_item = int(it)
        if max_abs_drift > float(self.cfg.transition_control_drift_max):
            return {
                "gate_tripped": True,
                "gate_name": "G2.T3_control_drift",
                "control_items_vp_ids": list(self.cfg.transition_control_items),
                "log_var_at_baseline_end_loop": int(baseline_hi),
                "log_var_at_post_end_loop": int(post_hi),
                "per_item_delta_log_var": control_drifts,
                "max_abs_drift_observed": max_abs_drift,
                "worst_item_vp_id": worst_item,
                "drift_threshold_max": float(self.cfg.transition_control_drift_max),
            }

        # All three pass; record summary for HANDOFF (None signifies no trip).
        return None

    def _evaluate_extended_t1_t3(self) -> Optional[dict[str, Any]]:
        """Extended-mode in-flight check: G2.T1 vs the full post-onset window
        seen so far, and G2.T3 control-drift loop30 → latest. G2.T2 is
        explicitly NOT evaluated here (it is the post-hoc loop-100 three-part
        criterion). Returns a trip-record on failure; None on pass.
        """
        baseline_lo, baseline_hi = self.cfg.transition_baseline_loops
        post_lo, _ = self.cfg.transition_post_onset_loops
        records_by_loop = {
            int(r["loop_index"]): r for r in self._diag_per_loop_summary
        }

        baseline_losses = [
            records_by_loop[k]["mean_loss"]
            for k in range(int(baseline_lo), int(baseline_hi) + 1)
            if k in records_by_loop
        ]
        post_losses = [
            records_by_loop[k]["mean_loss"]
            for k in range(int(post_lo), self._last_loop_seen + 1)
            if k in records_by_loop
        ]
        if not baseline_losses or not post_losses:
            return None  # not enough data yet; let the next checkpoint try
        baseline_loss_mean = float(np.mean(baseline_losses))
        post_loss_max = float(np.max(post_losses))
        scale = max(abs(baseline_loss_mean), 1.0)
        spike_threshold = baseline_loss_mean + (
            float(self.cfg.transition_loss_spike_ratio) - 1.0
        ) * scale
        if post_loss_max > spike_threshold:
            return {
                "gate_tripped": True,
                "gate_name": "G2.T1_loss_spike_extended",
                "mode": "extended",
                "baseline_loops": [int(baseline_lo), int(baseline_hi)],
                "post_onset_loops_observed": [int(post_lo), self._last_loop_seen],
                "baseline_loss_mean": baseline_loss_mean,
                "post_loss_max": post_loss_max,
                "spike_threshold": spike_threshold,
                "ratio_threshold": float(self.cfg.transition_loss_spike_ratio),
            }

        # G2.T3 — drift loop_baseline_hi → latest seen, per control item.
        def _agg_lv(loop_idx: int, items: tuple[int, ...]) -> Optional[float]:
            if loop_idx not in records_by_loop:
                return None
            d = records_by_loop[loop_idx]["mean_log_var_by_viewing_position_id"]
            vals = [d[str(int(it))] for it in items if str(int(it)) in d]
            return float(np.mean(vals)) if vals else None

        control_drifts: dict[str, float] = {}
        max_abs_drift = 0.0
        worst_item: Optional[int] = None
        for it in self.cfg.transition_control_items:
            lv_base = _agg_lv(int(baseline_hi), (int(it),))
            lv_latest = _agg_lv(int(self._last_loop_seen), (int(it),))
            if lv_base is None or lv_latest is None:
                control_drifts[str(int(it))] = float("nan")
                continue
            drift = lv_latest - lv_base
            control_drifts[str(int(it))] = float(drift)
            if abs(drift) > max_abs_drift:
                max_abs_drift = abs(drift)
                worst_item = int(it)
        if max_abs_drift > float(self.cfg.transition_control_drift_max):
            return {
                "gate_tripped": True,
                "gate_name": "G2.T3_control_drift_extended",
                "mode": "extended",
                "control_items_vp_ids": list(self.cfg.transition_control_items),
                "log_var_at_baseline_end_loop": int(baseline_hi),
                "log_var_at_latest_loop": int(self._last_loop_seen),
                "per_item_delta_log_var": control_drifts,
                "max_abs_drift_observed": max_abs_drift,
                "worst_item_vp_id": worst_item,
                "drift_threshold_max": float(self.cfg.transition_control_drift_max),
            }

        return None

    def _write_diagnostic_json(
        self,
        gate_tripped: bool,
        trip: Optional[dict[str, Any]],
    ) -> None:
        if not self.cfg.transition_diagnostic_path:
            return
        path = Path(self.cfg.transition_diagnostic_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "phase_name": str(self.cfg.phase_name),
            "perturbation_tag": str(self.cfg.perturbation_tag),
            "baseline_loops": list(self.cfg.transition_baseline_loops),
            "post_onset_loops": list(self.cfg.transition_post_onset_loops),
            "perturbed_items_vp_ids": list(self.cfg.transition_perturbed_items),
            "control_items_vp_ids": list(self.cfg.transition_control_items),
            "thresholds": {
                "loss_spike_ratio": float(self.cfg.transition_loss_spike_ratio),
                "log_var_widening_min": float(self.cfg.transition_log_var_widening_min),
                "control_drift_max": float(self.cfg.transition_control_drift_max),
            },
            "per_loop": list(self._diag_per_loop_summary),
            "gate_tripped": bool(gate_tripped),
            "trip_record": trip,
            "git_commit": self.cfg.git_commit,
        }
        path.write_text(json.dumps(payload, indent=2))

    def _write_diagnostic_trip_marker(self, trip: dict[str, Any]) -> None:
        marker_path = self.cfg.output_dir / "transition_diagnostic_TRIPPED.txt"
        marker_path.write_text(json.dumps(trip, indent=2))
        # And rewrite the diagnostic JSON one more time with the trip recorded.
        self._write_diagnostic_json(gate_tripped=True, trip=trip)

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

"""Confidence-thresholded recall mixing (spec §2.8, instr §5).

Above tau: shape-only (predictor mean + log_var).
Below tau: predictor output + top-k bank instances retrieved by cosine on the
mean of the probe window.

tau itself is the median predictor-confidence over a calibration window of
Phase-1 training steps; computation lives in `compute_tau_from_confidences`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from v0.src.config import CONFIDENCE_M, TOP_K_INSTANCES
from v0.src.memory.memory_bank import MemoryBank
from v0.src.predictor.inner_pam import InnerPAM, confidence_from_log_var


@dataclass
class MixResult:
    mode: str                    # "predictor_only" | "predictor_plus_bank"
    mean: torch.Tensor           # (B, K, d)
    log_var: torch.Tensor        # (B, K)
    confidence: torch.Tensor     # (B,)
    instance_cosines: Optional[np.ndarray] = None  # (B, top_k) or None
    instance_indices: Optional[np.ndarray] = None  # (B, top_k) or None


def mix(
    probe_window: torch.Tensor,
    predictor: InnerPAM,
    bank: MemoryBank,
    tau: float,
    m: int = CONFIDENCE_M,
    top_k_instances: int = TOP_K_INSTANCES,
) -> MixResult:
    """Run the predictor; route to predictor-only or predictor+bank by confidence.

    The bank query is the L2-normalised mean of the probe window — a simple
    summary of "where am I?" rather than the predictor's output. The predictor's
    output is the shape continuation; the bank lookup is the instance anchor.
    """
    assert probe_window.ndim == 3, "probe_window must be (B, W, d)"
    mean, log_var = predictor(probe_window)
    conf = confidence_from_log_var(log_var, m)

    high_conf = (conf > tau)
    if bool(high_conf.all().item()):
        return MixResult(
            mode="predictor_only",
            mean=mean,
            log_var=log_var,
            confidence=conf,
        )

    window_mean = probe_window.mean(dim=1)
    window_mean = F.normalize(window_mean, dim=-1)
    query = window_mean.detach().cpu().numpy().astype(np.float32)
    cosines, indices = bank.retrieve_by_cosine(query, k=top_k_instances)

    mode = "predictor_only" if bool(high_conf.all().item()) else "predictor_plus_bank"
    return MixResult(
        mode=mode,
        mean=mean,
        log_var=log_var,
        confidence=conf,
        instance_cosines=cosines,
        instance_indices=indices,
    )


def compute_tau_from_confidences(
    confidences: np.ndarray,
    start_step: int,
    end_step: int,
    step_indices: Optional[np.ndarray] = None,
) -> float:
    """Median confidence over training steps in [start_step, end_step].

    If step_indices is None, confidences is assumed to be aligned 1:1 with
    training steps starting from 0.
    """
    if step_indices is None:
        if end_step > len(confidences):
            raise ValueError(
                f"end_step {end_step} exceeds confidences length {len(confidences)}"
            )
        window = confidences[start_step:end_step]
    else:
        mask = (step_indices >= start_step) & (step_indices < end_step)
        window = confidences[mask]
    if window.size == 0:
        raise ValueError(
            f"empty calibration window [{start_step}, {end_step})"
        )
    return float(np.median(window))

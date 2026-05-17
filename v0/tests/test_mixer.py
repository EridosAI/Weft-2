"""Recall mixer + tau calibration tests."""

import numpy as np
import pytest
import torch

from v0.src.config import CONFIDENCE_M, EMBED_DIM, WINDOW_W
from v0.src.memory.memory_bank import BankEntry, MemoryBank
from v0.src.mixing.recall_mixer import compute_tau_from_confidences, mix
from v0.src.predictor.inner_pam import InnerPAM


class _ConstantPredictor(InnerPAM):
    """Predictor whose output log_var is a constant so we can drive mixer mode deterministically."""

    def __init__(self, fixed_log_var: float):
        super().__init__()
        self.fixed_log_var = fixed_log_var

    def forward(self, window):
        b = window.shape[0]
        mean = torch.zeros(b, 16, EMBED_DIM, device=window.device)
        log_var = torch.full((b, 16), self.fixed_log_var, device=window.device)
        return mean, log_var


def _stocked_bank() -> MemoryBank:
    bank = MemoryBank(cap=50, allow_eviction=False)
    rng = np.random.RandomState(0)
    for i in range(20):
        v = rng.randn(EMBED_DIM).astype(np.float32)
        v /= np.linalg.norm(v)
        bank.append(v, BankEntry(
            frame_idx=i, loop_idx=i // 5, viewing_position_id=None,
            phase="transit", phase_name="phase1", perturbation="none", depth=float(i),
        ))
    return bank


def test_mix_predictor_only_when_confidence_above_tau():
    pred = _ConstantPredictor(fixed_log_var=-1.0)  # confidence = -(-1.0) = 1.0
    bank = _stocked_bank()
    window = torch.randn(1, WINDOW_W, EMBED_DIM)
    res = mix(window, pred, bank, tau=0.0)
    assert res.mode == "predictor_only"
    assert res.instance_cosines is None


def test_mix_falls_back_to_bank_when_below_tau():
    pred = _ConstantPredictor(fixed_log_var=1.0)  # confidence = -1.0
    bank = _stocked_bank()
    window = torch.randn(1, WINDOW_W, EMBED_DIM)
    res = mix(window, pred, bank, tau=0.0)
    assert res.mode == "predictor_plus_bank"
    assert res.instance_cosines is not None
    assert res.instance_indices is not None


def test_compute_tau_from_confidences_median():
    confidences = np.arange(0.0, 100.0, 1.0)
    tau = compute_tau_from_confidences(confidences, start_step=10, end_step=20)
    # Window is [10, 20) = values 10..19; median is 14.5.
    assert abs(tau - 14.5) < 1e-9


def test_compute_tau_rejects_empty_window():
    with pytest.raises(ValueError):
        compute_tau_from_confidences(np.zeros(5), start_step=10, end_step=20)

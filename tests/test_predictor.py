"""Predictor shape / loss / parameter-count invariants."""

import math

import numpy as np
import pytest
import torch

from src.config import EMBED_DIM, PREDICT_K, WINDOW_W
from src.predictor.inner_pam import (
    InnerPAM,
    all_parameter_count,
    confidence_from_log_var,
    gaussian_nll_loss,
    trainable_parameter_count,
)


def test_predictor_forward_shapes():
    torch.manual_seed(0)
    pred = InnerPAM()
    window = torch.zeros(3, WINDOW_W, EMBED_DIM)
    mean, log_var = pred(window)
    assert mean.shape == (3, PREDICT_K, EMBED_DIM)
    assert log_var.shape == (3, PREDICT_K)
    assert torch.isfinite(mean).all()
    assert torch.isfinite(log_var).all()


def test_predictor_param_count_within_tolerance():
    pred = InnerPAM()
    n = trainable_parameter_count(pred)
    target = 21_000_000
    assert abs(n - target) <= target * 0.10, (
        f"trainable params {n} outside 10% of target {target}"
    )
    assert all_parameter_count(pred) == n  # everything is trainable in v0


def test_predictor_log_var_clamped():
    pred = InnerPAM()
    # Manually push a forward through extreme inputs.
    window = torch.randn(2, WINDOW_W, EMBED_DIM) * 100.0
    _, log_var = pred(window)
    assert float(log_var.min()) >= -10.0 - 1e-6
    assert float(log_var.max()) <= 10.0 + 1e-6


def test_gaussian_nll_matches_closed_form():
    # mu, target, sigma^2 = 1 -> L_k = 0.5 * (||target - mu||^2 + d * 0) per step.
    d = 4
    k = 2
    mu = torch.tensor([[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]])
    target = torch.tensor([[[1.0, 0.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]]])
    log_var = torch.zeros(1, k)  # log_var = 0 -> sigma^2 = 1
    # step 1: sq_err = 0; step 2: sq_err = 1; d*log_var = 0.
    # Loss = 0.5 * (0 + 1) summed over K = 0.5
    loss = gaussian_nll_loss(mu, log_var, target)
    assert abs(loss.item() - 0.5) < 1e-6


def test_gaussian_nll_penalises_overconfidence():
    # If mu is wrong and log_var is very negative (overconfident), loss explodes.
    d = 4
    k = 1
    mu = torch.tensor([[[0.0, 0.0, 0.0, 0.0]]])
    target = torch.tensor([[[1.0, 1.0, 1.0, 1.0]]])
    log_var_overconfident = torch.tensor([[-5.0]])  # sigma^2 ~= 0.0067
    log_var_calibrated = torch.tensor([[0.0]])
    loss_oc = gaussian_nll_loss(mu, log_var_overconfident, target)
    loss_cal = gaussian_nll_loss(mu, log_var_calibrated, target)
    assert loss_oc.item() > loss_cal.item() * 10


def test_gaussian_nll_target_detached_from_grad():
    pred = InnerPAM()
    window = torch.randn(1, WINDOW_W, EMBED_DIM, requires_grad=False)
    target = torch.randn(1, PREDICT_K, EMBED_DIM, requires_grad=True)
    mean, log_var = pred(window)
    loss = gaussian_nll_loss(mean, log_var, target)
    loss.backward()
    # Target had requires_grad=True but should not have received a gradient because
    # gaussian_nll_loss internally detaches it.
    assert target.grad is None


def test_confidence_from_log_var():
    log_var = torch.tensor([[1.0, 2.0, 3.0, 4.0], [-1.0, -2.0, -3.0, -4.0]])
    conf = confidence_from_log_var(log_var, m=3)
    assert conf.shape == (2,)
    # -mean(log_var[:, :3]) = -mean([1,2,3]) = -2.0; -mean([-1,-2,-3]) = 2.0
    assert abs(conf[0].item() - (-2.0)) < 1e-6
    assert abs(conf[1].item() - 2.0) < 1e-6

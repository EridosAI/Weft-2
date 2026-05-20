"""Predictor architecture tests for v1 (spec §7.2-§7.4, §3.3, §4.1)."""

from __future__ import annotations

import pytest
import torch

from v1.src.config import EMBED_DIM, PREDICT_K, PRED_HIDDEN, WINDOW_W
from v1.src.predictor.inner_pam_v1_ablation1 import InnerPAM_v1_Ablation1
from v1.src.predictor.inner_pam_v1_ablation2 import InnerPAM_v1_Ablation2
from v1.src.predictor.inner_pam_v1_primary import InnerPAM_v1_Primary
from v1.src.predictor.inner_pam_v1_shared import (
    LOG_VAR_CLAMP_MAX,
    LOG_VAR_CLAMP_MIN,
    path_prediction_loss,
    trainable_parameter_count,
)


DECODER_N_LAYERS = 2  # PRE-C placeholder for tests


def _random_window(b: int = 2) -> torch.Tensor:
    torch.manual_seed(0)
    w = torch.randn(b, WINDOW_W, EMBED_DIM)
    return w / w.norm(dim=-1, keepdim=True)


def _random_target(b: int = 2) -> torch.Tensor:
    torch.manual_seed(1)
    t = torch.randn(b, PREDICT_K, EMBED_DIM)
    return t / t.norm(dim=-1, keepdim=True)


# --------------------------------------------------------------------------
# Spec §7.2.1: decoder_n_layers must be explicit (no silent default)
# --------------------------------------------------------------------------


def test_primary_raises_on_default_decoder_n_layers():
    with pytest.raises(ValueError, match="decoder_n_layers"):
        InnerPAM_v1_Primary()


def test_ablation1_raises_on_default_decoder_n_layers():
    with pytest.raises(ValueError, match="decoder_n_layers"):
        InnerPAM_v1_Ablation1()


# --------------------------------------------------------------------------
# Spec §7.2 / §7.3: Primary and Ablation 1 output shape contracts
# --------------------------------------------------------------------------


def test_primary_forward_shape():
    model = InnerPAM_v1_Primary(decoder_n_layers=DECODER_N_LAYERS)
    window = _random_window()
    mean, log_var = model(window)
    assert mean.shape == (window.shape[0], PREDICT_K, EMBED_DIM)
    assert log_var.shape == (window.shape[0], PREDICT_K)


def test_ablation1_forward_shape():
    model = InnerPAM_v1_Ablation1(decoder_n_layers=DECODER_N_LAYERS)
    window = _random_window()
    mean, log_var = model(window)
    assert mean.shape == (window.shape[0], PREDICT_K, EMBED_DIM)
    assert log_var.shape == (window.shape[0], PREDICT_K)


def test_ablation2_forward_shape():
    model = InnerPAM_v1_Ablation2()
    window = _random_window()
    mean, log_var = model(window)
    assert mean.shape == (window.shape[0], PREDICT_K, EMBED_DIM)
    assert log_var.shape == (window.shape[0], PREDICT_K)


# --------------------------------------------------------------------------
# Spec §7.2.4 P1 / §7.3.4 A1.1: K output queries per-K parameters
# --------------------------------------------------------------------------


def test_primary_output_queries_per_k_parameters():
    model = InnerPAM_v1_Primary(decoder_n_layers=DECODER_N_LAYERS)
    assert model.output_queries.numel() == PREDICT_K * PRED_HIDDEN


def test_ablation1_output_queries_per_k_parameters():
    model = InnerPAM_v1_Ablation1(decoder_n_layers=DECODER_N_LAYERS)
    assert model.output_queries.numel() == PREDICT_K * PRED_HIDDEN


# --------------------------------------------------------------------------
# Spec §7.3.4 A1.3: Ablation 1 variance is parameter-shared across K
# --------------------------------------------------------------------------


def test_ablation1_shared_log_var_is_scalar():
    model = InnerPAM_v1_Ablation1(decoder_n_layers=DECODER_N_LAYERS)
    assert model.shared_log_var.numel() == 1
    # All K log-variance outputs read from the same parameter:
    window = _random_window()
    mean, log_var = model(window)
    # Per-K log_var values should be identical (broadcast from one scalar
    # before clamping).
    assert torch.allclose(log_var, log_var[:, :1].expand_as(log_var))


# --------------------------------------------------------------------------
# Spec §7.4.4: Ablation 2 architecture matches v0 verbatim
# --------------------------------------------------------------------------


def test_ablation2_output_proj_dimensions():
    import torch.nn as nn

    model = InnerPAM_v1_Ablation2()
    assert isinstance(model.output_proj, nn.Linear)
    assert model.output_proj.out_features == PREDICT_K * (EMBED_DIM + 1)


def test_ablation2_has_no_output_queries_or_decoder():
    model = InnerPAM_v1_Ablation2()
    keys = list(model.state_dict().keys())
    assert not any(k.startswith("output_queries") for k in keys)
    assert not any(k.startswith("decoder") for k in keys)


# --------------------------------------------------------------------------
# Spec §7.2.4 P3 / §7.3.4 (A1 analogue): per-K parameter isolation in head
# --------------------------------------------------------------------------


def test_primary_per_k_variance_head_isolation():
    """log_var[:, k] should depend only on decoded[:, k, :] in Primary."""
    model = InnerPAM_v1_Primary(decoder_n_layers=DECODER_N_LAYERS)
    window = _random_window()
    b = window.shape[0]
    # Reconstruct intermediate decoded with gradient tracking.
    x = model.input_proj(window)
    positions = torch.arange(WINDOW_W)
    x = x + model.pos_emb(positions).unsqueeze(0)
    memory = model.encoder(x)
    queries = model.output_queries.unsqueeze(0).expand(b, -1, -1)
    decoded = model.decoder(queries, memory)
    decoded = decoded.detach().clone().requires_grad_(True)
    log_var = model.output_proj_log_var(decoded).squeeze(-1)
    k_target = 0
    out = torch.zeros_like(log_var)
    out[:, k_target] = 1.0
    log_var.backward(out)
    mask_other = torch.ones(PREDICT_K, dtype=torch.bool)
    mask_other[k_target] = False
    assert decoded.grad[:, mask_other, :].abs().sum().item() == 0.0
    assert decoded.grad[:, k_target, :].abs().sum().item() > 0.0


# --------------------------------------------------------------------------
# Spec §4.1: path-prediction loss
# --------------------------------------------------------------------------


def test_path_prediction_loss_shapes_and_finite():
    model = InnerPAM_v1_Primary(decoder_n_layers=DECODER_N_LAYERS)
    window = _random_window()
    target = _random_target()
    mean, log_var = model(window)
    loss = path_prediction_loss(mean, log_var, target)
    assert loss.ndim == 0
    assert torch.isfinite(loss).item()


def test_path_prediction_loss_target_detached():
    """Loss gradient should not flow into target (target is the encoder output;
    encoder is frozen, but the detach is explicit in path_prediction_loss)."""
    model = InnerPAM_v1_Primary(decoder_n_layers=DECODER_N_LAYERS)
    window = _random_window()
    target = _random_target().requires_grad_(True)
    mean, log_var = model(window)
    loss = path_prediction_loss(mean, log_var, target)
    loss.backward()
    assert target.grad is None or target.grad.abs().sum().item() == 0.0


def test_log_var_within_clamp():
    model = InnerPAM_v1_Primary(decoder_n_layers=DECODER_N_LAYERS)
    window = _random_window()
    _, log_var = model(window)
    assert (log_var >= LOG_VAR_CLAMP_MIN).all().item()
    assert (log_var <= LOG_VAR_CLAMP_MAX).all().item()


# --------------------------------------------------------------------------
# Spec §4.1: per-step Gaussian NLL value-correctness (analytical check)
# --------------------------------------------------------------------------


def test_path_prediction_loss_value_isotropic():
    """For target = mean exactly, loss = 0.5 * d * log_var; verify
    analytical agreement at a deterministic point."""
    torch.manual_seed(7)
    b, k, d = 1, 1, 4
    mean = torch.randn(b, k, d)
    log_var = torch.full((b, k), 0.3)
    target = mean.clone()
    expected = 0.5 * d * 0.3
    loss = path_prediction_loss(mean, log_var, target)
    assert abs(loss.item() - expected) < 1e-5


def test_path_prediction_loss_value_with_error():
    """Quadratic-error term contribution adds to the analytical baseline."""
    torch.manual_seed(7)
    b, k, d = 1, 1, 4
    mean = torch.zeros(b, k, d)
    log_var = torch.zeros((b, k))  # sigma^2 = 1
    target = torch.tensor([[[1.0, 2.0, 0.0, 0.0]]])  # sq_err = 1 + 4 = 5
    expected = 0.5 * (5.0 / 1.0 + d * 0.0)
    loss = path_prediction_loss(mean, log_var, target)
    assert abs(loss.item() - expected) < 1e-5


# --------------------------------------------------------------------------
# Primary and Ablation 1 share the encoder body architecturally
# --------------------------------------------------------------------------


def test_primary_and_ablation1_share_encoder_architecture():
    p = InnerPAM_v1_Primary(decoder_n_layers=DECODER_N_LAYERS)
    a = InnerPAM_v1_Ablation1(decoder_n_layers=DECODER_N_LAYERS)
    # State dict keys for encoder + input_proj + pos_emb must match.
    p_keys = sorted(k for k in p.state_dict() if k.startswith(("encoder", "input_proj", "pos_emb")))
    a_keys = sorted(k for k in a.state_dict() if k.startswith(("encoder", "input_proj", "pos_emb")))
    assert p_keys == a_keys


# --------------------------------------------------------------------------
# Source-text discipline: no [:, -1, :] pattern in Primary / Ablation 1
# --------------------------------------------------------------------------


def test_primary_source_has_no_last_token_pattern():
    import inspect
    src = inspect.getsource(
        inspect.getmodule(InnerPAM_v1_Primary)  # type: ignore[arg-type]
    )
    import re
    assert not re.search(r"\[:,\s*-1,\s*:\]", src)


def test_ablation1_source_has_no_last_token_pattern():
    import inspect
    src = inspect.getsource(
        inspect.getmodule(InnerPAM_v1_Ablation1)  # type: ignore[arg-type]
    )
    import re
    assert not re.search(r"\[:,\s*-1,\s*:\]", src)


def test_ablation2_inherits_v0_last_token_pattern():
    """Ablation 2 is v0 InnerPAM; the parent module should contain the
    pooled readout pattern."""
    import inspect
    import re
    from v0.src.predictor.inner_pam import InnerPAM as V0InnerPAM
    src = inspect.getsource(inspect.getmodule(V0InnerPAM))  # type: ignore[arg-type]
    assert re.search(r"\[:,\s*-1,\s*:\]", src) is not None


# --------------------------------------------------------------------------
# Spec §7.5: three predictor classes are architecturally distinct
# --------------------------------------------------------------------------


def test_predictor_classes_are_distinct_types():
    assert InnerPAM_v1_Primary is not InnerPAM_v1_Ablation1
    assert InnerPAM_v1_Primary is not InnerPAM_v1_Ablation2
    assert InnerPAM_v1_Ablation1 is not InnerPAM_v1_Ablation2
    # Different module paths:
    assert InnerPAM_v1_Primary.__module__ != InnerPAM_v1_Ablation1.__module__
    assert InnerPAM_v1_Primary.__module__ != InnerPAM_v1_Ablation2.__module__

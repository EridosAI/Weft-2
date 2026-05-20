"""V2-PRE-D1a unit tests — trajectory assessment on negative NLL (regression guard).

These lock the fix for the two negative-loss bugs (a ">0" plateau guard and a
multiplicative loss-increase tolerance) that produced spurious "100% not
plateaued" / "unstable" flags on the path-prediction NLL (which goes negative).
"""

from __future__ import annotations

from v2.src.preflight.pre_d1a_endpoint_stability import (
    assess_trajectory, baseline_configs, endpoint_configs,
)


def test_plateaued_negative_loss_is_stable_and_not_descending():
    # Negative, nearly flat tail -> plateaued, stable. (Old ">0" guard wrongly
    # returned rel_imp=1.0; old *1.1 wrongly flagged unstable.)
    rel_imp, still_desc, flag = assess_trajectory([-2900.0, -2905.0], nan_inf=False)
    assert abs(rel_imp - (5.0 / 2900.0)) < 1e-6
    assert still_desc is False
    assert flag == "stable"


def test_still_descending_negative_loss():
    rel_imp, still_desc, flag = assess_trajectory([-2700.0, -2920.0], nan_inf=False)
    assert rel_imp > 0.02 and still_desc is True
    assert flag == "stable"          # decreasing (improving) loss is not "unstable"


def test_loss_diverging_upward_is_unstable():
    # Loss got substantially worse (less negative) end-to-end -> unstable.
    _, still_desc, flag = assess_trajectory([-2900.0, -2700.0], nan_inf=False)
    assert flag == "unstable"
    assert still_desc is False


def test_nan_inf_is_divergent():
    _, still_desc, flag = assess_trajectory([1.0, 2.0], nan_inf=True)
    assert flag == "divergent" and still_desc is False


def test_config_counts():
    assert len(endpoint_configs()) == 20      # 5 axes x 2 endpoints x 2 L_d_main
    assert len(baseline_configs()) == 20       # n=20 bit-identical baseline (§11.7)
    # All endpoint L_d_main values are envelope endpoints {1, 4}.
    assert {c[2] for c in endpoint_configs()} == {1, 4}

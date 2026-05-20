"""Weft Inner PAM v1 — Ablation 2 arm predictor (readout-topology ablation).

Spec §7.4. v0 `InnerPAM` architecture inherited verbatim. Pooled
`last_token` readout, K · (d + 1) flat output projection, scalar log-var.

Per spec §7.4.5 and instr §3.3, this arm is implemented by *subclassing
v0's `InnerPAM` with a class-name override only* — any change to v0's
`InnerPAM` propagates automatically, and PRE-D's property assertions catch
unintended divergence.

What this ablation isolates (spec §7.4.5): whether v1's substrate change
(stronger perturbation) alone reproduces v0's coupling result.
"""

from __future__ import annotations

from v0.src.predictor.inner_pam import InnerPAM


class InnerPAM_v1_Ablation2(InnerPAM):
    """v0 InnerPAM architecture inherited verbatim for the readout-topology ablation.

    Spec §7.4.2: "Ablation 2 should be code-equivalent to v0's `InnerPAM`
    (modulo class name)." Subclassing without modification enforces this.
    """

    pass

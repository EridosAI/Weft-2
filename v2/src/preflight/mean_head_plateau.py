"""V2 recalibration Stage 1 — mean-head-aware plateau detection.

Recalibration instructions §3.4. Replaces the legacy NLL-loss-plateau criterion
for `V2_TRAINING_STEPS` calibration. The broken-mechanics episode (HANDOFF /
WEFT_INNER_PAM_v2_SESSION_BRIEF §5) established that the variance head saturates
first and masks the mean head, so loss flatness is a trap. This detector keys on
`cos(mean, target)` at k=1 — the mean head's actual fit to the target — and
returns the *post-asymptotic* step per the §3.3 lock criterion.

DEPRECATION POINTER: the legacy NLL-plateau detector
(`v2.src.preflight.pre_d1a_endpoint_stability.assess_trajectory`, driven by
`config.V2_PLATEAU_REL_IMPROVEMENT` / `V2_SMOKE_*`) is retained for the PRE-A
smoke-calibration *record* only. It must NOT drive `V2_TRAINING_STEPS` anymore —
see recalibration instructions §3.4 and §1.5. Use this module instead.
"""

from __future__ import annotations

from typing import Optional, Sequence

# §3.3 lock-criterion constants.
WITHIN_MAX_FRAC: float = 0.99   # condition 1: cos_k1 >= 0.99 * max(cos_k1)
NEXT_GAIN_ABS: float = 0.01     # condition 2: next ckpt cos_k1 - this <= 0.01 (absolute)


def mean_head_plateau_step(
    curve: Sequence[dict],
    within_max_frac: float = WITHIN_MAX_FRAC,
    next_gain_abs: float = NEXT_GAIN_ABS,
) -> Optional[int]:
    """Smallest checkpoint step at which `cos_k1` has reached the post-asymptotic
    regime (recalibration instructions §3.3), or ``None`` if the curve is empty.

    Args:
        curve: list of dicts, each with integer ``"steps"`` and float ``"cos_k1"``.
            Need not be pre-sorted; sorted internally by ascending steps.
        within_max_frac: condition-1 fraction-of-max (default 0.99 → "within 1%").
        next_gain_abs: condition-2 absolute forward-gain tolerance (default 0.01).

    A step qualifies when BOTH hold:
      (1) ``cos_k1 >= within_max_frac * max(cos_k1 over the whole curve)``, AND
      (2) the NEXT checkpoint's ``cos_k1`` does not exceed this one by more than
          ``next_gain_abs``. The final checkpoint has no successor, so condition 2
          is vacuously satisfied there (nothing more is gained within budget).

    Returns the ``steps`` of the earliest qualifying checkpoint. A curve that is
    still climbing meaningfully at the end will only qualify at its last point;
    the caller applies the §3.5 STOP check (lock_step_candidate > 175000).
    """
    if not curve:
        return None
    pts = sorted(curve, key=lambda e: e["steps"])
    cos_max = max(e["cos_k1"] for e in pts)
    threshold = within_max_frac * cos_max
    for i, e in enumerate(pts):
        if e["cos_k1"] < threshold:
            continue  # not yet within 1% of the asymptote
        if i + 1 < len(pts) and (pts[i + 1]["cos_k1"] - e["cos_k1"]) > next_gain_abs:
            continue  # still climbing > next_gain_abs into the next checkpoint
        return int(e["steps"])
    return None


def not_grokked_within_budget(
    curve: Sequence[dict],
    trivial_baseline_cos_k1: float,
    min_clearance: float = 0.10,
) -> bool:
    """True if max `cos_k1` never clears the trivial baseline by `min_clearance`
    (recalibration instructions §3.3 / §3.5 not-grokked-within-budget condition).
    """
    if not curve:
        return True
    cos_max = max(e["cos_k1"] for e in curve)
    return (cos_max - trivial_baseline_cos_k1) < min_clearance


# Back-compat alias matching the §3.4 prose name.
mean_head_plateau_detected = mean_head_plateau_step

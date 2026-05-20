"""PRE-A: Substrate verification (spec §6.1 / §8.3, instr §6.1).

Runs the v0 §5 protocol on the v1 substrate plus v1-specific extensions
(§5.4 magnitude verification, §5.8 locality re-run on v1 mechanism — those
two are handled in PRE-B once the mechanism is selected).

PRE-A checks the *unperturbed* substrate state — environment, encoder
behaviour, render determinism patterns — independent of which perturbation
mechanism PRE-B will select.

This module produces the PRE-A report. The driver script
`scripts/run_pre_a_substrate.py` is responsible for opening the AI2-THOR
controller, capturing frames, and calling the per-check entry points
exposed here.

Operating-mode note. AI2-THOR is not assumed present in every Python
process that imports this module. Helpers below are pure-Python /
NumPy-only; the AI2-THOR-dependent driver script imports them
conditionally.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np

from v1.src.config import (
    FINDING_3_COS_MAX,
    SUBSTRATE_COMBINED_GAP_MIN,
    SUBSTRATE_CROSS_ELEMENT_DISTINGUISH_MAX,
    SUBSTRATE_CROSS_INSTANCE_STABILITY_MIN,
)


# --------------------------------------------------------------------------
# §5.1 cross-instance stability
# --------------------------------------------------------------------------


@dataclass
class CheckResult:
    name: str
    value: float
    threshold: float
    threshold_direction: str   # "greater" or "less"
    passed: bool
    notes: str = ""


def cross_instance_stability(
    embeddings_by_item: dict[int, np.ndarray],
) -> CheckResult:
    """§5.1: mean cosine of embeddings across instances of the same item.

    `embeddings_by_item[item]` is an (N_instances, d) array of L2-normalised
    embeddings collected at the same viewing position across loops.
    """
    cosines: list[float] = []
    for item, embs in embeddings_by_item.items():
        if embs.shape[0] < 2:
            continue
        # Pair-wise cosines across all distinct ordered pairs (excluding self).
        sims = embs @ embs.T
        n = sims.shape[0]
        mask = ~np.eye(n, dtype=bool)
        cosines.extend(sims[mask].tolist())
    mean_cos = float(np.mean(cosines)) if cosines else float("nan")
    return CheckResult(
        name="5.1_cross_instance_stability",
        value=mean_cos,
        threshold=SUBSTRATE_CROSS_INSTANCE_STABILITY_MIN,
        threshold_direction="greater",
        passed=mean_cos > SUBSTRATE_CROSS_INSTANCE_STABILITY_MIN,
        notes=f"n_pairs={len(cosines)}",
    )


def cross_element_distinguishability(
    embeddings_by_item: dict[int, np.ndarray],
) -> CheckResult:
    """§5.2: mean cosine across different items."""
    item_means = {
        i: e.mean(axis=0) / (np.linalg.norm(e.mean(axis=0)) + 1e-12)
        for i, e in embeddings_by_item.items()
    }
    items = sorted(item_means.keys())
    cosines: list[float] = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            a = item_means[items[i]]
            b = item_means[items[j]]
            cosines.append(float(np.dot(a, b)))
    mean_cos = float(np.mean(cosines)) if cosines else float("nan")
    return CheckResult(
        name="5.2_cross_element_distinguishability",
        value=mean_cos,
        threshold=SUBSTRATE_CROSS_ELEMENT_DISTINGUISH_MAX,
        threshold_direction="less",
        passed=mean_cos < SUBSTRATE_CROSS_ELEMENT_DISTINGUISH_MAX,
        notes=f"n_pairs={len(cosines)}",
    )


def combined_gap(stability: CheckResult, distinguish: CheckResult) -> CheckResult:
    """§5.3: §5.1 − §5.2 ≥ 0.15."""
    gap = stability.value - distinguish.value
    return CheckResult(
        name="5.3_combined_gap",
        value=gap,
        threshold=SUBSTRATE_COMBINED_GAP_MIN,
        threshold_direction="greater",
        passed=gap >= SUBSTRATE_COMBINED_GAP_MIN,
    )


# --------------------------------------------------------------------------
# Eight v0 substrate-finding checks (instr §6.1.3 / spec §8.4)
# --------------------------------------------------------------------------


def python_version_check() -> CheckResult:
    """v0 finding 1: Python 3.12.3 environment."""
    import sys
    pv = sys.version_info
    expected = (3, 12, 3)
    actual = (pv.major, pv.minor, pv.micro)
    passed = actual == expected
    return CheckResult(
        name="finding_1_python_version",
        value=float(pv.minor + pv.micro / 100.0),
        threshold=12.03,
        threshold_direction="equal",
        passed=passed,
        notes=f"actual={pv.major}.{pv.minor}.{pv.micro}, expected {expected}",
    )


def embeddings_full_population_check(embeddings: np.ndarray) -> CheckResult:
    """v0 finding 2: no zero rows, L2 norms in tolerance."""
    if embeddings.size == 0:
        return CheckResult(
            "finding_2_embeddings_full_population",
            0.0,
            1.0,
            "equal",
            False,
            "empty embeddings array",
        )
    norms = np.linalg.norm(embeddings, axis=1)
    zero_rows = int((norms == 0).sum())
    tolerance_violations = int(((norms - 1.0).__abs__() > 1e-5).sum())
    return CheckResult(
        name="finding_2_embeddings_full_population",
        value=float(zero_rows + tolerance_violations),
        threshold=0,
        threshold_direction="equal",
        passed=(zero_rows == 0 and tolerance_violations == 0),
        notes=f"zero_rows={zero_rows}, l2_norm_violations={tolerance_violations}",
    )


def continuous_motion_check(embeddings: np.ndarray, cos_max: float = FINDING_3_COS_MAX) -> CheckResult:
    """v0 finding 3: no 30-frame static dwell.

    Verified by checking that no consecutive embedding pair has cosine
    > `cos_max`. Threshold recalibrated 0.9999 → 0.999 on 2026-05-19 per
    design-chat determination (substrate-as-feature classification on the
    25 transit→close_up structural-handoff duplicates; v0 STOP_REPORT
    precedent). SCAFFOLDING constant `FINDING_3_COS_MAX` in v1/src/config.py.
    """
    if embeddings.shape[0] < 2:
        return CheckResult(
            "finding_3_continuous_motion",
            0.0,
            cos_max,
            "less",
            False,
            "fewer than 2 embeddings",
        )
    a = embeddings[:-1]
    b = embeddings[1:]
    cos = (a * b).sum(axis=1) / (
        np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12
    )
    bad = int((cos > cos_max).sum())
    return CheckResult(
        name="finding_3_continuous_motion",
        value=float(bad),
        threshold=0,
        threshold_direction="equal",
        passed=(bad == 0),
        notes=f"pairs_with_cos>{cos_max}: {bad}/{len(cos)}",
    )


# --------------------------------------------------------------------------
# Report writing
# --------------------------------------------------------------------------


def write_pre_a_report(checks: list[CheckResult], path: Path) -> bool:
    """Write the PRE-A report. Returns True iff all checks PASS."""
    path.parent.mkdir(parents=True, exist_ok=True)
    all_pass = all(c.passed for c in checks)
    path.write_text(
        json.dumps(
            {
                "all_passed": all_pass,
                "checks": [c.__dict__ for c in checks],
            },
            indent=2,
        )
    )
    return all_pass

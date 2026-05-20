"""PRE-B: Perturbation mechanism selection (spec §8.2, instr §6.2).

Tests four candidate mechanisms against the spec §8.2.1 verification
criteria (magnitude, locality, reproducibility, no substrate corruption),
plus per-candidate loop-length calibration. Selects the first candidate (in
spec §8.2.2 priority order) that passes all criteria.

Selected candidate is written to `selected.json` at
`PATHS.results_pre_b / "selected.json"` with keys:
  {"mechanism": str, "frames_per_loop": int, "magnitude": float, ...}

Candidates (spec §8.2.2):
  1. per_object_material_setting
  2. asset_replacement
  3. hand_built_texture_swaps
  4. alternate_procthor_scene

The AI2-THOR interaction itself happens in the driver script; this module
provides:
  - The criteria-evaluation function for a candidate's measured metrics.
  - The candidate-priority ordering.
  - The report-writing helpers.

A candidate's measurements are obtained by the driver script (which runs
AI2-THOR + DINOv2) and passed in as a `CandidateMeasurements` object.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from v1.src.config import (
    PERTURBATION_LOCALITY_MAX,
    PERTURBATION_MAGNITUDE_MAX,
    PERTURBATION_MAGNITUDE_MIN,
    PERTURBATION_REPRODUCIBILITY_TOL,
)


CANDIDATE_PRIORITY: tuple[str, ...] = (
    "per_object_material_setting",
    "asset_replacement",
    "hand_built_texture_swaps",
    "alternate_procthor_scene",
)


@dataclass
class CandidateMeasurements:
    """Measurements for a single candidate mechanism produced by the driver.

    All cosine drops are 1 − cosine; all values are at viewing position 1 of
    each item per spec §8.2.1.
    """

    candidate: str
    perturbed_item_cosine_drops: dict[int, float]  # vp_id -> cosine drop
    unperturbed_item_cosine_drops: dict[int, float]  # vp_id -> cosine drop
    reproducibility_run1: dict[int, float]          # vp_id -> cosine drop (run 1)
    reproducibility_run2: dict[int, float]          # vp_id -> cosine drop (run 2)
    eight_finding_checks: dict[str, bool]
    frames_per_loop: int
    api_success: bool


@dataclass
class CandidateVerdict:
    candidate: str
    magnitude_ok: bool
    locality_ok: bool
    reproducibility_ok: bool
    substrate_ok: bool
    api_ok: bool
    overall_pass: bool
    detail: dict


def evaluate_candidate(m: CandidateMeasurements) -> CandidateVerdict:
    """Apply spec §8.2.1 verification criteria to a candidate's measurements."""
    # Magnitude: every perturbed item's drop must be in [MIN, MAX].
    in_band = [
        PERTURBATION_MAGNITUDE_MIN <= v <= PERTURBATION_MAGNITUDE_MAX
        for v in m.perturbed_item_cosine_drops.values()
    ]
    magnitude_ok = all(in_band) and len(in_band) > 0

    # Locality: every unperturbed item's drop must be < PERTURBATION_LOCALITY_MAX.
    locality_ok = all(
        v < PERTURBATION_LOCALITY_MAX for v in m.unperturbed_item_cosine_drops.values()
    )

    # Reproducibility: per-item |run1 - run2| < tol.
    diffs = []
    for vp in m.reproducibility_run1:
        if vp in m.reproducibility_run2:
            diffs.append(abs(m.reproducibility_run1[vp] - m.reproducibility_run2[vp]))
    reproducibility_ok = all(d < PERTURBATION_REPRODUCIBILITY_TOL for d in diffs)

    # Substrate: eight-finding checks all pass.
    substrate_ok = all(m.eight_finding_checks.values()) and len(m.eight_finding_checks) > 0

    api_ok = m.api_success
    overall_pass = (
        magnitude_ok and locality_ok and reproducibility_ok and substrate_ok and api_ok
    )

    return CandidateVerdict(
        candidate=m.candidate,
        magnitude_ok=magnitude_ok,
        locality_ok=locality_ok,
        reproducibility_ok=reproducibility_ok,
        substrate_ok=substrate_ok,
        api_ok=api_ok,
        overall_pass=overall_pass,
        detail={
            "perturbed_item_cosine_drops": m.perturbed_item_cosine_drops,
            "unperturbed_item_cosine_drops": m.unperturbed_item_cosine_drops,
            "reproducibility_diffs": diffs,
            "eight_finding_checks": m.eight_finding_checks,
            "frames_per_loop": m.frames_per_loop,
            "api_success": m.api_success,
        },
    )


def select_candidate(verdicts: list[CandidateVerdict]) -> Optional[CandidateVerdict]:
    """Select the highest-priority candidate that passes all criteria.

    Returns None if no candidate passes.
    """
    pri = {c: i for i, c in enumerate(CANDIDATE_PRIORITY)}
    passing = [v for v in verdicts if v.overall_pass]
    if not passing:
        return None
    passing.sort(key=lambda v: pri.get(v.candidate, 10_000))
    return passing[0]


def write_candidate_report(
    measurements: CandidateMeasurements,
    verdict: CandidateVerdict,
    output_dir: Path,
) -> None:
    """Write per-candidate report: pre_b_report.{md,json}."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "pre_b_report.json").write_text(
        json.dumps(
            {
                "candidate": measurements.candidate,
                "measurements": measurements.__dict__,
                "verdict": {
                    "magnitude_ok": verdict.magnitude_ok,
                    "locality_ok": verdict.locality_ok,
                    "reproducibility_ok": verdict.reproducibility_ok,
                    "substrate_ok": verdict.substrate_ok,
                    "api_ok": verdict.api_ok,
                    "overall_pass": verdict.overall_pass,
                    "detail": verdict.detail,
                },
            },
            indent=2,
        )
    )


def write_summary(verdicts: list[CandidateVerdict], output_dir: Path) -> Optional[CandidateVerdict]:
    """Write summary.{json}; return the selected candidate or None."""
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = select_candidate(verdicts)
    payload = {
        "candidates": [
            {
                "candidate": v.candidate,
                "magnitude_ok": v.magnitude_ok,
                "locality_ok": v.locality_ok,
                "reproducibility_ok": v.reproducibility_ok,
                "substrate_ok": v.substrate_ok,
                "api_ok": v.api_ok,
                "overall_pass": v.overall_pass,
            }
            for v in verdicts
        ],
        "selected": selected.candidate if selected is not None else None,
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2))
    return selected


def write_selection_lock(
    selected: CandidateVerdict,
    measurements: CandidateMeasurements,
    lock_path: Path,
) -> None:
    """Write the config.py-readable selection lock at PATHS.results_pre_b / 'selected.json'."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(
        json.dumps(
            {
                "mechanism": selected.candidate,
                "frames_per_loop": int(measurements.frames_per_loop),
                "magnitude_band": [
                    PERTURBATION_MAGNITUDE_MIN,
                    PERTURBATION_MAGNITUDE_MAX,
                ],
                "measured_perturbed_drops": measurements.perturbed_item_cosine_drops,
                "measured_unperturbed_drops": measurements.unperturbed_item_cosine_drops,
            },
            indent=2,
        )
    )

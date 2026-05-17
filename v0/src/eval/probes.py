"""Probe construction for the v0 evaluation harness (instr §6.1).

Two kinds of probes per phase, drawn from the held-out region:

- steady-state: window of W=16 dwell frames at the same viewing position,
  target = next K=16 actual frames.
- cue: window straddling the dwell-to-transit boundary (last 8 dwell at A,
  first 8 transit toward B), target = next K=16 actual frames.

Each probe records (phase, probe_type, from_item, to_item|None, loop_idx,
window_start_frame, window_end_frame).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence

import numpy as np

from v0.src.config import (
    PREDICT_K,
    PROBES_PER_TRANSITION,
    PROBES_PER_VIEWING_POSITION,
    ROUTE_TRANSITIONS,
    SEED_PROBE_SAMPLING,
    VIEWING_POSITION_IDS,
    WINDOW_W,
)


@dataclass(frozen=True)
class Probe:
    probe_type: str          # "steady" | "cue"
    from_item: int           # viewing_position_id
    to_item: Optional[int]   # next item (cue only); None for steady
    loop_idx: int
    window_start: int        # inclusive
    window_end: int          # inclusive (window of W frames: window_end - window_start + 1 == W)
    target_start: int        # inclusive (target K frames: target_start..target_start+K-1)


def _annotation_phase(ann: dict[str, Any]) -> str:
    return str(ann.get("phase", "transit"))


def _viewing_position(ann: dict[str, Any]) -> Optional[int]:
    vp = ann.get("viewing_position_id")
    if vp in (None, 0):
        return None
    return int(vp)


def _loop_index(ann: dict[str, Any]) -> int:
    return int(ann.get("loop_index", -1))


def build_steady_state_probes(
    annotations: Sequence[dict[str, Any]],
    held_out_start: int,
    held_out_end: int,
    rng: np.random.Generator,
    per_position: int = PROBES_PER_VIEWING_POSITION,
) -> list[Probe]:
    """Sample steady-state probes whose window+target lie in the held-out region."""
    n = held_out_end - held_out_start
    if n < WINDOW_W + PREDICT_K:
        return []
    candidates_by_vp: dict[int, list[int]] = {vp: [] for vp in VIEWING_POSITION_IDS}
    last_valid_window_end = held_out_end - PREDICT_K - 1
    for window_end in range(held_out_start + WINDOW_W - 1, last_valid_window_end + 1):
        window_anns = annotations[window_end - WINDOW_W + 1 : window_end + 1]
        first_phase = _annotation_phase(window_anns[0])
        if first_phase != "dwell":
            continue
        first_vp = _viewing_position(window_anns[0])
        if first_vp is None:
            continue
        if not all(_annotation_phase(a) == "dwell" for a in window_anns):
            continue
        if not all(_viewing_position(a) == first_vp for a in window_anns):
            continue
        candidates_by_vp[first_vp].append(window_end)
    out: list[Probe] = []
    for vp, ends in candidates_by_vp.items():
        if not ends:
            continue
        ends_arr = np.asarray(ends, dtype=np.int64)
        chosen = rng.choice(
            ends_arr, size=min(per_position, len(ends_arr)), replace=False
        )
        for window_end in chosen:
            window_end = int(window_end)
            loop_idx = _loop_index(annotations[window_end])
            out.append(
                Probe(
                    probe_type="steady",
                    from_item=int(vp),
                    to_item=None,
                    loop_idx=int(loop_idx),
                    window_start=window_end - WINDOW_W + 1,
                    window_end=window_end,
                    target_start=window_end + 1,
                )
            )
    return out


def build_cue_probes(
    annotations: Sequence[dict[str, Any]],
    held_out_start: int,
    held_out_end: int,
    rng: np.random.Generator,
    per_transition: int = PROBES_PER_TRANSITION,
    half_w: int = WINDOW_W // 2,
) -> list[Probe]:
    """Sample cue probes whose window straddles a dwell-to-transit boundary."""
    last_valid_window_end = held_out_end - PREDICT_K - 1
    candidates_by_pair: dict[tuple[int, int], list[Probe]] = {
        t: [] for t in ROUTE_TRANSITIONS
    }
    for window_end in range(held_out_start + WINDOW_W - 1, last_valid_window_end + 1):
        window_anns = annotations[window_end - WINDOW_W + 1 : window_end + 1]
        first_half = window_anns[:half_w]
        second_half = window_anns[half_w:]
        if not all(_annotation_phase(a) == "dwell" for a in first_half):
            continue
        from_vp = _viewing_position(first_half[0])
        if from_vp is None:
            continue
        if not all(_viewing_position(a) == from_vp for a in first_half):
            continue
        if not all(_annotation_phase(a) == "transit" for a in second_half):
            continue
        # Determine "toward" by scanning forward through the stream until the
        # next dwell frame appears. Transit segments are typically 60+ frames
        # while PREDICT_K is 16, so an earlier version of this loop that
        # peeked only K frames ahead skipped almost every candidate.
        to_vp = None
        for j in range(window_end + 1, len(annotations)):
            if _annotation_phase(annotations[j]) == "dwell":
                to_vp = _viewing_position(annotations[j])
                break
        if to_vp is None:
            continue
        pair = (int(from_vp), int(to_vp))
        if pair not in candidates_by_pair:
            continue  # not a recognised route transition
        candidates_by_pair[pair].append(
            Probe(
                probe_type="cue",
                from_item=int(from_vp),
                to_item=int(to_vp),
                loop_idx=_loop_index(window_anns[0]),
                window_start=window_end - WINDOW_W + 1,
                window_end=window_end,
                target_start=window_end + 1,
            )
        )
    out: list[Probe] = []
    for pair, probes in candidates_by_pair.items():
        if not probes:
            continue
        idx = rng.choice(
            len(probes), size=min(per_transition, len(probes)), replace=False
        )
        for i in idx:
            out.append(probes[int(i)])
    return out


def build_probes(
    annotations: Sequence[dict[str, Any]],
    held_out_start: int,
    held_out_end: int,
    seed: int = SEED_PROBE_SAMPLING,
) -> dict[str, list[Probe]]:
    rng_steady = np.random.default_rng(seed)
    rng_cue = np.random.default_rng(seed + 1)
    return {
        "steady": build_steady_state_probes(annotations, held_out_start, held_out_end, rng_steady),
        "cue": build_cue_probes(annotations, held_out_start, held_out_end, rng_cue),
    }


def compute_held_out_boundary(
    annotations: Sequence[dict[str, Any]],
    held_out_loops: int,
) -> tuple[int, int]:
    """Find the frame index marking the start of the last `held_out_loops` loops."""
    if not annotations:
        return 0, 0
    last_loop = max(_loop_index(a) for a in annotations)
    first_held_out_loop = last_loop - held_out_loops + 1
    if first_held_out_loop <= 0:
        raise ValueError(
            f"only {last_loop + 1} loops available; cannot reserve {held_out_loops}"
        )
    for i, a in enumerate(annotations):
        if _loop_index(a) == first_held_out_loop:
            return int(i), int(len(annotations))
    raise ValueError(f"could not locate first held-out loop {first_held_out_loop}")

"""Probe construction tests on a synthetic stream."""

import numpy as np

from v0.src.config import PREDICT_K, WINDOW_W
from v0.src.eval.probes import (
    Probe,
    build_cue_probes,
    build_steady_state_probes,
    compute_held_out_boundary,
)


def _synth_annotations(n_loops: int = 12, dwell_per_item: int = 30,
                       transit_per_segment: int = 60) -> list[dict]:
    """Build annotations for a deterministic 5-item route over n_loops loops."""
    items = [1, 2, 3, 4, 5]
    out: list[dict] = []
    fi = 0
    for loop in range(n_loops):
        for vp in items:
            for _ in range(dwell_per_item):
                out.append({"frame_idx": fi, "loop_index": loop,
                            "phase": "dwell", "viewing_position_id": vp})
                fi += 1
            for _ in range(transit_per_segment):
                out.append({"frame_idx": fi, "loop_index": loop,
                            "phase": "transit", "viewing_position_id": 0})
                fi += 1
    return out


def test_held_out_boundary():
    anns = _synth_annotations(n_loops=12)
    start, end = compute_held_out_boundary(anns, held_out_loops=2)
    assert end == len(anns)
    # First held-out loop is index 10; its first frame is at index 10 * loop_length.
    expected_loop_length = 5 * 30 + 5 * 60
    assert start == 10 * expected_loop_length


def test_steady_state_probes_have_uniform_dwell_window():
    anns = _synth_annotations(n_loops=12)
    start, end = compute_held_out_boundary(anns, held_out_loops=2)
    rng = np.random.default_rng(0)
    probes = build_steady_state_probes(anns, start, end, rng, per_position=5)
    # Up to 5 probes per viewing position × 5 positions = 25.
    assert 1 <= len(probes) <= 25
    for p in probes:
        assert p.probe_type == "steady"
        assert p.to_item is None
        for i in range(p.window_start, p.window_end + 1):
            assert anns[i]["phase"] == "dwell"
            assert anns[i]["viewing_position_id"] == p.from_item
        # Target lives inside the stream.
        assert p.target_start + PREDICT_K - 1 < len(anns)


def test_cue_probes_straddle_dwell_to_transit():
    anns = _synth_annotations(n_loops=12)
    start, end = compute_held_out_boundary(anns, held_out_loops=2)
    rng = np.random.default_rng(0)
    probes = build_cue_probes(anns, start, end, rng, per_transition=5,
                              half_w=WINDOW_W // 2)
    half = WINDOW_W // 2
    for p in probes:
        assert p.probe_type == "cue"
        first_half = anns[p.window_start : p.window_start + half]
        second_half = anns[p.window_start + half : p.window_end + 1]
        assert all(a["phase"] == "dwell" for a in first_half)
        assert all(a["phase"] == "transit" for a in second_half)
        assert all(a["viewing_position_id"] == p.from_item for a in first_half)
        # to_item must be a valid next-item.
        assert p.to_item in {1, 2, 3, 4, 5}

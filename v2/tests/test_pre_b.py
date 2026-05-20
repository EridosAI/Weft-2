"""V2-PRE-B unit tests — cache-independent helpers (segmentation, §2.4, summarise).

The worked-example measurement runs on the gitignored 65k DINOv2 cache, so the
full run isn't a portable test; these cover the pure logic.
"""

from __future__ import annotations

import numpy as np

from v2.src.preflight import pre_b_worked_example_measurement as pb


def _ann(loop, vpid, seg_, apex, pert):
    return {"loop_index": loop, "viewing_position_id": vpid, "phase_segment": seg_,
            "close_up_apex_flag": apex, "perturbation_active": pert}


def test_segment_stage_and_ordinal():
    # Loop 0 clean: transit, then Bed close-up (3 frames, apex middle).
    # Loop 1 perturbed: transit, Bed close-up (2 frames).
    ann = [
        _ann(0, 0, "transit", False, False),
        _ann(0, 1, "close_up", False, False),
        _ann(0, 1, "close_up", True, False),
        _ann(0, 1, "close_up", False, False),
        _ann(1, 0, "transit", False, True),
        _ann(1, 1, "close_up", False, True),
        _ann(1, 1, "close_up", True, True),
    ]
    seg = pb.segment(ann)
    assert seg["n_loops"] == 2
    assert seg["clean_loops"] == [0] and seg["pert_loops"] == [1]
    assert list(seg["stage"]) == ["A", "A", "A", "A", "B", "B", "B"]
    # close_up_ordinal: -1 for transit, then 0,1,2 within loop0 Bed segment; 0,1 within loop1.
    assert list(seg["close_up_ordinal"]) == [-1, 0, 1, 2, -1, 0, 1]


def test_extract_s24_ranges():
    s = pb.extract_s24_ranges()
    assert s["magnitude"]["range"] == [0.0, 1.0] and not s["magnitude"]["ambiguous"]
    assert s["manifold_dim"]["range"] == [1.0, 1024.0]
    assert not s["continuity"]["ambiguous"] and not s["locality"]["ambiguous"]
    # Repetition is genuinely ambiguous -> surfaced for design chat.
    assert s["repetition"]["range"] is None and s["repetition"]["ambiguous"] is True
    # Every bound carries a trace (research_operations §4.1).
    assert all("trace" in v and v["trace"] for v in s.values())


def test_summarise_multimodal_detection():
    rng = np.random.default_rng(0)
    bimodal = np.concatenate([rng.normal(0.01, 0.002, 80), rng.normal(0.05, 0.002, 80)])
    out = pb._summarise(bimodal, bic_threshold=10.0)
    assert out["n"] == 160
    assert out["multimodal"]["multimodal"] is True
    assert min(out["multimodal"]["component_medians"]) < 0.03 < max(out["multimodal"]["component_medians"])

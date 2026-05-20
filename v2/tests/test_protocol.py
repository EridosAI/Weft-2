"""V2-PRE-A unit tests — protocol end-to-end + grid mapping (spec §6.2, §6.3)."""

from __future__ import annotations

import numpy as np

from v2.config import load_calibrated_thresholds
from v2.src.protocol.grid_mapping import detect_multimodal, nearest_grid_point
from v2.src.protocol.protocol import apply_protocol, estimate_reference_state
from v2.src.substrate.base_manifold_trajectory import load_or_create_U
from v2.src.substrate.stream_builder import StreamParams, build_stream


def test_protocol_end_to_end_keys_and_values():
    U = load_or_create_U()
    bs = build_stream(StreamParams(magnitude_M=0.5, fidelity_F=0.999, locality_L=0.5), U)
    rec = apply_protocol(bs.stream, load_calibrated_thresholds())
    for key in (
        "period_P", "fidelity_F", "repetition_coverage", "magnitude_M",
        "locality_L", "locality_defined", "continuity_C", "continuity_C_curv",
        "manifold_D_local", "manifold_D_global", "n_perturbed_detected",
    ):
        assert key in rec, f"missing protocol output key: {key}"
    # Recovers the constructed period and magnitude within tolerance.
    assert rec["period_P"] == bs.construction["period_P"]
    assert abs(rec["magnitude_M"] - bs.construction["magnitude_M"]) < 0.05
    assert rec["locality_defined"] is True


def test_reference_state_recovers_unperturbed_baseline():
    U = load_or_create_U()
    bs = build_stream(StreamParams(magnitude_M=0.5, fidelity_F=0.999), U)
    ref = estimate_reference_state(
        bs.stream / np.linalg.norm(bs.stream, axis=1, keepdims=True),
        bs.construction["period_P"],
    )
    assert ref.shape == bs.stream.shape
    # Reference at a perturbed position should be close to the clean base value
    # (median over the majority-clean repetitions), i.e. far from the perturbed
    # vector itself.
    assert np.allclose(np.linalg.norm(ref, axis=1), 1.0, atol=1e-6)


def test_locality_undefined_without_repetition():
    # A non-repeating stream => reference undefined => locality undefined (§4.2).
    rng = np.random.default_rng(0)
    stream = rng.standard_normal((400, 1024))
    stream /= np.linalg.norm(stream, axis=1, keepdims=True)
    rec = apply_protocol(stream, load_calibrated_thresholds())
    assert rec["period_P"] is None
    assert rec["locality_defined"] is False
    assert rec["locality_L"] is None


def test_grid_mapping_multimodal_detection():
    rng = np.random.default_rng(7)
    bimodal = np.concatenate([rng.normal(0.1, 0.01, 60), rng.normal(0.8, 0.01, 60)])
    unimodal = rng.normal(0.4, 0.02, 120)
    assert detect_multimodal(bimodal, bic_improvement_threshold=10.0)["multimodal"] is True
    assert detect_multimodal(unimodal, bic_improvement_threshold=10.0)["multimodal"] is False
    comp = detect_multimodal(bimodal, 10.0)["component_medians"]
    assert comp is not None and min(comp) < 0.3 < max(comp)


def test_nearest_grid_point():
    grid = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    out = nearest_grid_point(0.4, grid)
    assert out["nearest_value"] == 0.5
    assert abs(out["interpolation_distance"] - 0.1) < 1e-9

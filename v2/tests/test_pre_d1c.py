"""V2-PRE-D1c unit tests — position/classification helpers (instr §7.4)."""

from __future__ import annotations

from v2.src.preflight.pre_d1c_corner_reachability import _classify, _position


def test_position_normalisation():
    assert _position(0.5, [0.0, 1.0]) == 0.5
    assert _position(1.0, [1.0, 1024.0]) == 0.0
    assert abs(_position(13.75, [1.0, 1024.0]) - (12.75 / 1023.0)) < 1e-9


def test_classify_bands():
    assert _classify(0.5) == "central"
    assert _classify(0.25) == "central"        # within middle 60%
    assert _classify(0.15) == "off-center"     # outer 40%, not outer 10%
    assert _classify(0.85) == "off-center"
    assert _classify(0.05) == "near-extreme"   # outer 10%
    assert _classify(0.95) == "near-extreme"

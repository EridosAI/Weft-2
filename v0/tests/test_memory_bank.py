"""Memory bank append / retrieve / hard-cap / serialise tests."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from v0.src.memory.memory_bank import BankCapExceededError, BankEntry, MemoryBank


def _emb(d: int = 1024) -> np.ndarray:
    v = np.random.RandomState(0).randn(d).astype(np.float32)
    v /= max(float(np.linalg.norm(v)), 1e-12)
    return v


def _entry(frame_idx: int) -> BankEntry:
    return BankEntry(
        frame_idx=frame_idx, loop_idx=frame_idx // 10,
        viewing_position_id=None, phase="transit",
        phase_name="phase1", perturbation="none",
        depth=float(frame_idx),
    )


def test_bank_append_and_size():
    bank = MemoryBank(cap=10, allow_eviction=False)
    assert bank.size() == 0
    for i in range(5):
        bank.append(_emb(), _entry(i))
    assert bank.size() == 5
    assert len(bank) == 5
    assert bank.vectors.shape == (5, 1024)


def test_bank_retrieve_by_cosine_returns_top_k():
    bank = MemoryBank(cap=10, allow_eviction=False)
    rng = np.random.RandomState(0)
    targets = []
    for i in range(5):
        v = rng.randn(1024).astype(np.float32)
        v /= np.linalg.norm(v)
        targets.append(v)
        bank.append(v, _entry(i))
    # Query with one of the stored vectors; expect it to be top-1.
    cos, idx = bank.retrieve_by_cosine(targets[2], k=3)
    assert cos.shape == (1, 3)
    assert idx.shape == (1, 3)
    assert int(idx[0, 0]) == 2
    assert float(cos[0, 0]) > 0.999


def test_bank_cap_exceeded_raises_when_eviction_disabled():
    bank = MemoryBank(cap=3, allow_eviction=False)
    for i in range(3):
        bank.append(_emb(), _entry(i))
    with pytest.raises(BankCapExceededError):
        bank.append(_emb(), _entry(99))


def test_bank_cap_exceeded_evicts_when_eviction_enabled():
    bank = MemoryBank(cap=3, allow_eviction=True)
    for i in range(3):
        bank.append(_emb(), _entry(i))
    # Should now succeed (FIFO eviction of oldest).
    bank.append(_emb(), _entry(99))
    assert bank.size() == 3
    assert bank.entries[0].frame_idx == 1
    assert bank.entries[-1].frame_idx == 99


def test_bank_save_load_round_trip():
    bank = MemoryBank(cap=10, allow_eviction=False)
    rng = np.random.RandomState(7)
    for i in range(5):
        v = rng.randn(1024).astype(np.float32)
        v /= np.linalg.norm(v)
        bank.append(v, _entry(i))
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "bank"
        bank.save(path)
        loaded = MemoryBank.load(path)
    assert loaded.size() == bank.size()
    assert loaded.cap == bank.cap
    assert loaded.allow_eviction == bank.allow_eviction
    np.testing.assert_allclose(loaded.vectors, bank.vectors, atol=1e-6)
    for a, b in zip(loaded.entries, bank.entries):
        assert a.frame_idx == b.frame_idx
        assert a.loop_idx == b.loop_idx
        assert a.phase == b.phase

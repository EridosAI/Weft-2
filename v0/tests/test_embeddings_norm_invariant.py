"""Verify the on-disk embeddings file is L2-normed to fp32 tolerance.

Source of truth for §7.2 / §4.7 norm-check; the trainer's init-time check
is a runtime mirror of this test.
"""

from pathlib import Path

import numpy as np
import pytest

from v0.src.config import PATHS


@pytest.mark.skipif(not PATHS.embeddings.is_file(),
                    reason="embeddings file not present (run encode first)")
def test_phase1_embeddings_shape_and_norms():
    e = np.load(PATHS.embeddings, mmap_mode="r")
    assert e.shape == (100_000, 1024)
    assert e.dtype == np.float32
    rng = np.random.default_rng(0)
    sample = e[rng.integers(0, e.shape[0], size=2000)]
    norms = np.linalg.norm(sample, axis=1)
    assert float(norms.min()) >= 1.0 - 1e-5
    assert float(norms.max()) <= 1.0 + 1e-5


@pytest.mark.skipif(not PATHS.embeddings.is_file(),
                    reason="embeddings file not present (run encode first)")
def test_phase1_embeddings_no_zero_rows():
    """Guards against the session-1 finding: dwell-only embeddings had 67k zero rows."""
    e = np.load(PATHS.embeddings, mmap_mode="r")
    norms = np.linalg.norm(e, axis=1)
    n_zero = int((norms == 0.0).sum())
    assert n_zero == 0, (
        f"{n_zero} zero rows in embeddings.npy — likely the dwell-only artefact"
    )

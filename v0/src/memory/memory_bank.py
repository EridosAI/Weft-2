"""Append-only memory bank backed by FAISS IndexFlatIP.

Spec §2.7: shapes live in predictor weights; instances live in the bank.
Instructions §13.1: hard cap, FIFO eviction *disabled* in v0 — BankCapExceededError
is the loud failure that prevents silent attrition from polluting M5.

Each entry carries:
  - embedding (1024-d, L2-normalised; cosine == inner product)
  - frame_idx (position in the source stream)
  - loop_idx
  - viewing_position_id (None for transit frames)
  - phase tag (dwell | transit)
  - phase_name tag (phase1 | phase2 | phase3)
  - perturbation tag (none | livingroom_retexture | ...)
  - depth (recency depth in v0; surprise/error-modulated in v1+)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import faiss
import numpy as np

from v0.src.config import BANK_ALLOW_EVICTION, BANK_CAP, EMBED_DIM


class BankCapExceededError(RuntimeError):
    """Raised when append would exceed BANK_CAP and BANK_ALLOW_EVICTION is False."""


@dataclass
class BankEntry:
    frame_idx: int
    loop_idx: int
    viewing_position_id: Optional[int]
    phase: str          # dwell | transit
    phase_name: str     # phase1 | phase2 | phase3
    perturbation: str   # none | livingroom_retexture | ...
    depth: float        # recency depth in v0


class MemoryBank:
    """Append-only embedding store, recency-indexed, FAISS-backed for cosine retrieval."""

    def __init__(
        self,
        cap: int = BANK_CAP,
        allow_eviction: bool = BANK_ALLOW_EVICTION,
        embed_dim: int = EMBED_DIM,
    ):
        self.cap = int(cap)
        self.allow_eviction = bool(allow_eviction)
        self.embed_dim = int(embed_dim)
        self._index = faiss.IndexFlatIP(self.embed_dim)
        self._entries: list[BankEntry] = []
        # FAISS does not return vectors with cosine; keep a parallel float32 array
        # so we can recover an embedding by row index (used by checkpoint serialisation).
        self._vectors = np.zeros((0, self.embed_dim), dtype=np.float32)

    # ---- inspection ------------------------------------------------------

    def size(self) -> int:
        return len(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def vectors(self) -> np.ndarray:
        """A view of all bank vectors (rows index aligned with entries)."""
        return self._vectors[: len(self._entries)]

    @property
    def entries(self) -> Sequence[BankEntry]:
        return tuple(self._entries)

    # ---- mutation --------------------------------------------------------

    def append(self, embedding: np.ndarray, entry: BankEntry) -> None:
        assert embedding.shape == (self.embed_dim,), (
            f"bank append: embedding shape {embedding.shape} != ({self.embed_dim},)"
        )
        assert embedding.dtype == np.float32, (
            f"bank append: dtype {embedding.dtype} != float32"
        )
        if len(self._entries) + 1 > self.cap:
            if not self.allow_eviction:
                raise BankCapExceededError(
                    f"bank size {len(self._entries)} +1 > cap {self.cap}; "
                    "eviction disabled per BANK_ALLOW_EVICTION=False"
                )
            self._evict_oldest()
        if len(self._entries) >= self._vectors.shape[0]:
            self._grow_vectors()
        self._vectors[len(self._entries)] = embedding
        self._index.add(embedding.reshape(1, -1))
        self._entries.append(entry)

    def _evict_oldest(self) -> None:
        # FIFO eviction. Disabled by default in v0 (raise instead) per BANK_ALLOW_EVICTION.
        # This path only runs when the operator has explicitly enabled eviction for v1+ work.
        self._entries.pop(0)
        self._vectors = np.roll(self._vectors, shift=-1, axis=0)
        rebuilt = faiss.IndexFlatIP(self.embed_dim)
        rebuilt.add(self._vectors[: len(self._entries)])
        self._index = rebuilt

    def _grow_vectors(self) -> None:
        new_size = max(1024, self._vectors.shape[0] * 2)
        new_size = min(new_size, self.cap)
        if new_size <= self._vectors.shape[0]:
            return
        new = np.zeros((new_size, self.embed_dim), dtype=np.float32)
        new[: self._vectors.shape[0]] = self._vectors
        self._vectors = new

    # ---- retrieval -------------------------------------------------------

    def retrieve_by_cosine(
        self, query: np.ndarray, k: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Top-k by inner product (== cosine on L2-normed vectors).

        Returns (cosines, row_indices) of shapes (B, k) and (B, k).
        Caller is responsible for L2-normalising the query.
        """
        if query.ndim == 1:
            query = query.reshape(1, -1)
        assert query.shape[1] == self.embed_dim
        assert query.dtype == np.float32
        if len(self._entries) == 0:
            return (
                np.full((query.shape[0], k), -1.0, dtype=np.float32),
                np.full((query.shape[0], k), -1, dtype=np.int64),
            )
        cos, idx = self._index.search(query, min(k, len(self._entries)))
        if idx.shape[1] < k:
            pad_idx = np.full((query.shape[0], k - idx.shape[1]), -1, dtype=np.int64)
            pad_cos = np.full((query.shape[0], k - cos.shape[1]), -1.0, dtype=np.float32)
            idx = np.concatenate([idx, pad_idx], axis=1)
            cos = np.concatenate([cos, pad_cos], axis=1)
        return cos.astype(np.float32), idx.astype(np.int64)

    # ---- serialisation ---------------------------------------------------

    def save(self, dir_path: Path) -> None:
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(dir_path / "faiss.index"))
        np.save(dir_path / "vectors.npy", self.vectors)
        meta = {
            "cap": self.cap,
            "allow_eviction": self.allow_eviction,
            "embed_dim": self.embed_dim,
            "entries": [
                {
                    "frame_idx": e.frame_idx,
                    "loop_idx": e.loop_idx,
                    "viewing_position_id": e.viewing_position_id,
                    "phase": e.phase,
                    "phase_name": e.phase_name,
                    "perturbation": e.perturbation,
                    "depth": e.depth,
                }
                for e in self._entries
            ],
        }
        (dir_path / "bank_meta.json").write_text(json.dumps(meta))

    @classmethod
    def load(cls, dir_path: Path) -> "MemoryBank":
        dir_path = Path(dir_path)
        meta = json.loads((dir_path / "bank_meta.json").read_text())
        bank = cls(
            cap=int(meta["cap"]),
            allow_eviction=bool(meta["allow_eviction"]),
            embed_dim=int(meta["embed_dim"]),
        )
        bank._index = faiss.read_index(str(dir_path / "faiss.index"))
        vectors = np.load(dir_path / "vectors.npy")
        assert vectors.dtype == np.float32 and vectors.shape[1] == bank.embed_dim
        # Re-seat vectors so subsequent appends grow in place.
        bank._vectors = np.zeros(
            (max(1024, vectors.shape[0] * 2), bank.embed_dim), dtype=np.float32
        )
        bank._vectors[: vectors.shape[0]] = vectors
        bank._entries = [
            BankEntry(
                frame_idx=int(e["frame_idx"]),
                loop_idx=int(e["loop_idx"]),
                viewing_position_id=e["viewing_position_id"],
                phase=str(e["phase"]),
                phase_name=str(e["phase_name"]),
                perturbation=str(e["perturbation"]),
                depth=float(e["depth"]),
            )
            for e in meta["entries"]
        ]
        return bank

"""Frozen DINOv2 encoder wrapper for Phase 2 / Phase 3 frame encoding.

Phase 1 uses pre-computed embeddings; this module is only used by the
Phase 2 / Phase 3 collect+encode pipelines. The protocol mirrors the
substrate verification batch exactly (see
`scripts/run_encoder_verification_dinov2.py`):

  - facebook/dinov2-large, frozen, fp16 eval
  - 256x256 RGB -> center-crop 224x224 -> ImageNet mean/std
  - CLS token (last_hidden_state[:, 0, :]) -> L2-normalise

This file is also imported by the encoder-sanity-recheck used at Phase 2/3
encode time (instr §13.4): if encoder forward produces non-L2-normalised
output for any reason, the trainer halts.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset


DINOV2_MODEL_ID = "facebook/dinov2-large"
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)
_CROP = 224


class FrameDataset(Dataset):
    def __init__(self, frames_dir: Path, indices: list[int]):
        self.frames_dir = Path(frames_dir)
        self.indices = list(indices)
        self.mean = torch.tensor(_IMAGENET_MEAN).view(3, 1, 1)
        self.std = torch.tensor(_IMAGENET_STD).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, k: int):
        fi = self.indices[k]
        path = self.frames_dir / f"frame_{fi:08d}.png"
        im = Image.open(path).convert("RGB")
        w, h = im.size
        assert w == 256 and h == 256, f"unexpected frame size {im.size}"
        left = (w - _CROP) // 2
        top = (h - _CROP) // 2
        im = im.crop((left, top, left + _CROP, top + _CROP))
        arr = np.asarray(im, dtype=np.float32) / 255.0
        t = torch.from_numpy(arr).permute(2, 0, 1)
        t = (t - self.mean) / self.std
        return fi, t


def load_frozen_dinov2(
    device: torch.device, model_id: str = DINOV2_MODEL_ID
) -> torch.nn.Module:
    from transformers import AutoModel
    model = AutoModel.from_pretrained(model_id, torch_dtype=torch.float16)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def encode_frames(
    model: torch.nn.Module,
    frames_dir: Path,
    indices: list[int],
    device: torch.device,
    batch_size: int = 64,
    num_workers: int = 4,
) -> np.ndarray:
    """Returns (len(indices), 1024) float32 L2-normalised."""
    ds = FrameDataset(frames_dir, indices)
    loader = DataLoader(
        ds, batch_size=batch_size, num_workers=num_workers, shuffle=False,
        pin_memory=True,
    )
    out = np.zeros((len(indices), 1024), dtype=np.float32)
    index_to_row = {idx: row for row, idx in enumerate(indices)}
    with torch.no_grad():
        for fi_batch, x_batch in loader:
            x_batch = x_batch.to(device, dtype=torch.float16, non_blocking=True)
            res = model(pixel_values=x_batch)
            cls = res.last_hidden_state[:, 0, :].float()
            cls = F.normalize(cls, dim=1, eps=1e-12)
            cls_np = cls.cpu().numpy()
            for k in range(cls_np.shape[0]):
                out[index_to_row[int(fi_batch[k].item())]] = cls_np[k]
    return out

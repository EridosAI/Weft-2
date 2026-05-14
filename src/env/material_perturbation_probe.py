"""Shared AI2-THOR + DINOv2 helpers for the Phase 2 material-perturbation probes.

The Phase 2 preflight, extended diagnostic, and pose-search scripts all
need the same primitive operations against a Controller in the seed-7
house:

  - teleport the agent to (position, heading) and capture an RGB frame
  - call `RandomizeMaterials(inRoomTypes=["LivingRoom"], useTrainMaterials=True)`
  - capture a set of viewing positions in batch
  - encode captured frames through the frozen DINOv2-large protocol
    (matches `src/encoder/dinov2_encoder.py`: resize 256 → center crop 224
    → ImageNet mean/std → fp16 forward → CLS extract → L2-norm)
  - compute flat-RGB cosine between two uint8 frames

The functions are pure given the controller; they do not assume any
particular probe schedule or scoring scheme.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


# DINOv2 input protocol — must match src/encoder/dinov2_encoder.py.
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
_RESIZE = 256
_CROP = 224

# Item-set identifiers in the seed-7 route.
LIVINGROOM_ITEM_IDS: Tuple[int, ...] = (3, 4)        # Dresser, Sofa
BEDROOM_ITEM_IDS: Tuple[int, ...] = (1, 2, 5)        # Bed, DiningTable, Television


def items_by_id(route: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    return {int(it["item_id"]): it for it in route["items"]}


def pixel_cosine(img_a: np.ndarray, img_b: np.ndarray) -> float:
    a = img_a.astype(np.float64).reshape(-1)
    b = img_b.astype(np.float64).reshape(-1)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def teleport_and_capture(
    controller,
    position: Dict[str, float],
    heading_deg: float,
    force_action: bool = False,
) -> np.ndarray:
    """Teleport the agent to `(position, heading_deg)` and capture the RGB frame.

    `force_action=True` bypasses AI2-THOR's grid-alignment check on the
    target position. Use it for off-grid sweeps (e.g., the 0.20 m close-up
    path inside the explorer's continuous-motion segment); the resulting
    frames are still rendered at the exact requested pose. Leave it
    `False` for grid-aligned NavMesh-snapped poses (e.g., viewing positions).
    """
    kwargs = {
        "action": "Teleport",
        "position": dict(position),
        "rotation": {"x": 0.0, "y": float(heading_deg), "z": 0.0},
        "horizon": 0.0,
        "standing": True,
    }
    if force_action:
        kwargs["forceAction"] = True
    event = controller.step(**kwargs)
    if not event.metadata.get("lastActionSuccess", False):
        raise RuntimeError(
            f"Teleport to {position} heading={heading_deg} failed: "
            f"{event.metadata.get('errorMessage', '?')}"
        )
    return np.asarray(event.frame, dtype=np.uint8)


def teleport_only(
    controller,
    position: Dict[str, float],
    heading_deg: float,
    force_action: bool = False,
) -> bool:
    """Same as `teleport_and_capture` but returns success/failure without capturing.

    Used by NavMesh navigability checks where we only care whether the
    pose is reachable. `force_action=True` bypasses the grid-alignment
    check on the target position.
    """
    kwargs = {
        "action": "Teleport",
        "position": dict(position),
        "rotation": {"x": 0.0, "y": float(heading_deg), "z": 0.0},
        "horizon": 0.0,
        "standing": True,
    }
    if force_action:
        kwargs["forceAction"] = True
    event = controller.step(**kwargs)
    return bool(event.metadata.get("lastActionSuccess", False))


def capture_all_items(
    controller, items: Dict[int, Dict[str, Any]]
) -> Dict[int, np.ndarray]:
    out: Dict[int, np.ndarray] = {}
    for item_id, it in items.items():
        out[item_id] = teleport_and_capture(
            controller, it["viewing_position"], float(it["viewing_heading_deg"])
        )
    return out


def randomize_livingroom(controller) -> Tuple[bool, Any]:
    event = controller.step(
        action="RandomizeMaterials",
        inRoomTypes=["LivingRoom"],
        useTrainMaterials=True,
    )
    return (
        bool(event.metadata.get("lastActionSuccess", False)),
        event.metadata.get("actionReturn"),
    )


def dinov2_encode_batch(
    model: torch.nn.Module,
    frames: List[np.ndarray],
    device: torch.device,
) -> np.ndarray:
    """Encode a list of uint8 (H, W, 3) RGB frames via the frozen DINOv2 protocol.

    Returns (N, 1024) float32 L2-normalised CLS embeddings.
    """
    batch: List[torch.Tensor] = []
    for frame in frames:
        im = Image.fromarray(frame).convert("RGB")
        if im.size != (_RESIZE, _RESIZE):
            im = im.resize((_RESIZE, _RESIZE), resample=Image.BILINEAR)
        w, h = im.size
        left = (w - _CROP) // 2
        top = (h - _CROP) // 2
        im = im.crop((left, top, left + _CROP, top + _CROP))
        arr = np.asarray(im, dtype=np.float32) / 255.0
        t = torch.from_numpy(arr).permute(2, 0, 1)
        t = (t - _IMAGENET_MEAN) / _IMAGENET_STD
        batch.append(t)
    x = torch.stack(batch).to(device, dtype=torch.float16, non_blocking=True)
    with torch.no_grad():
        res = model(pixel_values=x)
    cls = res.last_hidden_state[:, 0, :].float()
    cls = F.normalize(cls, dim=1, eps=1e-12)
    return cls.cpu().numpy()


def check_navmesh_reachable(controller, position: Dict[str, float]) -> bool:
    """Verify a viewing position is in the NavMesh-reachable set.

    Faster than teleport_only because it doesn't actually move the agent.
    """
    event = controller.step(action="GetReachablePositions")
    if not event.metadata.get("lastActionSuccess", False):
        return False
    reach = event.metadata.get("actionReturn") or []
    target_x, target_z = float(position["x"]), float(position["z"])
    # Accept if any reachable point is within one grid step (0.25 m) on x/z.
    tol = 0.30
    for p in reach:
        if abs(float(p["x"]) - target_x) < tol and abs(float(p["z"]) - target_z) < tol:
            return True
    return False


__all__ = [
    "LIVINGROOM_ITEM_IDS",
    "BEDROOM_ITEM_IDS",
    "items_by_id",
    "pixel_cosine",
    "teleport_and_capture",
    "teleport_only",
    "capture_all_items",
    "randomize_livingroom",
    "dinov2_encode_batch",
    "check_navmesh_reachable",
]

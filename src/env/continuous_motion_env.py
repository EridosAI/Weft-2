"""AI2-THOR environment wrapper for the continuous-motion route.

Replaces AI2ThorFurnitureEnv (from the previous repo) for v0+ work.
Loads the seed-7 ProcTHOR house and drives the new
ContinuousMotionExplorer.

Public API mirrors the prior wrapper enough that the calibration and
collection scripts can use it the same way:

  - `next_frame() -> uint8 (frame_size, frame_size, 3) RGB ndarray`
  - `current_room_name() -> str`
  - `episode_boundary_flag` — True if the last returned frame was a
    loop boundary.
  - `last_observation` — full per-frame metadata dict.
  - `explorer_stats` — dict of explorer counters.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from src.env.continuous_motion_explorer import ContinuousMotionExplorer
from src.env.procthor_house import (
    load_house,
    make_controller,
    room_for_position,
)


_ENCODER_FRAME_SIZE = 256


class ContinuousMotionEnv:
    """ProcTHOR house + ContinuousMotionExplorer; consume one frame at a time."""

    def __init__(
        self,
        *,
        house_seed: int,
        route_items: List[Dict[str, Any]],
        width: int = 300,
        height: int = 300,
        frame_size: int = _ENCODER_FRAME_SIZE,
        grid_size: float = 0.25,
        min_rooms: int = 4,
        close_up_length_m: float = 2.0,
        close_up_step_m: float = 0.20,
        densify_step_m: float = 0.20,
        corner_rotation_step_deg: float = 5.0,
    ) -> None:
        self._house_seed = int(house_seed)
        self._frame_size = int(frame_size)
        self._house = load_house(seed=self._house_seed, min_rooms=int(min_rooms))
        self._rooms = self._house.get("rooms", [])
        self._controller = make_controller(
            self._house,
            width=int(width),
            height=int(height),
            grid_size=float(grid_size),
        )
        self._route_items = [dict(it) for it in route_items]
        self._explorer = ContinuousMotionExplorer(
            controller=self._controller,
            route_items=self._route_items,
            close_up_length_m=float(close_up_length_m),
            close_up_step_m=float(close_up_step_m),
            densify_step_m=float(densify_step_m),
            corner_rotation_step_deg=float(corner_rotation_step_deg),
        )
        self._last_obs: Dict[str, Any] = {}
        self._episode_boundary_flag: bool = False

    def next_frame(self) -> np.ndarray:
        obs = self._explorer.next_micro_step()
        self._last_obs = obs
        self._episode_boundary_flag = bool(obs.get("loop_boundary_flag", False))
        frame = obs["frame"]
        if frame.shape[:2] != (self._frame_size, self._frame_size):
            from PIL import Image
            frame = np.asarray(
                Image.fromarray(frame).resize(
                    (self._frame_size, self._frame_size),
                    resample=Image.BILINEAR,
                )
            )
        return frame

    @property
    def last_observation(self) -> Dict[str, Any]:
        return dict(self._last_obs)

    @property
    def episode_boundary_flag(self) -> bool:
        return self._episode_boundary_flag

    @property
    def explorer_stats(self) -> Dict[str, int]:
        return self._explorer.stats()

    def current_room_name(self) -> str:
        pos = self._last_obs.get("position")
        if not pos or not self._rooms:
            return "?"
        room_type, _room_id = room_for_position(
            self._rooms, float(pos["x"]), float(pos["z"])
        )
        return str(room_type)

    def close(self) -> None:
        try:
            self._controller.stop()
        except Exception:
            pass


__all__ = ["ContinuousMotionEnv"]

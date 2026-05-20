"""ProcTHOR house loader and room detection for Weft.

Adapts v14's house-loading + room-polygon machinery (see
`alan_v14/environments/procthor_env.py`) into a minimal API for Weft.
The higher-level wrapper (`AI2ThorWeftEnv`) owns the controller
lifecycle and per-frame stepping; this module only builds and inspects.

Public API:
  - `load_house(seed, min_rooms, split, scan_limit) -> dict`
  - `make_controller(house, width, height, grid_size, server_timeout) -> Controller`
  - `room_for_position(rooms, x, z) -> (room_type, room_id)`
  - `reachable_positions_by_room(controller, rooms) -> dict[room_type, list[pos]]`
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Tuple


def _point_in_polygon(x: float, z: float, polygon: List[Dict[str, float]]) -> bool:
    """Ray-casting point-in-polygon on the XZ plane.

    `polygon` is the ProcTHOR `floorPolygon`: a list of vertex dicts
    with keys `x`, `y`, `z`.
    """
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, zi = polygon[i]["x"], polygon[i]["z"]
        xj, zj = polygon[j]["x"], polygon[j]["z"]
        if ((zi > z) != (zj > z)) and (x < (xj - xi) * (z - zi) / (zj - zi) + xi):
            inside = not inside
        j = i
    return inside


def _polygon_centroid(polygon: List[Dict[str, float]]) -> Tuple[float, float]:
    cx = sum(p["x"] for p in polygon) / len(polygon)
    cz = sum(p["z"] for p in polygon) / len(polygon)
    return cx, cz


def load_house(
    seed: int = 42,
    min_rooms: int = 4,
    split: str = "train",
    scan_limit: int = 500,
    revision: str | None = None,
) -> Dict[str, Any]:
    """Load one ProcTHOR-10K house with at least `min_rooms` rooms.

    Selection is deterministic in `seed`. Requires the `prior` package.
    Returned dict is directly passable to `ai2thor.controller.Controller(scene=...)`.

    Args:
      revision: optional procthor-10k git revision (e.g.
        ``ab3cacd0fc17754d4c080a3fd50b18395fae8647`` for the
        ai2thor-5.0.0-aligned revision per the AI2-THOR warning's
        downgrade recommendation). Default None preserves the v0-pinned
        behaviour (latest cached revision, currently `4391935...`). v1
        scripts set this explicitly per the build-config investigation
        2026-05-19 design-chat decision; v0 scripts left untouched.
    """
    import prior  # lazy import — heavy

    if revision is None:
        dataset = prior.load_dataset("procthor-10k")
    else:
        dataset = prior.load_dataset("procthor-10k", revision=revision)
    data = dataset[split]

    candidates: List[Tuple[int, Dict[str, Any]]] = []
    limit = min(scan_limit, len(data))
    for i in range(limit):
        house = data[i]
        if len(house.get("rooms", [])) >= min_rooms:
            candidates.append((i, house))

    if not candidates:
        raise RuntimeError(
            f"No houses with >= {min_rooms} rooms in the first {scan_limit} "
            f"entries of split '{split}'. Try increasing scan_limit or "
            f"reducing min_rooms."
        )

    rng = random.Random(seed)
    return rng.choice(candidates)[1]


def make_controller(
    house: Dict[str, Any],
    width: int = 300,
    height: int = 300,
    grid_size: float = 0.25,
    server_timeout: int = 120,
):
    """Construct an AI2-THOR controller bound to `house`.

    Controller defaults: `rotateStepDegrees=90`, `snapToGrid=True`. The
    SmoothWrapper overrides per call by passing `degrees=` to step().
    """
    from ai2thor.controller import Controller

    return Controller(
        scene=house,
        width=width,
        height=height,
        gridSize=grid_size,
        server_timeout=server_timeout,
    )


def room_for_position(
    rooms: List[Dict[str, Any]],
    x: float,
    z: float,
) -> Tuple[str, str]:
    """Return `(room_type, room_id)` for an `(x, z)` coordinate.

    Falls back to the nearest room centroid when the point is not strictly
    inside any floor polygon (e.g. agent in a doorway).
    """
    for room in rooms:
        if _point_in_polygon(x, z, room["floorPolygon"]):
            return room["roomType"], room["id"]

    min_dist = float("inf")
    best: Tuple[str, str] = ("Unknown", "unknown")
    for room in rooms:
        cx, cz = _polygon_centroid(room["floorPolygon"])
        d = (x - cx) ** 2 + (z - cz) ** 2
        if d < min_dist:
            min_dist = d
            best = (room["roomType"], room["id"])
    return best


def reachable_positions_by_room(
    controller,
    rooms: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, float]], Dict[str, List[Dict[str, float]]]]:
    """Query AI2-THOR for reachable positions, group by room_type.

    Returns `(all_positions, by_room)` where `by_room` maps room_type
    (e.g. "Kitchen") to the list of positions inside that room. Positions
    that fall outside every floor polygon (typical for doorways) are
    grouped by the nearest-centroid fallback in `room_for_position`.
    """
    event = controller.step(action="GetReachablePositions")
    positions: List[Dict[str, float]] = event.metadata["actionReturn"] or []
    by_room: Dict[str, List[Dict[str, float]]] = {}
    for p in positions:
        rt, _rid = room_for_position(rooms, p["x"], p["z"])
        by_room.setdefault(rt, []).append(p)
    return positions, by_room


__all__ = [
    "load_house",
    "make_controller",
    "room_for_position",
    "reachable_positions_by_room",
]

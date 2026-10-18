"""Person tracking and line-crossing counting.

Pure Python (no OpenCV / YOLO needed), so it can be unit-tested with fake
detections. This replaces the old nearest-centroid matcher, which lost people
between inference ticks and double-counted anyone standing near the line.

What it does, in plain words:
  1. PersonTracker  - keeps a stable ID for each person from one tick to the next,
                      using each person's speed to guess where they will be next.
  2. LineCounter    - counts an entry/exit only when a person *clearly* moves from one
                      side of the door line to the other. A "dead zone" (band) around the
                      line stops jitter from creating fake counts.

Honest limits: this is still a lightweight tracker. In a very crowded doorway, or if
people are hidden for several seconds, counts can be off. Always compare against a
manual count on real footage before quoting accuracy to anyone.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

VERTICAL_DIRECTIONS = ("left_to_right", "right_to_left")
HORIZONTAL_DIRECTIONS = ("top_to_bottom", "bottom_to_top")
# Entry means moving from side -1 (left/top) to side +1 (right/bottom), or the reverse.
_ENTRY_SIDE_CHANGE = {
    "left_to_right": (-1, 1), "right_to_left": (1, -1),
    "top_to_bottom": (-1, 1), "bottom_to_top": (1, -1),
}


@dataclass
class Track:
    id: int
    x: float
    y: float
    scale: float          # size of the person's box in pixels (used to judge "how far is too far")
    first_seen: float
    last_seen: float
    vx: float = 0.0       # pixels / second
    vy: float = 0.0
    hits: int = 1
    side: int | None = None          # committed side of the line: -1, +1, or None (unknown yet)
    last_count_at: float = -1e9


class PersonTracker:
    """Greedy nearest-match tracker with constant-velocity prediction (SORT-like)."""

    def __init__(self, gate: float = 1.0, max_age_seconds: float = 3.0):
        self.gate = gate                      # max match distance, in "person sizes"
        self.max_age_seconds = max_age_seconds
        self.tracks: dict[int, Track] = {}
        self._next_id = 1

    def reset(self) -> None:
        self.tracks.clear()

    def update(self, points: Sequence[tuple[float, float, float]], now: float) -> list[Track | None]:
        """points = [(x, y, scale), ...]. Returns one Track per input point (same order)."""
        pairs: list[tuple[float, int, int]] = []
        for ti, tr in self.tracks.items():
            dt = max(0.0, min(now - tr.last_seen, self.max_age_seconds))
            px, py = tr.x + tr.vx * dt, tr.y + tr.vy * dt
            for di, (x, y, scale) in enumerate(points):
                size = max(scale, tr.scale, 1.0)
                d_pred = ((x - px) ** 2 + (y - py) ** 2) ** 0.5
                d_last = ((x - tr.x) ** 2 + (y - tr.y) ** 2) ** 0.5
                cost = min(d_pred, d_last) / size
                if cost <= self.gate:
                    pairs.append((cost, ti, di))
        pairs.sort()
        used_tracks: set[int] = set()
        assigned: dict[int, Track] = {}
        for _cost, ti, di in pairs:
            if ti in used_tracks or di in assigned:
                continue
            used_tracks.add(ti)
            tr = self.tracks[ti]
            x, y, scale = points[di]
            dt = max(0.05, now - tr.last_seen)
            tr.vx = 0.6 * ((x - tr.x) / dt) + 0.4 * tr.vx
            tr.vy = 0.6 * ((y - tr.y) / dt) + 0.4 * tr.vy
            tr.x, tr.y, tr.scale, tr.last_seen = x, y, scale, now
            tr.hits += 1
            assigned[di] = tr
        for di, (x, y, scale) in enumerate(points):
            if di not in assigned:
                tr = Track(id=self._next_id, x=x, y=y, scale=scale, first_seen=now, last_seen=now)
                self._next_id += 1
                self.tracks[tr.id] = tr
                assigned[di] = tr
        self.tracks = {i: t for i, t in self.tracks.items() if now - t.last_seen <= self.max_age_seconds}
        return [assigned[i] for i in range(len(points))]


@dataclass
class CountEvent:
    kind: str            # "entry" or "exit"
    track_id: int
    direction: str       # e.g. "left_to_right"


class LineCounter:
    """Counts entries/exits when a tracked person crosses a line (with a dead-zone band)."""

    def __init__(self, orientation: str = "vertical", entry_direction: str = "left_to_right",
                 line_ratio: float = 0.5, band_ratio: float = 0.04, min_hits: int = 2,
                 cooldown_seconds: float = 1.0, tracker: PersonTracker | None = None):
        if orientation not in ("vertical", "horizontal"):
            raise ValueError("line_orientation must be 'vertical' or 'horizontal'")
        allowed = VERTICAL_DIRECTIONS if orientation == "vertical" else HORIZONTAL_DIRECTIONS
        if entry_direction not in allowed:
            raise ValueError(f"entry_direction '{entry_direction}' does not match a {orientation} line; use one of {allowed}")
        self.orientation, self.entry_direction = orientation, entry_direction
        self.line_ratio, self.band_ratio = float(line_ratio), float(band_ratio)
        self.min_hits, self.cooldown_seconds = int(min_hits), float(cooldown_seconds)
        self.tracker = tracker or PersonTracker()
        self.entries = 0
        self.exits = 0

    def _axis(self, x1: float, y1: float, x2: float, y2: float) -> float:
        # Vertical line -> use horizontal position of the person's centre.
        # Horizontal line -> use the feet (bottom edge), which is steadier than the box centre.
        return (x1 + x2) / 2.0 if self.orientation == "vertical" else float(y2)

    def update(self, boxes: Iterable[Sequence[float]], width: int, height: int, now: float) -> tuple[list[CountEvent], list[int]]:
        """boxes = [(x1,y1,x2,y2), ...] of people. Returns (events, track_id_per_box)."""
        boxes = list(boxes)
        pts = [((b[0] + b[2]) / 2.0, float(b[3]), max(b[2] - b[0], b[3] - b[1])) for b in boxes]
        tracks = self.tracker.update(pts, now)
        dim = width if self.orientation == "vertical" else height
        line, band = self.line_ratio * dim, self.band_ratio * dim
        enter_from, enter_to = _ENTRY_SIDE_CHANGE[self.entry_direction]
        names = ("left", "right") if self.orientation == "vertical" else ("top", "bottom")
        events: list[CountEvent] = []
        for box, tr in zip(boxes, tracks):
            pos = self._axis(*box)
            zone = -1 if pos < line - band else (1 if pos > line + band else 0)
            if zone == 0:
                continue                       # inside the dead zone: change nothing
            if tr.side is None:
                tr.side = zone                 # first clear sighting: remember side, count nothing
                continue
            if zone == tr.side:
                continue
            if tr.hits < self.min_hits or now - tr.last_count_at < self.cooldown_seconds:
                continue                       # too new / just counted: wait, side stays unchanged
            direction = f"{names[0] if tr.side < 0 else names[1]}_to_{names[0] if zone < 0 else names[1]}"
            kind = "entry" if (tr.side, zone) == (enter_from, enter_to) else "exit"
            if kind == "entry":
                self.entries += 1
            else:
                self.exits += 1
            tr.side, tr.last_count_at = zone, now
            events.append(CountEvent(kind=kind, track_id=tr.id, direction=direction))
        return events, [t.id for t in tracks]

"""Autopilot paths and a "carrot" that walks along them for the position controller to chase."""
from __future__ import annotations

import json
import math
from pathlib import Path

Point = tuple[float, float]


def _center_and_size(inner: tuple[float, float, float, float]) -> tuple[float, float, float]:
    x0, y0, x1, y1 = inner
    return (x0 + x1) / 2, (y0 + y1) / 2, min(x1 - x0, y1 - y0)


def square(inner: tuple[float, float, float, float]) -> list[Point]:
    cx, cy, size = _center_and_size(inner)
    h = 0.35 * size
    return [(cx - h, cy + h), (cx + h, cy + h), (cx + h, cy - h), (cx - h, cy - h), (cx - h, cy + h)]


def circle(inner: tuple[float, float, float, float], n: int = 24) -> list[Point]:
    cx, cy, size = _center_and_size(inner)
    r = 0.35 * size
    return [(cx + r * math.sin(2 * math.pi * i / n), cy + r * math.cos(2 * math.pi * i / n)) for i in range(n + 1)]


def figure8(inner: tuple[float, float, float, float], n: int = 32) -> list[Point]:
    cx, cy, size = _center_and_size(inner)
    a = 0.4 * size
    pts = []
    for i in range(n + 1):
        t = 2 * math.pi * i / n
        d = 1 + math.sin(t) ** 2  # lemniscate of Gerono-style figure eight, sideways
        pts.append((cx + a * math.cos(t) / d, cy + a * math.sin(t) * math.cos(t) / d))
    return pts


SHAPES = {"square": square, "circle": circle, "figure8": figure8}


def build_path(spec: str, inner: tuple[float, float, float, float]) -> list[Point]:
    """A built-in shape name, or a JSON file of [[x, y], ...] in arena meters (from nav.setup path)."""
    if spec in SHAPES:
        return SHAPES[spec](inner)
    pts = [tuple(map(float, p)) for p in json.loads(Path(spec).read_text())]
    if len(pts) < 1:
        raise ValueError(f"{spec}: no waypoints")
    return pts


class Mission:
    """Carrot at distance `s` along start -> waypoints -> home.

    It only advances while the drone keeps up, and waits at each waypoint until the drone gets within
    `arrive_tol`, so corners get flown instead of cut.
    """

    def __init__(
        self,
        start: Point,
        waypoints: list[Point],
        home: Point,
        speed: float = 0.3,
        max_lead: float = 0.5,
        arrive_tol: float = 0.2,
    ):
        self.points = [start, *waypoints, home]
        self.speed = speed
        self.max_lead = max_lead
        self.arrive_tol = arrive_tol
        self.s = 0.0
        self._lengths = [math.dist(a, b) for a, b in zip(self.points, self.points[1:])]
        self.total = sum(self._lengths)
        self._vertex_s = [sum(self._lengths[:i]) for i in range(len(self.points))]

    def carrot(self) -> Point:
        s = self.s
        for (a, b), length in zip(zip(self.points, self.points[1:]), self._lengths):
            if s <= length and length > 0:
                f = s / length
                return a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1])
            s -= length
        return self.points[-1]

    def advance(self, dt: float, drone: Point) -> Point:
        if math.dist(drone, self.carrot()) >= self.max_lead:
            return self.carrot()
        new_s = min(self.total, self.s + self.speed * dt)
        for vs, p in zip(self._vertex_s[1:], self.points[1:]):
            if self.s - 1e-9 <= vs <= new_s and math.dist(drone, p) > self.arrive_tol:
                new_s = vs  # hold the carrot on this waypoint until the drone gets there
                break
        self.s = new_s
        return self.carrot()

    def done(self, drone: Point, tolerance: float = 0.15) -> bool:
        return self.s >= self.total and math.dist(drone, self.points[-1]) < tolerance

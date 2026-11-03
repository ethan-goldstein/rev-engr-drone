"""Virtual barrier: a rectangle the drone may not leave.

Three layers, softest first:
  * targets (hold point, autopilot carrot) are clamped inside the inner rectangle,
  * inside the margin band the drone is pushed back toward the middle, and user sticks pointing
    outward are cut,
  * actually outside the rectangle for `breach_s` means land.
"""
from __future__ import annotations

from nav.pose import Pose


class Geofence:
    def __init__(
        self,
        rect: tuple[float, float, float, float],
        margin: float = 0.3,
        push: float = 0.8,
        breach_s: float = 0.5,
    ):
        self.xmin, self.ymin, self.xmax, self.ymax = rect
        self.margin = margin
        self.push = push  # stick per meter of intrusion into the margin band
        self.breach_s = breach_s
        self._outside_since: float | None = None

    @property
    def inner(self) -> tuple[float, float, float, float]:
        m = self.margin
        return (self.xmin + m, self.ymin + m, self.xmax - m, self.ymax - m)

    def clamp(self, p: tuple[float, float]) -> tuple[float, float]:
        x0, y0, x1, y1 = self.inner
        return min(max(p[0], x0), x1), min(max(p[1], y0), y1)

    def contains(self, pose: Pose) -> bool:
        return self.xmin <= pose.x <= self.xmax and self.ymin <= pose.y <= self.ymax

    def push_back(self, pose: Pose) -> tuple[float, float]:
        """World-frame stick pushing the drone out of the margin band, toward the middle."""
        x0, y0, x1, y1 = self.inner
        ux = self.push * (max(0.0, x0 - pose.x) - max(0.0, pose.x - x1))
        uy = self.push * (max(0.0, y0 - pose.y) - max(0.0, pose.y - y1))
        return ux, uy

    def limit_user(self, ux: float, uy: float, pose: Pose) -> tuple[float, float]:
        """Cut the outward part of a world-frame user command once the drone is in the margin band."""
        x0, y0, x1, y1 = self.inner
        if (pose.x <= x0 and ux < 0) or (pose.x >= x1 and ux > 0):
            ux = 0.0
        if (pose.y <= y0 and uy < 0) or (pose.y >= y1 and uy > 0):
            uy = 0.0
        return ux, uy

    def breached(self, pose: Pose, now: float) -> bool:
        if self.contains(pose):
            self._outside_since = None
            return False
        if self._outside_since is None:
            self._outside_since = now
        return now - self._outside_since >= self.breach_s

    def reset(self) -> None:
        self._outside_since = None

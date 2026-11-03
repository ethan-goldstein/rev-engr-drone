"""Shared control types: stick axes and one-shot flight commands."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


@dataclass(frozen=True)
class Axes:
    """Normalized stick positions, each in [-1, 1]. Positive = up / turn right / forward / right."""

    throttle: float = 0.0
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0

    def clamped(self, limit: float = 1.0) -> "Axes":
        def c(v: float) -> float:
            return max(-limit, min(limit, v))

        return Axes(c(self.throttle), c(self.yaw), c(self.pitch), c(self.roll))

    def scaled(self, k: float) -> "Axes":
        return Axes(self.throttle * k, self.yaw * k, self.pitch * k, self.roll * k)

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.throttle, self.yaw, self.pitch, self.roll)

    def is_zero(self) -> bool:
        return all(v == 0.0 for v in self.as_tuple())


ZERO = Axes()


class Command(str, Enum):
    TAKEOFF = "takeoff"
    LAND = "land"
    ESTOP = "estop"

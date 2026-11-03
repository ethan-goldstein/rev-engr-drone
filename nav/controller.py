"""PID position hold: world-frame error in meters -> roll/pitch sticks.

The drone reports nothing about its position, so this runs on the tracking camera's pose. The
integral term is what cancels steady drift (wind, a slightly off-balance drone); the derivative term
damps overshoot.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

from nav.pose import Pose


@dataclass(frozen=True)
class Gains:
    kp: float = 0.6  # stick per meter of error
    ki: float = 0.15  # stick per meter-second of accumulated error
    kd: float = 0.3  # stick per m/s of velocity (damping)

    @classmethod
    def parse(cls, text: str) -> "Gains":
        kp, ki, kd = (float(v) for v in text.split(","))
        return cls(kp, ki, kd)


def clamp_vec(x: float, y: float, limit: float) -> tuple[float, float]:
    """Scale (x, y) down to length `limit`, keeping its direction."""
    n = math.hypot(x, y)
    return (x, y) if n <= limit or n == 0 else (x * limit / n, y * limit / n)


def body_sticks(ux: float, uy: float, heading: float) -> tuple[float, float]:
    """World-frame (ux, uy) -> (pitch, roll) for a drone facing `heading`. Inverse of drivers/sim.py."""
    s, c = math.sin(heading), math.cos(heading)
    return ux * s + uy * c, ux * c - uy * s


def world_sticks(pitch: float, roll: float, heading: float) -> tuple[float, float]:
    """(pitch, roll) for a drone facing `heading` -> world-frame (ux, uy)."""
    s, c = math.sin(heading), math.cos(heading)
    return pitch * s + roll * c, pitch * c - roll * s


class PositionController:
    def __init__(self, gains: Gains = Gains(), limit: float = 0.3, i_limit: float = 0.2, vel_window_s: float = 0.25):
        self.gains = gains
        self.limit = limit  # max correction stick magnitude
        self.i_limit = i_limit  # max stick the integral alone may contribute
        self.vel_window_s = vel_window_s  # velocity from poses this far apart, so camera noise averages out
        self.reset()

    def reset(self) -> None:
        self._ix = self._iy = 0.0
        self._poses: deque[Pose] = deque()
        self._out = (0.0, 0.0)

    def velocity(self) -> tuple[float, float]:
        if len(self._poses) < 2:
            return 0.0, 0.0
        a, b = self._poses[0], self._poses[-1]
        dt = b.t - a.t
        return ((b.x - a.x) / dt, (b.y - a.y) / dt) if dt > 0 else (0.0, 0.0)

    def update(self, pose: Pose, target: tuple[float, float], integrate: bool = True) -> tuple[float, float]:
        """World-frame correction (ux, uy). Call once per new pose; a repeated pose returns the last output.

        Pass integrate=False while the target is moving (autopilot): the drone always lags a moving
        target, and integrating that lag winds up an overshoot at the end. The wind estimate already
        in the integral is kept.
        """
        if self._poses and pose.t <= self._poses[-1].t:
            return self._out
        dt = pose.t - self._poses[-1].t if self._poses else 0.0
        self._poses.append(pose)
        while pose.t - self._poses[0].t > self.vel_window_s and len(self._poses) > 2:
            self._poses.popleft()

        g = self.gains
        ex, ey = target[0] - pose.x, target[1] - pose.y
        if g.ki > 0 and integrate:
            # Anti-windup: the integral term alone can never ask for more than i_limit.
            self._ix, self._iy = clamp_vec(self._ix + ex * dt, self._iy + ey * dt, self.i_limit / g.ki)
        vx, vy = self.velocity()
        ux = g.kp * ex + g.ki * self._ix - g.kd * vx
        uy = g.kp * ey + g.ki * self._iy - g.kd * vy
        self._out = clamp_vec(ux, uy, self.limit)
        return self._out

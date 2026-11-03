"""On-screen fake drone so the whole gesture pipeline can be flown without hardware."""
from __future__ import annotations

import math
import random
import time
from collections import deque
from typing import Callable

import cv2
import numpy as np

from drivers.base import Driver
from flight.types import ZERO, Axes

ROOM_M = 8.0  # square room, meters
CEILING_M = 2.5
HOVER_M = 1.0
MAX_SPEED = 1.5  # m/s at full stick
MAX_CLIMB = 1.0  # m/s at full stick
MAX_YAW = math.radians(120)  # rad/s at full stick


class SimDriver(Driver):
    name = "sim"

    def __init__(
        self,
        clock: Callable[[], float] = time.monotonic,
        wind: tuple[float, float] = (0.0, 0.0),
        gust: float = 0.0,
        rng: random.Random | None = None,
        room_m: float = ROOM_M,
    ):
        self._clock = clock
        self.room_m = room_m
        self._last = clock()
        self.wind = wind  # m/s (x, y) pushing the drone while it's in the air, like the real one's drift
        self.gust = gust  # m/s size of slowly changing random gusts on top
        self._gust = (0.0, 0.0)
        self._rng = rng or random.Random()
        self.axes = ZERO
        self.x = self.y = 0.0  # meters, room center is the origin
        self.z = 0.0
        self.heading = 0.0  # radians, 0 = facing "up" on the map
        self.flying = False
        self.phase = "landed"  # landed | taking_off | flying | landing | crashed
        self.bumped = False
        self.trail: deque[tuple[float, float]] = deque(maxlen=200)

    def connect(self) -> None:
        self._last = self._clock()

    def takeoff(self) -> None:
        self._advance()
        if self.phase in ("landed", "crashed"):
            self.phase = "taking_off"
            self.flying = True

    def land(self) -> None:
        self._advance()
        if self.flying:
            self.phase = "landing"

    def estop(self) -> None:
        self._advance()
        if self.flying:
            self.phase = "crashed"
        self.flying = False
        self.z = 0.0
        self.axes = ZERO

    def set_axes(self, axes: Axes) -> None:
        self._advance()
        self.axes = axes

    def status(self) -> str:
        return f"sim {self.phase} z={self.z:.1f}m" + (" BUMP" if self.bumped else "")

    def latest_frame(self) -> np.ndarray:
        return self.render(480, 480)

    # ------------------------------------------------------------------ physics
    def update(self, dt: float) -> None:
        a = self.axes
        if self.phase == "taking_off":
            self.z = min(HOVER_M, self.z + 0.8 * dt)
            if self.z >= HOVER_M:
                self.phase = "flying"
        elif self.phase == "landing":
            self.z = max(0.0, self.z - 0.6 * dt)
            if self.z <= 0.0:
                self.phase, self.flying = "landed", False
        elif self.phase == "flying":
            self.heading += a.yaw * MAX_YAW * dt
            fwd, right = a.pitch * MAX_SPEED * dt, a.roll * MAX_SPEED * dt
            self.x += fwd * math.sin(self.heading) + right * math.cos(self.heading)
            self.y += fwd * math.cos(self.heading) - right * math.sin(self.heading)
            self.z = min(CEILING_M, max(0.2, self.z + a.throttle * MAX_CLIMB * dt))
        if self.phase in ("taking_off", "flying", "landing"):
            self._drift(dt)
            half = self.room_m / 2 - 0.2
            self.bumped = abs(self.x) > half or abs(self.y) > half
            self.x, self.y = max(-half, min(half, self.x)), max(-half, min(half, self.y))
        self.trail.append((self.x, self.y))

    def _drift(self, dt: float) -> None:
        if self.gust:
            k = math.exp(-dt / 2.0)  # gusts wander over a couple of seconds
            n = self.gust * math.sqrt(1 - k * k)
            self._gust = (self._gust[0] * k + self._rng.gauss(0, n), self._gust[1] * k + self._rng.gauss(0, n))
        self.x += (self.wind[0] + self._gust[0]) * dt
        self.y += (self.wind[1] + self._gust[1]) * dt

    def _advance(self) -> None:
        now = self._clock()
        dt, self._last = now - self._last, now
        # Integrate in small steps so a slow frame doesn't teleport the drone.
        while dt > 0:
            step = min(dt, 0.02)
            self.update(step)
            dt -= step

    # ------------------------------------------------------------------ drawing
    def render(self, w: int, h: int) -> np.ndarray:
        img = np.full((h, w, 3), 28, np.uint8)
        bar_w = 50
        size = min(w - bar_w, h) - 20
        ox, oy = 10, (h - size) // 2
        scale = size / self.room_m

        def to_px(x: float, y: float) -> tuple[int, int]:
            return int(ox + (x + self.room_m / 2) * scale), int(oy + (self.room_m / 2 - y) * scale)

        cv2.rectangle(img, (ox, oy), (ox + size, oy + size), (90, 90, 90), 1)
        for i in range(1, int(self.room_m)):
            p = int(i * scale)
            cv2.line(img, (ox + p, oy), (ox + p, oy + size), (45, 45, 45), 1)
            cv2.line(img, (ox, oy + p), (ox + size, oy + p), (45, 45, 45), 1)

        pts = [to_px(x, y) for x, y in self.trail]
        for p, q in zip(pts, pts[1:]):
            cv2.line(img, p, q, (90, 140, 90), 1)

        cx, cy = to_px(self.x, self.y)
        r = 10 + int(self.z * 4)  # bigger = higher
        nose = (cx + int(r * math.sin(self.heading)), cy - int(r * math.cos(self.heading)))
        left = (cx + int(r * math.sin(self.heading - 2.5)), cy - int(r * math.cos(self.heading - 2.5)))
        right = (cx + int(r * math.sin(self.heading + 2.5)), cy - int(r * math.cos(self.heading + 2.5)))
        color = {
            "flying": (80, 220, 80),
            "taking_off": (80, 220, 220),
            "landing": (80, 220, 220),
            "crashed": (60, 60, 230),
        }.get(self.phase, (170, 170, 170))
        cv2.fillPoly(img, [np.array([nose, left, right])], color)

        # Altitude bar on the right.
        bx, top, bottom = w - bar_w + 15, oy, oy + size
        cv2.rectangle(img, (bx, top), (bx + 18, bottom), (90, 90, 90), 1)
        fill = bottom - int((self.z / CEILING_M) * size)
        cv2.rectangle(img, (bx + 1, fill), (bx + 17, bottom - 1), color, -1)
        cv2.putText(img, f"{self.z:.1f}m", (bx - 8, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1)
        cv2.putText(img, f"SIM  {self.phase}", (ox, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
        return img

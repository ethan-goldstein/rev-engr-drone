"""Where the drone is, in arena meters, plus a pose source that reads the simulator.

Arena frame (same as drivers/sim.py and the map panel): x = right, y = forward / up on the map,
heading 0 = facing +y, increasing clockwise (turning right).
"""
from __future__ import annotations

import random
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Protocol


@dataclass(frozen=True)
class Pose:
    x: float
    y: float
    heading: float  # radians
    t: float  # monotonic time of the measurement


class PoseSource(Protocol):
    def latest(self) -> Pose | None:
        """Newest fresh pose, or None while the drone isn't being tracked."""


class SimPoseSource:
    """Stands in for the tracking camera: the sim's true position plus noise and latency."""

    def __init__(
        self,
        sim,
        noise: float = 0.02,
        latency: float = 0.05,
        clock: Callable[[], float] = time.monotonic,
        rng: random.Random | None = None,
    ):
        self.sim = sim
        self.noise = noise
        self.latency = latency
        self._clock = clock
        self._rng = rng or random.Random()
        self._history: deque[Pose] = deque()
        self.enabled = True  # tests switch this off to simulate losing the marker

    def latest(self) -> Pose | None:
        now = self._clock()
        jitter = lambda: self._rng.gauss(0.0, self.noise)  # noqa: E731
        self._history.append(Pose(self.sim.x + jitter(), self.sim.y + jitter(), self.sim.heading, now))
        while len(self._history) > 1 and now - self._history[1].t >= self.latency:
            self._history.popleft()
        if not self.enabled:
            return None
        oldest = self._history[0]  # the newest sample that is at least `latency` old, if there is one
        return oldest if now - oldest.t >= self.latency else None

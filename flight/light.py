"""Too dark to fly: the F19 holds its spot with a small camera underneath that watches the ground.

In the dark that camera sees nothing and the drone wanders off. On 2 Nov it went more than 10 m in about
4 s, into a building. The front camera's brightness is a good stand-in. The daylight flights measured
106-140 out of 255, and every flight that drifted measured 3-17. So takeoff waits for light.
"""
from __future__ import annotations

from collections import deque

import cv2
import numpy as np

DARK = 40  # mean brightness (0-255) of the drone's front camera below which it can't hold position


class LightCheck:
    def __init__(self, window: int = 15, dark: float = DARK):
        self.dark = dark
        self._values: deque[float] = deque(maxlen=window)

    def update(self, frame: np.ndarray | None) -> None:
        if frame is not None:
            small = cv2.resize(frame, (64, 36), interpolation=cv2.INTER_AREA)
            self._values.append(float(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).mean()))

    @property
    def brightness(self) -> float | None:
        return float(np.median(self._values)) if self._values else None

    @property
    def too_dark(self) -> bool:
        """False until there's video: no picture isn't proof of darkness."""
        b = self.brightness
        return b is not None and b < self.dark

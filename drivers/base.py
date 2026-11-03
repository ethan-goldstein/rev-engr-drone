"""Driver interface plus a stall watchdog shared by the real-drone drivers."""
from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from typing import Callable

import numpy as np

from flight.types import Axes


class Driver(ABC):
    name = "driver"

    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def takeoff(self) -> None: ...

    @abstractmethod
    def land(self) -> None: ...

    @abstractmethod
    def estop(self) -> None:
        """Cut the motors immediately."""

    @abstractmethod
    def set_axes(self, axes: Axes) -> None:
        """Called every frame with governed stick values in [-1, 1]."""

    def calibrate(self) -> None:
        """Level the gyro. Only while landed on a flat surface; a no-op where unsupported."""

    def status(self) -> str:
        return "ok"

    def latest_frame(self) -> np.ndarray | None:
        """Most recent BGR image from the drone camera (or sim view), if any."""
        return None

    def close(self) -> None:
        pass


class StallWatchdog:
    """Zeroes the sticks if set_axes stops arriving (e.g. the app's main loop froze).

    Real-drone drivers keep re-sending the last sticks on their own threads, so without
    this a hung webcam read would leave the drone flying on stale input.
    """

    def __init__(self, zero_sticks: Callable[[], None], timeout_s: float = 0.4):
        self._zero = zero_sticks
        self._timeout = timeout_s
        self._last_feed = time.monotonic()
        self._tripped = False
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="StallWatchdog", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def feed(self) -> None:
        self._last_feed = time.monotonic()
        self._tripped = False

    @property
    def tripped(self) -> bool:
        return self._tripped

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        while not self._stop.wait(0.05):
            if not self._tripped and time.monotonic() - self._last_feed > self._timeout:
                self._tripped = True
                self._zero()

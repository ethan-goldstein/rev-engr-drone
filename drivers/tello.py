"""Driver for a Ryze/DJI Tello via its official Wi-Fi SDK (djitellopy). Untested: no Tello on hand.

Install with `uv pip install djitellopy`, join the TELLO-XXXXXX network, then run
`main.py --driver tello`.
"""
from __future__ import annotations

import threading

from drivers.base import Driver, StallWatchdog
from flight.types import Axes


class TelloDriver(Driver):
    name = "tello"

    def __init__(self):
        self._tello = None
        self._lock = threading.Lock()
        self._watchdog: StallWatchdog | None = None
        self._battery = "?"

    def connect(self) -> None:
        from djitellopy import Tello

        self._tello = Tello()
        self._tello.connect()
        self._battery = str(self._tello.get_battery())
        self._watchdog = StallWatchdog(lambda: self._send(0, 0, 0, 0))
        self._watchdog.start()

    def takeoff(self) -> None:
        # djitellopy blocks until the Tello acknowledges; run it off the UI thread.
        threading.Thread(target=self._tello.takeoff, daemon=True).start()

    def land(self) -> None:
        threading.Thread(target=self._tello.land, daemon=True).start()

    def estop(self) -> None:
        self._tello.emergency()

    def set_axes(self, axes: Axes) -> None:
        self._watchdog.feed()
        # Tello order: left/right, forward/back, up/down, yaw; each -100..100.
        self._send(axes.roll, axes.pitch, axes.throttle, axes.yaw)

    def _send(self, lr: float, fb: float, ud: float, yaw: float) -> None:
        with self._lock:
            self._tello.send_rc_control(*(int(round(v * 100)) for v in (lr, fb, ud, yaw)))

    def status(self) -> str:
        return f"tello battery {self._battery}%"

    def close(self) -> None:
        if self._watchdog:
            self._watchdog.stop()
        if self._tello:
            self._send(0, 0, 0, 0)
            self._tello.end()

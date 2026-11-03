"""Real-drone driver for "WiFi UAV"-app drones (e.g. Contixo F19) via turbodrone.

Wires turbodrone's WiFi UAV model, RC adapter, video service and FlightController the
same way vendor/turbodrone/backend/web_server.py does (pinned commit 2c09cf8), minus
the web server. turbodrone is imported lazily in connect() so nothing from vendor/ runs
unless this driver is selected.

Join the drone's Wi-Fi (e.g. "Contixo-F19-XXXX") before starting.
Env overrides (same names turbodrone uses): DRONE_IP, CONTROL_PORT, VIDEO_PORT.
"""
from __future__ import annotations

import os
import queue
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from drivers.base import Driver, StallWatchdog
from flight.types import Axes

BACKEND_DIR = Path(__file__).resolve().parent.parent / "vendor" / "turbodrone" / "backend"
ESTOP_REPEAT_S = 0.5  # keep sending the motor-stop bit this long after SPACE

# WiFi UAV takeoff and land share one toggle bit, sent in a single packet. A missed or extra toggle
# leaves the app and the drone disagreeing about whether it's flying, so the toggle is only used for
# takeoff. Landing is the governor holding the throttle down and then sending the motor-stop bit.


class TurbodroneDriver(Driver):
    name = "turbodrone"

    def __init__(self, drone_type: str = "wifi_uav", control_rate: float = 80.0):
        self.drone_type = drone_type
        self.control_rate = control_rate
        self._fc = None
        self._receiver = None
        self._frames: queue.Queue | None = None
        self._last_frame: np.ndarray | None = None
        self._last_frame_at = 0.0
        self._watchdog: StallWatchdog | None = None
        self._estop_until = 0.0

    def connect(self) -> None:
        if not BACKEND_DIR.is_dir():
            raise RuntimeError(f"turbodrone not found at {BACKEND_DIR}; see README for the pinned clone.")
        if str(BACKEND_DIR) not in sys.path:
            sys.path.append(str(BACKEND_DIR))

        from control.strategies import DirectStrategy
        from models.wifi_uav_rc import WifiUavRcModel
        from protocols.wifi_uav_rc_protocol_adapter import WifiUavRcProtocolAdapter
        from protocols.wifi_uav_video_protocol import WifiUavVideoProtocolAdapter
        from services.flight_controller import FlightController
        from services.video_receiver import VideoReceiverService
        from utils.wifi_uav_variants import WIFI_UAV_DRONE_TYPES, resolve_wifi_uav_variant

        if self.drone_type not in WIFI_UAV_DRONE_TYPES:
            raise ValueError(f"{self.drone_type!r} is not a WiFi UAV type; options: {sorted(WIFI_UAV_DRONE_TYPES)}")

        variant = resolve_wifi_uav_variant(self.drone_type)
        drone_ip = os.getenv("DRONE_IP", "192.168.169.1")
        ctrl_port = int(os.getenv("CONTROL_PORT", 8800))
        video_port = int(os.getenv("VIDEO_PORT", 8800))

        model = WifiUavRcModel()
        model.set_strategy(DirectStrategy())  # absolute stick positions, like the web client's "abs" mode
        # Linear sticks: turbodrone's default expo curve shrinks 0.1 to 0.03, which would flatten the
        # small corrections position hold sends.
        model.expo_factor = 0.0
        rc_proto = WifiUavRcProtocolAdapter(drone_ip, ctrl_port, variant=variant)

        # The video service owns the UDP socket and hands it to the RC adapter, so it starts first.
        self._receiver = VideoReceiverService(
            protocol_adapter_class=WifiUavVideoProtocolAdapter,
            protocol_adapter_args={
                "drone_ip": drone_ip,
                "control_port": ctrl_port,
                "video_port": video_port,
                "variant": variant,
                "debug": False,
            },
            rc_adapter=rc_proto,
        )
        self._receiver.start()
        self._frames = self._receiver.get_frame_queue()
        time.sleep(1.0)

        self._fc = FlightController(model, rc_proto, self.control_rate)
        self._fc.start()
        self._watchdog = StallWatchdog(lambda: self._fc.set_axes_from("watchdog", 0, 0, 0, 0))
        self._watchdog.start()

    def takeoff(self) -> None:
        self._fc.model.takeoff()

    def land(self) -> None:
        pass  # deliberately no land toggle; see the comment at the top of this file

    def estop(self) -> None:
        self._fc.set_axes_from("estop", 0, 0, 0, 0)
        self._fc.model.emergency_stop()
        # The stop flag is one-shot (one UDP packet); keep re-arming it so a lost packet can't swallow it.
        self._estop_until = time.monotonic() + ESTOP_REPEAT_S

    def calibrate(self) -> None:
        # Same one-shot flag as the WiFi UAV app's gyro-calibrate button; the adapter clears it after sending.
        self._fc.model.calibration_flag = True

    def set_axes(self, axes: Axes) -> None:
        self._watchdog.feed()
        if time.monotonic() < self._estop_until:
            self._fc.model.emergency_stop()
        self._fc.set_axes_from("gesture", axes.throttle, axes.yaw, axes.pitch, axes.roll)

    def latest_frame(self) -> np.ndarray | None:
        if self._frames is None:
            return None
        newest = None
        try:
            while True:
                newest = self._frames.get_nowait()
        except queue.Empty:
            pass
        if newest is not None:
            data = getattr(newest, "data", newest)
            img = cv2.imdecode(np.frombuffer(bytes(data), np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                self._last_frame, self._last_frame_at = img, time.monotonic()
        return self._last_frame

    @property
    def frame_time(self) -> float | None:
        """When the newest drone frame was decoded (monotonic), so callers can skip frames they've seen."""
        return self._last_frame_at or None

    def status(self) -> str:
        if self._fc is None:
            return "turbodrone: not connected"
        video = "video ok" if time.monotonic() - self._last_frame_at < 2.0 else "no video"
        stalled = " WATCHDOG" if self._watchdog and self._watchdog.tripped else ""
        return f"turbodrone {self.drone_type}: {video}{stalled}"

    def close(self) -> None:
        if self._watchdog:
            self._watchdog.stop()
        if self._fc:
            self._fc.set_axes_from("close", 0, 0, 0, 0)
            self._fc.stop()
        if self._receiver:
            self._receiver.stop()

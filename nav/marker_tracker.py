"""Finds the drone's ArUco tag in the tracking camera and turns it into a Pose.

The camera looks down at the flying area (an iPhone as a Continuity Camera, or a USB webcam). A
background thread reads frames so a slow camera never stalls the gesture loop.
"""
from __future__ import annotations

import threading
import time
from typing import Callable

import cv2
import numpy as np

from nav.arena import Arena
from nav.pose import Pose

DICTIONARY = cv2.aruco.DICT_4X4_50
MARKER_ID = 0


def make_detector() -> cv2.aruco.ArucoDetector:
    return cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(DICTIONARY), cv2.aruco.DetectorParameters())


def open_camera(index: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # more pixels on the small tag
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    if not cap.isOpened():
        raise SystemExit(
            f"Could not open tracking camera {index}. An iPhone only shows up as a camera while it's nearby "
            "and locked (a USB cable makes it reliable); a USB webcam usually takes the next free number. "
            "Try --track-camera 0, 1 or 2."
        )
    return cap


class MarkerTracker:
    def __init__(
        self,
        arena: Arena,
        camera: int | None = None,
        marker_id: int = MARKER_ID,
        stale_s: float = 0.3,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.arena = arena
        self.marker_id = marker_id
        self.stale_s = stale_s  # a pose older than this counts as "not tracked"
        self._clock = clock
        self._detector = make_detector()
        self._lock = threading.Lock()
        self._pose: Pose | None = None
        self._frame: np.ndarray | None = None
        self._cap = open_camera(camera) if camera is not None else None
        self._running = self._cap is not None
        self._thread: threading.Thread | None = None
        if self._running:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def process(self, frame: np.ndarray, now: float) -> Pose | None:
        """Find the tag in one camera frame. Called by the thread; tests call it directly."""
        corners, ids, _ = self._detector.detectMarkers(frame)
        pose = None
        if ids is not None:
            for c, marker in zip(corners, ids.ravel()):
                if marker == self.marker_id:
                    pose = Pose(*self.arena.heading_from_marker(c), now)
                    break

        annotated = frame.copy()
        outline = np.int32(self.arena.image_corners).reshape(-1, 1, 2)
        cv2.polylines(annotated, [outline], True, (60, 60, 230), 3)
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(annotated, corners, ids)
        with self._lock:
            if pose is not None:
                self._pose = pose
            self._frame = annotated
        return pose

    def latest(self) -> Pose | None:
        with self._lock:
            pose = self._pose
        if pose is None or self._clock() - pose.t > self.stale_s:
            return None
        return pose

    def latest_frame(self) -> np.ndarray | None:
        with self._lock:
            return self._frame

    def close(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._cap is not None:
            self._cap.release()

    def _run(self) -> None:
        while self._running:
            ok, frame = self._cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            self.process(frame, self._clock())

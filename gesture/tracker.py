"""Webcam hand tracking with MediaPipe's GestureRecognizer (Tasks API)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    GestureRecognizer,
    GestureRecognizerOptions,
    RunningMode,
)

DEFAULT_MODEL = Path(__file__).resolve().parent.parent / "assets" / "gesture_recognizer.task"

# Canned gestures the stock model knows. "None" means a hand is visible but no gesture matched.
CANNED_GESTURES = (
    "Closed_Fist",
    "Open_Palm",
    "Pointing_Up",
    "Thumb_Down",
    "Thumb_Up",
    "Victory",
    "ILoveYou",
)


@dataclass(frozen=True)
class HandObservation:
    gesture: str | None  # one of CANNED_GESTURES, or None when unrecognized
    score: float
    landmarks: tuple[tuple[float, float, float], ...]  # 21 points, x/y normalized to the frame


class GestureTracker:
    """Runs the recognizer on BGR frames. Feed frames in timestamp order (VIDEO mode)."""

    def __init__(self, model_path: Path | str = DEFAULT_MODEL, min_detection: float = 0.6):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Gesture model not found at {model_path}. Download it with:\n"
                "  curl -fsSL -o assets/gesture_recognizer.task "
                "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/"
                "gesture_recognizer/float16/1/gesture_recognizer.task"
            )
        options = GestureRecognizerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=min_detection,
            min_hand_presence_confidence=min_detection,
            min_tracking_confidence=0.5,
        )
        self._recognizer = GestureRecognizer.create_from_options(options)
        self._last_ts = -1

    def process(self, frame_bgr: np.ndarray, timestamp_ms: int) -> HandObservation | None:
        # MediaPipe rejects non-increasing timestamps in VIDEO mode.
        timestamp_ms = max(timestamp_ms, self._last_ts + 1)
        self._last_ts = timestamp_ms

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self._recognizer.recognize_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp_ms
        )
        if not result.hand_landmarks:
            return None

        landmarks = tuple((p.x, p.y, p.z) for p in result.hand_landmarks[0])
        gesture, score = None, 0.0
        if result.gestures and result.gestures[0]:
            top = result.gestures[0][0]
            score = float(top.score)
            if top.category_name in CANNED_GESTURES:
                gesture = top.category_name
        return HandObservation(gesture=gesture, score=score, landmarks=landmarks)

    def close(self) -> None:
        self._recognizer.close()

    def __enter__(self) -> "GestureTracker":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

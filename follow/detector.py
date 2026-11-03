"""Finds people in the drone's video with MediaPipe's object detector (EfficientDet-Lite0, COCO "person")."""
from __future__ import annotations

from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import ObjectDetector, ObjectDetectorOptions, RunningMode

from follow.types import Detection

ASSETS = Path(__file__).resolve().parent.parent / "assets"
DEFAULT_MODEL = ASSETS / "efficientdet_lite0.tflite"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float16/1/efficientdet_lite0.tflite"
)


class PersonDetector:
    def __init__(self, model_path: Path | str = DEFAULT_MODEL, min_score: float = 0.4, max_people: int = 5):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Person model not found at {model_path}. Download it with:\n  curl -fsSL -o {model_path} {MODEL_URL}")
        options = ObjectDetectorOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=RunningMode.VIDEO,
            category_allowlist=["person"],
            score_threshold=min_score,
            max_results=max_people,
        )
        self._detector = ObjectDetector.create_from_options(options)
        self._last_ms = -1

    def detect(self, frame: np.ndarray, t: float) -> list[Detection]:
        ms = max(int(t * 1000), self._last_ms + 1)  # VIDEO mode needs increasing timestamps
        self._last_ms = ms
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self._detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ms)
        out = []
        for d in result.detections:
            b = d.bounding_box
            score = d.categories[0].score if d.categories else 0.0
            out.append(Detection(float(b.origin_x), float(b.origin_y), float(b.origin_x + b.width), float(b.origin_y + b.height), score))
        return out

    def close(self) -> None:
        self._detector.close()

"""Finds people in the drone's video.

PersonDetector: MediaPipe's object detector (EfficientDet-Lite0, COCO "person") for everyone in view.
FollowDetector: that, plus MediaPipe's pose tracker, which carries one person from frame to frame. On the
3 Nov field test EfficientDet alone found the walking pilot in 28% of frames (the laptop they carried hid
their torso); the pose tracker found them in 85%.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    ObjectDetector,
    ObjectDetectorOptions,
    PoseLandmarker,
    PoseLandmarkerOptions,
    RunningMode,
)

from follow.types import Detection, iou

ASSETS = Path(__file__).resolve().parent.parent / "assets"
DEFAULT_MODEL = ASSETS / "efficientdet_lite0.tflite"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float16/1/efficientdet_lite0.tflite"
)
POSE_MODEL = ASSETS / "pose_landmarker_lite.task"


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


def pose_box(landmarks, w: int, h: int, min_visibility: float = 0.3) -> Detection | None:
    """A person box from pose landmarks ((x, y, visibility), 0-1), or None if too little of the body shows."""
    pts = [(x * w, y * h, v) for x, y, v in landmarks if v >= min_visibility]
    if len(pts) < 8:
        return None
    xs, ys, vs = zip(*pts)
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    bw, bh = x1 - x0, y1 - y0
    if bh < 20:
        return None
    # The landmarks stop at the nose and heels: pad up for the top of the head, and a little all round.
    x0, x1, y0, y1 = x0 - 0.1 * bw, x1 + 0.1 * bw, y0 - 0.1 * bh, y1 + 0.03 * bh
    return Detection(max(0.0, x0), max(0.0, y0), min(float(w), x1), min(float(h), y1), score=float(np.mean(vs)))


def merge(dets: list[Detection], extra: Detection | None, same: float = 0.4) -> list[Detection]:
    """Add the pose tracker's box unless the object detector already has that person."""
    if extra is None or any(iou((d.x0, d.y0, d.x1, d.y1), (extra.x0, extra.y0, extra.x1, extra.y1)) >= same for d in dets):
        return dets
    return dets + [extra]


class FollowDetector:
    def __init__(self, people: PersonDetector | None = None, pose_model: Path | str = POSE_MODEL):
        self.people = people or PersonDetector()
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(pose_model)),
            running_mode=RunningMode.VIDEO,  # tracks from the last frame's landmarks instead of searching anew
            num_poses=1,
            min_pose_detection_confidence=0.4,
            min_pose_presence_confidence=0.4,
            min_tracking_confidence=0.4,
        )
        self._pose = PoseLandmarker.create_from_options(options)
        self._last_ms = -1

    def detect(self, frame: np.ndarray, t: float) -> list[Detection]:
        dets = self.people.detect(frame, t)
        ms = max(int(t * 1000), self._last_ms + 1)
        self._last_ms = ms
        rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        result = self._pose.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ms)
        extra = None
        if result.pose_landmarks:
            lm = [(p.x, p.y, p.visibility if p.visibility is not None else 1.0) for p in result.pose_landmarks[0]]
            extra = pose_box(lm, frame.shape[1], frame.shape[0])
        return merge(dets, extra)

    def close(self) -> None:
        self.people.close()
        self._pose.close()

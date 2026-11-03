"""Both arms up, as seen by the drone's own camera, means land. Works after you've walked away from the laptop.

PoseLandmarker (lite) runs on a crop around the locked person: 33 body points, each with a
visibility score. "Arms up" = both wrists clearly visible and above the nose.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode

from follow.types import Detection

NOSE, LEFT_EYE, RIGHT_EYE, LEFT_WRIST, RIGHT_WRIST = 0, 2, 5, 15, 16
FACE = range(11)  # nose, eyes, ears, mouth
FACE_LEFT_SIDE = (1, 2, 3, 7, 9)  # the person's own left: eye inner/center/outer, ear, mouth corner
DEFAULT_MODEL = Path(__file__).resolve().parent.parent / "assets" / "pose_landmarker_lite.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)

Landmarks = Sequence[tuple[float, float, float]]  # (x, y, visibility), y grows downward


def arms_up(lm: Landmarks, min_visibility: float = 0.5) -> bool:
    nose, lw, rw = lm[NOSE], lm[LEFT_WRIST], lm[RIGHT_WRIST]
    return lw[2] >= min_visibility and rw[2] >= min_visibility and lw[1] < nose[1] and rw[1] < nose[1]


def face_box(lm: Landmarks, rect: tuple[int, int, int, int], min_visibility: float = 0.6):
    """The face in frame pixels (x0, y0, x1, y1), or None unless it's turned toward the camera.

    `rect` is the crop the landmarks came from. This finds a face; it doesn't recognize whose it is.
    """
    if min(lm[i][2] for i in (NOSE, LEFT_EYE, RIGHT_EYE)) < min_visibility:
        return None
    if lm[LEFT_EYE][0] <= lm[RIGHT_EYE][0]:  # their left eye is on the image's right only when they face us
        return None
    x0, y0, x1, y1 = rect
    xs = [x0 + lm[i][0] * (x1 - x0) for i in FACE]
    ys = [y0 + lm[i][1] * (y1 - y0) for i in FACE]
    size = max(max(xs) - min(xs), 4.0) * 1.3
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    return cx - size / 2, cy - 0.7 * size, cx + size / 2, cy + 0.6 * size


class ArmsUpCommand:
    def __init__(self, pose_fn: Callable[[np.ndarray], Landmarks | None], hold_s: float = 1.5):
        self.pose_fn = pose_fn  # crop -> landmarks, or None if no body was found
        self.hold_s = hold_s
        self.progress = 0.0  # 0..1 while arms are held up, for the HUD
        self.landmarks: Landmarks | None = None  # from the latest update, with the crop they're relative to
        self.rect = (0, 0, 0, 0)
        self._since: float | None = None
        self._fired = False

    def update(self, frame: np.ndarray, box: Detection, t: float) -> bool:
        """True exactly once per arms-up hold. Lower the arms to re-arm."""
        h, w = frame.shape[:2]
        pad_x, pad_y = 0.3 * box.w, 0.25 * box.h  # raised arms can reach past the detector's box
        x0, y0 = int(max(0, box.x0 - pad_x)), int(max(0, box.y0 - pad_y))
        x1, y1 = int(min(w, box.x1 + pad_x)), int(min(h, box.y1 + pad_y))
        lm = self.pose_fn(frame[y0:y1, x0:x1]) if x1 - x0 >= 16 and y1 - y0 >= 32 else None
        self.landmarks, self.rect = lm, (x0, y0, x1, y1)
        if lm is None or not arms_up(lm):
            self._since, self._fired, self.progress = None, False, 0.0
            return False
        if self._since is None:
            self._since = t
        self.progress = min(1.0, (t - self._since) / self.hold_s)
        if self.progress >= 1.0 and not self._fired:
            self._fired = True
            return True
        return False


class MediaPipePose:
    def __init__(self, model_path: Path | str = DEFAULT_MODEL):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Pose model not found at {model_path}. Download it with:\n  curl -fsSL -o {model_path} {MODEL_URL}")
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=RunningMode.IMAGE,
            num_poses=1,
        )
        self._landmarker = PoseLandmarker.create_from_options(options)

    def __call__(self, crop: np.ndarray) -> Landmarks | None:
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        result = self._landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)))
        if not result.pose_landmarks:
            return None
        return [(p.x, p.y, p.visibility if p.visibility is not None else 1.0) for p in result.pose_landmarks[0]]

    def close(self) -> None:
        self._landmarker.close()


def sim_pose(person) -> Callable[[np.ndarray], Landmarks]:
    """Fake landmarks for a SimPerson: wrists above the head while person.arms_up is set (U key in the sim),
    and a face only while they aren't walking away from the drone."""

    def pose(_crop: np.ndarray) -> Landmarks:
        lm = [(0.5, 0.5, 1.0)] * 33
        side = 1 if person.facing_drone else -1  # facing away puts their left eye on the image's left
        for i in FACE:
            lm[i] = (0.5 + side * (0.08 if i in FACE_LEFT_SIDE else -0.08), 0.22, 1.0)
        lm[NOSE] = (0.5, 0.23, 1.0)  # the head, in the crop ArmsUpCommand pads around the box
        wrist_y = 0.05 if person.arms_up else 0.55
        lm[LEFT_WRIST], lm[RIGHT_WRIST] = (0.3, wrist_y, 1.0), (0.7, wrist_y, 1.0)
        return lm

    return pose

"""A detected person in the drone camera, plus the clothing-color fingerprint used to re-lock on them."""
from __future__ import annotations

from dataclasses import dataclass, replace

import cv2
import numpy as np


@dataclass(frozen=True, eq=False)
class Detection:
    x0: float  # box corners in drone-frame pixels
    y0: float
    x1: float
    y1: float
    score: float = 1.0
    appearance: np.ndarray | None = None  # torso color histogram, filled in from the frame

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2

    @property
    def w(self) -> float:
        return self.x1 - self.x0

    @property
    def h(self) -> float:
        return self.y1 - self.y0

    def with_appearance(self, appearance: np.ndarray | None) -> "Detection":
        return replace(self, appearance=appearance)


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def torso_histogram(frame: np.ndarray, det: Detection) -> np.ndarray | None:
    """Hue/saturation histogram of the shirt area (middle of the box), so the same person can be found again."""
    h, w = frame.shape[:2]
    x0 = int(max(0, det.x0 + 0.25 * det.w))
    x1 = int(min(w, det.x1 - 0.25 * det.w))
    y0 = int(max(0, det.y0 + 0.22 * det.h))
    y1 = int(min(h, det.y0 + 0.55 * det.h))
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    hsv = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
    return cv2.normalize(hist, hist).flatten()


def appearance_distance(a: np.ndarray | None, b: np.ndarray | None) -> float:
    """0 = identical colors, 1 = nothing in common. Unknown counts as a mismatch."""
    if a is None or b is None:
        return 1.0
    return float(cv2.compareHist(a.astype(np.float32), b.astype(np.float32), cv2.HISTCMP_BHATTACHARYYA))

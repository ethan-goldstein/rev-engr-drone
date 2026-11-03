"""The flying area as the tracking camera sees it: 4 clicked floor corners plus the measured size.

Corners are clicked in screen order: top-left, top-right, bottom-right, bottom-left. A perspective
transform maps camera pixels to arena meters with the origin at the middle of the area, so the map
looks like the camera picture.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class Arena:
    width_m: float
    depth_m: float
    image_corners: tuple[tuple[float, float], ...]  # 4 pixel points, screen order

    @property
    def rect(self) -> tuple[float, float, float, float]:
        """(xmin, ymin, xmax, ymax) in meters."""
        return (-self.width_m / 2, -self.depth_m / 2, self.width_m / 2, self.depth_m / 2)

    def _homography(self) -> np.ndarray:
        w, d = self.width_m / 2, self.depth_m / 2
        world = np.float32([(-w, d), (w, d), (w, -d), (-w, -d)])
        return cv2.getPerspectiveTransform(np.float32(self.image_corners), world)

    def to_world(self, px: tuple[float, float]) -> tuple[float, float]:
        out = cv2.perspectiveTransform(np.float32([[px]]), self._homography())
        return float(out[0, 0, 0]), float(out[0, 0, 1])

    def to_pixel(self, xy: tuple[float, float]) -> tuple[float, float]:
        out = cv2.perspectiveTransform(np.float32([[xy]]), np.linalg.inv(self._homography()))
        return float(out[0, 0, 0]), float(out[0, 0, 1])

    def heading_from_marker(self, corners_px: np.ndarray) -> tuple[float, float, float]:
        """(x, y, heading) from an ArUco marker's 4 corners (marker order: TL, TR, BR, BL).

        The marker's top edge points at the drone's nose. Heading is measured in the world, after
        the perspective transform, so a tilted camera doesn't skew it.
        """
        c = np.asarray(corners_px, dtype=np.float32).reshape(4, 2)
        center = self.to_world(tuple(c.mean(axis=0)))
        nose = self.to_world(tuple((c[0] + c[1]) / 2))
        heading = math.atan2(nose[0] - center[0], nose[1] - center[1])
        return center[0], center[1], heading

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps({"width_m": self.width_m, "depth_m": self.depth_m, "image_corners": self.image_corners}, indent=2)
        )

    @classmethod
    def load(cls, path: str | Path) -> "Arena":
        data = json.loads(Path(path).read_text())
        corners = tuple(tuple(map(float, p)) for p in data["image_corners"])
        if len(corners) != 4:
            raise ValueError(f"{path}: expected 4 image_corners, got {len(corners)}")
        return cls(float(data["width_m"]), float(data["depth_m"]), corners)

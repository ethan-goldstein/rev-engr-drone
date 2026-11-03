"""Person box -> sticks: turn to keep them centered, fly forward/back to hold the follow distance.

Distance comes from the box height through the pinhole model: distance = f * person_height / box_height,
where f follows from the camera's field of view (60 degrees on the F19).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from flight.types import Axes


@dataclass(frozen=True)
class FollowConfig:
    distance: float = 4.0  # m to hold
    min_distance: float = 3.0  # closer than this, back away
    max_distance: float = 8.0  # farther than this, stop advancing (Wi-Fi range, line of sight)
    person_height: float = 1.7  # m
    hfov_deg: float = 60.0
    k_yaw: float = 1.5  # yaw stick per unit of horizontal offset (-1 = left edge, +1 = right edge)
    max_yaw: float = 0.8
    k_dist: float = 0.8  # pitch stick per meter of distance error
    k_rate: float = 1.0  # extra pitch per m/s the person is walking away (keeps up instead of lagging)
    backoff: float = 0.6  # pitch stick backward when too close
    lead_s: float = 0.25  # predict this far ahead: about the F19's video lag


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


class FollowController:
    def __init__(self, cfg: FollowConfig = FollowConfig()):
        self.cfg = cfg

    def focal(self, frame_w: float) -> float:
        return (frame_w / 2) / math.tan(math.radians(self.cfg.hfov_deg / 2))

    def distance(self, box_h: float, frame_w: float) -> float:
        return self.focal(frame_w) * self.cfg.person_height / max(box_h, 1.0)

    def command(
        self, cx: float, box_h: float, vh: float, frame_w: float, cut_top: bool = False, cut_bottom: bool = False
    ) -> tuple[Axes, float]:
        """(yaw + pitch sticks in [-1, 1], estimated distance in m).

        `vh` is how fast the box height is changing (px/s). `cut_top` / `cut_bottom` mean the box
        touches that edge of the frame: part of the person is out of view, so the box is too short
        and the distance reads too far. Never fly toward someone measured like that.
        """
        c = self.cfg
        f = self.focal(frame_w)
        d = self.distance(box_h, frame_w)
        offset = (cx - frame_w / 2) / (frame_w / 2)
        yaw = clamp(c.k_yaw * offset, -c.max_yaw, c.max_yaw)
        walking_away = -f * c.person_height / max(box_h, 1.0) ** 2 * vh  # m/s, from the box shrinking
        pitch = c.k_dist * (d - c.distance) + c.k_rate * walking_away
        if cut_top or cut_bottom:
            pitch = min(pitch, 0.0)
        if (cut_top and cut_bottom) or d < c.min_distance:
            pitch = min(pitch, -c.backoff)
        if d > c.max_distance:
            pitch = min(pitch, 0.0)
        return Axes(yaw=yaw, pitch=clamp(pitch, -1.0, 1.0)), d

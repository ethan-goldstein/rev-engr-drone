"""Drag mode: the drone moves while your open palm moves, and holds still when your palm does.

Open-palm offsets are measured from an anchor instead of the screen center. The anchor is set
where the palm appears, then jumps to the palm whenever it has rested for `still_s`. So a still
hand always means centered sticks, wherever it is, and a slowly sagging arm doesn't steer.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from gesture.mapper import MapperConfig, palm_center
from gesture.tracker import HandObservation

# Offsets start at the resting palm, so the dead zone only has to absorb tracking jitter.
DRAG_MAPPER_CONFIG = MapperConfig(dead_zone=0.04, span=0.15)


@dataclass(frozen=True)
class DragConfig:
    still_s: float = 0.3  # palm must rest this long before the drone stops
    still_tol: float = 0.02  # palm wander (normalized) that still counts as resting
    release_s: float = 0.3  # palm missing this long drops the anchor; the next palm starts fresh


class DragAnchor:
    def __init__(self, cfg: DragConfig = DragConfig()):
        self.cfg = cfg
        self.anchor: tuple[float, float] | None = None
        self._rest: tuple[float, float] | None = None
        self._rest_since = 0.0
        self._palm_seen_at: float | None = None

    def update(self, obs: HandObservation | None, now: float) -> tuple[float, float] | None:
        """Feed every frame. Returns the anchor to measure open-palm offsets from, or None."""
        if obs is None or obs.gesture != "Open_Palm":
            # Ride out a dropped frame or two; after that, the next palm gets a fresh anchor.
            if self._palm_seen_at is None or now - self._palm_seen_at > self.cfg.release_s:
                self.anchor = self._rest = None
            return self.anchor

        palm = palm_center(obs.landmarks)
        self._palm_seen_at = now
        if self.anchor is None:
            self.anchor = palm
        if self._rest is None or math.dist(palm, self._rest) > self.cfg.still_tol:
            self._rest, self._rest_since = palm, now  # palm is moving
        elif now - self._rest_since >= self.cfg.still_s:
            self.anchor = palm  # palm has rested: re-center so the sticks go to zero
        return self.anchor

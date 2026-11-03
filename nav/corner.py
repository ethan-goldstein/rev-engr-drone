"""The AUTO box: an open palm held in a corner of the webcam view toggles the autopilot."""
from __future__ import annotations

from gesture.mapper import palm_center
from gesture.tracker import HandObservation


class CornerZone:
    def __init__(self, corner: str = "top_right", size: float = 0.22, hold_s: float = 1.0):
        self.corner = corner
        self.size = size  # fraction of the frame width/height
        self.hold_s = hold_s
        self._since: float | None = None
        self._fired = False
        self.progress = 0.0  # 0..1 while a hold is under way, for the HUD

    @property
    def rect(self) -> tuple[float, float, float, float]:
        """(x0, y0, x1, y1), normalized to the (mirrored) webcam frame."""
        s = self.size
        x0 = 1 - s if self.corner.endswith("right") else 0.0
        y0 = 0.0 if self.corner.startswith("top") else 1 - s
        return x0, y0, x0 + s, y0 + s

    def contains(self, obs: HandObservation | None) -> bool:
        if obs is None or obs.gesture != "Open_Palm":
            return False
        x, y = palm_center(obs.landmarks)
        x0, y0, x1, y1 = self.rect
        return x0 <= x <= x1 and y0 <= y <= y1

    def update(self, obs: HandObservation | None, now: float) -> bool:
        """True exactly once per hold. Take the palm out of the box to re-arm."""
        if not self.contains(obs):
            self._since, self._fired, self.progress = None, False, 0.0
            return False
        if self._since is None:
            self._since = now
        self.progress = min(1.0, (now - self._since) / self.hold_s)
        if self.progress >= 1.0 and not self._fired:
            self._fired = True
            return True
        return False

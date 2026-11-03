"""Trim: a small, constant stick offset that cancels a steady drift.

A drone that always slides the same way while "holding" usually has a gyro calibrated on uneven ground,
a chipped or bent prop, or a weaker motor. Trim leans it back the other way, like the trim buttons on a
remote. Keys: [ and ] nudge roll left/right, - and = nudge pitch back/forward. It's saved to trim.json so
the next flight starts trimmed. It's added after the governor, so the speed cap doesn't shrink it, and
only while the drone is in the air.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

from flight.governor import Decision, FlightState
from flight.types import Axes, Command

STEP = 0.02  # about 2.5 of the 255 stick steps per key press
LIMIT = 0.3  # more than this means something's broken (a prop, a motor), not out of trim


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


@dataclass
class Trim:
    roll: float = 0.0  # + leans right
    pitch: float = 0.0  # + leans forward

    def nudge(self, roll: float = 0.0, pitch: float = 0.0) -> str:
        self.roll = round(_clamp(self.roll + roll, -LIMIT, LIMIT), 3)
        self.pitch = round(_clamp(self.pitch + pitch, -LIMIT, LIMIT), 3)
        return self.label()

    def label(self) -> str:
        side = "left" if self.roll < 0 else "right"
        way = "back" if self.pitch < 0 else "forward"
        return f"trim: {abs(self.roll):.2f} {side}, {abs(self.pitch):.2f} {way}"

    def applied(self, decision: Decision) -> Decision:
        """The decision with the trim added, if the drone is in the air and not being shut off."""
        if decision.state not in (FlightState.FLYING, FlightState.LANDING) or decision.command is Command.ESTOP:
            return decision
        a = decision.axes
        axes = Axes(a.throttle, a.yaw, _clamp(a.pitch + self.pitch, -1.0, 1.0), _clamp(a.roll + self.roll, -1.0, 1.0))
        return replace(decision, axes=axes)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"roll": self.roll, "pitch": self.pitch}, indent=2) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "Trim":
        try:
            data = json.loads(Path(path).read_text())
            return cls(_clamp(float(data.get("roll", 0.0)), -LIMIT, LIMIT), _clamp(float(data.get("pitch", 0.0)), -LIMIT, LIMIT))
        except (OSError, ValueError, TypeError, AttributeError):
            return cls()

"""Turns one frame's hand observation into a flight intent.

Gesture vocabulary (selfie view, so "right" is right on screen):
  Open_Palm    steer: palm position is a virtual joystick (left/right = roll, up/down = throttle).
               Held still inside the center box while landed = takeoff (the governor times it).
  Thumb_Up     fly forward
  Victory      fly backward
  Pointing_Up  turn: finger left/right of center = yaw
  Closed_Fist  land
  anything else, or no hand: hover

With `hover_only` (the default "simple" mode) an open hand only takes off and a fist lands; nothing
else moves the drone. With `vertical_only` ("updown" mode) the open palm only climbs or descends.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from flight.types import ZERO, Axes
from gesture.tracker import HandObservation

PALM_POINTS = (0, 5, 9, 13, 17)  # wrist + the four knuckles
INDEX_TIP = 8


class IntentKind(str, Enum):
    NONE = "none"  # no hand in view
    HOVER = "hover"
    MOVE = "move"
    LAND = "land"


@dataclass(frozen=True)
class Intent:
    kind: IntentKind
    label: str  # stable name used for debouncing, e.g. "steer", "forward"
    axes: Axes = ZERO
    takeoff_ready: bool = False  # open palm held inside the center box


NO_HAND = Intent(IntentKind.NONE, "no hand")


@dataclass(frozen=True)
class MapperConfig:
    dead_zone: float = 0.12  # normalized offset from center that counts as "centered"
    span: float = 0.20  # offset beyond the dead zone where the stick saturates
    min_score: float = 0.55  # gesture confidence below this is treated as unrecognized
    takeoff_zone: float = 0.12  # half-size of the screen-center box where an open palm arms takeoff
    vertical_only: bool = False  # up/down only: no sideways, forward/back or turning
    hover_only: bool = False  # open hand = take off and hover, fist = land; no stick movement at all


def stick(offset: float, cfg: MapperConfig) -> float:
    """Dead-zoned, linear, saturating stick value for a normalized offset from center."""
    mag = abs(offset) - cfg.dead_zone
    if mag <= 0:
        return 0.0
    return (1.0 if offset > 0 else -1.0) * min(1.0, mag / cfg.span)


def palm_center(landmarks) -> tuple[float, float]:
    xs = [landmarks[i][0] for i in PALM_POINTS]
    ys = [landmarks[i][1] for i in PALM_POINTS]
    return sum(xs) / len(xs), sum(ys) / len(ys)


def map_observation(
    obs: HandObservation | None,
    cfg: MapperConfig = MapperConfig(),
    anchor: tuple[float, float] = (0.5, 0.5),
) -> Intent:
    """`anchor` is the neutral palm position: the screen center, or where the palm last rested (drag mode).
    Takeoff always needs the palm in the screen-center box, wherever the anchor is."""
    if obs is None:
        return NO_HAND

    gesture = obs.gesture if obs.score >= cfg.min_score else None
    cx, cy = palm_center(obs.landmarks)
    dx, dy = cx - anchor[0], anchor[1] - cy  # dy > 0 means the hand is above the anchor
    centered = abs(cx - 0.5) < cfg.takeoff_zone and abs(cy - 0.5) < cfg.takeoff_zone

    if cfg.hover_only:
        if gesture == "Open_Palm":
            return Intent(IntentKind.HOVER, "open hand", takeoff_ready=centered)
        if gesture == "Closed_Fist":
            return Intent(IntentKind.LAND, "land")
        return Intent(IntentKind.HOVER, "hover")
    if gesture == "Open_Palm":
        axes = Axes(throttle=stick(dy, cfg), roll=0.0 if cfg.vertical_only else stick(dx, cfg))
        kind = IntentKind.HOVER if axes.is_zero() else IntentKind.MOVE
        return Intent(kind, "up/down" if cfg.vertical_only else "steer", axes, takeoff_ready=centered)
    if gesture == "Closed_Fist":
        return Intent(IntentKind.LAND, "land")
    if cfg.vertical_only:
        return Intent(IntentKind.HOVER, "hover")
    if gesture == "Thumb_Up":
        return Intent(IntentKind.MOVE, "forward", Axes(pitch=1.0))
    if gesture == "Victory":
        return Intent(IntentKind.MOVE, "back", Axes(pitch=-1.0))
    if gesture == "Pointing_Up":
        tip_dx = obs.landmarks[INDEX_TIP][0] - 0.5
        axes = Axes(yaw=stick(tip_dx, cfg))
        return Intent(IntentKind.HOVER if axes.is_zero() else IntentKind.MOVE, "turn", axes)
    return Intent(IntentKind.HOVER, "hover")

"""OpenCV drawing for the pilot's view: hand skeleton, center box, stick bars, status."""
from __future__ import annotations

import cv2
import numpy as np

from flight.governor import Decision, FlightState
from gesture.mapper import MapperConfig
from gesture.tracker import HandObservation

HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
)

TRIM_KEYS = "trim a steady drift: [ ] left/right   - = back/forward"
LEGEND = (
    "OPEN PALM  steer (hold centered 1s = takeoff)",
    "THUMB UP   forward     VICTORY  back",
    "POINT UP   turn        FIST     land",
    TRIM_KEYS,
    "keys: SPACE e-stop  L land  T takeoff  C calibrate  Q quit",
)
DRAG_LEGEND = ("OPEN PALM  drag (palm still = drone still, hand gone = land)",) + LEGEND[1:]
UPDOWN_LEGEND = (
    "OPEN PALM  hold centered 1s = takeoff",
    "palm high = UP    middle = HOLD    low = DOWN",
    "FIST or hand gone = land",
    TRIM_KEYS,
    LEGEND[-1],
)
SIMPLE_LEGEND = (
    "OPEN HAND, hold 1s = take off and hover",
    "FIST = land, then motors off",
    TRIM_KEYS,
    "keys: SPACE motors off now   C calibrate (landed)   Q land + quit",
)
LEGENDS = {"simple": SIMPLE_LEGEND, "updown": UPDOWN_LEGEND, "joystick": LEGEND, "drag": DRAG_LEGEND}

WHITE, GREEN, AMBER, RED, GRAY = (235, 235, 235), (80, 220, 80), (40, 190, 250), (60, 60, 230), (140, 140, 140)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _text(img, s, org, scale=0.5, color=WHITE, thick=1):
    cv2.putText(img, s, org, FONT, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, s, org, FONT, scale, color, thick, cv2.LINE_AA)


def draw_hand(img: np.ndarray, obs: HandObservation | None) -> None:
    if obs is None:
        return
    h, w = img.shape[:2]
    pts = [(int(x * w), int(y * h)) for x, y, _ in obs.landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(img, pts[a], pts[b], GREEN, 2, cv2.LINE_AA)
    for p in pts:
        cv2.circle(img, p, 3, WHITE, -1, cv2.LINE_AA)


def draw_center_box(
    img: np.ndarray, cfg: MapperConfig, decision: Decision, center: tuple[float, float] | None = (0.5, 0.5)
) -> None:
    """Landed: the takeoff box at screen center. Flying: the hover box (white) and full-stick box (gray)
    around `center`, which is the screen center or the drag-mode anchor (None draws nothing), or in
    up/down mode the UP / HOLD / DOWN bands."""
    h, w = img.shape[:2]

    def box(cx: float, cy: float, half: float, color) -> None:
        cv2.rectangle(img, (int((cx - half) * w), int((cy - half) * h)), (int((cx + half) * w), int((cy + half) * h)), color, 1)

    if decision.state is FlightState.LANDED:
        if cfg.takeoff_zone < 0.5:  # 0.5 means anywhere on screen, so there's no box to show
            box(0.5, 0.5, cfg.takeoff_zone, WHITE)
        if decision.takeoff_progress > 0:
            r = int(min(cfg.takeoff_zone, 0.12) * min(w, h)) + 12
            cv2.ellipse(img, (w // 2, h // 2), (r, r), -90, 0, 360 * decision.takeoff_progress, AMBER, 4, cv2.LINE_AA)
            _text(img, "TAKEOFF", (w // 2 - 38, h // 2 - r - 8), 0.6, AMBER, 2)
    elif cfg.hover_only:
        pass  # nothing to steer with
    elif decision.state is FlightState.FLYING and cfg.vertical_only:
        top, bottom = int((0.5 - cfg.dead_zone) * h), int((0.5 + cfg.dead_zone) * h)
        cv2.line(img, (0, top), (w, top), WHITE, 1)
        cv2.line(img, (0, bottom), (w, bottom), WHITE, 1)
        _text(img, "UP", (w // 2 - 16, top - 14), 0.9, WHITE, 2)
        _text(img, "HOLD", (w // 2 - 32, h // 2 + 10), 0.9, GRAY, 2)
        _text(img, "DOWN", (w // 2 - 36, bottom + 36), 0.9, WHITE, 2)
    elif decision.state is FlightState.FLYING and center is not None:
        box(*center, cfg.dead_zone, WHITE)
        box(*center, cfg.dead_zone + cfg.span, GRAY)


def draw_sticks(img: np.ndarray, decision: Decision, authority: float, origin: tuple[int, int]) -> None:
    x0, y0 = origin
    for i, (name, v) in enumerate(zip(("THR", "YAW", "PIT", "ROL"), decision.axes.as_tuple())):
        y = y0 + i * 20
        _text(img, name, (x0, y + 10), 0.45)
        left, mid, right = x0 + 40, x0 + 110, x0 + 180
        cv2.rectangle(img, (left, y), (right, y + 12), GRAY, 1)
        cv2.line(img, (mid, y - 2), (mid, y + 14), WHITE, 1)
        end = mid + int(max(-1.0, min(1.0, v / max(authority, 1e-6))) * (right - mid))
        cv2.rectangle(img, (min(mid, end), y + 2), (max(mid, end), y + 10), GREEN if v else GRAY, -1)


def draw_status(img: np.ndarray, decision: Decision, driver_status: str, fps: float) -> None:
    color = {FlightState.FLYING: GREEN, FlightState.LANDING: AMBER}.get(decision.state, WHITE)
    state = decision.state.value.upper()
    if decision.state is FlightState.LANDING:
        state += "  (motors off after)"
    elif decision.state is FlightState.FLYING:
        # "holding" means every stick is centered, so any movement then is the drone's own drift.
        state += "  holding" if decision.axes.is_zero() else "  moving"
    _text(img, state, (12, 28), 0.8, color, 2)
    _text(img, f"gesture: {decision.label}", (12, 54), 0.55)
    _text(img, driver_status, (12, 78), 0.5, AMBER if "no video" in driver_status or "WATCHDOG" in driver_status else WHITE)
    _text(img, f"{fps:4.1f} fps", (img.shape[1] - 90, 24), 0.5, GRAY)
    if decision.auto_land_in is not None:
        _text(img, f"no hand: landing in {decision.auto_land_in:.1f}s", (12, 108), 0.8, AMBER, 2)


def draw_warning(img: np.ndarray, lines: tuple[str, ...]) -> None:
    for i, line in enumerate(lines):
        _text(img, line, (12, 130 + i * 30), 0.8, AMBER, 2)


def draw_trim(img: np.ndarray, label: str) -> None:
    _text(img, label, (img.shape[1] - 250, 48), 0.5, AMBER)


def draw_note(img: np.ndarray, text: str) -> None:
    """A message (follow on/off, and why not) across the middle of the webcam view."""
    h, w = img.shape[:2]
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    x, y = max(8, (w - tw) // 2), int(h * 0.62)
    cv2.rectangle(img, (x - 10, y - th - 10), (min(w, x + tw + 10), y + 10), (20, 20, 20), -1)
    _text(img, text, (x, y), 0.7, AMBER, 2)


def draw_legend(img: np.ndarray, lines: tuple[str, ...] = LEGEND) -> None:
    h = img.shape[0]
    for i, line in enumerate(reversed(lines)):
        _text(img, line, (12, h - 12 - i * 18), 0.45, WHITE if i else GRAY)


def draw_corner(
    img: np.ndarray, rect: tuple[float, float, float, float], progress: float, active: bool, label: str = "AUTO"
) -> None:
    """A corner box (AUTO = autopilot path, FOLLOW = follow mode): hold an open palm in it for 1 s to toggle."""
    h, w = img.shape[:2]
    x0, y0, x1, y1 = int(rect[0] * w), int(rect[1] * h), int(rect[2] * w) - 2, int(rect[3] * h)
    cv2.rectangle(img, (x0, y0), (x1, y1), AMBER if active else GRAY, 2)
    _text(img, label + (" ON" if active else ""), (x0 + 12, y0 + 34), 0.8, AMBER if active else WHITE, 2)
    if progress > 0:
        cv2.rectangle(img, (x0, y1 - 12), (x0 + int((x1 - x0) * progress), y1), AMBER, -1)


def compose(
    left: np.ndarray, right: np.ndarray | None, height: int = 480, extra: np.ndarray | None = None
) -> np.ndarray:
    def fit(img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        return cv2.resize(img, (int(w * height / h), height))

    left = fit(left)
    if right is None:
        right = np.full((height, height, 3), 28, np.uint8)
        _text(right, "no drone video", (height // 2 - 70, height // 2), 0.6, GRAY)
    panels = [left, fit(right)] + ([fit(extra)] if extra is not None else [])
    return np.hstack(panels)

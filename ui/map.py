"""Top-down map for position hold: barrier, autopilot path, target, home and the tracked drone."""
from __future__ import annotations

import math

import cv2
import numpy as np

from nav.navigator import Navigator, NavState

PAD_M = 0.5
STATE_COLORS = {
    NavState.OFF: (170, 170, 170),
    NavState.HOLD: (80, 220, 80),
    NavState.MANUAL: (235, 235, 235),
    NavState.AUTO: (40, 190, 250),
    NavState.NO_FIX: (60, 60, 230),
}


def render_map(
    nav: Navigator,
    size: int = 480,
    camera_frame: np.ndarray | None = None,
    note: str = "",
    people: tuple[tuple[float, float], ...] = (),
) -> np.ndarray:
    img = np.full((size, size, 3), 24, np.uint8)
    f = nav.fence
    x0, y0, x1, y1 = f.xmin - PAD_M, f.ymin - PAD_M, f.xmax + PAD_M, f.ymax + PAD_M
    scale = (size - 50) / max(x1 - x0, y1 - y0)
    ox = (size - (x1 - x0) * scale) / 2
    oy = 40 + (size - 50 - (y1 - y0) * scale) / 2

    def px(p: tuple[float, float]) -> tuple[int, int]:
        return int(ox + (p[0] - x0) * scale), int(oy + (y1 - p[1]) * scale)

    for gx in range(math.ceil(x0), math.floor(x1) + 1):  # 1 m grid
        cv2.line(img, px((gx, y0)), px((gx, y1)), (40, 40, 40), 1)
    for gy in range(math.ceil(y0), math.floor(y1) + 1):
        cv2.line(img, px((x0, gy)), px((x1, gy)), (40, 40, 40), 1)

    cv2.rectangle(img, px((f.xmin, f.ymax)), px((f.xmax, f.ymin)), (60, 60, 230), 2)  # the barrier
    ix0, iy0, ix1, iy1 = f.inner
    cv2.rectangle(img, px((ix0, iy1)), px((ix1, iy0)), (50, 50, 120), 1)  # where the push-back starts

    path = nav.mission.points if nav.mission is not None else [f.clamp(p) for p in nav.path]
    if len(path) > 1:
        cv2.polylines(img, [np.int32([px(p) for p in path]).reshape(-1, 1, 2)], False, (150, 110, 60), 1, cv2.LINE_AA)

    trail = list(nav.trail)
    if len(trail) > 1:
        cv2.polylines(img, [np.int32([px(p) for p in trail]).reshape(-1, 1, 2)], False, (90, 140, 90), 1, cv2.LINE_AA)

    if nav.home is not None:
        cv2.circle(img, px(nav.home), 9, (80, 220, 80), 1, cv2.LINE_AA)
        cv2.putText(img, "H", (px(nav.home)[0] - 5, px(nav.home)[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 220, 80), 1)
    if nav.target is not None and nav.state is not NavState.OFF:
        tx, ty = px(nav.target)
        color = (40, 190, 250) if nav.state is NavState.AUTO else (235, 235, 235)
        cv2.drawMarker(img, (tx, ty), color, cv2.MARKER_CROSS, 16, 2)

    for p in people:  # simulated people being followed
        cv2.circle(img, px(p), 7, (40, 140, 250), -1, cv2.LINE_AA)

    if nav.pose is not None:
        p, hdg = px((nav.pose.x, nav.pose.y)), nav.pose.heading
        r = 11
        tri = [
            (p[0] + int(r * math.sin(hdg + a)), p[1] - int(r * math.cos(hdg + a)))
            for a in (0.0, 2.5, -2.5)
        ]
        color = STATE_COLORS[NavState.NO_FIX] if nav.state is NavState.NO_FIX else STATE_COLORS[nav.state]
        cv2.fillPoly(img, [np.array(tri)], color)

    label = f"NAV {nav.state.value.upper()}"
    cv2.putText(img, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, STATE_COLORS[nav.state], 2)
    if note:
        cv2.putText(img, note[:48], (10, size - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1)

    if camera_frame is not None:  # the tracking camera, picture-in-picture
        h, w = camera_frame.shape[:2]
        pw = size // 3
        ph = int(h * pw / w)
        img[44 : 44 + ph, size - pw - 6 : size - 6] = cv2.resize(camera_frame, (pw, ph))
    return img

"""Draw the map: turbo height colors, the single-file three.js viewer (map.html), and a fly-around (map.mp4)."""
from __future__ import annotations

import base64
import html
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from mapping.geometry import focal_px, look_at

VIEWER = Path(__file__).with_name("viewer.html")
DRONE_BGR = (255, 255, 255)
PERSON_BGR = (26, 140, 255)  # orange
BACKGROUND_BGR = (11, 7, 5)


@dataclass
class MapData:
    """Everything in map frame: metres, z up, ground near z = 0."""

    points: np.ndarray  # (N, 3)
    colors: np.ndarray  # (N, 3) uint8 RGB
    drone: np.ndarray  # (M, 3) path in time order
    person: np.ndarray  # (K, 3) path in time order
    zrange: tuple[float, float] = (0.0, 1.0)  # heights at the two ends of the color bar
    info: list[tuple[str, str]] = field(default_factory=list)  # shown in the viewer's panel
    drone_t: np.ndarray = field(default_factory=lambda: np.zeros(0))  # seconds into the recording, per drone point
    person_t: np.ndarray = field(default_factory=lambda: np.zeros(0))  # same, per person point


TURBO_SPAN = (0.1, 0.92)  # turbo's very ends are near-black: the ground would vanish into the background


def turbo(values: np.ndarray) -> np.ndarray:
    """0..1 -> (N, 3) uint8 RGB, blue to red, on the visible part of OpenCV's turbo colormap."""
    lo, hi = TURBO_SPAN
    v = lo + np.clip(np.asarray(values, float), 0, 1) * (hi - lo)
    bgr = cv2.applyColorMap(np.round(v * 255).astype(np.uint8).reshape(-1, 1), cv2.COLORMAP_TURBO)
    return bgr.reshape(-1, 3)[:, ::-1].copy()


def height_colors(z: np.ndarray, lo_pct: float = 2, hi_pct: float = 98) -> tuple[np.ndarray, tuple[float, float]]:
    """Color by height like a LiDAR map: low = blue, high = red. Percentiles keep a few stray points
    from squashing everything else into one color. Returns (colors, (z at blue, z at red))."""
    z = np.asarray(z, float)
    if len(z) == 0:
        return np.zeros((0, 3), np.uint8), (0.0, 1.0)
    lo, hi = (float(x) for x in np.percentile(z, [lo_pct, hi_pct]))
    if hi - lo < 1e-6:
        hi = lo + 1.0
    return turbo((z - lo) / (hi - lo)), (lo, hi)


def _b64(a: np.ndarray, dtype) -> str:
    return base64.b64encode(np.ascontiguousarray(a, dtype=dtype).tobytes()).decode("ascii")


def write_html(path: Path, data: MapData, title: str) -> None:
    center, radius = _extent(data)
    payload = {
        "view": {"center": np.round(center, 3).tolist(), "radius": round(radius, 3)},
        "title": title,
        "points": _b64(data.points, "<f4"),
        "colors": _b64(data.colors, np.uint8),
        "drone": np.round(data.drone, 3).tolist(),
        "person": np.round(data.person, 3).tolist(),
        "zrange": [round(data.zrange[0], 3), round(data.zrange[1], 3)],
        "turbo": ["#%02x%02x%02x" % tuple(c) for c in turbo(np.linspace(0, 1, 9))],
        "info": [[str(k), str(v)] for k, v in data.info],
    }
    # Inside <script type="application/json"> only "</script" could break out; escaping < > & covers it.
    blob = json.dumps(payload, separators=(",", ":")).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    page = VIEWER.read_text().replace("__TITLE__", html.escape(title)).replace("/*MAP_DATA*/", blob)
    Path(path).write_text(page)


def _project(X: np.ndarray, eye: np.ndarray, R: np.ndarray, f: float, w: int, h: int):
    Xc = (X - eye) @ R
    z = Xc[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = f * Xc[:, 0] / z + w / 2
        v = f * Xc[:, 1] / z + h / 2
    return u, v, z


def _polyline(img, pts, eye, R, f, color, thickness, near=0.1):
    if len(pts) < 2:
        return
    u, v, z = _project(pts, eye, R, f, img.shape[1], img.shape[0])
    for i in range(len(pts) - 1):
        if z[i] > near and z[i + 1] > near:
            a = (int(round(u[i])), int(round(v[i])))
            b = (int(round(u[i + 1])), int(round(v[i + 1])))
            cv2.line(img, a, b, color, thickness, cv2.LINE_AA)


def _splat(img, pts, colors_bgr, eye, R, f):
    """Nearest point wins each pixel (a real z-buffer, not painter's order), 2x2 px per point."""
    h, w = img.shape[:2]
    u, v, z = _project(pts, eye, R, f, w, h)
    ok = z > 0.1
    ui, vi = np.floor(u[ok]).astype(int), np.floor(v[ok]).astype(int)
    uu = np.concatenate([ui, ui + 1, ui, ui + 1])
    vv = np.concatenate([vi, vi, vi + 1, vi + 1])
    zs, cs = np.tile(z[ok], 4), np.tile(colors_bgr[ok], (4, 1))
    on = (uu >= 0) & (uu < w) & (vv >= 0) & (vv < h)
    pix, zs, cs = vv[on] * w + uu[on], zs[on], cs[on]
    order = np.lexsort((zs, pix))  # by pixel, nearest first
    first = np.ones(len(order), bool)
    first[1:] = pix[order][1:] != pix[order][:-1]
    keep = order[first]
    img.reshape(-1, 3)[pix[keep]] = cs[keep]


def _extent(data: MapData) -> tuple[np.ndarray, float]:
    parts = [p for p in (data.points, data.drone, data.person) if len(p)]
    allp = np.vstack(parts) if parts else np.zeros((1, 3))
    lo, hi = np.percentile(allp, 5, axis=0), np.percentile(allp, 95, axis=0)
    if len(data.drone):
        lo, hi = np.minimum(lo, data.drone.min(axis=0)), np.maximum(hi, data.drone.max(axis=0))
    if len(data.person):
        lo, hi = np.minimum(lo, data.person.min(axis=0)), np.maximum(hi, data.person.max(axis=0))
    center = (lo + hi) / 2
    radius = max(1.0, float(np.linalg.norm((hi - lo)[:2]) / 2), float(hi[2] - lo[2]) / 2)
    return center, radius


def write_flyaround(path: Path, data: MapData, title: str, size=(960, 540), seconds: float = 10.0, fps: int = 30) -> int:
    """One slow orbit around the map, drawn with OpenCV. Returns the number of frames written."""
    W, H = size
    f = focal_px(W)
    center, radius = _extent(data)
    dist, elev = 1.15 * radius / math.tan(math.radians(30)), math.radians(32)
    step = 1.0 if radius < 15 else 5.0
    span = math.ceil(radius * 1.4 / step) * step
    gx = np.arange(-span, span + step / 2, step)
    ts = np.linspace(-span, span, 40)
    grid = [np.column_stack([np.full_like(ts, center[0] + x), center[1] + ts, np.zeros_like(ts)]) for x in gx]
    grid += [np.column_stack([center[0] + ts, np.full_like(ts, center[1] + y), np.zeros_like(ts)]) for y in gx]
    colors_bgr = np.ascontiguousarray(data.colors[:, ::-1]) if len(data.colors) else np.zeros((0, 3), np.uint8)
    bar = turbo(np.linspace(0, 1, 160))[:, ::-1]

    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    n = max(1, int(round(seconds * fps)))
    for i in range(n):
        az = 2 * math.pi * i / n - math.pi / 2  # start behind the takeoff point, looking along +y
        eye = center + dist * np.array([math.cos(elev) * math.cos(az), math.cos(elev) * math.sin(az), math.sin(elev)])
        R = look_at(eye, center)
        img = np.zeros((H, W, 3), np.uint8)
        img[:] = BACKGROUND_BGR
        for line in grid:
            _polyline(img, line, eye, R, f, (48, 40, 36), 1)
        if len(data.points):
            _splat(img, data.points, colors_bgr, eye, R, f)
        _polyline(img, data.drone, eye, R, f, DRONE_BGR, 2)
        _polyline(img, data.person, eye, R, f, PERSON_BGR, 2)

        cv2.putText(img, title, (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 230, 230), 1, cv2.LINE_AA)
        cv2.putText(img, "drone", (16, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.5, DRONE_BGR, 1, cv2.LINE_AA)
        cv2.putText(img, "person", (80, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.5, PERSON_BGR, 1, cv2.LINE_AA)
        if len(data.points):
            img[H - 34:H - 24, 16:16 + len(bar)] = bar[None, :, :]
            lo, hi = data.zrange
            cv2.putText(img, f"{lo:.1f} m", (16, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.putText(img, f"{hi:.1f} m", (16 + len(bar) - 40, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (200, 200, 200), 1, cv2.LINE_AA)
        out.write(img)
    out.release()
    return n

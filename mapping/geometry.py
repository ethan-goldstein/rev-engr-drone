"""Camera geometry for the map: pinhole rays, which way is up, metric scale, and the person path.

Cameras use the COLMAP/OpenCV convention: x right, y down, z forward. A camera is (C, R) with C its
center in world coordinates and R = world_from_cam rotation (columns are the camera axes in world).
The finished map is metres, z up, the first camera at x = y = 0 facing +y (the drone's heading 0),
and the ground (low points) at z = 0.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

HFOV_DEG = 60.0  # printed on the F19's camera
PERSON_CENTER_M = 0.85  # half the 1.7 m person the follow controller assumes


def focal_px(width: float, hfov_deg: float = HFOV_DEG) -> float:
    return (width / 2) / math.tan(math.radians(hfov_deg / 2))


def box_rays(cx: np.ndarray, cy: np.ndarray, f: float, w: float, h: float) -> np.ndarray:
    """Camera-frame rays through normalized box centers, scaled to unit depth (z = 1).

    Unit depth rather than unit length because distance_m is pinhole depth (f * height / box height).
    """
    cx, cy = np.asarray(cx, float), np.asarray(cy, float)
    return np.stack([(cx * w - w / 2) / f, (cy * h - h / 2) / f, np.ones_like(cx)], axis=-1)


def person_positions(C: np.ndarray, R: np.ndarray, cx, cy, dist, f: float, w: float, h: float) -> np.ndarray:
    """The person at each frame: along the box-center ray, `dist` deep. C (K,3), R (K,3,3) -> (K,3)."""
    rays = box_rays(cx, cy, f, w, h) * np.asarray(dist, float)[:, None]
    return C + np.einsum("kij,kj->ki", R, rays)


def look_at(eye, target, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """world_from_cam rotation for a camera at `eye` looking at `target`, image top toward `up`."""
    fwd = np.asarray(target, float) - np.asarray(eye, float)
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, up)
    right /= np.linalg.norm(right)
    return np.column_stack([right, np.cross(fwd, right), fwd])


def camera_from_heading(heading_rad: float) -> np.ndarray:
    """world_from_cam for a level camera facing `heading` (0 = +y, clockwise positive, as in nav.pose)."""
    s, c = math.sin(heading_rad), math.cos(heading_rad)
    return np.array([[c, 0.0, s], [-s, 0.0, c], [0.0, -1.0, 0.0]])


def estimate_up(C: np.ndarray, R: np.ndarray, points: np.ndarray | None = None,
                max_tilt_deg: float = 35.0) -> tuple[np.ndarray, str]:
    """World up direction in the reconstruction's (arbitrary) frame, and which cue gave it.

    Best is the ground: the biggest flat patch of points below the drone, roughly level. Without one,
    the drone flies roughly level, so when its path spreads out in two directions that plane's normal
    is up (but a slow altitude drift tilts it, which is why it comes second). A straight line, a hover
    or a takeoff climb pins nothing down; last resort is the cameras' own up (-y), which is off by
    however much the camera looks down.
    """
    cam_up = -R[:, :, 1].mean(axis=0)
    cam_up /= np.linalg.norm(cam_up)
    cos_max = math.cos(math.radians(max_tilt_deg))
    if points is not None:
        ground = fit_ground(points, C, cam_up, cos_max)
        if ground is not None:
            return plane_normal(ground, cam_up), "ground plane"
    if len(C) >= 4:
        _, s, vt = np.linalg.svd(C - C.mean(axis=0), full_matrices=False)
        n = vt[2] if cos_between(vt[2], cam_up) >= 0 else -vt[2]
        flat = s[1] > 1e-9 and s[1] > 3 * s[2]
        if flat and cos_between(n, cam_up) > cos_max:
            return n, "flight path plane"
    return cam_up, "camera tilt"


def fit_ground(points: np.ndarray, C: np.ndarray, near_up: np.ndarray, cos_max: float,
               iters: int = 400, min_share: float = 0.1) -> np.ndarray | None:
    """Points on the plane holding the most points, among planes within acos(cos_max) of near_up with
    every camera above them (RANSAC). None if the best one holds under min_share of the points."""
    if len(points) < 50:
        return None
    rng = np.random.default_rng(0)  # deterministic: the same flight always gives the same map
    pts = points if len(points) <= 20000 else points[rng.choice(len(points), 20000, replace=False)]
    tol = 0.01 * float(np.median(np.linalg.norm(pts - C.mean(axis=0), axis=1)))  # ~5 cm at 5 m; units are arbitrary
    best, best_count = None, 0
    for a, b, c in rng.integers(0, len(pts), (iters, 3)):
        n = np.cross(pts[b] - pts[a], pts[c] - pts[a])
        if np.linalg.norm(n) < 1e-12:
            continue
        n /= np.linalg.norm(n)
        n = n if n @ near_up >= 0 else -n
        if n @ near_up < cos_max or ((C - pts[a]) @ n <= 0).any():
            continue
        inliers = np.abs((pts - pts[a]) @ n) < tol
        if inliers.sum() > best_count:
            best, best_count = inliers, int(inliers.sum())
    if best is None or best_count < min_share * len(pts):
        return None
    return pts[best]


def plane_normal(on: np.ndarray, near_up: np.ndarray) -> np.ndarray:
    """Least-squares normal of points on a plane, pointing the same way as near_up."""
    n = np.linalg.svd(on - on.mean(axis=0), full_matrices=False)[2][2]
    return n if n @ near_up >= 0 else -n


def ground_height(points: np.ndarray, C: np.ndarray, up: np.ndarray) -> tuple[float, bool]:
    """Height of the ground along `up` (reconstruction units), and whether a real ground plane was found.
    Without one, the 2nd percentile of point heights stands in (a few bad points below the floor don't count)."""
    ground = fit_ground(points, C, up, math.cos(math.radians(10)))
    if ground is not None:
        return float(np.median(ground @ up)), True
    heights = points @ up if len(points) else C @ up
    return float(np.percentile(heights, 2)), False


def cos_between(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def scale_from_ground(C, R, boxes, dist, f, w, h, up, ground, center_m=PERSON_CENTER_M) -> tuple[float | None, int]:
    """Metres per reconstruction unit: the scale at which the person, put distance_m along the
    box-center ray, stands with their middle center_m above the reconstructed ground.

    Works whether or not COLMAP kept the person (it drops anyone who moves). Per frame, the camera's
    height above the ground in metres is center_m minus the ray's drop over distance_m; in
    reconstruction units it is the camera's height above `ground`. Returns (scale or None, frames used).
    """
    rays = np.einsum("kij,kj->ki", R, box_rays(boxes[:, 0], boxes[:, 1], f, w, h))
    metres = center_m - dist * (rays @ up)
    units = C @ up - ground
    ok = (units > 0) & (metres > 0.3)  # a drone skimming the ground can't tell scale this way
    if not ok.any():
        return None, 0
    return float(np.median(metres[ok] / units[ok])), int(ok.sum())


def scale_from_depth(points, C, R, boxes, dist, f, w, h, min_points: int = 5) -> tuple[float | None, int]:
    """Metres per reconstruction unit, from how far the person is (distance_m) vs. how deep the
    reconstruction is around the box-center ray. Returns (scale or None, frames used).

    Only right when COLMAP reconstructed the person, i.e. they held still: otherwise the points in their
    box are the background behind them and the map comes out too small. Hence the fallback to
    scale_from_ground. Per frame: points in the middle half of the box, near quartile of their depth.
    """
    depths, used = [], []
    for k in range(len(C)):
        cx, cy, bw, bh = boxes[k]
        Xc = (points - C[k]) @ R[k]
        z = Xc[:, 2]
        front = z > 1e-6
        u = f * Xc[front, 0] / z[front] + w / 2
        v = f * Xc[front, 1] / z[front] + h / 2
        inside = (np.abs(u - cx * w) <= bw * w / 4) & (np.abs(v - cy * h) <= bh * h / 4)
        if inside.sum() >= min_points:
            depths.append(np.percentile(z[front][inside], 25))
            used.append(dist[k])
    if not depths:
        return None, 0
    return float(np.median(used) / np.median(depths)), len(depths)


def rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Smallest rotation taking direction a onto direction b."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v, c = np.cross(a, b), float(np.dot(a, b))
    if c < -1 + 1e-9:  # opposite: half turn about any axis perpendicular to a
        axis = np.cross(a, [1.0, 0.0, 0.0] if abs(a[0]) < 0.9 else [0.0, 1.0, 0.0])
        axis /= np.linalg.norm(axis)
        return 2 * np.outer(axis, axis) - np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1 + c)


@dataclass(frozen=True)
class Similarity:
    """x -> s * R x + t"""

    s: float
    R: np.ndarray
    t: np.ndarray

    def points(self, X: np.ndarray) -> np.ndarray:
        return self.s * np.asarray(X, float).reshape(-1, 3) @ self.R.T + self.t

    def rotations(self, R: np.ndarray) -> np.ndarray:
        return np.einsum("ij,kjl->kil", self.R, R)


def align(C: np.ndarray, R: np.ndarray, up: np.ndarray, scale: float, ground: float) -> Similarity:
    """Reconstruction frame -> map frame: z up, metres, first camera at the origin facing +y, ground
    (its height along `up`, in reconstruction units) at z = 0."""
    Rup = rotation_between(up, np.array([0.0, 0.0, 1.0]))
    fwd = Rup @ R[0][:, 2]
    if math.hypot(fwd[0], fwd[1]) > 1e-6:
        a = math.atan2(fwd[0], fwd[1])  # its heading; turning the world by that much points it at +y
        Rz = np.array([[math.cos(a), -math.sin(a), 0.0], [math.sin(a), math.cos(a), 0.0], [0.0, 0.0, 1.0]])
        Rup = Rz @ Rup
    c0 = scale * Rup @ C[0]
    return Similarity(scale, Rup, np.array([-c0[0], -c0[1], -scale * ground]))

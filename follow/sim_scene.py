"""Practice and test following without the drone: people walking in the sim, seen through a virtual F19 camera.

The camera matches the real one: 640x360 video (turbodrone's research notes) and a 60 degree
field of view (printed on the drone's camera).
"""
from __future__ import annotations

import math
import random

import cv2
import numpy as np

from follow.types import Detection

FRAME_W, FRAME_H = 640, 360
HFOV_DEG = 60.0
PATHS = {
    # Start 4 m in front of the drone's takeoff spot (the drone faces +y), then walk a loop.
    "square": [(0.0, 4.0), (2.5, 4.0), (2.5, -1.5), (-2.5, -1.5), (-2.5, 4.0)],
    "line": [(0.0, 4.0), (3.0, 4.0), (-3.0, 4.0)],
}


class SimPerson:
    def __init__(
        self,
        path: str = "square",
        speed: float = 0.4,
        pause_s: float = 2.0,
        shirt: tuple[int, int, int] = (40, 60, 200),  # BGR
        start: tuple[float, float] | None = None,
        rng: random.Random | None = None,
    ):
        self._rng = rng or random.Random()
        self.path = path
        self.waypoints = list(PATHS[path]) if path in PATHS else []
        self.x, self.y = start or (self.waypoints[0] if self.waypoints else (0.0, 4.0))
        self.speed = speed
        self.pause_s = pause_s  # stand still at each corner, like a person looking around
        self.shirt = shirt
        self.height, self.width = 1.7, 0.45
        self.visible = True
        self.arms_up = False  # U key in the sim: the "land" signal
        self.facing_drone = True  # SimScene clears it while they walk away from the drone
        self.walk_dir: tuple[float, float] | None = None  # unit vector while walking, None while standing
        self._i = 1 % max(1, len(self.waypoints))
        self._goal = self._next_goal()
        self._pause_until = 0.0
        self._last: float | None = None

    def _next_goal(self) -> tuple[float, float]:
        if self.waypoints:
            return self.waypoints[self._i]
        return self._rng.uniform(-3.0, 3.0), self._rng.uniform(-1.5, 4.5)

    def update(self, now: float) -> None:
        dt = 0.0 if self._last is None else max(0.0, now - self._last)
        self._last = now
        self.walk_dir = None
        if now < self._pause_until or self.speed <= 0:
            return
        gx, gy = self._goal
        dist = math.hypot(gx - self.x, gy - self.y)
        if dist > 1e-9:
            self.walk_dir = ((gx - self.x) / dist, (gy - self.y) / dist)
        step = self.speed * dt
        if step >= dist:
            self.x, self.y = gx, gy
            self._pause_until = now + self.pause_s
            if self.waypoints:
                self._i = (self._i + 1) % len(self.waypoints)
            self._goal = self._next_goal()
        else:
            self.x += (gx - self.x) * step / dist
            self.y += (gy - self.y) * step / dist


class SimCamera:
    """Pinhole camera on the drone, level (keep the real camera tilted level for following)."""

    def __init__(self, width: int = FRAME_W, height: int = FRAME_H, hfov_deg: float = HFOV_DEG, grid: bool = True):
        self.width, self.height = width, height
        self.grid = grid  # the ground grid shows motion on screen; tests skip it for speed
        self.f = (width / 2) / math.tan(math.radians(hfov_deg / 2))

    def project(self, pose, x: float, y: float, z: float) -> tuple[float, float, float] | None:
        """World point -> (u, v, depth) for a drone at pose (x, y, z, heading). None if behind the camera."""
        px, py, pz, hdg = pose
        dx, dy = x - px, y - py
        depth = dx * math.sin(hdg) + dy * math.cos(hdg)
        right = dx * math.cos(hdg) - dy * math.sin(hdg)
        if depth < 0.2:
            return None
        return self.width / 2 + self.f * right / depth, self.height / 2 - self.f * (z - pz) / depth, depth

    def extent(self, pose, person: SimPerson) -> tuple[float, float, float, float] | None:
        """The person's full outline in pixels (may run past the frame edges), or None if behind the camera."""
        px, py, _, hdg = pose
        if (person.x - px) * math.sin(hdg) + (person.y - py) * math.cos(hdg) < 0.5:
            return None
        rx, ry = math.cos(hdg) * person.width / 2, -math.sin(hdg) * person.width / 2  # camera's right axis
        pts = [self.project(pose, person.x + s * rx, person.y + s * ry, z) for s in (-1, 1) for z in (0.0, person.height)]
        if any(p is None for p in pts):
            return None
        return min(p[0] for p in pts), min(p[1] for p in pts), max(p[0] for p in pts), max(p[1] for p in pts)

    def box(self, pose, person: SimPerson) -> Detection | None:
        """What a detector would report: the outline clipped to the frame."""
        full = self.extent(pose, person)
        if full is None:
            return None
        x0, y0 = max(0.0, full[0]), max(0.0, full[1])
        x1, y1 = min(float(self.width), full[2]), min(float(self.height), full[3])
        if x1 - x0 < 3 or y1 - y0 < 6:
            return None  # out of frame
        return Detection(x0, y0, x1, y1)

    def render(self, pose, people: list[SimPerson]) -> np.ndarray:
        img = np.empty((self.height, self.width, 3), np.uint8)
        horizon = int(self.height / 2)
        img[:horizon] = (205, 170, 120)  # sky
        img[horizon:] = (70, 110, 70)  # grass
        px, py, pz, hdg = pose
        for i in range(-12, 13) if self.grid else ():  # 1 m grid on the ground, so turning and moving are visible
            for a, b in (((i, -12), (i, 12)), ((-12, i), (12, i))):
                seg = []
                for k in range(25):
                    t = k / 24
                    p = self.project(pose, a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]), 0.0)
                    if p is not None and p[2] < 14:
                        seg.append((int(p[0]), int(p[1])))
                    elif len(seg) > 1:
                        cv2.polylines(img, [np.int32(seg)], False, (85, 130, 85), 1)
                        seg = []
                if len(seg) > 1:
                    cv2.polylines(img, [np.int32(seg)], False, (85, 130, 85), 1)
        visible = [p for p in people if p.visible]
        for person in sorted(visible, key=lambda p: -math.hypot(p.x - px, p.y - py)):  # far ones first
            full = self.extent(pose, person)
            if full is None or self.box(pose, person) is None:
                continue
            x0, y0, x1, y1 = full
            hip, shoulder = y1 - 0.48 * (y1 - y0), y1 - 0.82 * (y1 - y0)
            cv2.rectangle(img, (int(x0), int(hip)), (int(x1), int(y1)), (60, 50, 45), -1)  # trousers
            cv2.rectangle(img, (int(x0), int(shoulder)), (int(x1), int(hip)), person.shirt, -1)
            r = max(2, int((x1 - x0) * 0.32))
            cv2.circle(img, (int((x0 + x1) / 2), int(shoulder - r)), r, (140, 170, 210), -1)  # head
            if person.arms_up:
                arm = max(2, int((x1 - x0) * 0.2))
                for x in (x0, x1):
                    cv2.line(img, (int(x), int(shoulder)), (int(x), int(shoulder - 3 * r)), person.shirt, arm)
        return img


class SimScene:
    """The people, the camera, and the sim drone's pose: renders what the drone would see."""

    def __init__(self, sim, people: list[SimPerson], camera: SimCamera | None = None):
        self.sim = sim
        self.people = people
        self.camera = camera or SimCamera()

    def pose(self) -> tuple[float, float, float, float]:
        return self.sim.x, self.sim.y, self.sim.z, self.sim.heading

    def update(self, now: float) -> None:
        for p in self.people:
            p.update(now)
            if p.walk_dir is None:
                p.facing_drone = True  # standing: they look at the drone
            else:
                tx, ty = self.sim.x - p.x, self.sim.y - p.y
                p.facing_drone = p.walk_dir[0] * tx + p.walk_dir[1] * ty > -0.3 * math.hypot(tx, ty)

    def render(self) -> np.ndarray:
        return self.camera.render(self.pose(), self.people)


class SimDetector:
    """Person boxes straight from the scene geometry, with pixel noise and occasional misses like a real detector."""

    def __init__(self, scene: SimScene, noise_px: float = 2.0, miss_rate: float = 0.03, rng: random.Random | None = None):
        self.scene = scene
        self.noise_px = noise_px
        self.miss_rate = miss_rate
        self._rng = rng or random.Random()
        self.blind_until = 0.0  # tests set this to hide everyone, like walking behind a tree

    def detect(self, frame: np.ndarray, t: float) -> list[Detection]:
        if t < self.blind_until:
            return []
        out = []
        pose = self.scene.pose()
        for person in self.scene.people:
            if not person.visible or self._rng.random() < self.miss_rate:
                continue
            box = self.scene.camera.box(pose, person)
            if box is None:
                continue
            n = lambda: self._rng.gauss(0.0, self.noise_px)  # noqa: E731
            out.append(Detection(box.x0 + n(), box.y0 + n(), box.x1 + n(), box.y1 + n(), score=0.9))
        return out

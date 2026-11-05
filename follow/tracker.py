"""Keeps track of the people the drone sees, and which one it's locked on.

Each person gets a Kalman filter on their box (center x, center y, height, and how fast each is
changing), so their position can be predicted through short gaps and ahead by the video lag.

Lock states:
  SEARCH    following is on but nobody's locked yet
  LOCKED    the target was seen in the latest frame
  COASTING  the target was missed for less than `coast_s`; the filter's prediction stands in
  LOST      missed for longer; only someone with matching clothing colors gets the lock back

Tuned on the 3 Nov field footage (a pilot walking with a laptop, in sun and shade): the target gets first
pick of the detections, its clothing colors adapt slowly to the light, its search area widens while it's
missing, and a guess stops sliding once the person has been out of sight for a moment.
"""
from __future__ import annotations

from enum import Enum

import numpy as np

from follow.types import Detection, appearance_distance, iou


class LockState(str, Enum):
    SEARCH = "search"
    LOCKED = "locked"
    COASTING = "coasting"
    LOST = "lost"


class Track:
    ACCEL = 160.0  # px/s^2: how quickly a box is allowed to change speed
    MEAS = 5.0  # px: detector box jitter

    def __init__(self, track_id: int, det: Detection, t: float):
        self.id = track_id
        self.x = np.array([det.cx, det.cy, det.h, 0.0, 0.0, 0.0])
        self.P = np.diag([25.0, 25.0, 25.0, 900.0, 900.0, 900.0])
        self.aspect = det.w / max(det.h, 1.0)
        self.t = t
        self.last_seen = t
        self.hits = 1
        self.appearance = det.appearance
        self.last_det = det

    def predict(self, t: float) -> None:
        dt = t - self.t
        if dt <= 0:
            return
        unseen = t - self.last_seen
        F = np.eye(6)
        F[0, 3] = F[1, 4] = F[2, 5] = dt
        q = self.ACCEL**2
        Q = np.zeros((6, 6))
        for i in range(3):  # constant-velocity model with random acceleration
            Q[i, i] = q * dt**4 / 4
            Q[i, i + 3] = Q[i + 3, i] = q * dt**3 / 2
            Q[i + 3, i + 3] = q * dt**2
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        if unseen > 0.3:  # out of sight a while: they've likely stopped or turned, so stop sliding the guess
            self.x[3:] *= np.exp(-dt / 0.5)
        self.t = t

    def correct(self, det: Detection, t: float) -> None:
        self.predict(t)
        H = np.hstack([np.eye(3), np.zeros((3, 3))])
        y = np.array([det.cx, det.cy, det.h]) - H @ self.x
        S = H @ self.P @ H.T + np.eye(3) * self.MEAS**2
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ H) @ self.P
        self.aspect = 0.8 * self.aspect + 0.2 * det.w / max(det.h, 1.0)
        if det.appearance is not None:
            self.appearance = det.appearance if self.appearance is None else 0.85 * self.appearance + 0.15 * det.appearance
        self.last_seen = t
        self.hits += 1
        self.last_det = det

    def state_at(self, t: float) -> np.ndarray:
        """(cx, cy, h, vcx, vcy, vh) extrapolated to time t, without changing the filter."""
        dt = t - self.t
        x = self.x.copy()
        x[:3] += x[3:] * dt
        return x

    def box_at(self, t: float) -> tuple[float, float, float, float]:
        cx, cy, h = self.state_at(t)[:3]
        w = h * self.aspect
        return cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2


class TargetTracker:
    def __init__(
        self,
        coast_s: float = 1.5,
        drop_s: float = 3.0,
        iou_min: float = 0.15,
        gate: float = 0.2,
        same_person: float = 0.5,
        adapt: float = 0.1,
        confirm_hits: int = 2,
    ):
        self.coast_s = coast_s
        self.drop_s = drop_s  # forget other people after this long unseen
        self.iou_min = iou_min
        self.gate = gate  # max center jump (fraction of frame width) to count as the same box
        self.same_person = same_person  # max clothing-color distance to count as the same person
        self.adapt = adapt  # how fast the locked person's stored colors follow the light (sun, shade)
        self.confirm_hits = confirm_hits
        self.tracks: list[Track] = []
        self.target: Track | None = None
        self.target_label: int | None = None  # the ID shown on screen; kept when re-locking
        self.target_appearance: np.ndarray | None = None
        self.state = LockState.SEARCH
        self._next_id = 1
        self._width = 640.0

    # ------------------------------------------------------------------ per frame
    def update(self, dets: list[Detection], t: float, frame_w: float = 640.0) -> None:
        self._width = frame_w
        for tr in self.tracks:
            tr.predict(t)
        matched = self._associate(dets, t)
        for i, det in enumerate(dets):
            if i not in matched:
                self.tracks.append(Track(self._next_id, det, t))
                self._next_id += 1
        self.tracks = [tr for tr in self.tracks if tr is self.target or t - tr.last_seen <= self.drop_s]
        self._update_state(t)

    def _target_match(self, dets: list[Detection], t: float) -> int | None:
        """The detection that's the locked person: near where they should be (a wider area the longer they've
        been missing) and dressed the same. The target picks first, so a duplicate box can't starve it."""
        tr = self.target
        if tr is None or tr not in self.tracks:
            return None
        pred = tr.box_at(t)
        pcx, pcy = (pred[0] + pred[2]) / 2, (pred[1] + pred[3]) / 2
        gate = self.gate + 0.3 * min(1.0, (t - tr.last_seen) / self.coast_s)
        best, best_score = None, -np.inf
        for di, det in enumerate(dets):
            overlap = iou(pred, (det.x0, det.y0, det.x1, det.y1))
            jump = np.hypot(det.cx - pcx, det.cy - pcy) / self._width
            if overlap < self.iou_min and jump > gate:
                continue
            looks = 0.0
            if self.target_appearance is not None and det.appearance is not None:
                looks = appearance_distance(self.target_appearance, det.appearance)
                if looks > self.same_person:
                    continue  # never hand the lock to someone dressed differently, even if they walk right past
            score = overlap - jump - looks  # where they should be AND dressed like them: a passer-by loses on both
            if score > best_score:
                best, best_score = di, score
        return best

    def _associate(self, dets: list[Detection], t: float) -> set[int]:
        matched_t, matched_d = set(), set()
        di = self._target_match(dets, t)
        if di is not None:
            self.target.correct(dets[di], t)
            box = (dets[di].x0, dets[di].y0, dets[di].x1, dets[di].y1)
            alone = all(iou(box, (d.x0, d.y0, d.x1, d.y1)) < 0.05 for j, d in enumerate(dets) if j != di)
            if alone and self.target_appearance is not None and dets[di].appearance is not None:
                # Only learn from a clean view: someone overlapping would mix their colors into the template.
                self.target_appearance = (1 - self.adapt) * self.target_appearance + self.adapt * dets[di].appearance
            matched_t.add(self.tracks.index(self.target))
            matched_d.add(di)
        pairs = []
        for ti, tr in enumerate(self.tracks):
            if tr is self.target:
                continue
            pred = tr.box_at(t)
            pcx, pcy = (pred[0] + pred[2]) / 2, (pred[1] + pred[3]) / 2
            for di, det in enumerate(dets):
                overlap = iou(pred, (det.x0, det.y0, det.x1, det.y1))
                jump = np.hypot(det.cx - pcx, det.cy - pcy) / self._width
                if overlap < self.iou_min and jump > self.gate:
                    continue
                pairs.append((overlap - jump, ti, di))
        for _, ti, di in sorted(pairs, reverse=True):
            if ti in matched_t or di in matched_d:
                continue
            self.tracks[ti].correct(dets[di], t)
            matched_t.add(ti)
            matched_d.add(di)
        return matched_d

    def _update_state(self, t: float) -> None:
        if self.target is None:
            self.state = LockState.SEARCH
            return
        missed = t - self.target.last_seen
        if missed <= 1e-6:
            self.state = LockState.LOCKED
        elif missed < self.coast_s:
            self.state = LockState.COASTING
            self._relock(t, near=True)  # their box may have been picked up as a new track (a jump, a second box)
        else:
            self.state = LockState.LOST
            self._relock(t)

    def _relock(self, t: float, near: bool = False) -> None:
        """Hand the lock to a track that's the same person. While coasting (`near`), only one close to where
        the target should be, and only if it's the only such track: two candidates means someone's walking
        past, and guessing could swap people."""
        pred = self.target.box_at(t)
        pcx, pcy = (pred[0] + pred[2]) / 2, (pred[1] + pred[3]) / 2
        gate = self.gate + 0.3 * min(1.0, (t - self.target.last_seen) / self.coast_s)
        candidates = []
        for tr in self.tracks:
            if tr is self.target or tr.last_seen < t or tr.hits < self.confirm_hits:
                continue
            if near and np.hypot(tr.x[0] - pcx, tr.x[1] - pcy) / self._width > gate:
                continue
            d = appearance_distance(self.target_appearance, tr.appearance)
            if d <= self.same_person:
                candidates.append((d, tr))
        if near and len(candidates) != 1:
            return
        best = min(candidates, key=lambda c: c[0])[1] if candidates else None
        if best is not None:
            self.target = best
            self.state = LockState.LOCKED

    # ------------------------------------------------------------------ lock control
    def lock_center(self, frame_w: float, frame_h: float, t: float, min_h: float = 0.0) -> bool:
        """Lock onto the confirmed, currently visible person nearest the middle of the frame, if their box is at
        least `min_h` px tall (close enough to be the pilot, not someone jogging past far away)."""
        seen = [tr for tr in self.tracks
                if tr.hits >= self.confirm_hits and t - tr.last_seen <= 0.25 and tr.x[2] >= min_h]
        if not seen:
            return False
        best = min(seen, key=lambda tr: np.hypot(tr.x[0] - frame_w / 2, tr.x[1] - frame_h / 2))
        self.target, self.target_label = best, best.id
        self.target_appearance = None if best.appearance is None else best.appearance.copy()
        self.state = LockState.LOCKED
        return True

    def unlock(self) -> None:
        self.target = self.target_label = self.target_appearance = None
        self.state = LockState.SEARCH

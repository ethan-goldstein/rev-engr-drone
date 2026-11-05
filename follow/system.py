"""Follow mode: the one object main.py talks to.

Each loop it takes the newest drone-camera frame, finds people, keeps the lock on one of them, and
returns the sticks to follow them. main.py hands those to the governor as if a pilot were steering,
so the speed cap, slew limit and landing all still apply.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

from flight.types import ZERO, Axes
from follow.controller import FollowConfig, FollowController
from follow.pose_command import ArmsUpCommand, face_box
from follow.tracker import LockState, TargetTracker
from follow.types import Detection, torso_histogram
from ui.target_card import draw_target_card


@dataclass(frozen=True)
class FollowOutput:
    axes: Axes | None  # None = follow is off: use the hand gestures as usual
    land: bool = False
    cut: bool = False  # motors off now, wherever the drone is (same as SPACE)
    state: LockState | None = None
    target: int | None = None
    distance: float | None = None
    note: str = ""


@dataclass(frozen=True)
class Sample:
    """One telemetry point for the strip charts and the flight log."""

    t: float
    state: LockState | None
    distance: float | None
    yaw: float
    pitch: float
    score: float


STATE_COLORS = {  # BGR
    LockState.SEARCH: (170, 170, 170),
    LockState.LOCKED: (80, 220, 80),
    LockState.COASTING: (40, 190, 250),
    LockState.LOST: (60, 60, 230),
}


class FollowSystem:
    def __init__(
        self,
        detector,
        cfg: FollowConfig = FollowConfig(),
        lost_off_s: float = 5.0,
        search_yaw: float = 0.6,
        search_min_offset: float = 0.5,
        history_s: float = 20.0,
        arms_up: ArmsUpCommand | None = None,
        profile: list[tuple[str, str]] = (),
    ):
        self.detector = detector  # anything with detect(frame, t) -> list[Detection]
        self.arms_up = arms_up  # both arms raised at the drone = land (its pose points also find the face)
        self.profile = list(profile)  # rows for the FACE LOCK card (follow/profile.py)
        self.cfg = cfg
        self.controller = FollowController(cfg)
        self.tracker = TargetTracker()
        self.lost_off_s = lost_off_s  # still can't see the locked person this long after losing them: motors off
        self.search_yaw = search_yaw  # yaw stick while turning toward the side they left from
        self.search_min_offset = search_min_offset  # only turn if they left past this far off-center (0-1)
        self.history_s = history_s
        self.active = False
        self.history: deque[Sample] = deque()
        self.detections: list[Detection] = []
        self.distance: float | None = None
        self.search_dir = 0  # while LOST: +1 turning right, -1 turning left, 0 holding still
        self.cutoff_in: float | None = None  # while LOST: seconds until the motors cut
        self.face: tuple[float, float, float, float] | None = None  # locked person's face, when turned to the drone
        self._locked_at: float | None = None
        self._now = 0.0
        self._frame_size = (640, 360)
        self._last_frame_time: float | None = None
        self._arms_up_fired = False

    # ------------------------------------------------------------------ on / off
    def toggle(self, now: float, flying: bool, climbing: bool) -> str:
        if self.active:
            self.stop()
            return "follow off: hovering"
        if not flying:
            return "take off first, then press F to follow"
        if climbing:
            return "still climbing: press F again in a second"
        self.active = True
        if self.tracker.lock_center(*self._frame_size, now, self._lock_min_h()):
            return f"follow on: locked #{self.tracker.target_label}"
        return "follow on: looking for a person (stand in front of the drone)"

    def stop(self) -> None:
        self.active = False
        self.tracker.unlock()
        self.distance = None
        self.search_dir, self.cutoff_in = 0, None
        self.face, self._locked_at = None, None
        self._arms_up_fired = False

    # ------------------------------------------------------------------ per loop
    def update(self, frame: np.ndarray | None, frame_time: float | None, now: float, flying: bool, climbing: bool) -> FollowOutput:
        if frame is not None and frame_time is not None and frame_time != self._last_frame_time:
            self._last_frame_time = frame_time
            h, w = frame.shape[:2]
            self._frame_size = (w, h)
            self.detections = [d.with_appearance(torso_histogram(frame, d)) for d in self.detector.detect(frame, frame_time)]
            self.tracker.update(self.detections, frame_time, w)
            self.face = None
            if self.active and self.arms_up is not None and self.tracker.state is LockState.LOCKED:
                self._arms_up_fired |= self.arms_up.update(frame, self.tracker.target.last_det, frame_time)
                if self.arms_up.landmarks is not None:
                    self.face = face_box(self.arms_up.landmarks, self.arms_up.rect)
        self._now = now

        if not self.active:
            return FollowOutput(None)
        if not flying:
            self.stop()
            return FollowOutput(None, note="follow off: landed")

        tr = self.tracker
        if self._arms_up_fired:
            self._arms_up_fired = False
            label = tr.target_label
            self.stop()
            return FollowOutput(ZERO, land=True, state=LockState.LOCKED, target=label, note="arms up: landing")
        if tr.target is None:
            tr.lock_center(*self._frame_size, now, self._lock_min_h())
        if tr.target is not None and self._locked_at is None:
            self._locked_at = now
        state = tr.state
        axes, self.distance, note, land = ZERO, None, "", False
        self.search_dir, self.cutoff_in = 0, None
        if state is LockState.LOST:
            missing = now - tr.target.last_seen
            if missing >= self.lost_off_s:
                label = tr.target_label
                self.stop()
                note = f"lost #{label} for {self.lost_off_s:.0f}s: motors off"
                return FollowOutput(ZERO, cut=True, state=LockState.LOST, target=label, note=note)
            self.cutoff_in = self.lost_off_s - missing
            # Turn (never fly) toward where they walked out; re-locks if the same clothes come into view.
            self.search_dir = self._exit_side()
            axes = Axes(yaw=self.search_dir * self.search_yaw)

        if state in (LockState.LOCKED, LockState.COASTING) and not climbing:
            cx, _, box_h, _, _, vh = (float(v) for v in tr.target.state_at(now + self.cfg.lead_s))
            w, h = self._frame_size
            last = tr.target.last_det
            # Box height extrapolates wildly once they're out of view (942 m on the first field flight).
            box_h = min(max(box_h, 0.7 * last.h), 1.4 * last.h)
            axes, self.distance = self.controller.command(cx, box_h, vh, w, last.y0 <= 2, last.y1 >= h - 2)
            if state is LockState.COASTING:  # fade the sticks out while the person is hidden
                fade = max(0.0, 1.0 - (now - tr.target.last_seen) / tr.coast_s)
                axes = axes.scaled(fade)

        score = max((d.score for d in self.detections), default=0.0)
        self._record(Sample(now, state, self.distance, axes.yaw, axes.pitch, score))
        return FollowOutput(axes, land=land, state=state, target=tr.target_label, distance=self.distance, note=note)

    def _exit_side(self) -> int:
        """Which side the locked person left the frame: +1 right, -1 left, 0 if they vanished near the middle."""
        tr = self.tracker
        w = self._frame_size[0]
        # Where the filter thought they were heading when coasting ran out (its mean moves in a straight line).
        cx = float(tr.target.state_at(tr.target.last_seen + tr.coast_s)[0])
        offset = (cx - w / 2) / (w / 2)
        if abs(offset) < self.search_min_offset:
            return 0  # behind something in front of the drone: turning away would only lose them
        return 1 if offset > 0 else -1

    def _record(self, sample: Sample) -> None:
        self.history.append(sample)
        while self.history and sample.t - self.history[0].t > self.history_s:
            self.history.popleft()

    # ------------------------------------------------------------------ drawing
    def draw(self, frame: np.ndarray) -> np.ndarray:
        """The drone's view with every detected person, the locked box, and status chips."""
        img = frame.copy()
        h, w = img.shape[:2]
        for d in self.detections:
            cv2.rectangle(img, (int(d.x0), int(d.y0)), (int(d.x1), int(d.y1)), (150, 150, 150), 1)
        tr = self.tracker
        if self.active and tr.target is not None:
            color = STATE_COLORS[tr.state]
            if tr.state is LockState.LOCKED:
                box = (tr.target.last_det.x0, tr.target.last_det.y0, tr.target.last_det.x1, tr.target.last_det.y1)
            else:
                box = tr.target.box_at(self._last_frame_time or tr.target.t)
            x0, y0, x1, y1 = (int(v) for v in box)
            cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)
            if tr.state is LockState.LOCKED:
                draw_target_card(img, box, "FACE LOCK" if self.face else "BODY LOCK", self.profile,
                                 self._live_rows(), color, self.face)
            else:
                label = f"{tr.state.value.upper()} #{tr.target_label}"
                if self.distance is not None:
                    label += f"  {self.distance:.1f} m"
                _chip(img, label, (x0, max(14, y0 - 6)), color)
        cv2.drawMarker(img, (w // 2, h // 2), (230, 230, 230), cv2.MARKER_CROSS, 14, 1)
        chips = [("FOLLOW ON" if self.active else "FOLLOW OFF: PRESS F", (80, 220, 80) if self.active else (150, 150, 150))]
        if self.active:
            chips.append((tr.state.value.upper(), STATE_COLORS[tr.state]))
            if tr.state is LockState.SEARCH:
                chips.append(("STAND IN FRONT, 4-6 M AWAY", STATE_COLORS[LockState.SEARCH]))
            if self.search_dir:
                chips.append(("SEARCHING " + ("RIGHT" if self.search_dir > 0 else "LEFT"), STATE_COLORS[LockState.LOST]))
            warning = self.view_warning()
            if warning:
                chips.append((warning, STATE_COLORS[LockState.COASTING]))
            if self.cutoff_in is not None:
                chips.append((f"MOTORS OFF IN {self.cutoff_in:.1f}s", STATE_COLORS[LockState.LOST]))
            if self.arms_up is not None and self.arms_up.progress > 0:
                chips.append((f"ARMS UP {self.arms_up.progress:.0%}: LAND", (60, 60, 230)))
        x = 8
        for text, color in chips:
            x = _chip(img, text, (x, 18), color) + 6
        return img


    def _lock_min_h(self) -> float:
        """Box height of a person at max_distance: anyone smaller is too far off to be locked onto.
        On 3 Nov, F pressed before the pilot stepped in front locked a jogger about 16 m away instead."""
        return self.controller.focal(self._frame_size[0]) * self.cfg.person_height / self.cfg.max_distance

    def view_warning(self) -> str | None:
        """The locked person's head is out of the top of the frame. If the box is short, the camera is aimed
        too low (on 3 Nov it had been knocked down and saw only legs); if it fills the frame, they're too close."""
        tr = self.tracker
        if not self.active or tr.state is not LockState.LOCKED or tr.target.last_det.y0 > 2:
            return None
        if tr.target.last_det.h >= 0.8 * self._frame_size[1]:
            return "TOO CLOSE: STEP BACK"
        return "HEAD OUT OF VIEW: TILT CAMERA UP"

    def _live_rows(self) -> list[tuple[str, str]]:
        tr = self.tracker
        rows = [("TRACK", f"#{tr.target_label}")]
        if self.distance is not None:
            rows.append(("RANGE", f"{self.distance:.1f} M"))
        if self._locked_at is not None:
            held = max(0, int(self._now - self._locked_at))
            rows.append(("LOCKED", f"{held // 60:02d}:{held % 60:02d}"))
        rows.append(("CONF", f"{tr.target.last_det.score:.0%}"))
        return rows


def _chip(img: np.ndarray, text: str, org: tuple[int, int], color) -> int:
    """A small filled label; returns its right edge."""
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    x, y = org
    cv2.rectangle(img, (x - 3, y - th - 4), (x + tw + 3, y + 4), color, -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1, cv2.LINE_AA)
    return x + tw + 3

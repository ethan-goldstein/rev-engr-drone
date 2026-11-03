"""Position hold, barrier and autopilot on top of the governor.

Each frame main.py hands over the governor's Decision and gets back the axes to send. Corrections
are only added while the drone is flying and past its own takeoff climb, and only to roll and pitch,
so takeoff and the throttle-down landing are unchanged.

States:
  HOLD    no steering: hold the target (the takeoff spot, or wherever you let go of the sticks)
  MANUAL  you're steering: your sticks pass through (outward ones cut at the barrier)
  AUTO    the autopilot carrot walks the path and back home
  NO_FIX  the tracker lost the drone: no corrections, autopilot paused (lands after a while)
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum

from flight.governor import Decision, FlightState
from flight.types import Axes
from nav.controller import PositionController, body_sticks, clamp_vec, world_sticks
from nav.geofence import Geofence
from nav.mission import Mission, Point
from nav.pose import Pose, PoseSource


class NavState(str, Enum):
    OFF = "off"  # on the ground, landing, or in the takeoff climb
    HOLD = "hold"
    MANUAL = "manual"
    AUTO = "auto"
    NO_FIX = "no fix"


@dataclass(frozen=True)
class NavOutput:
    axes: Axes
    state: NavState
    land: bool = False  # main.py should start the governor's landing
    note: str = ""


class Navigator:
    def __init__(
        self,
        source: PoseSource,
        fence: Geofence,
        path: list[Point],
        controller: PositionController | None = None,
        max_correction: float = 0.4,
        no_fix_land_s: float = 3.0,
    ):
        self.source = source
        self.fence = fence
        self.path = path
        self.controller = controller or PositionController()
        self.max_correction = max_correction
        self.no_fix_land_s = no_fix_land_s  # during autopilot only; otherwise the drone just hovers
        self.state = NavState.OFF
        self.pose: Pose | None = None
        self.home: Point | None = None
        self.target: Point | None = None
        self.mission: Mission | None = None
        self.trail: deque[Point] = deque(maxlen=400)
        self._lost_since: float | None = None
        self._was_manual = False
        self._last_update: float | None = None

    # ------------------------------------------------------------------ autopilot switch
    def toggle_mission(self) -> str:
        """Start the path from wherever the drone is, or cancel it and hold. Returns a message for the user."""
        if self.mission is not None:
            self.mission = None
            if self.pose is not None:
                self.target = self.fence.clamp((self.pose.x, self.pose.y))
            return "autopilot off: holding here"
        if self.state not in (NavState.HOLD, NavState.MANUAL) or self.pose is None or self.home is None:
            return "autopilot needs the drone flying and tracked"
        here = (self.pose.x, self.pose.y)
        self.mission = Mission(here, [self.fence.clamp(p) for p in self.path], self.home)
        return "autopilot on: flying the path"

    # ------------------------------------------------------------------ per frame
    def update(self, decision: Decision, now: float, climbing: bool) -> NavOutput:
        dt = 0.0 if self._last_update is None else min(0.1, max(0.0, now - self._last_update))
        self._last_update = now
        pose = self.source.latest()
        if pose is not None:
            self.pose = pose
            self.trail.append((pose.x, pose.y))

        if decision.state is not FlightState.FLYING:
            self._reset()
            return NavOutput(decision.axes, NavState.OFF)
        if self.home is None and pose is not None:
            # First fix of this flight: the takeoff spot is home, and the first thing to hold.
            self.home = self.target = self.fence.clamp((pose.x, pose.y))
        if climbing:
            self.state = NavState.OFF  # the drone ignores the sticks during its own climb
            return NavOutput(decision.axes, self.state, note="takeoff climb")

        if pose is None:
            if self._lost_since is None:
                self._lost_since = now
            lost = now - self._lost_since
            self.state = NavState.NO_FIX
            land = self.mission is not None and lost >= self.no_fix_land_s
            return NavOutput(decision.axes, self.state, land=land, note=f"drone not tracked for {lost:.1f}s")
        self._lost_since = None

        if self.fence.breached(pose, now):
            return NavOutput(decision.axes, self.state, land=True, note="left the barrier: landing")

        here = (pose.x, pose.y)
        user_x, user_y = world_sticks(decision.axes.pitch, decision.axes.roll, pose.heading)
        note = ""
        if user_x or user_y:
            if self.mission is not None:
                self.mission = None
                note = "steering took over from the autopilot"
            self.state = NavState.MANUAL
            self._was_manual = True
            self.controller.update(pose, here)  # keep its velocity estimate fresh; output unused
            ux, uy = self.fence.limit_user(user_x, user_y, pose)
            px, py = self.fence.push_back(pose)
            ux, uy = ux + px, uy + py
        else:
            if self._was_manual:
                self._was_manual = False
                self.target = self.fence.clamp(here)  # let go of the sticks: hold right here
            if self.mission is not None:
                self.state = NavState.AUTO
                self.target = self.fence.clamp(self.mission.advance(dt, here))
                if self.mission.done(here):
                    self.mission = None
                    self.target = self.home
                    note = "autopilot done: holding at home"
            else:
                self.state = NavState.HOLD
            target = self.fence.clamp(self.target or here)
            cx, cy = self.controller.update(pose, target, integrate=self.state is NavState.HOLD)
            px, py = self.fence.push_back(pose)
            ux, uy = clamp_vec(cx + px, cy + py, self.max_correction)

        pitch, roll = body_sticks(ux, uy, pose.heading)
        axes = Axes(decision.axes.throttle, decision.axes.yaw, pitch, roll).clamped(1.0)
        return NavOutput(axes, self.state, note=note)

    def _reset(self) -> None:
        self.state = NavState.OFF
        self.home = self.target = self.mission = None
        self.controller.reset()
        self.fence.reset()
        self._lost_since = None
        self._was_manual = False

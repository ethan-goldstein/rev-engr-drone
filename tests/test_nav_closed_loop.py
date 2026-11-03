"""Whole loop in the simulator: governor -> navigator -> sim drone -> fake tracking camera, with wind."""
import math
import random
from dataclasses import replace

from drivers.sim import SimDriver
from flight.governor import FlightState, Governor, GovernorConfig
from flight.types import Axes
from gesture.mapper import NO_HAND, Intent, IntentKind
from main import apply
from nav.geofence import Geofence
from nav.mission import build_path
from nav.navigator import Navigator, NavState
from nav.pose import SimPoseSource
from tests.test_main import FakeClock

DT = 1 / 30
STEER_RIGHT = Intent(IntentKind.MOVE, "steer", Axes(roll=1.0))


class Rig:
    def __init__(self, wind=(0.0, 0.0), track=True):
        self.clock = FakeClock()
        self.sim = SimDriver(clock=self.clock, wind=wind, rng=random.Random(1))
        self.gov = Governor(GovernorConfig())
        self.source = SimPoseSource(self.sim, clock=self.clock, rng=random.Random(2))
        fence = Geofence((-2.0, -2.0, 2.0, 2.0))
        self.nav = Navigator(self.source, fence, build_path("square", fence.inner)) if track else None
        self.max_x = -math.inf

    def takeoff(self):
        apply(self.sim, self.gov.takeoff(self.clock()))

    def run(self, seconds, intent=NO_HAND, until=None):
        for _ in range(round(seconds / DT)):
            now = self.clock()
            d = self.gov.step(intent, now)
            if self.nav is not None:
                out = self.nav.update(d, now, self.gov.climbing(now))
                landing = self.gov.land(now) if out.land else None
                d = landing if landing is not None else replace(d, axes=out.axes)
            apply(self.sim, d)
            self.max_x = max(self.max_x, self.sim.x)
            self.clock.sleep(DT)
            if until is not None and until():
                return True
        return False

    @property
    def offset(self):
        return math.hypot(self.sim.x, self.sim.y)


def test_holds_the_takeoff_spot_against_wind():
    rig = Rig(wind=(0.2, 0.1))
    rig.takeoff()
    rig.run(22.0)
    assert rig.nav.state is NavState.HOLD
    assert rig.offset < 0.25


def test_without_tracking_the_wind_carries_it_away():
    rig = Rig(wind=(0.2, 0.1), track=False)
    rig.takeoff()
    rig.run(22.0)
    assert rig.offset > 2.0


def test_autopilot_flies_the_square_and_comes_home():
    rig = Rig(wind=(0.1, 0.0))
    rig.takeoff()
    rig.run(3.0)
    assert rig.nav.toggle_mission().startswith("autopilot on")
    corners = list(rig.nav.mission.points[1:-1])
    visited = set()

    def check():
        for i, c in enumerate(corners):
            if math.dist(c, (rig.sim.x, rig.sim.y)) < 0.3:
                visited.add(i)
        return rig.nav.mission is None

    assert rig.run(90.0, until=check), "autopilot never finished"
    assert visited == set(range(len(corners)))
    rig.run(3.0)
    assert rig.offset < 0.25  # home is the takeoff spot


def test_steering_into_the_barrier_stops_there():
    rig = Rig()
    rig.takeoff()
    rig.run(3.0)
    rig.run(10.0, intent=STEER_RIGHT)
    assert rig.gov.state is FlightState.FLYING  # stopped by the barrier, not by a breach landing
    assert 1.5 < rig.max_x < 2.0
    rig.run(3.0)  # let go: holds near where it stopped
    assert rig.nav.state is NavState.HOLD and 1.3 < rig.sim.x < 1.8


def test_losing_tracking_drops_corrections_and_lands_an_autopilot_flight():
    rig = Rig(wind=(0.2, 0.0))
    rig.takeoff()
    rig.run(3.0)
    rig.nav.toggle_mission()
    rig.run(1.0)
    rig.source.enabled = False
    rig.run(1.0)
    assert rig.nav.state is NavState.NO_FIX
    out = rig.nav.update(rig.gov.step(NO_HAND, rig.clock()), rig.clock(), False)
    assert out.axes.pitch == out.axes.roll == 0  # flying blind: no corrections
    rig.run(2.5)
    assert rig.gov.state is FlightState.LANDING


def test_hold_mode_without_autopilot_just_hovers_when_tracking_is_lost():
    rig = Rig()
    rig.takeoff()
    rig.run(3.0)
    rig.source.enabled = False
    rig.run(5.0)
    assert rig.gov.state is FlightState.FLYING and rig.nav.state is NavState.NO_FIX

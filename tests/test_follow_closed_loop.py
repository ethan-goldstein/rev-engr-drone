"""Whole follow loop in the simulator: sim drone -> virtual camera -> detector -> tracker -> governor."""
import math
import random
from dataclasses import replace

import pytest

from drivers.sim import SimDriver
from flight.governor import FlightState, Governor, GovernorConfig
from follow.sim_scene import SimCamera, SimDetector, SimPerson, SimScene
from follow.system import FollowSystem
from follow.tracker import LockState
from gesture.mapper import NO_HAND, Intent, IntentKind
from main import apply
from nav.geofence import Geofence
from nav.navigator import Navigator
from nav.pose import SimPoseSource
from tests.test_main import FakeClock

DT = 1 / 30


class FollowRig:
    def __init__(self, people=None, track=True):
        self.clock = FakeClock()
        self.sim = SimDriver(clock=self.clock, rng=random.Random(1), room_m=16.0)
        self.people = people or [SimPerson("square", 0.4, rng=random.Random(2))]
        self.scene = SimScene(self.sim, self.people, SimCamera(grid=False))
        self.detector = SimDetector(self.scene, rng=random.Random(3))
        self.follow = FollowSystem(self.detector)
        self.gov = Governor(GovernorConfig())
        self.nav = None
        if track:  # the same position stack the app runs in the sim
            source = SimPoseSource(self.sim, clock=self.clock, rng=random.Random(4))
            self.nav = Navigator(source, Geofence((-7.0, -7.0, 7.0, 7.0)), [])
        self.distances: list[float] = []
        self.centered: list[bool] = []
        self.states: list[LockState] = []

    def start(self):
        apply(self.sim, self.gov.takeoff(self.clock()))
        self.run(3.0)
        msg = self.follow.toggle(self.clock(), self.gov.state is FlightState.FLYING, self.gov.climbing(self.clock()))
        assert msg.startswith("follow on: locked"), msg

    def run(self, seconds, measure=False):
        for _ in range(round(seconds / DT)):
            now = self.clock()
            self.scene.update(now)
            frame = self.scene.render()
            flying, climbing = self.gov.state is FlightState.FLYING, self.gov.climbing(now)
            out = self.follow.update(frame, now, now, flying, climbing)
            intent = NO_HAND
            if out.axes is not None:
                intent = Intent(IntentKind.HOVER if out.axes.is_zero() else IntentKind.MOVE, "follow", out.axes)
            d = self.gov.step(intent, now)
            if out.cut:
                d = self.gov.estop(now)
            elif out.land:
                d = self.gov.land(now) or d
            if self.nav is not None:
                nav_out = self.nav.update(d, now, self.gov.climbing(now))
                d = replace(d, axes=nav_out.axes)
            apply(self.sim, d)
            if measure:
                p = self.people[0]
                self.distances.append(math.hypot(p.x - self.sim.x, p.y - self.sim.y))
                box = self.scene.camera.box(self.scene.pose(), p)
                self.centered.append(box is not None and 160 <= box.cx <= 480)
                self.states.append(out.state)
            self.clock.sleep(DT)


def fraction(flags):
    return sum(flags) / len(flags)


def test_follows_a_person_walking_a_square():
    rig = FollowRig()
    rig.start()
    rig.run(5.0)  # settle
    rig.run(70.0, measure=True)  # more than one lap
    ok = [3.0 <= d <= 5.5 for d in rig.distances]
    assert fraction(ok) >= 0.9, f"in range only {fraction(ok):.0%}"
    assert min(rig.distances) > 2.7
    assert fraction(rig.centered) >= 0.85, f"centered only {fraction(rig.centered):.0%}"
    assert rig.gov.state is FlightState.FLYING


def test_coasts_through_a_dropout_and_keeps_the_same_id():
    rig = FollowRig()
    rig.start()
    rig.run(5.0)
    label = rig.follow.tracker.target_label
    rig.detector.blind_until = rig.clock() + 1.0  # walks behind a tree
    rig.run(1.0, measure=True)
    assert LockState.COASTING in rig.states
    rig.run(1.0, measure=True)
    assert rig.states[-1] is LockState.LOCKED
    assert rig.follow.tracker.target_label == label


def standing_person() -> SimPerson:
    return SimPerson(speed=0.0, start=(0.0, 4.0), pause_s=1000.0)


def test_motors_cut_5_seconds_after_losing_the_person():
    rig = FollowRig(people=[standing_person()])
    rig.start()
    rig.run(2.0)
    heading = rig.sim.heading
    rig.people[0].visible = False  # gone mid-frame, like behind a tree: no reason to turn
    rig.run(4.5, measure=True)
    assert rig.states[-1] is LockState.LOST and rig.follow.search_dir == 0
    assert abs(rig.sim.heading - heading) < math.radians(5)
    assert rig.gov.state is FlightState.FLYING  # still waiting for them
    rig.run(1.0)
    assert rig.gov.state is FlightState.LANDED and rig.sim.phase == "crashed"  # props stopped where it was
    assert not rig.follow.active


@pytest.mark.parametrize("side", [1, -1])
def test_turns_toward_where_the_person_ran_out_and_finds_them(side):
    rig = FollowRig(people=[standing_person()])
    rig.start()
    rig.run(2.0)
    label, heading = rig.follow.tracker.target_label, rig.sim.heading
    person = rig.people[0]
    person.waypoints, person.speed = [(side * 4.0, -1.0)], 5.0  # sprints around behind the drone's view
    person._goal = person.waypoints[0]
    searched = []
    for _ in range(round(5.0 / DT)):
        rig.run(DT, measure=True)
        searched.append(rig.follow.search_dir)
    assert LockState.LOST in rig.states  # coasting alone couldn't catch up...
    assert set(searched) == {0, side}  # ...so it turned toward the side they left from...
    assert rig.states[-1] is LockState.LOCKED and rig.follow.tracker.target_label == label  # ...and found them
    assert side * (rig.sim.heading - heading) > math.radians(80)
    assert rig.gov.state is FlightState.FLYING


def test_stays_on_the_target_when_someone_else_walks_through():
    target = SimPerson("line", 0.3, shirt=(40, 60, 200), rng=random.Random(5))
    other = SimPerson(speed=0.0, shirt=(60, 180, 60), start=(-3.0, 4.6))
    other.waypoints, other.speed = [(-3.0, 4.6), (3.0, 4.6)], 0.5  # crosses right behind the target
    other._goal = (3.0, 4.6)
    rig = FollowRig(people=[target, other])
    rig.start()
    label = rig.follow.tracker.target_label
    rig.run(14.0)
    assert rig.follow.tracker.target_label == label
    box = rig.scene.camera.box(rig.scene.pose(), target)
    tracked = rig.follow.tracker.target.box_at(rig.clock())
    assert abs((tracked[0] + tracked[2]) / 2 - box.cx) < 25


def test_follow_needs_the_drone_flying():
    rig = FollowRig()
    msg = rig.follow.toggle(rig.clock(), False, False)
    assert "take off first" in msg and not rig.follow.active


def test_arms_up_lands():
    from follow.pose_command import ArmsUpCommand, sim_pose

    rig = FollowRig()
    rig.follow.arms_up = ArmsUpCommand(sim_pose(rig.people[0]))
    rig.start()
    rig.run(4.0)
    rig.people[0].arms_up = True
    rig.run(1.2)
    assert rig.gov.state is FlightState.FLYING  # not held long enough yet
    rig.run(0.6)
    assert rig.gov.state is FlightState.LANDING and not rig.follow.active


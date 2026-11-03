from drivers.sim import HOVER_M, SimDriver
from flight.types import Axes


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def fly(sim, clock, axes, seconds):
    for _ in range(int(seconds * 30)):
        clock.t += 1 / 30
        sim.set_axes(axes)


def airborne():
    clock = FakeClock()
    sim = SimDriver(clock)
    sim.connect()
    sim.takeoff()
    fly(sim, clock, Axes(), 2.0)
    return sim, clock


def test_takeoff_climbs_to_hover_height():
    sim, _ = airborne()
    assert sim.phase == "flying"
    assert sim.z == HOVER_M


def test_sticks_move_the_drone():
    sim, clock = airborne()
    fly(sim, clock, Axes(roll=0.5), 1.0)
    assert sim.x > 0.5 and abs(sim.y) < 1e-6
    fly(sim, clock, Axes(pitch=0.5), 1.0)
    assert sim.y > 0.5
    fly(sim, clock, Axes(throttle=0.5), 0.5)
    assert sim.z > HOVER_M


def test_land_and_estop():
    sim, clock = airborne()
    sim.land()
    fly(sim, clock, Axes(), 3.0)
    assert sim.phase == "landed" and sim.z == 0.0

    sim.takeoff()
    fly(sim, clock, Axes(), 2.0)
    sim.estop()
    assert sim.phase == "crashed" and not sim.flying and sim.z == 0.0


def test_render_shape():
    sim, _ = airborne()
    assert sim.render(320, 240).shape == (240, 320, 3)

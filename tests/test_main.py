"""The exit path must finish the landing and turn the motors off before the app lets go."""
from drivers.sim import SimDriver
from flight.governor import FlightState, Governor, GovernorConfig
from main import land_and_stop


class FakeClock:
    def __init__(self):
        self.t = 100.0

    def __call__(self) -> float:
        return self.t

    def sleep(self, dt: float) -> None:
        self.t += dt


class RecordingSim(SimDriver):
    def __init__(self, clock):
        super().__init__(clock)
        self.calls = []

    def land(self):
        self.calls.append("land")
        super().land()

    def estop(self):
        self.calls.append("estop")
        super().estop()


def test_quitting_mid_flight_lands_then_turns_motors_off():
    clock = FakeClock()
    sim = RecordingSim(clock)
    gov = Governor(GovernorConfig())
    gov.takeoff(clock())
    sim.takeoff()
    clock.sleep(5.0)
    land_and_stop(gov, sim, clock=clock, sleep=clock.sleep)
    assert sim.calls == ["land", "estop"]
    assert gov.state is FlightState.LANDED
    assert not sim.flying


def test_quitting_while_landed_sends_nothing():
    clock = FakeClock()
    sim = RecordingSim(clock)
    land_and_stop(Governor(GovernorConfig()), sim, clock=clock, sleep=clock.sleep)
    assert sim.calls == []

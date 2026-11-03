import json
import random

from drivers.sim import MAX_SPEED, SimDriver
from flight.governor import Governor
from flight.trim import LIMIT, STEP, Trim
from flight.types import ZERO, Axes, Command
from gesture.mapper import NO_HAND
from main import apply
from tests.test_main import FakeClock


def test_nudges_add_up_and_stop_at_the_limit():
    trim = Trim()
    assert trim.nudge(roll=-STEP) == "trim: 0.02 left, 0.00 forward"
    for _ in range(50):
        trim.nudge(pitch=STEP)
    assert trim.pitch == LIMIT


def test_only_added_while_in_the_air():
    gov, trim, clock = Governor(), Trim(roll=-0.1, pitch=0.04), FakeClock()
    landed = gov.step(NO_HAND, clock())
    assert trim.applied(landed).axes == ZERO
    takeoff = gov.takeoff(clock())
    assert trim.applied(takeoff).axes == Axes(0.0, 0.0, 0.04, -0.1)
    assert trim.applied(gov.estop(clock())).axes == ZERO  # motors-off packets stay clean
    assert gov.estop(clock()).command is Command.ESTOP


def test_saved_between_flights(tmp_path):
    path = tmp_path / "trim.json"
    Trim(roll=-0.06).save(path)
    assert Trim.load(path) == Trim(roll=-0.06)
    path.write_text(json.dumps({"roll": 5}))
    assert Trim.load(path).roll == LIMIT
    path.write_text("not json")
    assert Trim.load(path) == Trim()
    assert Trim.load(tmp_path / "missing.json") == Trim()


def hover_drift(trim: Trim, seconds: float = 10.0) -> float:
    clock = FakeClock()
    sim = SimDriver(clock=clock, wind=(0.15, 0.0), rng=random.Random(1))  # slides right 15 cm/s
    gov = Governor()
    apply(sim, trim.applied(gov.takeoff(clock())))
    for _ in range(round(seconds * 30)):
        apply(sim, trim.applied(gov.step(NO_HAND, clock())))
        clock.sleep(1 / 30)
    return sim.x


def test_trim_cancels_a_steady_drift_in_the_sim():
    assert hover_drift(Trim()) > 1.0
    assert abs(hover_drift(Trim(roll=-0.15 / MAX_SPEED))) < 0.2


def test_keys_typed_in_the_terminal_reach_the_app():
    import os
    import time

    from ui.terminal_keys import TerminalKeys

    master, slave = os.openpty()
    stream = os.fdopen(slave, "r")
    keys = TerminalKeys(stream)
    os.write(master, b"f\x1b[A[")  # F, an arrow key (ignored, not ESC/quit), then trim-left
    got, deadline = [], time.monotonic() + 1.0
    while len(got) < 2 and time.monotonic() < deadline:
        k = keys.get()
        if k != -1:
            got.append(k)
        time.sleep(0.005)
    keys.close()
    os.close(master)  # hang up first, so the reader's blocked read returns before its fd closes
    time.sleep(0.05)
    stream.close()
    assert got == [ord("f"), ord("[")]

import random

import numpy as np
import pytest

from follow.tracker import LockState, TargetTracker
from follow.types import Detection

DT = 1 / 15  # the F19's video rate


def color(i: int) -> np.ndarray:
    """A fake clothing-color histogram: all the weight in one bin."""
    h = np.zeros(128, np.float32)
    h[i] = 1.0
    return h


def person(cx, cy=180.0, h=200.0, look=0, noise=0.0, rng=None):
    rng = rng or random.Random(0)
    n = lambda: rng.gauss(0, noise)  # noqa: E731
    w = h * 0.4
    return Detection(cx - w / 2 + n(), cy - h / 2 + n(), cx + w / 2 + n(), cy + h / 2 + n(), 0.9, color(look))


def locked_tracker(cx=320.0, look=0):
    tr = TargetTracker()
    for i in range(3):
        tr.update([person(cx, look=look)], i * DT)
    assert tr.lock_center(640, 360, 2 * DT)
    return tr, 2 * DT


def test_smooths_a_jittery_box():
    rng = random.Random(1)
    tr = TargetTracker()
    raw, filtered = [], []
    for i in range(90):
        det = person(300.0, noise=4.0, rng=rng)
        tr.update([det], i * DT)
        if i >= 30:  # after it settles
            raw.append(det.cx - 300.0)
            filtered.append(tr.tracks[0].state_at(i * DT)[0] - 300.0)
    rms = lambda v: float(np.sqrt(np.mean(np.square(v))))  # noqa: E731
    assert rms(filtered) < 0.75 * rms(raw)
    assert abs(np.mean(filtered)) < 1.0  # and it isn't biased


def test_locks_on_the_person_nearest_the_middle():
    tr = TargetTracker()
    for i in range(3):
        tr.update([person(100.0, look=1), person(330.0, look=2), person(560.0, look=3)], i * DT)
    assert tr.lock_center(640, 360, 2 * DT)
    assert tr.target.x[0] == pytest.approx(330.0, abs=1)


def test_coasting_then_lost():
    tr, t = locked_tracker()
    tr.update([], t + 0.5)
    assert tr.state is LockState.COASTING
    tr.update([], t + 2.0)
    assert tr.state is LockState.LOST


def test_keeps_the_same_id_through_a_one_second_gap():
    tr = TargetTracker()
    t = 0.0
    for i in range(15):  # walking right at 60 px/s
        t = i * DT
        tr.update([person(200.0 + 60 * t)], t)
    tr.lock_center(640, 360, t)
    label, track = tr.target_label, tr.target
    for i in range(1, 16):
        tr.update([], t + i * DT)  # hidden for 1 s
    t2 = t + 16 * DT
    tr.update([person(200.0 + 60 * t2)], t2)  # reappears where the filter expects
    assert tr.state is LockState.LOCKED
    assert tr.target is track and tr.target_label == label


def test_relocks_only_on_matching_clothes():
    tr, t = locked_tracker(look=5)
    label = tr.target_label
    tr.update([], t + 2.0)
    assert tr.state is LockState.LOST
    for i in range(3):  # someone in different clothes walks into view
        tr.update([person(500.0, look=9)], t + 2.1 + i * DT)
    assert tr.state is LockState.LOST
    for i in range(3):  # the original person reappears somewhere else
        tr.update([person(150.0, look=5)], t + 2.5 + i * DT)
    assert tr.state is LockState.LOCKED and tr.target_label == label
    assert tr.target.x[0] == pytest.approx(150.0, abs=5)


def test_a_passer_by_cannot_steal_the_lock():
    tr, t = locked_tracker(cx=320.0, look=5)
    track = tr.target
    # Someone dressed differently walks right through the target's spot while the target stands still.
    for i in range(1, 30):
        tr.update([person(320.0, look=5), person(200.0 + 9 * i, look=9)], t + i * DT)
        assert tr.target is track
    assert tr.target.x[0] == pytest.approx(320.0, abs=3)


def test_unlock_resets():
    tr, _ = locked_tracker()
    tr.unlock()
    assert tr.target is None and tr.state is LockState.SEARCH

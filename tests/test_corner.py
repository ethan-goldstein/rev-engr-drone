from nav.corner import CornerZone
from tests.fixtures import hand

DT = 1 / 30


def hold(zone, obs, start, seconds):
    fired, t = 0, start
    for _ in range(round(seconds / DT)):
        fired += zone.update(obs, t)
        t += DT
    return fired, t


def test_open_palm_in_the_corner_fires_once_after_a_second():
    zone = CornerZone()
    fired, t = hold(zone, hand("Open_Palm", palm=(0.9, 0.1)), 0.0, 0.9)
    assert fired == 0 and 0.8 < zone.progress < 1.0
    fired, t = hold(zone, hand("Open_Palm", palm=(0.9, 0.1)), t, 2.0)
    assert fired == 1


def test_leaving_the_box_re_arms_it():
    zone = CornerZone()
    _, t = hold(zone, hand("Open_Palm", palm=(0.9, 0.1)), 0.0, 1.2)
    _, t = hold(zone, None, t, 0.2)
    assert zone.progress == 0.0
    fired, _ = hold(zone, hand("Open_Palm", palm=(0.9, 0.1)), t, 1.2)
    assert fired == 1


def test_other_places_and_gestures_do_nothing():
    zone = CornerZone()
    assert hold(zone, hand("Open_Palm", palm=(0.5, 0.5)), 0.0, 2.0)[0] == 0
    assert hold(zone, hand("Open_Palm", palm=(0.1, 0.1)), 0.0, 2.0)[0] == 0  # wrong corner
    assert hold(zone, hand("Closed_Fist", palm=(0.9, 0.1)), 0.0, 2.0)[0] == 0

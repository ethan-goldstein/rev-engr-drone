from nav.geofence import Geofence
from nav.pose import Pose

FENCE = (-2.0, -2.0, 2.0, 2.0)


def at(x, y):
    return Pose(x, y, 0.0, 0.0)


def test_targets_are_clamped_inside_the_margin():
    f = Geofence(FENCE, margin=0.3)
    assert f.clamp((5.0, -5.0)) == (1.7, -1.7)
    assert f.clamp((0.5, 0.5)) == (0.5, 0.5)


def test_push_back_points_inward_only_in_the_margin_band():
    f = Geofence(FENCE, margin=0.3)
    assert f.push_back(at(0.0, 0.0)) == (0.0, 0.0)
    ux, uy = f.push_back(at(1.9, -1.9))
    assert ux < 0 < uy


def test_outward_user_sticks_are_cut_at_the_band():
    f = Geofence(FENCE, margin=0.3)
    assert f.limit_user(0.3, 0.2, at(1.8, 0.0)) == (0.0, 0.2)  # outward x cut, y untouched
    assert f.limit_user(-0.3, 0.0, at(1.8, 0.0)) == (-0.3, 0.0)  # heading back in is fine
    assert f.limit_user(0.3, 0.0, at(1.0, 0.0)) == (0.3, 0.0)  # nowhere near the edge


def test_breach_needs_half_a_second_outside():
    f = Geofence(FENCE, breach_s=0.5)
    assert not f.breached(at(2.1, 0.0), 10.0)
    assert not f.breached(at(2.1, 0.0), 10.4)
    assert f.breached(at(2.1, 0.0), 10.6)
    assert not f.breached(at(1.9, 0.0), 10.7)  # back inside resets the timer
    assert not f.breached(at(2.1, 0.0), 11.0)

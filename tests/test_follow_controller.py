import pytest

from follow.controller import FollowConfig, FollowController

W = 640
C = FollowController(FollowConfig())
F = C.focal(W)


def h_at(distance: float) -> float:
    return F * 1.7 / distance


def test_focal_length_from_the_60_degree_lens():
    assert F == pytest.approx(554.3, abs=0.5)


def test_pinhole_distance():
    assert C.distance(h_at(4.0), W) == pytest.approx(4.0)
    assert C.distance(h_at(6.5), W) == pytest.approx(6.5)


def test_turns_toward_the_person():
    right, _ = C.command(480, h_at(4), 0, W)
    left, _ = C.command(160, h_at(4), 0, W)
    centered, _ = C.command(320, h_at(4), 0, W)
    assert right.yaw > 0 > left.yaw
    assert centered.yaw == 0


def test_holds_the_follow_distance():
    far, _ = C.command(320, h_at(6.0), 0, W)
    near, _ = C.command(320, h_at(3.5), 0, W)
    exact, _ = C.command(320, h_at(4.0), 0, W)
    assert far.pitch > 0 > near.pitch
    assert exact.pitch == pytest.approx(0, abs=1e-9)


def test_backs_away_when_too_close():
    axes, d = C.command(320, h_at(2.5), 0, W)
    assert d < 3 and axes.pitch <= -0.6


def test_keeps_up_with_someone_walking_away():
    still, _ = C.command(320, h_at(4.0), 0.0, W)
    leaving, _ = C.command(320, h_at(4.0), -20.0, W)  # box shrinking 20 px/s
    assert leaving.pitch > still.pitch


def test_never_advances_on_a_cut_off_box():
    short = h_at(7.0)  # looks far away only because the head is out of frame
    assert C.command(320, short, 0, W, cut_top=True)[0].pitch <= 0
    assert C.command(320, short, 0, W, cut_bottom=True)[0].pitch <= 0
    assert C.command(320, 360, 0, W, cut_top=True, cut_bottom=True)[0].pitch <= -0.6


def test_stops_advancing_beyond_max_distance():
    axes, d = C.command(320, h_at(9.0), 0, W)
    assert d > 8 and axes.pitch == 0


def test_outputs_are_clamped():
    axes, _ = C.command(639, h_at(30.0), -500.0, W)
    assert abs(axes.yaw) <= 0.8 and -1 <= axes.pitch <= 1

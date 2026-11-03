import math

import pytest

from follow.sim_scene import SimCamera, SimPerson

CAM = SimCamera()


def standing(x, y):
    return SimPerson(speed=0.0, start=(x, y))


def test_person_straight_ahead_at_4m():
    box = CAM.box((0.0, 0.0, 1.0, 0.0), standing(0.0, 4.0))  # drone 1 m up, facing +y
    assert box.cx == pytest.approx(320, abs=0.5)
    assert box.h == pytest.approx(CAM.f * 1.7 / 4.0, abs=1)


def test_left_right_and_behind():
    pose = (0.0, 0.0, 1.0, 0.0)
    assert CAM.box(pose, standing(1.0, 4.0)).cx > 320
    assert CAM.box(pose, standing(-1.0, 4.0)).cx < 320
    assert CAM.box(pose, standing(0.0, -4.0)) is None


def test_turning_the_drone_moves_the_person_in_frame():
    p = standing(0.0, 4.0)
    turned_right = CAM.box((0.0, 0.0, 1.0, math.radians(10)), p)
    assert turned_right.cx < 320  # turn right and the person slides left on screen


def test_render_shows_the_person():
    p = standing(0.0, 4.0)
    img = CAM.render((0.0, 0.0, 1.0, 0.0), [p])
    assert img.shape == (360, 640, 3)
    assert (img == p.shirt).all(axis=2).sum() > 500


def test_walker_follows_its_path():
    p = SimPerson("line", speed=1.0, pause_s=0.0)
    for i in range(31):
        p.update(i * 0.1)
    assert (p.x, p.y) == pytest.approx((3.0, 4.0), abs=0.05)

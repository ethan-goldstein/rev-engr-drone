import math

import pytest

from nav.controller import Gains, PositionController, body_sticks, clamp_vec, world_sticks
from nav.pose import Pose


@pytest.mark.parametrize(
    "heading, world, expected",
    [
        (0.0, (1.0, 0.0), (0.0, 1.0)),  # facing +y: +x is to the right -> roll right
        (0.0, (0.0, 1.0), (1.0, 0.0)),  # facing +y: +y is ahead -> pitch forward
        (math.pi / 2, (1.0, 0.0), (1.0, 0.0)),  # facing +x: +x is ahead
        (math.pi, (0.0, 1.0), (-1.0, 0.0)),  # facing -y: +y is behind
    ],
)
def test_body_sticks(heading, world, expected):
    assert body_sticks(*world, heading) == pytest.approx(expected, abs=1e-9)


def test_world_and_body_sticks_are_inverses():
    for heading in (0.3, 1.9, -2.4):
        assert world_sticks(*body_sticks(0.2, -0.7, heading), heading) == pytest.approx((0.2, -0.7))


def test_clamp_vec_keeps_direction():
    x, y = clamp_vec(3.0, 4.0, 1.0)
    assert math.hypot(x, y) == pytest.approx(1.0)
    assert x / y == pytest.approx(0.75)


def test_output_is_limited():
    c = PositionController(limit=0.3)
    c.update(Pose(0, 0, 0, 0.0), (10.0, 10.0))
    ux, uy = c.update(Pose(0, 0, 0, 0.1), (10.0, 10.0))
    assert math.hypot(ux, uy) == pytest.approx(0.3)


def settle(gains: Gains, wind: float = 0.2, seconds: float = 30.0) -> float:
    """1-D drone that moves 1.5 m/s per unit stick, pushed by wind; returns where it ends up."""
    c, x, dt = PositionController(gains), 0.0, 1 / 30
    for i in range(round(seconds / dt)):
        u, _ = c.update(Pose(x, 0.0, 0.0, i * dt), (0.0, 0.0))
        x += (1.5 * u + wind) * dt
    return x


def test_integral_cancels_steady_wind():
    assert abs(settle(Gains())) < 0.02
    assert abs(settle(Gains(ki=0.0))) > 0.1  # proportional alone is blown off target


def test_repeated_pose_returns_last_output():
    c = PositionController()
    c.update(Pose(0, 0, 0, 1.0), (1.0, 0.0))
    first = c.update(Pose(0, 0, 0, 1.1), (1.0, 0.0))
    assert c.update(Pose(0, 0, 0, 1.1), (1.0, 0.0)) == first


def test_gains_parse():
    assert Gains.parse("0.5,0.1,0.2") == Gains(0.5, 0.1, 0.2)

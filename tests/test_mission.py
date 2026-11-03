import json
import math

import pytest

from nav.geofence import Geofence
from nav.mission import SHAPES, Mission, build_path


def test_carrot_advances_at_speed_while_the_drone_keeps_up():
    m = Mission((0.0, 0.0), [(1.0, 0.0)], (0.0, 0.0), speed=0.5)
    assert m.advance(1.0, m.carrot()) == pytest.approx((0.5, 0.0))


def test_carrot_waits_for_a_lagging_drone():
    m = Mission((0.0, 0.0), [(2.0, 0.0)], (0.0, 0.0), speed=0.5, max_lead=0.5)
    m.advance(1.0, (0.0, 0.0))
    stuck = m.carrot()
    assert m.advance(1.0, (-1.0, 0.0)) == stuck


def test_visits_every_waypoint_and_ends_home():
    waypoints = [(1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    m = Mission((0.0, 0.0), waypoints, (0.2, 0.2), speed=0.3)
    seen, dt = [], 1 / 30
    for _ in range(2000):
        seen.append(m.advance(dt, m.carrot()))
        if m.done(m.carrot()):
            break
    assert m.done(m.carrot())
    for w in waypoints:
        assert min(math.dist(w, p) for p in seen) < 0.02
    assert seen[-1] == pytest.approx((0.2, 0.2))
    assert len(seen) * dt == pytest.approx(m.total / 0.3, abs=0.1)


@pytest.mark.parametrize("name", list(SHAPES))
def test_built_in_shapes_fit_inside_the_barrier(name):
    inner = Geofence((-2.0, -1.5, 2.0, 1.5)).inner
    pts = build_path(name, inner)
    assert len(pts) >= 5
    assert all(inner[0] <= x <= inner[2] and inner[1] <= y <= inner[3] for x, y in pts)


def test_path_from_json(tmp_path):
    f = tmp_path / "path.json"
    f.write_text(json.dumps([[0.5, 0.5], [-0.5, 0.25]]))
    assert build_path(str(f), (-1, -1, 1, 1)) == [(0.5, 0.5), (-0.5, 0.25)]


def test_carrot_waits_at_a_waypoint_until_the_drone_arrives():
    m = Mission((0.0, 0.0), [(0.5, 0.0), (0.5, 1.0)], (0.0, 0.0), speed=1.0, arrive_tol=0.2)
    m.advance(0.4, (0.0, 0.0))
    for _ in range(10):
        m.advance(0.1, (0.25, 0.0))  # close behind, but not within 0.2 of the corner
    assert m.carrot() == pytest.approx((0.5, 0.0))
    m.advance(0.1, (0.45, 0.0))  # arrived: on to the next leg
    assert m.carrot()[1] > 0

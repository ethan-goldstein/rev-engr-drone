"""Showreel pieces that don't need COLMAP or the detector: map panel, map cache, night brightening."""
import numpy as np

from mapping.reel import PANEL, MapPanel, brighten, load_map, no_map_panel, save_map
from mapping.render import BACKGROUND_BGR, MapData, height_colors


def flight_map() -> MapData:
    rng = np.random.default_rng(0)
    points = rng.uniform([-4, 2, 0], [4, 10, 3], (2000, 3))
    colors, zrange = height_colors(points[:, 2])
    t = np.linspace(0, 20, 60)
    drone = np.column_stack([np.sin(t / 5), t / 4, np.full_like(t, 1.2)])
    person = np.column_stack([np.sin(t / 5), t / 4 + 4, np.full_like(t, 0.85)])
    return MapData(points, colors, drone, person, zrange, [("source", "test")], drone_t=t, person_t=t)


def path_pixels(img: np.ndarray) -> int:
    white = (img == 255).all(axis=2)  # the drone path is pure white; height colors never are
    return int(white.sum())


def test_paths_grow_with_the_flight():
    panel = MapPanel(flight_map(), duration=20.0)
    early, late = panel.draw(1.0), panel.draw(19.0)
    assert early.shape == late.shape == (PANEL[1], PANEL[0], 3)
    assert path_pixels(late) > path_pixels(early) > 0


def test_map_cache_round_trip(tmp_path):
    data = flight_map()
    save_map(tmp_path / "map.npz", data)
    found, back = load_map(tmp_path / "map.npz")
    assert found and np.allclose(back.drone, data.drone) and np.allclose(back.person_t, data.person_t)
    assert back.info == data.info and back.zrange == data.zrange
    save_map(tmp_path / "none.npz", None)
    assert load_map(tmp_path / "none.npz") == (True, None)
    assert load_map(tmp_path / "missing.npz") == (False, None)


def test_night_footage_is_brightened():
    rng = np.random.default_rng(1)
    night = rng.integers(0, 25, (360, 640, 3), dtype=np.uint8)
    assert brighten(night).mean() > 3 * night.mean()


def test_no_map_panel_explains_why():
    img = no_map_panel("too dark to map")
    assert img.shape == (PANEL[1], PANEL[0], 3)
    assert (img != np.array(BACKGROUND_BGR, np.uint8)).any()

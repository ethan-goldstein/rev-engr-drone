"""Post-flight 3D map, on synthetic data: no COLMAP needed.

The synthetic world: flat ground, a wall, one person standing still, and a drone camera flying a level
arc around them. Pushing it through a random similarity (rotation, scale, offset) gives what SfM would
hand back: right shape, arbitrary orientation and units.
"""
import base64
import csv
import json
import math
import re

import cv2
import numpy as np
import pytest

from mapping.geometry import (
    Similarity,
    camera_from_heading,
    estimate_up,
    focal_px,
    ground_height,
    look_at,
    person_positions,
    scale_from_depth,
    scale_from_ground,
)
from mapping.reconstruct import (
    Keyframe,
    Track,
    extract_keyframes,
    load_track,
    main,
    map_from_reconstruction,
    map_from_track,
    pick_sharpest,
    sharpness,
)
from mapping.render import MapData, height_colors, turbo, write_flyaround, write_html
from mapping.sfm import SfmResult

W, H = 640, 360
F = focal_px(W)
PERSON = np.array([0.0, 5.0])  # standing still, 5 m in front of the first camera
SFM_OFFSET = np.array([3.0, -2.0, 5.0])
TRACK_HEADER = ["t", "state", "id", "cx", "cy", "w", "h", "distance_m", "yaw_cmd", "pitch_cmd",
                "drone_x", "drone_y", "drone_heading_deg"]


def person_cloud(rng, n=600):
    a, z = rng.uniform(0, 2 * np.pi, n), rng.uniform(0, 1.7, n)
    return np.column_stack([PERSON[0] + 0.25 * np.cos(a), PERSON[1] + 0.25 * np.sin(a), z])


def scene_points(rng, person=True, ground=True):
    gx, gy = np.meshgrid(np.arange(-6, 6.01, 0.5), np.arange(-2, 12.01, 0.5))
    floor = np.column_stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)])
    wx, wz = np.meshgrid(np.arange(-6, 6.01, 0.25), np.arange(0.25, 3.01, 0.25))
    wall = np.column_stack([wx.ravel(), np.full(wx.size, 12.0), wz.ravel()])
    return np.vstack([floor] * ground + [wall] + [person_cloud(rng)] * person)


def arc_cameras(n=12):
    """Level flight (1.5 m, a few cm of wobble) on an arc around the person, always looking at them."""
    torso = np.array([*PERSON, 0.9])
    C, R = [], []
    for k in range(n):
        th, r = np.radians(60) * k / (n - 1), 5.0 - 0.8 * k / (n - 1)
        c = np.array([PERSON[0] + r * np.sin(th), PERSON[1] - r * np.cos(th), 1.5 + 0.03 * np.sin(3 * k)])
        C.append(c)
        R.append(look_at(c, torso))
    return np.array(C), np.array(R)


def observe_person(C, R):
    """What the follow system logs: normalized box (cx, cy, w, h) and pinhole distance, per camera."""
    body = person_cloud(np.random.default_rng(9), 2000)
    boxes, dist = [], []
    for c, r in zip(C, R):
        Xc = (body - c) @ r
        u, v = F * Xc[:, 0] / Xc[:, 2] + W / 2, F * Xc[:, 1] / Xc[:, 2] + H / 2
        boxes.append([(u.min() + u.max()) / 2 / W, (v.min() + v.max()) / 2 / H, np.ptp(u) / W, np.ptp(v) / H])
        dist.append(((np.array([*PERSON, 0.85]) - c) @ r)[2])
    return np.array(boxes), np.array(dist)


def as_sfm(points, C, R, seed=0, s=0.37):
    """The world as SfM would return it: some rotation, 0.37 units per metre, an offset."""
    q, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    sim = Similarity(s, q, SFM_OFFSET)
    return sim.points(points), sim.points(C), sim.rotations(R), q


def angle_deg(a, b):
    return math.degrees(math.acos(np.clip(np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b), -1, 1)))


def write_track(path, rows):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(TRACK_HEADER)
        for r in rows:
            w.writerow([r.get(k, "") for k in TRACK_HEADER])


# --- keyframes ---------------------------------------------------------------------------------------------

def textured(seed=0):
    rng = np.random.default_rng(seed)
    img = (rng.uniform(0, 1, (H // 8, W // 8)) > 0.5).astype(np.uint8) * 200 + 30
    return cv2.cvtColor(cv2.resize(img, (W, H), interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)


def test_sharpness_drops_with_blur():
    img = textured()
    scores = [sharpness(cv2.GaussianBlur(img, (0, 0), s)) if s else sharpness(img) for s in (0, 1, 2, 4)]
    assert scores == sorted(scores, reverse=True)
    assert scores[0] > 5 * scores[-1]


def test_picks_the_sharpest_frame_in_each_third_of_a_second():
    times = np.arange(30) / 15  # 2 s at 15 fps: six windows of five frames
    scores = np.ones(30)
    best = [2, 6, 14, 15, 23, 29]
    scores[best] = 10.0
    assert pick_sharpest(times, scores, 1 / 3) == best


def test_one_frame_per_window_even_with_uneven_timing():
    times = np.array([0.0, 0.05, 0.31, 0.9, 0.95, 2.0])  # the drone's frame rate wobbles; 1.0-2.0 s is a gap
    scores = np.array([1, 2, 3, 5, 4, 1.0])
    assert pick_sharpest(times, scores, 1 / 3) == [2, 3, 5]


def test_extracts_the_sharp_frames_from_a_video(tmp_path):
    video = tmp_path / "raw.mp4"
    out = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 15, (W, H))
    sharp = {2, 7, 12, 17}  # one per 5-frame window, away from the window edges
    for i in range(20):
        img = textured(i)
        out.write(img if i in sharp else cv2.GaussianBlur(img, (0, 0), 3))
    out.release()
    times = {i: round(i / 15, 3) for i in range(20)}  # as frames.csv stores them

    keys, size = extract_keyframes(video, times, tmp_path / "images", 1 / 3)
    assert size == (W, H)
    assert [k.frame for k in keys] == sorted(sharp)
    assert [k.t for k in keys] == [times[i] for i in sorted(sharp)]
    for k in keys:
        assert cv2.imread(str(tmp_path / "images" / k.name)).shape == (H, W, 3)


# --- height coloring ---------------------------------------------------------------------------------------

def test_height_colors_run_blue_to_red():
    z = np.linspace(0, 10, 101)
    colors, (lo, hi) = height_colors(z)
    assert colors.shape == (101, 3) and colors.dtype == np.uint8
    assert lo == pytest.approx(0.2) and hi == pytest.approx(9.8)
    assert (colors[0] == turbo([0.0])[0]).all() and (colors[-1] == turbo([1.0])[0]).all()
    low, mid, high = colors[0].astype(int), colors[50].astype(int), colors[-1].astype(int)
    assert low[2] > low[0]  # blue end
    assert high[0] > high[2]  # red end
    assert mid[1] == mid.max()  # green in the middle


def test_a_stray_point_does_not_flatten_the_colors():
    z = np.concatenate([np.random.default_rng(0).uniform(0, 2, 999), [500.0]])
    colors, (lo, hi) = height_colors(z)
    assert hi < 3
    bottom, top = colors[np.argmin(z[:-1])].astype(int), colors[np.argmax(z[:-1])].astype(int)
    assert bottom[2] > 150 > bottom[0] and top[0] > 150 > top[2]


# --- up axis, scale, alignment -----------------------------------------------------------------------------

def test_up_comes_from_the_ground():
    C, R = arc_cameras()
    pts, Cs, Rs, q = as_sfm(scene_points(np.random.default_rng(0)), C, R)
    up, source = estimate_up(Cs, Rs, pts)
    assert source == "ground plane"
    assert angle_deg(up, q @ [0, 0, 1]) < 0.5


def test_without_ground_up_comes_from_the_level_flight_path():
    C, R = arc_cameras()
    _, Cs, Rs, q = as_sfm(np.zeros((0, 3)), C, R)
    up, source = estimate_up(Cs, Rs)
    assert source == "flight path plane"
    assert angle_deg(up, q @ [0, 0, 1]) < 2


def line_flight(flight):
    rng = np.random.default_rng(1)
    if flight == "hover":
        C = np.array([0.0, 0.0, 1.2]) + rng.normal(0, 0.05, (10, 3))
    else:
        C = np.array([[0, 0, z] for z in np.linspace(0.2, 1.5, 5)] + [[0, y, 1.5] for y in np.linspace(0.3, 3, 10)])
    tilt = math.radians(8)  # camera pitched a little down
    return C, np.array([look_at(c, c + [0, 1, -math.tan(tilt)]) for c in C])


@pytest.mark.parametrize("flight", ["climb then straight", "hover"])
def test_up_falls_back_to_the_camera_when_nothing_else_says(flight):
    C, R = line_flight(flight)
    _, Cs, Rs, q = as_sfm(np.zeros((0, 3)), C, R, seed=3)
    up, source = estimate_up(Cs, Rs)
    assert source == "camera tilt"
    assert angle_deg(up, q @ [0, 0, 1]) < 10


def test_a_wall_is_not_the_ground():
    C, R = line_flight("climb then straight")
    walls = scene_points(np.random.default_rng(0), person=False, ground=False)
    pts, Cs, Rs, _ = as_sfm(walls, C, R, seed=3)
    assert estimate_up(Cs, Rs, pts)[1] == "camera tilt"


def test_ground_height_ignores_stray_points_below_the_floor():
    C, R = arc_cameras()
    junk = np.random.default_rng(5).uniform([-5, 0, -3], [5, 10, -0.5], (40, 3))  # bad triangulations
    pts, Cs, _, q = as_sfm(np.vstack([scene_points(np.random.default_rng(0)), junk]), C, R)
    ground, found = ground_height(pts, Cs, q @ [0, 0, 1])
    assert found
    assert ground == pytest.approx(SFM_OFFSET @ (q @ [0, 0, 1]), abs=0.01)  # where z = 0 went


def test_scale_from_the_person_standing_on_the_ground():
    """The usual case: the person keeps walking, so COLMAP drops them and only the ground and wall are left."""
    C, R = arc_cameras()
    boxes, dist = observe_person(C, R)
    pts, Cs, Rs, q = as_sfm(scene_points(np.random.default_rng(0), person=False), C, R)
    up = q @ [0, 0, 1]
    ground, _ = ground_height(pts, Cs, up)
    scale, used = scale_from_ground(Cs, Rs, boxes, dist, F, W, H, up, ground)
    assert used == len(C)
    assert scale == pytest.approx(1 / 0.37, rel=0.03)
    # Depth around the box would measure the wall behind them: the map would come out far too small.
    behind, _ = scale_from_depth(pts, Cs, Rs, boxes, dist, F, W, H)
    assert behind < 0.7 / 0.37


def test_scale_from_depth_when_the_person_was_reconstructed():
    C, R = arc_cameras()
    boxes, dist = observe_person(C, R)
    pts, Cs, Rs, _ = as_sfm(scene_points(np.random.default_rng(0)), C, R)
    scale, used = scale_from_depth(pts, Cs, Rs, boxes, dist, F, W, H)
    assert used == len(C)
    assert scale == pytest.approx(1 / 0.37, rel=0.08)


def test_no_depth_scale_without_points_around_the_person():
    C, R = arc_cameras()
    boxes, dist = observe_person(C, R)
    far_away = np.array([[50.0, -50.0, 0.0]] * 20)
    assert scale_from_depth(far_away, C, R, boxes, dist, F, W, H) == (None, 0)


def sfm_flight(drop=(), **scene):
    """The arc flight as COLMAP + the recorder would leave it. `drop`: keyframes COLMAP failed to register."""
    C, R = arc_cameras()
    boxes, dist = observe_person(C, R)
    pts, Cs, Rs, _ = as_sfm(scene_points(np.random.default_rng(0), **scene), C, R)
    keys = [Keyframe(5 * k, round(k / 3, 3), f"{5 * k:06d}.jpg") for k in range(len(C))]
    cams = {k.name: (Cs[i], Rs[i]) for i, k in enumerate(keys) if i not in drop}
    track = Track(
        t=np.array([k.t for k in keys]),
        state=np.array(["LOCKED"] * len(keys), dtype=object),
        cx=boxes[:, 0], cy=boxes[:, 1], w=boxes[:, 2], h=boxes[:, 3], dist=dist,
        drone_x=np.full(len(keys), np.nan), drone_y=np.full(len(keys), np.nan), heading=np.full(len(keys), np.nan),
    )
    return SfmResult(pts, cams, F, 1), keys, track, C


def test_map_frame_is_metric_z_up_and_starts_at_the_takeoff_spot():
    sfm, keys, track, C = sfm_flight(drop={4})
    data = map_from_reconstruction(sfm, keys, track, W, H)
    truth = np.delete(C, 4, axis=0)  # first camera is at (0, 0, 1.5) facing +y, ground at z = 0
    assert len(data.drone) == len(truth)
    assert np.abs(data.drone - truth).max() < 0.1
    assert (np.abs(data.points[:, 2]) < 0.02).sum() >= 600  # the floor (725 points, a few trimmed) is at z = 0
    assert dict(data.info)["up from"] == "ground plane"
    assert dict(data.info)["scale from"] == f"person on the ground, {len(truth)} frames"


def test_without_ground_scale_falls_back_to_depth_around_the_person():
    sfm, keys, track, C = sfm_flight(ground=False)
    data = map_from_reconstruction(sfm, keys, track, W, H)
    assert dict(data.info)["scale from"] == f"points around the person, {len(keys)} frames"
    assert np.abs(data.drone[:, :2] - C[:, :2]).max() < 0.3


def test_person_path_lands_where_they_stood():
    sfm, keys, track, _ = sfm_flight()
    data = map_from_reconstruction(sfm, keys, track, W, H)
    assert len(data.person) == len(keys)
    err = np.linalg.norm(data.person - [*PERSON, 0.85], axis=1)
    assert err.max() < 0.4


def test_coasting_cut_off_and_lost_boxes():
    sfm, keys, track, _ = sfm_flight()
    track.state[[0, 1]] = "LOST"
    track.state[2] = "COASTING"  # still goes on the path, but not into the scale
    track.cy[3], track.h[3] = 0.6, 0.85  # bottom of the box at the frame edge: distance reads long
    data = map_from_reconstruction(sfm, keys, track, W, H)
    assert len(data.person) == len(keys) - 3
    assert dict(data.info)["scale from"] == f"person on the ground, {len(keys) - 4} frames"


# --- person path from the track (no SfM) -------------------------------------------------------------------

def test_person_sits_on_the_box_center_ray_at_distance_m():
    R = camera_from_heading(0.0)[None]  # facing +y
    C = np.array([[1.0, 2.0, 1.2]])
    center = person_positions(C, R, [0.5], [0.5], [4.0], F, W, H)[0]
    assert center == pytest.approx([1.0, 6.0, 1.2])
    right_edge = person_positions(C, R, [1.0], [0.5], [4.0], F, W, H)[0]  # distance_m is depth, not range
    assert right_edge == pytest.approx([1.0 + 4 * math.tan(math.radians(30)), 6.0, 1.2])
    low = person_positions(C, R, [0.5], [0.75], [4.0], F, W, H)[0]
    assert low[2] < 1.2


def test_camera_from_heading_matches_nav_convention():
    R = camera_from_heading(math.radians(90))  # heading clockwise from +y: 90 faces +x
    assert R[:, 2] == pytest.approx([1, 0, 0], abs=1e-12)  # forward
    assert R[:, 0] == pytest.approx([0, -1, 0], abs=1e-12)  # right
    assert R == pytest.approx(look_at([0, 0, 1], [5, 0, 1]))


def test_from_track_maps_drone_and_person(tmp_path):
    rows = [{"t": "0.000", "state": "OFF"}]  # before follow was switched on: no box, no position
    for i in range(30):  # flying +x, facing +x, person 3 m straight ahead
        rows.append({"t": f"{(i + 1) / 15:.3f}", "state": "LOCKED", "id": "1", "cx": "0.5", "cy": "0.5",
                     "w": "0.1", "h": "0.6", "distance_m": "3.00", "drone_x": f"{1 + 0.05 * i:.3f}",
                     "drone_y": "2.000", "drone_heading_deg": "90.0"})
    rows.append({"t": "2.100", "state": "LOST", "drone_x": "2.5", "drone_y": "2.0", "drone_heading_deg": "90.0"})
    write_track(tmp_path / "track.csv", rows)

    data = map_from_track(load_track(tmp_path / "track.csv"), height=1.2)
    assert len(data.drone) == 31 and len(data.person) == 30 and len(data.points) == 0
    assert data.drone[0] == pytest.approx([1.0, 2.0, 1.2], abs=0.15)  # display smoothing pulls the ends in a bit
    assert data.person[:, 0] == pytest.approx(data.drone[:30, 0] + 3.0, abs=0.1)
    assert data.person[:, 1:] == pytest.approx(np.tile([2.0, 1.2], (30, 1)), abs=1e-6)


def test_from_track_needs_positions(tmp_path):
    write_track(tmp_path / "track.csv", [{"t": "0.1", "state": "LOCKED", "cx": "0.5", "cy": "0.5", "h": "0.5",
                                          "distance_m": "4"}])
    with pytest.raises(SystemExit, match="drone_x"):
        map_from_track(load_track(tmp_path / "track.csv"))


def test_track_rows_are_matched_by_time():
    tr = Track.empty()
    tr.t = np.array([0.0, 0.1, 0.2, 0.5])
    assert tr.nearest([0.04, 0.06, 0.19, 0.38, 2.0]).tolist() == [0, 1, 2, 3, -1]


# --- outputs -----------------------------------------------------------------------------------------------

def small_map():
    pts = np.array([[0, 0, 0], [1, 2, 0.5], [-1, 3, 2.0]], float)
    colors, zr = height_colors(pts[:, 2])
    return MapData(pts, colors, np.array([[0, 0, 1.2], [0.5, 0.5, 1.2]]), np.array([[0, 4, 0.9], [1, 4, 0.9]]), zr,
                   [("source", "test"), ("note", "a <b>bold</b> & claim")])


def embedded(page):
    return json.loads(re.search(r'<script type="application/json" id="map-data">(.*?)</script>', page, re.S).group(1))


def test_html_is_a_single_file_viewer_with_pinned_three(tmp_path):
    data = small_map()
    write_html(tmp_path / "map.html", data, "run1 - 3D map")
    page = (tmp_path / "map.html").read_text()
    assert "https://cdnjs.cloudflare.com/ajax/libs/three.js/0.180.0/three.module.min.js" in page
    assert "https://cdn.jsdelivr.net/npm/three@0.180.0/examples/jsm/controls/OrbitControls.js" in page
    assert page.count('"sha384-') == 3  # integrity for three, its core, and OrbitControls
    assert "/*MAP_DATA*/" not in page and "__TITLE__" not in page

    blob = embedded(page)
    pts = np.frombuffer(base64.b64decode(blob["points"]), "<f4").reshape(-1, 3)
    assert pts == pytest.approx(data.points)
    assert np.frombuffer(base64.b64decode(blob["colors"]), np.uint8).reshape(-1, 3).tolist() == data.colors.tolist()
    assert blob["drone"] == data.drone.tolist() and blob["person"] == data.person.tolist()
    assert blob["info"][1] == ["note", "a <b>bold</b> & claim"]


def test_html_data_cannot_break_out_of_its_script_tag(tmp_path):
    data = small_map()
    data.info.append(("x", "</script><script>alert(1)</script>"))
    write_html(tmp_path / "map.html", data, "</title><script>alert(2)</script>")
    page = (tmp_path / "map.html").read_text()
    assert "alert(1)</script>" not in page and "<script>alert(2)" not in page
    assert embedded(page)["title"] == "</title><script>alert(2)</script>"


def test_flyaround_video(tmp_path):
    n = write_flyaround(tmp_path / "map.mp4", small_map(), "t", size=(320, 180), seconds=1, fps=10)
    cap = cv2.VideoCapture(str(tmp_path / "map.mp4"))
    frames = []
    while True:
        ok, img = cap.read()
        if not ok:
            break
        frames.append(img)
    assert n == len(frames) == 10
    assert frames[0].shape == (180, 320, 3)
    assert all((f.max(axis=2) > 120).sum() > 50 for f in frames)  # paths and points, not just background


def test_cli_from_track_writes_html_and_video(tmp_path, capsys):
    run = tmp_path / "run7"
    run.mkdir()
    rows = [{"t": f"{i / 15:.3f}", "state": "LOCKED", "id": "1", "cx": "0.5", "cy": "0.5", "w": "0.1", "h": "0.6",
             "distance_m": "4", "drone_x": f"{math.sin(i / 20):.3f}", "drone_y": f"{i / 30:.3f}",
             "drone_heading_deg": "0"} for i in range(45)]
    write_track(run / "track.csv", rows)
    assert main([str(run), "--from-track"]) == 0
    assert embedded((run / "map.html").read_text())["title"] == "run7 - 3D map"
    assert cv2.VideoCapture(str(run / "map.mp4")).get(cv2.CAP_PROP_FRAME_COUNT) > 0
    assert "person samples: 45" in capsys.readouterr().out


def test_cli_without_footage_points_at_from_track(tmp_path):
    with pytest.raises(SystemExit, match="--from-track"):
        main([str(tmp_path)])

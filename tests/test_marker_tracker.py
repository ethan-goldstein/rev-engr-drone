"""Tracking on synthetic camera frames: a printed tag pasted at a known spot and angle."""
import math

import cv2
import numpy as np
import pytest

from nav.arena import Arena
from nav.marker_tracker import DICTIONARY, MARKER_ID, MarkerTracker, make_detector
from nav.setup import marker_image, png_with_dpi

# A 4 m x 3 m area filling most of a 1000x800 frame: 200 px per meter, centered at pixel (500, 400).
ARENA = Arena(4.0, 3.0, ((100.0, 100.0), (900.0, 100.0), (900.0, 700.0), (100.0, 700.0)))


def frame_with_tag(center_px, screen_rotation_deg, side=90):
    """White frame with the tag rotated clockwise on screen by `screen_rotation_deg`."""
    frame = np.full((800, 1000, 3), 255, np.uint8)
    tag = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(DICTIONARY), MARKER_ID, side)
    patch = np.full((side * 2, side * 2), 255, np.uint8)
    patch[side // 2 : side // 2 + side, side // 2 : side // 2 + side] = tag
    rot = cv2.getRotationMatrix2D((side, side), -screen_rotation_deg, 1.0)  # OpenCV: positive = counterclockwise
    patch = cv2.warpAffine(patch, rot, (side * 2, side * 2), borderValue=255)
    x, y = int(center_px[0]) - side, int(center_px[1]) - side
    frame[y : y + 2 * side, x : x + 2 * side] = cv2.cvtColor(patch, cv2.COLOR_GRAY2BGR)
    return frame


@pytest.mark.parametrize(
    "rotation, heading_deg",
    [(0, 0), (90, 90), (180, 180), (-45, -45)],  # tag top pointing up the screen = facing +y
)
def test_pose_from_a_tag(rotation, heading_deg):
    tracker = MarkerTracker(ARENA)
    pose = tracker.process(frame_with_tag((700, 250), rotation), now=5.0)
    assert pose is not None
    assert (pose.x, pose.y) == pytest.approx((1.0, 0.75), abs=0.03)
    diff = (math.degrees(pose.heading) - heading_deg + 180) % 360 - 180
    assert abs(diff) < 3


def test_no_tag_no_pose():
    tracker = MarkerTracker(ARENA)
    assert tracker.process(np.full((800, 1000, 3), 255, np.uint8), now=1.0) is None
    assert tracker.latest() is None
    assert tracker.latest_frame() is not None  # the annotated view still updates


def test_old_poses_go_stale():
    t = [5.0]
    tracker = MarkerTracker(ARENA, clock=lambda: t[0])
    tracker.process(frame_with_tag((500, 400), 0), now=5.0)
    t[0] = 5.2
    assert tracker.latest() is not None
    t[0] = 5.5
    assert tracker.latest() is None


def test_arena_pixels_and_meters_round_trip(tmp_path):
    assert ARENA.to_world((500, 400)) == pytest.approx((0.0, 0.0), abs=1e-4)
    assert ARENA.to_world((900, 100)) == pytest.approx((2.0, 1.5), abs=1e-4)
    assert ARENA.to_pixel((1.0, -0.5)) == pytest.approx((700, 500), abs=1e-3)
    path = tmp_path / "arena.json"
    ARENA.save(path)
    assert Arena.load(path) == ARENA


def test_printable_marker_is_detectable_and_carries_dpi():
    img = marker_image(5.0)
    _, ids, _ = make_detector().detectMarkers(img)
    assert ids is not None and ids.ravel().tolist() == [MARKER_ID]
    data = png_with_dpi(img)
    assert b"pHYs" in data[:64]
    assert cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE).shape == img.shape

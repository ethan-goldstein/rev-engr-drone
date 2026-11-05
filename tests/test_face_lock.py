"""The FACE LOCK card: profile file, face finding from pose points, and the panel on the drone view."""
import json

import numpy as np
import pytest

from follow.pose_command import FACE, FACE_LEFT_SIDE, NOSE, ArmsUpCommand, face_box, sim_pose
from follow.profile import MAX_ROWS, MAX_VALUE, load_profile
from tests.test_follow_closed_loop import FollowRig, standing_person
from ui.target_card import draw_target_card


def test_profile_rows_keep_their_order_in_capitals(tmp_path):
    path = tmp_path / "p.json"
    path.write_text(json.dumps({"Name": "Jane Doe", "Role": "pilot", "Occupation": "Student / SWE", "Flights": 12}))
    assert load_profile(path) == [("NAME", "JANE DOE"), ("ROLE", "PILOT"), ("OCCUPATION", "STUDENT / SWE"), ("FLIGHTS", "12")]


def test_profile_is_trimmed_to_fit_the_drone_view(tmp_path):
    path = tmp_path / "p.json"
    path.write_text(json.dumps({f"K{i}": "x" * 40 for i in range(MAX_ROWS + 3)} | {"K0": "café"}))
    rows = load_profile(path)
    assert len(rows) == MAX_ROWS
    assert rows[0] == ("K0", "CAF?")  # OpenCV can only draw ASCII
    assert all(len(v) <= MAX_VALUE for _, v in rows)


@pytest.mark.parametrize("text", ["[1, 2]", '{"NAME": {"first": "E"}}', "not json"])
def test_bad_profile_stops_with_a_message(tmp_path, text):
    path = tmp_path / "p.json"
    path.write_text(text)
    with pytest.raises(SystemExit, match="--profile"):
        load_profile(path)


def face(facing: bool, visibility: float = 1.0):
    lm = [(0.5, 0.5, 1.0)] * 33
    side = 1 if facing else -1
    for i in FACE:
        lm[i] = (0.5 + side * (0.1 if i in FACE_LEFT_SIDE else -0.1), 0.2, visibility)
    lm[NOSE] = (0.5, 0.22, visibility)
    return lm


def test_face_box_only_when_turned_toward_the_drone():
    rect = (100, 50, 200, 250)  # the crop the points are relative to
    x0, y0, x1, y1 = face_box(face(True), rect)
    assert x0 < 150 < x1 and y0 < 50 + 0.22 * 200 < y1  # around the nose, in frame pixels
    assert face_box(face(False), rect) is None  # back of the head
    assert face_box(face(True, visibility=0.3), rect) is None


@pytest.mark.parametrize("box_x", [100, 560])
def test_card_fits_beside_the_box(box_x):
    img = np.zeros((360, 640, 3), np.uint8)
    box = (box_x - 30, 60, box_x + 30, 300)
    rows = [("NAME", "JANE DOE"), ("OCCUPATION", "STUDENT / SWE")]
    x0, y0, x1, y1 = draw_target_card(img, box, "FACE LOCK", rows, [("RANGE", "4.0 M")], (80, 220, 80))
    assert 0 <= x0 and x1 <= 640 and 0 <= y0 and y1 <= 360
    assert x0 >= box[2] if box_x == 100 else x1 <= box[0]  # right of the box, unless that runs off the frame
    assert img[y0 + 5, x0 + 5].any()


def rig_with_card():
    rig = FollowRig(people=[standing_person()])
    rig.follow.arms_up = ArmsUpCommand(sim_pose(rig.people[0]))
    rig.follow.profile = [("NAME", "JANE DOE")]
    rig.start()
    rig.run(2.5)
    return rig


def test_face_lock_while_facing_and_body_lock_while_walking_away():
    rig = rig_with_card()
    assert rig.follow.face is not None
    rig.follow.draw(rig.scene.render())
    person = rig.people[0]
    person.waypoints, person.speed = [(0.0, 12.0)], 0.5  # walks straight away from the drone
    person._goal = person.waypoints[0]
    rig.run(2.0)
    assert person.facing_drone is False and rig.follow.face is None
    rig.follow.draw(rig.scene.render())


def test_lock_timer_counts_from_the_lock():
    rig = rig_with_card()
    rows = dict(rig.follow._live_rows())
    assert rows["TRACK"] == f"#{rig.follow.tracker.target_label}" and rows["LOCKED"] == "00:02"
    rig.follow.stop()
    assert rig.follow.face is None and rig.follow._locked_at is None


def test_warns_when_only_the_legs_are_in_view():
    from follow.types import Detection

    rig = rig_with_card()
    assert rig.follow.view_warning() is None  # whole body in view
    target = rig.follow.tracker.target
    target.last_det = Detection(300, 0, 340, 140)  # cut off at the top, short: camera aimed at the ground
    assert rig.follow.view_warning() == "HEAD OUT OF VIEW: TILT CAMERA UP"
    target.last_det = Detection(250, 0, 390, 330)  # cut off and filling the frame: just too close
    assert rig.follow.view_warning() == "TOO CLOSE: STEP BACK"

"""Real MediaPipe models on a frame from a real night flight (a seated person, seen from above).

tests/images/person.jpg isn't in git (it shows a real person and place); that test skips without it."""
from pathlib import Path

import cv2
import numpy as np
import pytest

from follow.detector import DEFAULT_MODEL as PERSON_MODEL
from follow.detector import PersonDetector
from follow.pose_command import DEFAULT_MODEL as POSE_MODEL
from follow.pose_command import ArmsUpCommand, MediaPipePose, arms_up
from follow.types import Detection

IMAGE = Path(__file__).parent / "images" / "person.jpg"


@pytest.mark.skipif(not PERSON_MODEL.exists() or not IMAGE.exists(), reason="person model or test image missing")
def test_finds_the_person_at_night():
    frame = cv2.imread(str(IMAGE))
    found = PersonDetector().detect(frame, 0.0)
    assert len(found) == 1
    d = found[0]
    assert d.score >= 0.4
    assert 0.55 < d.cx / frame.shape[1] < 0.75 and 0.45 < d.cy / frame.shape[0] < 0.6


@pytest.mark.skipif(not PERSON_MODEL.exists(), reason="person model missing")
def test_empty_scene_has_nobody():
    assert PersonDetector().detect(np.full((360, 640, 3), 90, np.uint8), 0.0) == []


@pytest.mark.skipif(not POSE_MODEL.exists(), reason="pose model missing")
def test_pose_model_runs_on_a_crop():
    pose = MediaPipePose()
    assert pose(np.full((200, 120, 3), 90, np.uint8)) is None  # no body, no landmarks


def body(wrists_y: float, visibility: float = 1.0):
    lm = [(0.5, 0.5, 1.0)] * 33
    lm[0] = (0.5, 0.2, 1.0)  # nose
    lm[15], lm[16] = (0.3, wrists_y, visibility), (0.7, wrists_y, visibility)
    return lm


def test_arms_up_needs_both_wrists_above_the_nose_and_visible():
    assert arms_up(body(0.1))
    assert not arms_up(body(0.6))
    assert not arms_up(body(0.1, visibility=0.2))
    one_up = body(0.6)
    one_up[15] = (0.3, 0.1, 1.0)
    assert not arms_up(one_up)


def test_arms_up_must_be_held():
    state = {"up": True}
    cmd = ArmsUpCommand(lambda crop: body(0.1 if state["up"] else 0.6), hold_s=1.5)
    frame = np.zeros((360, 640, 3), np.uint8)
    box = Detection(280, 60, 360, 300)
    fired = [cmd.update(frame, box, i / 15) for i in range(30)]
    assert fired.count(True) == 1 and fired.index(True) == 23  # 1.5 s at 15 fps
    state["up"] = False
    assert not cmd.update(frame, box, 2.1) and cmd.progress == 0

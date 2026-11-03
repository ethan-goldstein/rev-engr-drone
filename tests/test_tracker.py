"""Runs the real MediaPipe model on Google's sample gesture photos."""
from pathlib import Path

import cv2
import numpy as np
import pytest

from gesture.mapper import IntentKind, map_observation
from gesture.tracker import DEFAULT_MODEL, GestureTracker

IMAGES = Path(__file__).parent / "images"

pytestmark = pytest.mark.skipif(not DEFAULT_MODEL.exists(), reason="gesture model not downloaded")


@pytest.fixture
def tracker():
    with GestureTracker() as t:
        yield t


def test_blank_frame_has_no_hand(tracker):
    assert tracker.process(np.zeros((480, 640, 3), np.uint8), 0) is None


@pytest.mark.parametrize(
    "name, gesture, label",
    [
        ("thumbs_up", "Thumb_Up", "forward"),
        ("victory", "Victory", "back"),
        ("pointing_up", "Pointing_Up", "turn"),
        ("thumbs_down", "Thumb_Down", "hover"),
    ],
)
def test_sample_photos(tracker, name, gesture, label):
    path = IMAGES / f"{name}.jpg"
    if not path.exists():
        pytest.skip(f"{path.name} not downloaded")
    obs = tracker.process(cv2.imread(str(path)), 0)
    assert obs is not None and len(obs.landmarks) == 21
    assert obs.gesture == gesture
    intent = map_observation(obs)
    assert intent.label == label
    assert intent.kind is not IntentKind.NONE

from pathlib import Path

import cv2
import numpy as np
import pytest

from flight.light import DARK, LightCheck

FLIGHTS = Path(__file__).parent.parent / "flights"


def frame(level: int) -> np.ndarray:
    return np.full((360, 640, 3), level, np.uint8)


def test_no_video_is_not_called_dark():
    light = LightCheck()
    assert light.brightness is None and not light.too_dark
    light.update(None)
    assert not light.too_dark


def test_night_is_too_dark_and_daylight_is_fine():
    night, day = LightCheck(), LightCheck()
    for _ in range(15):
        night.update(frame(15))
        day.update(frame(130))
    assert night.too_dark and not day.too_dark


def test_one_bright_frame_doesnt_clear_a_dark_scene():
    light = LightCheck()
    for level in [12] * 10 + [200]:  # a lamp swinging into view
        light.update(frame(level))
    assert light.too_dark


@pytest.mark.parametrize("run, dark", [("field4", False), ("field8", True)])
def test_real_flights(run, dark):
    video = FLIGHTS / run / "raw.mp4"
    if not video.exists():
        pytest.skip(f"{run} recording not on this machine")
    cap, light = cv2.VideoCapture(str(video)), LightCheck()
    for _ in range(15):
        ok, f = cap.read()
        if ok:
            light.update(f)
    assert light.too_dark is dark, f"{run} brightness {light.brightness:.0f} vs threshold {DARK}"

import pytest

from gesture.drag import DRAG_MAPPER_CONFIG, DragAnchor, DragConfig
from gesture.mapper import IntentKind, map_observation
from tests.fixtures import hand

DT = 1 / 30
CFG = DragConfig()


def feed(drag, palm, start, seconds, gesture="Open_Palm"):
    """Feed the same palm position for `seconds`; return (last intent, end time)."""
    t, intent = start, None
    for _ in range(round(seconds / DT)):
        obs = hand(gesture, palm=palm) if palm else None
        anchor = drag.update(obs, t)
        intent = map_observation(obs, DRAG_MAPPER_CONFIG, anchor or (0.5, 0.5))
        t += DT
    return intent, t


def test_palm_anchors_where_it_first_appears():
    drag = DragAnchor()
    intent, _ = feed(drag, (0.8, 0.3), 0.0, 0.1)
    assert drag.anchor == pytest.approx((0.8, 0.3))
    assert intent.axes.is_zero()


def test_moving_palm_steers_then_resting_palm_stops():
    drag = DragAnchor()
    _, t = feed(drag, (0.5, 0.5), 0.0, 0.5)
    intent, t = feed(drag, (0.65, 0.5), t, DT)  # palm jumps right
    assert intent.kind is IntentKind.MOVE and intent.axes.roll > 0
    intent, t = feed(drag, (0.65, 0.5), t, CFG.still_s + 2 * DT)  # ...and rests there
    assert intent.axes.is_zero()
    assert drag.anchor == pytest.approx((0.65, 0.5))


def test_palm_moving_up_climbs():
    drag = DragAnchor()
    _, t = feed(drag, (0.5, 0.5), 0.0, 0.5)
    intent, _ = feed(drag, (0.5, 0.35), t, DT)
    assert intent.axes.throttle > 0 and intent.axes.roll == 0


def test_slow_sag_is_ignored():
    drag = DragAnchor()
    t = 0.0
    for i in range(90):  # 3 s of the palm sinking 0.02 per second
        obs = hand("Open_Palm", palm=(0.5, 0.5 + 0.02 * i * DT))
        intent = map_observation(obs, DRAG_MAPPER_CONFIG, drag.update(obs, t))
        assert intent.axes.is_zero()
        t += DT


def test_dropped_frame_keeps_the_anchor_but_a_lowered_hand_resets_it():
    drag = DragAnchor()
    _, t = feed(drag, (0.3, 0.3), 0.0, 0.2)
    _, t = feed(drag, None, t, 2 * DT)
    assert drag.anchor == pytest.approx((0.3, 0.3))
    _, t = feed(drag, None, t, CFG.release_s + DT)
    assert drag.anchor is None
    _, t = feed(drag, (0.7, 0.6), t, DT)
    assert drag.anchor == pytest.approx((0.7, 0.6))


def test_takeoff_still_needs_the_screen_center():
    drag = DragAnchor()
    off_center, _ = feed(drag, (0.85, 0.5), 0.0, 0.2)
    centered, _ = feed(DragAnchor(), (0.5, 0.5), 0.0, 0.2)
    assert not off_center.takeoff_ready
    assert centered.takeoff_ready


def test_thumbs_down_resets_the_anchor_without_hiding_the_hand():
    drag = DragAnchor()
    _, t = feed(drag, (0.7, 0.5), 0.0, 0.2)
    _, t = feed(drag, (0.4, 0.5), t, CFG.release_s + DT, gesture="Thumb_Down")  # move back while hovering
    assert drag.anchor is None
    intent, _ = feed(drag, (0.4, 0.5), t, DT)
    assert drag.anchor == pytest.approx((0.4, 0.5)) and intent.axes.is_zero()

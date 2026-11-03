import pytest

from gesture.mapper import IntentKind, MapperConfig, map_observation, stick
from tests.fixtures import hand

CFG = MapperConfig()


def test_no_hand_is_none():
    assert map_observation(None).kind is IntentKind.NONE


def test_centered_open_palm_hovers_and_is_takeoff_ready():
    intent = map_observation(hand("Open_Palm"))
    assert intent.kind is IntentKind.HOVER
    assert intent.label == "steer"
    assert intent.axes.is_zero()
    assert intent.takeoff_ready


@pytest.mark.parametrize(
    "palm, axis, sign",
    [
        ((0.85, 0.5), "roll", +1),  # hand right of center -> roll right
        ((0.15, 0.5), "roll", -1),
        ((0.5, 0.15), "throttle", +1),  # hand above center -> climb
        ((0.5, 0.85), "throttle", -1),
    ],
)
def test_open_palm_offsets_steer(palm, axis, sign):
    intent = map_observation(hand("Open_Palm", palm=palm))
    assert intent.kind is IntentKind.MOVE
    assert not intent.takeoff_ready
    assert getattr(intent.axes, axis) == pytest.approx(sign * 1.0)
    others = {"throttle", "yaw", "pitch", "roll"} - {axis}
    assert all(getattr(intent.axes, a) == 0 for a in others)


def test_thumb_up_forward_and_victory_back():
    assert map_observation(hand("Thumb_Up")).axes.pitch == 1.0
    assert map_observation(hand("Victory")).axes.pitch == -1.0


def test_pointing_up_turns_toward_finger():
    right = map_observation(hand("Pointing_Up", tip=(0.8, 0.3)))
    left = map_observation(hand("Pointing_Up", tip=(0.2, 0.3)))
    centered = map_observation(hand("Pointing_Up", tip=(0.52, 0.3)))
    assert right.axes.yaw > 0 > left.axes.yaw
    assert centered.kind is IntentKind.HOVER and centered.axes.is_zero()


def test_fist_lands():
    assert map_observation(hand("Closed_Fist")).kind is IntentKind.LAND


@pytest.mark.parametrize("gesture", [None, "Thumb_Down", "ILoveYou"])
def test_other_gestures_hover(gesture):
    intent = map_observation(hand(gesture, palm=(0.9, 0.1)))
    assert intent.kind is IntentKind.HOVER and intent.axes.is_zero()


def test_low_confidence_gesture_is_ignored():
    intent = map_observation(hand("Closed_Fist", score=0.3))
    assert intent.kind is IntentKind.HOVER


def test_stick_dead_zone_and_saturation():
    assert stick(0.05, CFG) == 0.0
    assert stick(CFG.dead_zone + CFG.span / 2, CFG) == pytest.approx(0.5)
    assert stick(-0.49, CFG) == -1.0


UPDOWN = MapperConfig(vertical_only=True)


@pytest.mark.parametrize("palm, throttle", [((0.9, 0.15), 1.0), ((0.1, 0.85), -1.0), ((0.9, 0.5), 0.0)])
def test_updown_palm_only_climbs_or_descends(palm, throttle):
    intent = map_observation(hand("Open_Palm", palm=palm), UPDOWN)
    assert intent.axes.throttle == pytest.approx(throttle)
    assert intent.axes.roll == intent.axes.pitch == intent.axes.yaw == 0


@pytest.mark.parametrize("gesture", ["Thumb_Up", "Victory", "Pointing_Up"])
def test_updown_ignores_other_movement_gestures(gesture):
    intent = map_observation(hand(gesture, tip=(0.9, 0.3)), UPDOWN)
    assert intent.kind is IntentKind.HOVER and intent.axes.is_zero()


def test_updown_fist_lands_and_center_palm_takes_off():
    assert map_observation(hand("Closed_Fist"), UPDOWN).kind is IntentKind.LAND
    assert map_observation(hand("Open_Palm"), UPDOWN).takeoff_ready


SIMPLE = MapperConfig(hover_only=True, takeoff_zone=0.5)


@pytest.mark.parametrize("palm", [(0.5, 0.5), (0.9, 0.1), (0.1, 0.9)])
def test_simple_open_hand_anywhere_takes_off_and_never_moves(palm):
    intent = map_observation(hand("Open_Palm", palm=palm), SIMPLE)
    assert intent.takeoff_ready
    assert intent.kind is IntentKind.HOVER and intent.axes.is_zero()


@pytest.mark.parametrize("gesture", ["Thumb_Up", "Victory", "Pointing_Up", "Thumb_Down", None])
def test_simple_other_gestures_do_nothing(gesture):
    intent = map_observation(hand(gesture, palm=(0.9, 0.1), tip=(0.9, 0.3)), SIMPLE)
    assert intent.kind is IntentKind.HOVER and intent.axes.is_zero() and not intent.takeoff_ready


def test_simple_fist_lands():
    assert map_observation(hand("Closed_Fist"), SIMPLE).kind is IntentKind.LAND

import pytest

from flight.governor import FlightState, Governor, GovernorConfig
from flight.types import Command
from gesture.mapper import map_observation
from tests.fixtures import hand

FPS = 30
DT = 1 / FPS
CFG = GovernorConfig()


def run(gov, obs, start, seconds):
    """Feed the same observation for `seconds`; return (decisions, end_time)."""
    out, t = [], start
    for _ in range(round(seconds * FPS)):
        out.append(gov.step(map_observation(obs), t))
        t += DT
    return out, t


def commands(decisions):
    return [d.command for d in decisions if d.command]


def flying_governor(cfg=CFG):
    """A governor that has taken off and finished its post-takeoff lockout."""
    gov = Governor(cfg)
    gov.takeoff(0.0)
    _, t = run(gov, hand("Open_Palm"), 0.0, cfg.post_takeoff_lockout_s + 0.1)
    return gov, t


def test_takeoff_needs_a_one_second_centered_palm():
    gov = Governor(CFG)
    ds, _ = run(gov, hand("Open_Palm"), 0.0, 1.3)
    assert commands(ds) == [Command.TAKEOFF]
    first = next(i for i, d in enumerate(ds) if d.command)
    # 3 frames to confirm the gesture, then 1.0 s of hold.
    assert first * DT == pytest.approx(CFG.takeoff_hold_s + (CFG.confirm_frames - 1) * DT, abs=2 * DT)
    assert gov.state is FlightState.FLYING


def test_releasing_early_resets_takeoff():
    gov = Governor(CFG)
    ds1, t = run(gov, hand("Open_Palm"), 0.0, 0.7)
    ds2, t = run(gov, None, t, 0.5)
    ds3, _ = run(gov, hand("Open_Palm"), t, 0.7)
    assert commands(ds1 + ds2 + ds3) == []
    assert gov.state is FlightState.LANDED


def test_off_center_palm_does_not_take_off():
    gov = Governor(CFG)
    ds, _ = run(gov, hand("Open_Palm", palm=(0.8, 0.5)), 0.0, 2.0)
    assert commands(ds) == []
    assert all(d.axes.is_zero() for d in ds)


def test_sticks_stay_zero_while_landed():
    gov = Governor(CFG)
    ds, _ = run(gov, hand("Thumb_Up"), 0.0, 1.0)
    assert all(d.axes.is_zero() for d in ds)


def test_sticks_stay_zero_during_post_takeoff_lockout():
    gov = Governor(CFG)
    gov.takeoff(0.0)
    ds, _ = run(gov, hand("Thumb_Up"), 0.0, CFG.post_takeoff_lockout_s - 0.1)
    assert all(d.axes.is_zero() for d in ds)


def test_authority_clamp_and_slew_ramp():
    gov, t = flying_governor()
    ds, _ = run(gov, hand("Thumb_Up"), t, 1.0)
    pitches = [d.axes.pitch for d in ds]
    assert max(pitches) == pytest.approx(CFG.authority)
    steps = [b - a for a, b in zip(pitches, pitches[1:])]
    assert max(steps) <= CFG.slew_per_s * DT + 1e-9
    assert pitches[2] < CFG.authority  # still ramping right after confirmation


def test_hand_removed_zeroes_sticks_within_hold_time():
    gov, t = flying_governor()
    _, t = run(gov, hand("Thumb_Up"), t, 1.0)
    ds, _ = run(gov, None, t, 1.0)
    first_zero = next(i for i, d in enumerate(ds) if d.axes.is_zero())
    assert first_zero * DT <= CFG.hold_s + DT
    assert all(d.axes.is_zero() for d in ds[first_zero:])


def test_brief_dropout_holds_sticks():
    gov, t = flying_governor()
    _, t = run(gov, hand("Thumb_Up"), t, 1.0)
    ds, _ = run(gov, None, t, 0.1)  # 3 frames lost
    assert all(d.axes.pitch == pytest.approx(CFG.authority) for d in ds)


def test_gesture_change_is_debounced():
    gov, t = flying_governor()
    _, t = run(gov, hand("Thumb_Up"), t, 1.0)
    d = gov.step(map_observation(hand("Victory")), t)
    assert d.label == "pending"
    assert d.axes.pitch > 0  # one stray frame doesn't reverse the drone


def test_fist_held_lands():
    gov, t = flying_governor()
    ds, _ = run(gov, hand("Closed_Fist"), t, 0.6)
    assert commands(ds) == [Command.LAND]
    assert gov.state is FlightState.LANDING


def test_landing_holds_throttle_down_whatever_the_hand_does():
    gov, t = flying_governor()
    _, t = run(gov, hand("Closed_Fist"), t, 0.6)
    ds, t = run(gov, hand("Open_Palm", palm=(0.5, 0.1)), t, CFG.landing_s - 0.2)  # palm says "climb"
    assert all(d.state is FlightState.LANDING for d in ds)
    assert all(d.axes.throttle <= 0 for d in ds)
    assert ds[-1].axes.throttle == pytest.approx(-1.0)  # full down, past the authority cap
    assert all(d.axes.pitch == d.axes.roll == d.axes.yaw == 0 for d in ds)
    ds, _ = run(gov, hand("Open_Palm", palm=(0.5, 0.1)), t, 0.4)
    assert commands(ds) == [Command.ESTOP]  # motors off once it's had time to reach the ground
    assert gov.state is FlightState.LANDED and ds[-1].axes.is_zero()


def test_land_waits_for_the_takeoff_climb():
    gov = Governor(CFG)
    gov.takeoff(0.0)
    ds, _ = run(gov, hand("Closed_Fist"), 0.0, CFG.post_takeoff_lockout_s + 0.3)
    assert commands(ds) == [Command.LAND]
    landed_at = next(i for i, d in enumerate(ds) if d.command) * DT
    assert landed_at == pytest.approx(CFG.post_takeoff_lockout_s, abs=2 * DT)


def test_brief_fist_does_not_land():
    gov, t = flying_governor()
    ds, _ = run(gov, hand("Closed_Fist"), t, 0.2)
    assert commands(ds) == []


def test_estop_cuts_sticks_and_lands():
    gov, t = flying_governor()
    _, t = run(gov, hand("Thumb_Up"), t, 1.0)
    d = gov.estop(t)
    assert d.command is Command.ESTOP and d.axes.is_zero()
    assert gov.state is FlightState.LANDED
    ds, _ = run(gov, hand("Thumb_Up"), t, 0.5)
    assert all(x.axes.is_zero() for x in ds)


def test_reversal_passes_through_zero():
    gov, t = flying_governor()
    _, t = run(gov, hand("Thumb_Up"), t, 1.0)
    ds, _ = run(gov, hand("Victory"), t, 1.0)
    pitches = [d.axes.pitch for d in ds]
    assert min(pitches) == pytest.approx(-CFG.authority)
    sign_change = next(i for i, p in enumerate(pitches) if p < 0)
    assert pitches[sign_change] >= -CFG.slew_per_s * DT - 1e-9


AUTO = GovernorConfig(auto_land_s=0.5)


def test_auto_land_after_hand_leaves():
    gov, t = flying_governor(AUTO)
    ds, _ = run(gov, None, t, AUTO.auto_land_s + 0.2)
    assert commands(ds) == [Command.LAND]
    assert gov.state is FlightState.LANDING
    landed_at = next(i for i, d in enumerate(ds) if d.command) * DT
    assert landed_at == pytest.approx(AUTO.auto_land_s, abs=2 * DT)
    countdown = [d.auto_land_in for d in ds if d.auto_land_in is not None]
    assert countdown[0] == pytest.approx(AUTO.auto_land_s) and countdown == sorted(countdown, reverse=True)


def test_hand_back_in_view_cancels_auto_land():
    gov, t = flying_governor(AUTO)
    _, t = run(gov, None, t, 0.4)
    ds, t = run(gov, hand("Thumb_Down"), t, DT)
    assert ds[-1].auto_land_in is None
    ds, _ = run(gov, None, t, 0.4)
    assert commands(ds) == [] and gov.state is FlightState.FLYING


def test_no_auto_land_by_default():
    gov, t = flying_governor()
    ds, _ = run(gov, None, t, 10.0)
    assert commands(ds) == [] and all(d.auto_land_in is None for d in ds)
    assert gov.state is FlightState.FLYING


def test_auto_land_waits_for_the_takeoff_climb():
    gov = Governor(AUTO)
    gov.takeoff(0.0)
    ds, _ = run(gov, None, 0.0, CFG.post_takeoff_lockout_s + 0.3)
    assert commands(ds) == [Command.LAND]
    landed_at = next(i for i, d in enumerate(ds) if d.command) * DT
    assert landed_at == pytest.approx(CFG.post_takeoff_lockout_s, abs=2 * DT)

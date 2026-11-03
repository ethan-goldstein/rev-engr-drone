"""Safety layer between gesture intents and the drone.

Every frame's intent passes through here. The governor:
  * debounces: a new gesture only counts after `confirm_frames` identical frames,
  * holds the last confirmed sticks through brief tracking dropouts, but never longer
    than `hold_s` (pull your hand away and the sticks are zero within that time),
  * times takeoff (open palm held centered) and landing (fist held),
  * lands by holding the throttle full down for `landing_s` and then turning the motors off
    (Command.ESTOP), so the props always stop and nothing depends on the drone's land toggle,
  * holds back landing until the drone has finished its own takeoff climb,
  * keeps sticks at zero on the ground and while the drone climbs after takeoff,
  * caps stick authority and slew-limits increases (decreases toward zero are instant),
  * optionally lands on its own once no hand has been seen for `auto_land_s`.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from flight.types import ZERO, Axes, Command
from gesture.mapper import Intent, IntentKind


class FlightState(str, Enum):
    LANDED = "landed"
    FLYING = "flying"
    LANDING = "landing"  # throttle held full down; motors turned off when landing_s runs out


@dataclass(frozen=True)
class GovernorConfig:
    authority: float = 0.35  # max stick deflection sent to the drone (0..1)
    confirm_frames: int = 3
    hold_s: float = 0.4
    takeoff_hold_s: float = 1.0
    takeoff_grace_s: float = 0.2  # tolerated gap in the takeoff hold before it resets
    land_hold_s: float = 0.3
    post_takeoff_lockout_s: float = 2.0
    slew_per_s: float = 1.5  # max increase in stick magnitude per second
    auto_land_s: float | None = None  # land after this long with no hand in view (None = just hover)
    landing_s: float = 3.0  # throttle held full down this long, then motors off


@dataclass(frozen=True)
class Decision:
    command: Command | None
    axes: Axes
    state: FlightState
    label: str  # confirmed gesture label, or "pending"
    takeoff_progress: float  # 0..1 while an open-palm takeoff hold is under way
    auto_land_in: float | None = None  # seconds left before a no-hand auto-land, while counting down


class Governor:
    def __init__(self, cfg: GovernorConfig = GovernorConfig()):
        self.cfg = cfg
        self.state = FlightState.LANDED
        self._candidate: str | None = None
        self._candidate_count = 0
        self._confirmed_target = ZERO
        self._confirmed_at: float | None = None
        self._takeoff_started: float | None = None
        self._takeoff_last_ready: float | None = None
        self._land_started: float | None = None
        self._hand_lost_at: float | None = None
        self._lockout_until = 0.0
        self._landing_until = 0.0
        self._out = ZERO
        self._last_step: float | None = None

    # ------------------------------------------------------------------ manual overrides
    def estop(self, now: float) -> Decision:
        self._to_landed()
        return self._decision(Command.ESTOP, "estop", 0.0)

    def land(self, now: float) -> Decision | None:
        if self.state is not FlightState.FLYING:
            return None
        self._to_landing(now)
        return self._decision(Command.LAND, "land", 0.0)

    def takeoff(self, now: float) -> Decision | None:
        if self.state is not FlightState.LANDED:
            return None
        self._to_flying(now)
        return self._decision(Command.TAKEOFF, "takeoff", 0.0)

    def climbing(self, now: float) -> bool:
        """True during the drone's own takeoff climb, when it ignores the sticks."""
        return self.state is FlightState.FLYING and now < self._lockout_until

    # ------------------------------------------------------------------ per-frame step
    def step(self, intent: Intent, now: float) -> Decision:
        dt = 0.0 if self._last_step is None else min(0.1, max(0.0, now - self._last_step))
        self._last_step = now

        confirmed = self._debounce(intent)
        # A missing hand never sets a new target; it only lets the hold below run out.
        if confirmed and intent.kind is not IntentKind.NONE:
            self._confirmed_target = intent.axes if intent.kind is IntentKind.MOVE else ZERO
            self._confirmed_at = now

        command: Command | None = None
        progress = 0.0
        auto_land_in: float | None = None

        if self.state is FlightState.LANDED:
            target = ZERO
            progress = self._takeoff_progress(intent, confirmed, now)
            if progress >= 1.0:
                self._to_flying(now)
                command = Command.TAKEOFF
                progress = 0.0
        elif self.state is FlightState.LANDING:
            if now >= self._landing_until:
                # Down long enough to be on the ground: turn the motors off so the props stop.
                self._to_landed()
                return self._decision(Command.ESTOP, "motors off", 0.0)
            # Gestures are ignored; descending is the safe direction, so this skips the authority cap.
            target = Axes(throttle=-1.0)
        else:
            # The drone ignores the sticks during its own takeoff climb, so landing waits for the lockout.
            climbing = now < self._lockout_until

            # Raw per-frame check: any frame with a hand in it restarts the countdown.
            if intent.kind is not IntentKind.NONE:
                self._hand_lost_at = None
            elif self._hand_lost_at is None:
                self._hand_lost_at = now
            if self.cfg.auto_land_s is not None and self._hand_lost_at is not None:
                auto_land_in = max(0.0, self.cfg.auto_land_s - (now - self._hand_lost_at))
                if auto_land_in <= 0 and not climbing:
                    self._to_landing(now)
                    return self._decision(Command.LAND, "auto land", 0.0)

            if confirmed and intent.kind is IntentKind.LAND:
                if self._land_started is None:
                    self._land_started = now
                if now - self._land_started >= self.cfg.land_hold_s and not climbing:
                    self._to_landing(now)
                    return self._decision(Command.LAND, "land", 0.0)
            else:
                self._land_started = None

            held = self._confirmed_at is not None and now - self._confirmed_at <= self.cfg.hold_s
            target = self._confirmed_target if held else ZERO
            if climbing:
                target = ZERO

        if self.state is not FlightState.LANDING:
            target = target.clamped(1.0).scaled(self.cfg.authority)
        self._out = self._slew(target, dt)
        label = intent.label if confirmed else "pending"
        return self._decision(command, label, progress, auto_land_in)

    # ------------------------------------------------------------------ internals
    def _debounce(self, intent: Intent) -> bool:
        if intent.label == self._candidate:
            self._candidate_count += 1
        else:
            self._candidate = intent.label
            self._candidate_count = 1
        return self._candidate_count >= self.cfg.confirm_frames

    def _takeoff_progress(self, intent: Intent, confirmed: bool, now: float) -> float:
        if confirmed and intent.takeoff_ready:
            if self._takeoff_started is None:
                self._takeoff_started = now
            self._takeoff_last_ready = now
        elif (
            self._takeoff_last_ready is None
            or now - self._takeoff_last_ready > self.cfg.takeoff_grace_s
        ):
            self._takeoff_started = None
            self._takeoff_last_ready = None
        if self._takeoff_started is None:
            return 0.0
        return min(1.0, (now - self._takeoff_started) / self.cfg.takeoff_hold_s)

    def _slew(self, target: Axes, dt: float) -> Axes:
        max_step = self.cfg.slew_per_s * dt
        out = []
        for cur, tgt in zip(self._out.as_tuple(), target.as_tuple()):
            toward_zero = abs(tgt) <= abs(cur) and (tgt == 0 or (tgt > 0) == (cur > 0))
            if toward_zero:
                out.append(tgt)
            else:
                # Growing or reversing: ramp from the current value (through zero if reversing).
                start = 0.0 if (cur > 0) != (tgt > 0) else cur
                step = max(-max_step, min(max_step, tgt - start))
                out.append(start + step)
        return Axes(*out)

    def _to_flying(self, now: float) -> None:
        self.state = FlightState.FLYING
        self._lockout_until = now + self.cfg.post_takeoff_lockout_s
        self._takeoff_started = self._takeoff_last_ready = None
        self._land_started = None
        self._hand_lost_at = None
        self._out = ZERO

    def _to_landing(self, now: float) -> None:
        self._to_landed()
        self.state = FlightState.LANDING
        self._landing_until = now + self.cfg.landing_s

    def _to_landed(self) -> None:
        self.state = FlightState.LANDED
        self._takeoff_started = self._takeoff_last_ready = None
        self._land_started = None
        self._hand_lost_at = None
        self._confirmed_target = ZERO
        self._confirmed_at = None
        self._out = ZERO

    def _decision(
        self, command: Command | None, label: str, progress: float, auto_land_in: float | None = None
    ) -> Decision:
        return Decision(command, self._out, self.state, label, progress, auto_land_in)

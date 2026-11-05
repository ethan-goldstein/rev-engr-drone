"""Gesture-controlled drone: webcam -> MediaPipe -> safety governor -> drone driver.

  python main.py                          # fly the on-screen simulator
  python main.py --driver turbodrone      # WiFi UAV drone (join its Wi-Fi first)
  python main.py --driver tello           # Ryze/DJI Tello

The simulator defaults to joystick mode (every movement) for practice. Real drones default to "simple"
mode: open hand held 1s = take off and hover; fist = land, then motors off.
Pick another with --mode simple | updown | joystick | drag.

Keys (in the video window): SPACE motors off now, L land, T takeoff, C calibrate gyro (landed),
Q/Esc land + quit. Quitting while flying always lands and turns the motors off first.

Position hold (needs tracking: automatic in the sim; a camera + nav/setup.py for a real drone): the
drone holds its takeoff spot, stays inside the red barrier, and an open palm held 1 s in the AUTO box
(top-right of the webcam view) flies the --path and comes back. Try: python main.py --sim-wind 0.2

Follow mode (--follow): after takeoff press F, or hold an open palm 1 s in the FOLLOW box (top-left).
The drone locks onto the person in front of its camera and follows at --follow-distance. Try it in the
sim first: python main.py --follow
"""
from __future__ import annotations

import argparse
import csv
import math
import time
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from drivers.base import Driver
from flight.governor import Decision, FlightState, Governor, GovernorConfig
from flight.types import ZERO, Command
from gesture.drag import DRAG_MAPPER_CONFIG, DragAnchor
from gesture.mapper import NO_HAND, Intent, IntentKind, MapperConfig, map_observation
from follow.controller import FollowConfig
from follow.pose_command import ArmsUpCommand, sim_pose
from flight.light import LightCheck
from flight.trim import STEP as TRIM_STEP
from flight.trim import Trim
from follow.profile import load_profile
from ui.terminal_keys import TerminalKeys
from follow.recorder import Recorder
from follow.sim_scene import SimDetector, SimPerson, SimScene
from follow.system import FollowSystem
from gesture.tracker import DEFAULT_MODEL, GestureTracker
from nav.arena import Arena
from nav.controller import Gains, PositionController
from nav.corner import CornerZone
from nav.geofence import Geofence
from nav.marker_tracker import MarkerTracker
from nav.mission import SHAPES, build_path
from nav.navigator import Navigator
from nav.pose import SimPoseSource
from ui import hud
from ui.map import render_map
from ui.telemetry import render_strip

WINDOW = "gesture drone"
DARK_MEAN = 6.0  # average pixel value below which a frame counts as black
DARK_WARN_S = 2.0  # how long frames must stay black before we warn
AUTO_LAND_S = 0.5  # updown/drag modes: land in place after this long with no hand in view
MOTOR_OFF_SETTLE_S = 0.6  # keep the loop alive after motors-off so the repeated stop packets go out
NOTE_S = 4.0  # how long a navigator message stays on the map


def make_driver(name: str, drone_type: str, sim_wind: float = 0.0, sim_room: float = 8.0) -> Driver:
    if name == "sim":
        from drivers.sim import SimDriver

        # Wind from one side plus gusts, so the sim drifts the way the real drone does.
        wind = (sim_wind * math.cos(math.radians(30)), sim_wind * math.sin(math.radians(30)))
        return SimDriver(wind=wind, gust=0.5 * sim_wind, room_m=sim_room)
    if name == "turbodrone":
        from drivers.turbodrone_driver import TurbodroneDriver

        return TurbodroneDriver(drone_type=drone_type)
    if name == "tello":
        from drivers.tello import TelloDriver

        return TelloDriver()
    raise ValueError(name)


def open_source(source: str) -> cv2.VideoCapture:
    if source.isdigit():
        cap = cv2.VideoCapture(int(source), cv2.CAP_AVFOUNDATION)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    else:
        cap = cv2.VideoCapture(source)  # recorded clip, for replaying sessions
    if not cap.isOpened():
        hint = ""
        if source.isdigit() and source != "0":
            hint = (
                " There may be no camera with that number: an iPhone only shows up as a camera while "
                "it's nearby and locked. Try --source 0."
            )
        raise SystemExit(
            f"Could not open video source {source!r}.{hint} For a webcam, allow camera access for your "
            "terminal app in System Settings > Privacy & Security > Camera, then try again."
        )
    return cap


def apply(driver: Driver, decision: Decision | None) -> None:
    if decision is None:
        return
    if decision.command is Command.TAKEOFF:
        driver.takeoff()
    elif decision.command is Command.LAND:
        driver.land()
    elif decision.command is Command.ESTOP:
        driver.estop()
    driver.set_axes(decision.axes)


def make_navigator(args: argparse.Namespace, driver: Driver) -> tuple[Navigator | None, MarkerTracker | None, str]:
    """Tracking source + barrier + autopilot path, or (None, None, why not)."""
    if args.track == "off":
        return None, None, "Position hold off (--track off)."
    camera: MarkerTracker | None = None
    if args.driver == "sim" and args.track == "auto":
        half = args.sim_fence / 2
        source, rect = SimPoseSource(driver), (-half, -half, half, half)
    else:
        arena_file = Path(args.arena)
        if not arena_file.exists():
            setup = f"python -m nav.setup arena --camera {args.track_camera}"
            if args.track == "camera":
                raise SystemExit(f"{arena_file} not found. Set up the tracking camera first: {setup}")
            return None, None, f"Position hold off: no {arena_file} yet (set it up with: {setup})."
        arena = Arena.load(arena_file)
        camera = MarkerTracker(arena, camera=args.track_camera)
        source, rect = camera, arena.rect
    fence = Geofence(rect)
    controller = PositionController(Gains.parse(args.nav_gains)) if args.nav_gains else None
    nav = Navigator(source, fence, build_path(args.path, fence.inner), controller)
    return nav, camera, "Position hold on. Open palm in the AUTO box (top right) for 1s = autopilot path."


def log_row(t: float, decision: Decision, nav: Navigator | None) -> list[str]:
    """One flight-log line: time, states, tracked pose, target, and the sticks actually sent."""
    pose = nav.pose if nav else None
    target = nav.target if nav else None
    return [
        f"{t:.3f}",
        decision.state.value,
        nav.state.value if nav else "",
        *((f"{pose.x:.3f}", f"{pose.y:.3f}", f"{math.degrees(pose.heading):.1f}") if pose else ("", "", "")),
        *((f"{target[0]:.3f}", f"{target[1]:.3f}") if target else ("", "")),
        *(f"{v:.3f}" for v in decision.axes.as_tuple()),
    ]


def make_follow(args: argparse.Namespace, driver: Driver) -> tuple[FollowSystem | None, SimScene | None]:
    """Follow mode's person detector and, in the sim, the simulated person and drone camera."""
    if not args.follow:
        return None, None
    cfg = FollowConfig(distance=args.follow_distance, min_distance=args.min_distance, person_height=args.person_height)
    profile = load_profile(args.profile) if Path(args.profile).exists() else []
    kind = args.detector if args.detector != "auto" else ("sim" if args.driver == "sim" else "mediapipe")
    if kind == "sim":
        if args.driver != "sim":
            raise SystemExit("--detector sim only works with the simulator (--driver sim).")
        scene = SimScene(driver, [SimPerson(args.sim_person, args.sim_person_speed)])
        arms = ArmsUpCommand(sim_pose(scene.people[0]))
        return FollowSystem(SimDetector(scene), cfg, arms_up=arms, profile=profile), scene
    from follow.detector import FollowDetector
    from follow.pose_command import MediaPipePose

    return FollowSystem(FollowDetector(), cfg, arms_up=ArmsUpCommand(MediaPipePose()), profile=profile), None


def land_and_stop(governor: Governor, driver: Driver, clock=time.monotonic, sleep=time.sleep) -> None:
    """Run the whole landing (throttle down, then motors off) before the app lets go of the drone."""
    apply(driver, governor.land(clock()))  # no-op unless flying
    while governor.state is FlightState.LANDING:
        apply(driver, governor.step(NO_HAND, clock()))
        sleep(1 / 30)
    end = clock() + MOTOR_OFF_SETTLE_S
    while clock() < end:
        driver.set_axes(ZERO)
        sleep(1 / 30)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--driver", choices=("sim", "turbodrone", "tello"), default="sim")
    ap.add_argument("--drone-type", default="wifi_uav", help="turbodrone DRONE_TYPE (default wifi_uav)")
    ap.add_argument("--source", default="0", help="webcam index (0, 1, ...) or a video file path")
    ap.add_argument("--authority", type=float, default=0.35, help="max stick deflection 0..1 (start low)")
    ap.add_argument(
        "--mode",
        choices=("simple", "updown", "joystick", "drag"),
        default=None,
        help="simple: open hand = take off and hover, fist = land and motors off; nothing else moves it. "
        "updown: palm high = climb, middle = hold, low = descend. "
        "joystick: palm offset from screen center = speed, plus forward/back/turn gestures. "
        "drag: drone moves while your palm moves and holds still when it stops. "
        "Default: joystick in the sim, simple on a real drone",
    )
    ap.add_argument(
        "--auto-land",
        type=float,
        default=None,
        help=f"seconds with no hand in view before landing (updown/drag default {AUTO_LAND_S}; "
        "simple/joystick default never; 0 = never)",
    )
    ap.add_argument(
        "--track",
        choices=("auto", "off", "camera"),
        default="auto",
        help="position tracking: auto = the sim's own position in the sim, the tracking camera for a real "
        "drone if arena.json exists; off = no position hold",
    )
    ap.add_argument("--track-camera", type=int, default=1, help="tracking camera number (default 1)")
    ap.add_argument("--arena", default="arena.json", help="from: python -m nav.setup arena")
    ap.add_argument(
        "--path", default="square", help=f"autopilot path: {', '.join(SHAPES)}, or a file from nav.setup path"
    )
    ap.add_argument("--nav-gains", help="position hold PID gains kp,ki,kd (default 0.6,0.15,0.3)")
    ap.add_argument("--sim-wind", type=float, default=0.0, help="sim only: wind in m/s, e.g. 0.2")
    ap.add_argument("--sim-fence", type=float, help="sim only: barrier size in meters (default 4, or 14 with --follow)")
    ap.add_argument("--log", help="write a flight log CSV (pose, target, sticks) for tuning")
    ap.add_argument("--allow-dark", action="store_true",
                    help="take off even when the drone camera says it's too dark for it to hold position")
    ap.add_argument("--follow", action="store_true", help="follow mode: F or the FOLLOW box locks onto a person")
    ap.add_argument("--detector", choices=("auto", "sim", "mediapipe"), default="auto",
                    help="person detector: auto = simulated in the sim, MediaPipe on a real drone")
    ap.add_argument("--follow-distance", type=float, default=4.0, help="meters to stay from the person")
    ap.add_argument("--min-distance", type=float, default=3.0, help="closer than this, the drone backs away")
    ap.add_argument("--person-height", type=float, default=1.7, help="height of the person followed, in meters")
    ap.add_argument("--trim", default="trim.json", help="where the [ ] - = trim is saved between flights")
    ap.add_argument("--profile", default="profile.json",
                    help="info card shown beside the locked person, if the file exists (see follow/profile.py)")
    ap.add_argument("--sim-person", choices=("square", "line", "random"), default="square",
                    help="sim only: how the simulated person walks")
    ap.add_argument("--sim-person-speed", type=float, default=0.4, help="sim only: walking speed in m/s")
    ap.add_argument("--record", metavar="DIR", help="save the drone video, overlay and track log to DIR (e.g. flights/run1)")
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--headless", action="store_true", help="no window; runs until the source ends")
    args = ap.parse_args()
    if args.mode is None:
        # Practice every movement in the sim; keep real drones on the simple, hover-only controls.
        args.mode = "joystick" if args.driver == "sim" else "simple"
    if args.sim_fence is None:
        args.sim_fence = 14.0 if args.follow else 4.0

    drag = DragAnchor() if args.mode == "drag" else None
    mapper_cfg = {
        "simple": MapperConfig(hover_only=True, takeoff_zone=0.5),  # open hand anywhere on screen
        "updown": MapperConfig(vertical_only=True),
        "joystick": MapperConfig(),
        "drag": DRAG_MAPPER_CONFIG,
    }[args.mode]
    default_auto_land = AUTO_LAND_S if args.mode in ("updown", "drag") else None
    auto_land_s = args.auto_land if args.auto_land is not None else default_auto_land
    if args.follow and auto_land_s:
        print("Follow mode: no-hand auto-land is off, since you'll walk away from the laptop.")
        auto_land_s = None
    governor = Governor(
        GovernorConfig(
            authority=max(0.0, min(1.0, args.authority)),
            auto_land_s=auto_land_s if auto_land_s else None,
        )
    )
    driver = make_driver(args.driver, args.drone_type, args.sim_wind, sim_room=args.sim_fence + 2)
    cap = open_source(args.source)
    tracker = GestureTracker(args.model)
    nav, track_camera, nav_message = make_navigator(args, driver)
    follow, scene = make_follow(args, driver)
    corner = CornerZone()
    follow_box = CornerZone("top_left")
    note, note_until = "", 0.0
    trim = Trim.load(args.trim)
    light = LightCheck() if args.driver != "sim" and not args.allow_dark else None
    dark_note = "TOO DARK: the drone can't hold its spot. Fly in daylight."
    if trim.roll or trim.pitch:
        print(f"Loaded {trim.label()} from {args.trim}")
    log_file = open(args.log, "w", newline="") if args.log else None
    log = csv.writer(log_file) if log_file else None
    if log:
        log.writerow(["t", "flight", "nav", "x", "y", "heading_deg", "target_x", "target_y",
                      "throttle", "yaw", "pitch", "roll"])

    print(f"Connecting to {driver.name}...")
    driver.connect()
    recorder = None
    if args.record:
        if args.driver == "sim" and scene is None:
            print("--record: the sim only has a drone camera in follow mode (add --follow); nothing will be recorded.")
        else:
            recorder = Recorder(args.record, time.monotonic())
            print(f"Recording to {args.record}/")
    last_recorded: float | None = None
    print(nav_message)
    if follow is not None:
        print("Follow mode: take off, then press F to follow (F again to stop).")
        print("Keys work in this terminal or in the drone window.")
    if args.mode == "simple":
        print("Ready. Hold an open hand up for 1s to take off; make a fist to land and turn the motors off.")
    else:
        print("Ready. Hold an open palm in the center box for 1s to take off.")

    t0 = time.monotonic()
    term_keys = TerminalKeys()  # keys typed in the terminal work too: it usually keeps the focus on macOS
    fps, last = 0.0, t0
    # macOS may hand index 0 to an iPhone (Continuity Camera) lying face down, which streams black.
    dark_since: float | None = None
    if args.source.isdigit():
        other = 1 if args.source == "0" else 0
        dark_warning = ("camera is sending black frames", f"quit (Q) and try --source {other}")
    else:
        dark_warning = ("video source is black",)
    warned = False
    try:
        while True:
            ok, frame = cap.read()
            now = time.monotonic()
            if not ok:
                if not args.source.isdigit():
                    break  # end of recorded clip
                # Camera hiccup: treat as "no hand" so the governor zeroes the sticks.
                decision = governor.step(map_observation(None, mapper_cfg), now)
                apply(driver, trim.applied(decision))
                time.sleep(0.01)
                continue

            frame = cv2.flip(frame, 1)  # selfie view: move right, drone goes right
            if frame[::8, ::8].mean() >= DARK_MEAN:
                dark_since = None
            elif dark_since is None:
                dark_since = now
            camera_black = dark_since is not None and now - dark_since >= DARK_WARN_S
            if camera_black and not warned:
                print("WARNING: " + "; ".join(dark_warning))
                warned = True

            obs = tracker.process(frame, int((now - t0) * 1000))
            anchor = drag.update(obs, now) if drag else None
            intent = map_observation(obs, mapper_cfg, anchor or (0.5, 0.5))
            corner_fired = False
            if nav is not None:
                if corner.contains(obs):
                    intent = Intent(IntentKind.HOVER, "AUTO box")  # reaching for the box isn't steering
                corner_fired = corner.update(obs, now)

            # The drone's own camera: the simulated view in sim follow mode, the real video otherwise.
            drone_frame, frame_time = None, None
            if scene is not None:
                scene.update(now)
                drone_frame, frame_time = scene.render(), now
            elif args.driver != "sim":
                drone_frame, frame_time = driver.latest_frame(), getattr(driver, "frame_time", None)
                if light is not None:
                    light.update(drone_frame)

            follow_out = None
            if follow is not None:
                if follow_box.contains(obs):
                    intent = Intent(IntentKind.HOVER, "FOLLOW box")
                flying, climbing = governor.state is FlightState.FLYING, governor.climbing(now)
                if follow_box.update(obs, now):
                    note, note_until = follow.toggle(now, flying, climbing), now + NOTE_S
                    print(f"[{now - t0:7.2f}s] {note}")
                follow_out = follow.update(drone_frame, frame_time, now, flying, climbing)
                if follow_out.note:
                    note, note_until = follow_out.note, now + NOTE_S
                    print(f"[{now - t0:7.2f}s] {follow_out.note}")
                # Follow steers like a pilot would; a fist at the laptop still lands.
                if follow_out.axes is not None and intent.kind is not IntentKind.LAND:
                    kind = IntentKind.HOVER if follow_out.axes.is_zero() else IntentKind.MOVE
                    intent = Intent(kind, "follow", follow_out.axes)

            if light is not None and light.too_dark and governor.state is FlightState.LANDED:
                if intent.takeoff_ready and note != dark_note:
                    print(f"[{now - t0:7.2f}s] no takeoff: drone camera brightness {light.brightness:.0f}/255 "
                          f"(needs {light.dark:.0f}+). Fly in daylight, or add --allow-dark.")
                if intent.takeoff_ready:
                    note, note_until = dark_note, now + NOTE_S
                intent = replace(intent, takeoff_ready=False)
            decision = governor.step(intent, now)
            if decision.command:
                print(f"[{now - t0:7.2f}s] {decision.command.value}")
            if follow_out is not None and follow_out.cut:
                decision = governor.estop(now)  # lost the person too long: props stop where it is
            elif follow_out is not None and follow_out.land:
                decision = governor.land(now) or decision

            if nav is not None:
                if corner_fired:
                    note, note_until = nav.toggle_mission(), now + NOTE_S
                    print(f"[{now - t0:7.2f}s] {note}")
                out = nav.update(decision, now, governor.climbing(now))
                if out.note:
                    # Steady or per-frame notes (climbing, "not tracked for 1.2s") go on the map only.
                    if out.note != note and not out.note.startswith(("takeoff", "drone not tracked")):
                        print(f"[{now - t0:7.2f}s] {out.note}")
                    note, note_until = out.note, now + NOTE_S
                landing = governor.land(now) if out.land else None
                if landing is not None:
                    decision = landing
                    print(f"[{now - t0:7.2f}s] land")
                else:
                    decision = replace(decision, axes=out.axes)
            apply(driver, trim.applied(decision))  # the HUD keeps showing the untrimmed sticks ("holding")
            if log:
                log.writerow(log_row(now - t0, decision, nav))

            drone_view = None  # the drone camera with the follow overlay and telemetry strip
            if drone_frame is not None and follow is not None and (not args.headless or recorder):
                view = follow.draw(drone_frame)
                drone_view = np.vstack([view, render_strip(follow.history, follow.cfg, width=view.shape[1])])
            if recorder and drone_frame is not None and frame_time is not None and frame_time != last_recorded:
                last_recorded = frame_time
                recorder.add(frame_time, drone_frame, drone_view, follow, decision.axes, nav.pose if nav else None)

            fps = 0.9 * fps + 0.1 / max(now - last, 1e-3)
            last = now
            if args.headless:
                continue

            hud.draw_center_box(frame, mapper_cfg, decision, anchor if drag else (0.5, 0.5))
            hud.draw_hand(frame, obs)
            hud.draw_status(frame, decision, driver.status(), fps)
            hud.draw_sticks(frame, decision, governor.cfg.authority, (frame.shape[1] - 200, 50))
            hud.draw_legend(frame, hud.LEGENDS[args.mode])
            if trim.roll or trim.pitch:
                hud.draw_trim(frame, trim.label())
            if nav is None and now < note_until:
                hud.draw_note(frame, note)  # with tracking on, the map shows notes instead
            if camera_black:
                hud.draw_warning(frame, dark_warning)
            nav_map = None
            if nav is not None:
                hud.draw_corner(frame, corner.rect, corner.progress, nav.mission is not None)
                camera_view = track_camera.latest_frame() if track_camera else None
                people = tuple((p.x, p.y) for p in scene.people) if scene else ()
                nav_map = render_map(nav, camera_frame=camera_view, note=note if now < note_until else "", people=people)
            if follow is not None:
                hud.draw_corner(frame, follow_box.rect, follow_box.progress, follow.active, label="FOLLOW")
            if drone_view is None:
                drone_view = driver.latest_frame()
            cv2.imshow(WINDOW, hud.compose(frame, drone_view, height=420 if nav_map is not None else 480,
                                           extra=nav_map))

            key = cv2.waitKey(1) & 0xFF
            if key == 0xFF:
                key = term_keys.get()
            if key == ord(" "):
                print("EMERGENCY STOP")
                apply(driver, governor.estop(now))
            elif key in (ord("l"), ord("L")):
                apply(driver, governor.land(now))
            elif key in (ord("t"), ord("T")):
                if light is not None and light.too_dark:
                    note, note_until = dark_note, now + NOTE_S
                else:
                    apply(driver, governor.takeoff(now))
            elif key in (ord("f"), ord("F")) and follow is not None:
                note = follow.toggle(now, governor.state is FlightState.FLYING, governor.climbing(now))
                note_until = now + NOTE_S
                print(f"[{now - t0:7.2f}s] {note}")
            elif key in (ord("u"), ord("U")) and scene is not None:
                scene.people[0].arms_up = not scene.people[0].arms_up  # sim: practice the arms-up landing
            elif key in (ord("["), ord("]"), ord("-"), ord("=")):
                d = TRIM_STEP
                note = trim.nudge(roll={ord("["): -d, ord("]"): d}.get(key, 0.0), pitch={ord("-"): -d, ord("="): d}.get(key, 0.0))
                note_until = now + NOTE_S
                trim.save(args.trim)
                print(f"[{now - t0:7.2f}s] {note}")
            elif key in (ord("c"), ord("C")) and governor.state is FlightState.LANDED:
                print("Calibrating gyro: keep the drone flat and still.")
                driver.calibrate()
            elif key in (ord("q"), ord("Q"), 27):
                break
    except KeyboardInterrupt:
        pass
    finally:
        term_keys.close()
        if governor.state is not FlightState.LANDED:
            print(f"Landing before exit ({governor.cfg.landing_s:.0f}s, then motors off)...")
        land_and_stop(governor, driver)
        driver.close()
        if track_camera:
            track_camera.close()
        if recorder:
            recorder.close()
            print(f"Saved {recorder.frames} frames to {args.record}/")
        if log_file:
            log_file.close()
        tracker.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

# Gesture drone

Fly a cheap Wi-Fi drone with hand gestures from a MacBook webcam. No soldering: the laptop
joins the drone's own Wi-Fi and speaks the same UDP protocol as its phone app.

```
webcam ─► MediaPipe GestureRecognizer ─► gesture/mapper.py ─► flight/governor.py ─► drivers/*
                                          (gesture → intent)   (safety layer)       sim | turbodrone | tello
```

**Target drone:** Contixo F19 Arvo (Best Buy, about $60). It uses the "WiFi UAV" app, whose
protocol is already reverse-engineered by [turbodrone](https://github.com/marshallrichards/turbodrone)
(vendored at `vendor/turbodrone`, pinned to commit `2c09cf864e7c26d2f5e674a11d3d87cc308c3a83`).

## Setup

```bash
git clone --recurse-submodules https://github.com/ethan-goldstein/rev-engr-drone.git
cd rev-engr-drone
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

`vendor/turbodrone` is a git submodule: [marshallrichards/turbodrone](https://github.com/marshallrichards/turbodrone)
(Apache-2.0), pinned to the commit this was built against. If you cloned without `--recurse-submodules`,
run `git submodule update --init`.

**Models** (Google MediaPipe, version 1, not in git). Download them, then check them against the hashes:

```bash
M=https://storage.googleapis.com/mediapipe-models
curl -fsSL -o assets/gesture_recognizer.task $M/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task
curl -fsSL -o assets/efficientdet_lite0.tflite $M/object_detector/efficientdet_lite0/float16/1/efficientdet_lite0.tflite
curl -fsSL -o assets/pose_landmarker_lite.task $M/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task
shasum -a 256 assets/*
```

| File | SHA-256 |
|---|---|
| `gesture_recognizer.task` | `97952348cf6a6a4915c2ea1496b4b37ebabc50cbbf80571435643c455f2b0482` |
| `efficientdet_lite0.tflite` | `4b59100025bea1235a84c1038879a6cccc9f6c49f5e41144e91e74d99e780993` |
| `pose_landmarker_lite.task` | `59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a` |

**Tests:** `.venv/bin/python -m pytest`. The photo tests skip unless MediaPipe's sample photos are in
`tests/images/`:

```bash
mkdir -p tests/images && for n in thumbs_up victory pointing_up thumbs_down; do
  curl -fsSL -o tests/images/$n.jpg https://storage.googleapis.com/mediapipe-tasks/gesture_recognizer/$n.jpg; done
```

**Your info card:** copy `profile.example.json` to `profile.json` and fill it in. It's in `.gitignore`.

Not in git: flight recordings and reels (`flights/`, `video/reels/`), models, test photos, `profile.json`,
`trim.json`, `arena.json`, and reverse-engineering captures (`re/`).

## Fly the simulator (no drone needed)

```bash
.venv/bin/python main.py
```

Run this from your own terminal (Terminal, iTerm or VS Code). macOS asks that app for camera access the first time.

If the camera panel stays black, macOS may have given index 0 to your iPhone (Continuity Camera). Quit and run
`main.py --source 1` to use the Mac's own camera. The app shows a warning after 2 s of black frames.
`--source 1` only exists while the iPhone is nearby and locked; otherwise the Mac's camera is 0 (the default).

**The simulator defaults to joystick mode** (every movement; table below) so you can practice. **A real
drone defaults to simple mode**, where the app never moves it sideways, forward or up and down:

| Gesture | Effect |
|---|---|
| Open hand, anywhere on screen, held 1 s | Take off. The drone climbs to its own hover height and hovers. |
| Fist, held 0.3 s | Land: throttle full down for 3 s, then **motors off** |
| Anything else, or no hand | Nothing; it keeps hovering |

Landing never uses the drone's land command. On WiFi UAV drones, takeoff and land share one toggle bit
that's sent in a single packet. A missed or extra toggle leaves the app and the drone disagreeing about
whether it's flying, which showed up as "the fist doesn't land it" and "props still spinning on the floor".
Instead, the app holds the throttle down to bring it to the ground, then sends the motor-stop bit
(repeated for 0.5 s). A fist in the first 2 s after takeoff waits until the drone's own climb is done.
**Quitting (Q, Ctrl-C or a crash) while flying runs the same landing first.**

The app can't stop the drone drifting on its own, because it has no idea where the drone is. If it wanders,
land, set it on a hard level table, press **C** to calibrate, and fly in good light over a patterned floor.

`--mode updown` adds climbing and descending: palm high on screen climbs, middle holds, low descends,
and no hand for 0.5 s lands.

`--mode joystick` brings back sideways and forward/back/turn:

| Gesture (joystick mode) | Effect |
|---|---|
| Open palm | Steer. Your palm position is a joystick: left/right moves sideways, up/down climbs or descends. |
| Thumb up | Fly forward |
| Victory ✌️ | Fly backward |
| Point up | Turn toward the side your finger is on |
| Fist (held 0.3 s) | Land |
| Anything else, or no hand | Hover |

Keys in the window (click the window first so it gets the keys): **SPACE** motors off right now
(resent for 0.5 s), **L** land, **T** takeoff, **C** calibrate the gyro (landed only), **Q** land and quit.

Safety defaults (in `flight/governor.py`):
- Sticks are capped at 35% (`--authority`).
- A gesture counts only after 3 matching frames.
- Sticks drop to zero within 0.4 s of losing your hand.
- Sticks stay at zero on the ground and for 2 s after takeoff.
- Increases are rate-limited; decreases happen immediately.

The real-drone drivers also zero the sticks if the app's main loop stalls, so a frozen camera can't leave the drone flying on old input.

### Drag mode

```bash
.venv/bin/python main.py --mode drag --driver turbodrone
```

In drag mode, the drone moves only while your open palm moves and stops when your palm stops.
- Raising your palm anywhere counts as "stopped", and the boxes appear around it.
- Move your palm to push the drone that way. Hold it still for about 0.3 s and the drone holds its position.
- A slowly sagging arm is ignored.
- To keep going in one direction, reset like lifting a mouse: show a thumbs-down (hover), move your hand back,
  then open your palm again. Don't drop your hand to reset; that lands.
- Takeoff is the same as the other modes: open palm held in the screen-center box for 1 s.
- If no hand is in view for **0.5 s**, it lands in place. Change the delay with `--auto-land 1.0`
  (seconds; `0` turns it off). This applies to up/down mode too.

The drone can't return to its takeoff spot: neither it nor the app knows where it is (no GPS, no position
reports), so it always lands where it is.

## Position hold, barrier and autopilot

The drone doesn't report where it is, so the app tracks it from the outside, like a motion-capture lab.
A second camera looks down at the flying area, and a printed tag on top of the drone gives its position
and heading about 30 times a second. A PID controller then nudges roll and pitch to:

- **hold the takeoff spot** (or wherever you let go of the sticks), cancelling drift,
- keep it inside a **barrier**:
  - near the edge it's pushed back and your outward sticks are cut,
  - actually outside for 0.5 s means it lands,
- fly an **autopilot path** when you hold an open palm in the **AUTO** box (top right of the webcam view)
  for 1 s:
  - it flies the path, stopping at each corner, then returns home and holds,
  - the AUTO box again or any steering takes over, and a fist still lands.

The window gets a third panel: a top-down map with the barrier (red), the path, the target (cross),
home (H) and the drone.

**Try it in the sim first.** The sim tracks itself, and `--sim-wind` makes it drift like the real one:

```bash
.venv/bin/python main.py --sim-wind 0.2               # holds against the wind
.venv/bin/python main.py --sim-wind 0.2 --track off   # compare: it blows away
.venv/bin/python main.py --path figure8               # autopilot shapes: square, circle, figure8
```

**Real drone setup (once):**
1. Run `.venv/bin/python -m nav.setup marker` and print `marker_0.png` at 100%, so the black square is 5 cm.
   Tape it flat on top of the drone with the arrow toward the nose.
2. Mount the tracking camera looking down at the flying area, about 2–2.5 m up (a balcony or a shelf).
   - An iPhone works as a Mac camera while it's nearby and locked; a USB cable makes it reliable.
   - A USB webcam works too.
   - Mark the area's corners on the floor with tape.
3. Run `.venv/bin/python -m nav.setup arena --camera 1`. Click the 4 corners and type the measured
   width and depth. This writes `arena.json`.
4. Optional: `.venv/bin/python -m nav.setup path --camera 1` lets you click your own waypoints, then
   fly them with `--path path.json`.
5. Props off: run `main.py --driver turbodrone` and slide the drone around by hand. The map should follow
   it, and the PIT/ROL bars should push back toward the target in the right direction whichever way it faces.
6. Props on, tethered first, in good light. Start gentle with `--nav-gains 0.4,0.1,0.2` and record a log
   (`--log flight.csv`). Raise the gains once it holds without oscillating.

Notes:
- With tracking on, the turbodrone driver sends linear sticks (turbodrone's expo curve would flatten
  the small corrections). If you use joystick mode on the real drone, start at `--authority 0.2`.
- If the tag isn't seen for 0.3 s, corrections stop and the drone just hovers on its own.
  During autopilot, 3 s without the tag means it lands.
- The camera needs light to see the tag, so night flying needs the area lit.

## Follow mode (person-following drone)

The drone locks onto a person in **its own camera** and follows them, like a security drone. A Kalman
filter keeps the lock through short gaps and re-locks on the same person by clothing color.
The drone view gets the lock box ("LOCKED #1 · 4.1 m") and live graphs of distance, yaw and pitch.

**Practice in the sim:** a simulated person walks around in front of a simulated F19 camera.

```bash
.venv/bin/python main.py --follow                        # person walks a square
.venv/bin/python main.py --follow --sim-person random --sim-wind 0.1
```

1. Take off with an open palm.
2. Press **F**, or hold an open palm 1 s in the **FOLLOW** box (top-left). It locks onto the person
   nearest the middle of the drone's view. Keys work in the terminal that started the app or in its
   window. Messages show in the middle of the webcam view.
3. Press **F** again to stop and hover.
4. In the sim, **U** raises the simulated person's arms, to practice the arms-up landing.

**How it flies:**
- It turns to keep you centered and flies forward or back to stay **4 m** away (`--follow-distance`).
- Closer than **3 m** (`--min-distance`), or if your head or feet are cut off in the frame, it backs
  away instead of approaching.
- Beyond 8 m it stops advancing.
- No sideways or up/down movement: altitude stays where takeoff put it.

**Lock states:**
- **SEARCH:** follow is on, nobody locked yet.
- **LOCKED:** you're in view.
- **COASTING:** hidden for under 1.5 s; it predicts where you are and eases off.
- **LOST:** if you left out the side of the frame, it turns in place toward that side to look for you
  (it never flies forward while searching). If you vanished near the middle (behind a tree), it holds
  still. It only re-locks on someone in the same clothes.
- **5 s after losing you, the motors cut off** wherever it is, the same as SPACE. It drops from hover
  height, so fly over grass. The countdown shows on the drone view.

**FACE LOCK card:** while locked, an info panel sits beside the person's box.
- The top rows come from `profile.json` (or `--profile other.json`), as `"LABEL": "value"` pairs, up to
  6 lines.
- The bottom rows are live: track ID, range, how long it's been locked, and detector confidence.
- The title reads **FACE LOCK**, with brackets on the face, while the person faces the drone, and
  **BODY LOCK** while they face away. The face comes from the pose model the arms-up landing already
  runs.
- It doesn't recognize anyone: whoever is locked gets the card, so swap the file when following a friend.

**Stopping it:**
- **Both arms raised above your head for 1.5 s** (seen by the drone's camera) lands it. This works once
  you've walked away from the laptop.
- A fist at the laptop, **L**, or **SPACE** always work.
- No-hand auto-land is off in follow mode, since you won't be at the laptop.

**Real drone:**
```bash
.venv/bin/python main.py --driver turbodrone --follow --record flights/run1
```
- **Props off first.** Power the drone, hold it at chest height, point it at a friend walking about 4 m
  away, take off, and press F. Check:
  - the box and distance look right;
  - the YAW bar goes positive when they're right of center;
  - the PITCH bar goes positive when they walk away.
- Then fly with **prop guards** fitted, tethered first, in an open, well-lit area, walking slowly.
  Set `--person-height` to the real person's height, since distance is estimated from it.
- Keep the drone's camera tilted level. The F19 streams 640x360 at about 15 fps with a 60° lens, so it
  can't see well past about 8 m.

**Recording** (`--record DIR`) saves:
- `raw.mp4`, the drone camera;
- `overlay.mp4`, what you saw;
- `frames.csv`, frame timing;
- `track.csv`, the lock, distance, sticks and tracked position.

These are the inputs for the after-flight 3D map (`python -m mapping.reconstruct DIR`, once that tool is built).

**Models** (downloaded from Google's MediaPipe model storage, version 1, into `assets/`):

| File | SHA-256 |
|---|---|
| `efficientdet_lite0.tflite` (person detector) | `4b59100025bea1235a84c1038879a6cccc9f6c49f5e41144e91e74d99e780993` |
| `pose_landmarker_lite.task` (arms-up landing) | `59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a` |

**Ground rules:**
- Follow yourself or friends who agreed to it. Following or recording people without permission can break
  privacy and harassment laws.
- Keep the drone in sight, and never launch it unattended.

## Fly the real drone (Contixo F19 / any "WiFi UAV" drone)

1. **Stock flight first.** Charge both batteries and fly once with the official WiFi UAV app.
   Check that the box or manual says "WiFi UAV".
2. **Keep the Mac online.** Plug the iPhone in by USB and turn on Personal Hotspot.
   In System Settings > Network > ⋯ > Set Service Order, drag **iPhone USB** above Wi-Fi.
3. **Join the drone's Wi-Fi.** Close the WiFi UAV app on the phone first, since the drone takes one controller at a time.
   Then connect the Mac to `Contixo-F19-XXXX`.
4. **Link test, props off.** From your own terminal:
   ```bash
   .venv/bin/python main.py --driver turbodrone
   ```
   Allow the camera and "find devices on local networks" prompts. The drone's video should appear in the right half
   of the window. Press **T** and the motors should spin up; press **SPACE** to stop them.
   If there's no video or the motors don't react, quit and retry with `--drone-type wifi_uav_uav`
   (turbodrone's other WiFi UAV transport; plain `wifi_uav` uses the FLD one).
5. **Gestures, props off.** Same command. After **T**, each gesture should spin the motors the expected way,
   and pulling your hand away should idle them.
6. **Props on.** Indoors, in an open room, at `--authority 0.35`. Raise it slowly once it feels predictable.

turbodrone's "land" can behave more like a motor cut on some variants, so land from low height.

### If it drifts

**First: is it dark?** The F19 holds its spot with a camera underneath that watches the ground, and at
night it sees nothing. Every flight that drifted badly was after sunset: the drone camera's brightness
was 3-17 out of 255, against 106-140 in daylight. One flew more than 10 m into a building in about 4 s.
So the app won't take off when the drone camera is that dark ("TOO DARK" on screen). `--allow-dark`
overrides this, but fly in daylight.

While flying, the HUD says **FLYING holding** when every stick is centered and **FLYING moving** otherwise.

- **Drifts while "moving":** your palm is outside the small box, so the app is still steering.
  Bring it back inside the box, or drop your hand.
- **Drifts while "holding":** that's the drone itself. Land, set it on a flat surface and press **C** to
  calibrate the gyro. Its hover sensor looks down at the floor, so fly in good light over a patterned floor or rug.
  A plain or shiny floor gives it nothing to lock onto.
- **Still slides the same way every flight:** check the props for chips or bends, which happen after a
  tip-over. Then trim it in the air: **[** / **]** lean it left/right and **-** / **=** back/forward,
  one small step per press. Press until it hovers in place. The trim is saved to `trim.json` and
  loaded on the next run, so the drone leans that way from takeoff. The keys work in the terminal too.

### If the F19's protocol differs (reverse-engineering fallback)

Keep everything under `re/` and treat it as untrusted: never install or run files from it.

- **Capture the app's traffic.** Plug the iPhone in, run `xcrun rvictl -s <UDID>`, and open Wireshark on `rvi0`.
  Fly with the WiFi UAV app and record idle, each stick at its extremes, takeoff and land. Then diff the packets.
- **Decompile the app.** Put the WiFi UAV APK in `re/apk/` and open it with jadx (ports, packet builders).
  Use Ghidra for the native `.so` libraries.
- **Write the adapter.** Mirror turbodrone's `build_control_packet` / `send_control_packet` adapter interface,
  then point `drivers/turbodrone_driver.py` at it.

## Tello (optional)

If you find a used Ryze/DJI Tello, install it with `uv pip install --python .venv/bin/python djitellopy`.
Join its Wi-Fi, then run `main.py --driver tello`. That driver hasn't been flown yet.

## Rules

- Under 250 g, so no FAA registration is needed. Flying outdoors recreationally needs the free FAA TRUST certificate.
- Check B4UFLY before any outdoor flight: controlled airspace around airports (Class B/C/D) needs authorization.

## Layout

- `gesture/tracker.py`: MediaPipe wrapper
- `gesture/mapper.py`: gesture → intent
- `flight/governor.py`: safety rules
- `drivers/`: sim, turbodrone (WiFi UAV), Tello, plus the stall watchdog
- `nav/`: position tracking (sim or camera + ArUco tag), PID position hold, barrier, autopilot paths, setup tool
- `ui/hud.py`: pilot overlay
- `ui/map.py`: top-down map panel
- `follow/`: person detection, lock tracker, follow controller, arms-up landing, sim people + camera, recorder
- `ui/telemetry.py`: strip charts under the drone view
- `hardware/`: printable parts (prop guard draft)
- `main.py`: app entry point
- `tests/`: unit tests, plus MediaPipe checks on Google's sample gesture photos in `tests/images/`

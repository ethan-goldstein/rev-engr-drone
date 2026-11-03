"""Showreel for a recorded flight: the drone's view with the follow overlay (lock box, FACE LOCK card,
telemetry graphs) next to the 3D map of the flight, with the paths drawn as the flight goes.

    python -m mapping.reel flights/field4 [flights/field3 ...]      -> video/reels/<run>.mp4
    python -m mapping.reel --all      every recorded flight that doesn't have a reel yet, plus index.md

Per run:
1. Replay: follow mode runs offline over raw.mp4, as if F had been pressed at the start, into
   <run>/replay/ in the recorder's format. Night footage is brightened first so there's something to see.
   An existing replay is reused.
2. Map: COLMAP SfM on the replay, which has the lock boxes the scale comes from. Also writes
   <run>/replay/map.html. If COLMAP can't register the footage (dark, or the drone sat still), the map
   panel says so.
3. Compose side by side, with the map orbiting slowly. Encoded to H.264 with ffmpeg so it plays anywhere.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

from flight.light import DARK
from mapping.geometry import focal_px, look_at
from mapping.reconstruct import map_from_sfm
from mapping.render import (
    BACKGROUND_BGR,
    DRONE_BGR,
    PERSON_BGR,
    MapData,
    _extent,
    _polyline,
    _splat,
    turbo,
    write_html,
)

ROOT = Path(__file__).resolve().parent.parent
PANEL = (640, 510)  # the drone view (640x360) with the telemetry strip under it
LOST_RESET_S = 3.0  # in a replay, let go of someone gone this long so the next person can be locked
_clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))


def brighten(frame: np.ndarray) -> np.ndarray:
    """Night footage (brightness 3-17 of 255) -> something watchable: gamma lift, local contrast, denoise."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    lift = np.power(lab[:, :, 0].astype(np.float32) / 255.0, 0.45) * 255.0
    lab[:, :, 0] = _clahe.apply(np.clip(lift, 0, 255).astype(np.uint8))
    return cv2.fastNlMeansDenoisingColored(cv2.cvtColor(lab, cv2.COLOR_LAB2BGR), None, 6, 6, 5, 11)


def frame_times(run: Path) -> list[float]:
    with open(run / "frames.csv") as f:
        return [float(r["t"]) for r in csv.DictReader(f)]


def footage_brightness(video: Path, samples: int = 30) -> float:
    cap = cv2.VideoCapture(str(video))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vals = []
    for i in range(0, max(n, 1), max(1, n // samples)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, f = cap.read()
        if ok:
            vals.append(float(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).mean()))
    return float(np.median(vals)) if vals else 0.0


def replay(run: Path) -> tuple[Path, bool]:
    """Follow mode over the recording -> <run>/replay/. Returns (folder, night)."""
    from follow.detector import PersonDetector
    from follow.pose_command import ArmsUpCommand, MediaPipePose
    from follow.profile import load_profile
    from follow.recorder import Recorder
    from follow.system import FollowSystem
    from follow.tracker import LockState
    from ui.telemetry import render_strip

    out = run / "replay"
    night = footage_brightness(run / "raw.mp4") < DARK
    if (out / ".done").exists():
        return out, night
    if out.exists():
        shutil.rmtree(out)  # a replay that was cut off: only ever this script's own output
    profile_file = ROOT / "profile.json"
    profile = load_profile(profile_file) if profile_file.exists() else []
    fs = FollowSystem(PersonDetector(), arms_up=ArmsUpCommand(MediaPipePose()), profile=profile, lost_off_s=1e9)
    rec = Recorder(out, 0.0)
    cap = cv2.VideoCapture(str(run / "raw.mp4"))
    locked = 0
    for i, t in enumerate(frame_times(run)):
        ok, frame = cap.read()
        if not ok:
            break
        if night:
            frame = brighten(frame)
        if i == 0:
            fs.toggle(t, True, False)
        o = fs.update(frame, t, t, True, False)
        tr = fs.tracker
        if tr.state is LockState.LOST and t - tr.target.last_seen > LOST_RESET_S:
            tr.unlock()
        locked += o.state is LockState.LOCKED
        overlay = np.vstack([fs.draw(frame), render_strip(fs.history, fs.cfg, width=frame.shape[1])])
        rec.add(t, frame, overlay, fs, o.axes, None)
    rec.close()
    (out / ".done").write_text(f"locked frames: {locked}\n")
    print(f"  replay: {rec.frames} frames, {locked} locked{' (brightened night footage)' if night else ''}")
    return out, night


def save_map(path: Path, data: MapData | None) -> None:
    if data is None:
        np.savez(path, none=True)
        return
    np.savez(path, points=data.points, colors=data.colors, drone=data.drone, person=data.person,
             zrange=np.array(data.zrange), drone_t=data.drone_t, person_t=data.person_t,
             info=np.array(data.info, dtype=str).reshape(-1, 2))


def load_map(path: Path) -> tuple[bool, MapData | None]:
    """(found, map): a finished COLMAP run is kept, so re-rendering a reel doesn't redo minutes of SfM."""
    if not path.exists():
        return False, None
    z = np.load(path)
    if "none" in z:
        return True, None
    return True, MapData(z["points"], z["colors"], z["drone"], z["person"], tuple(z["zrange"]),
                         [tuple(r) for r in z["info"]], drone_t=z["drone_t"], person_t=z["person_t"])


def build_map(folder: Path, windows: list[float]) -> MapData | None:
    """COLMAP is fickle on this camera: try a few keyframe spacings, keep whichever maps the most frames."""
    found, cached = load_map(folder / "map.npz")
    if found:
        print(f"  map: reusing {folder / 'map.npz'}" + ("" if cached else " (no map)"))
        return cached
    best: MapData | None = None
    for window_s, seed in [(w, seed) for seed in (0, 1) for w in windows]:
        try:
            data = map_from_sfm(folder, window_s, seed)
        except SystemExit as e:  # COLMAP registered too few frames
            print(f"  map ({window_s:.2f} s keyframes, seed {seed}): {e}")
            continue
        except Exception as e:  # noqa: BLE001 - a failed map shouldn't stop the reel
            print(f"  map ({window_s:.2f} s keyframes, seed {seed}): COLMAP failed ({type(e).__name__}: {e})")
            continue
        info = dict(data.info)
        print(f"  map ({window_s:.2f} s keyframes, seed {seed}): {info['keyframes']}, {info['points']} points")
        if best is None or len(data.drone) > len(best.drone):
            best = data
        registered, total = (int(x) for x in info["keyframes"].split(" registered")[0].split(" of "))
        if registered >= 0.5 * total:
            break
    save_map(folder / "map.npz", best)
    if best is not None:
        write_html(folder / "map.html", best, f"{folder.parent.name} - 3D map")
        for k, v in best.info:
            print(f"  map {k}: {v}")
    return best


class MapPanel:
    """The 3D map drawn at a moment in the flight: everything mapped, paths up to now, orbiting slowly."""

    def __init__(self, data: MapData, duration: float, size=PANEL):
        self.data, self.W, self.H = data, size[0], size[1]
        self.f = focal_px(self.W)
        self.center, radius = _extent(data)
        self.dist, self.elev = 1.15 * radius / math.tan(math.radians(30)), math.radians(30)
        self.period = max(24.0, duration)  # one slow turn over the flight
        step = 1.0 if radius < 15 else 5.0
        span = math.ceil(radius * 1.4 / step) * step
        gx, ts = np.arange(-span, span + step / 2, step), np.linspace(-span, span, 40)
        c = self.center
        self.grid = [np.column_stack([np.full_like(ts, c[0] + x), c[1] + ts, np.zeros_like(ts)]) for x in gx]
        self.grid += [np.column_stack([c[0] + ts, np.full_like(ts, c[1] + y), np.zeros_like(ts)]) for y in gx]
        self.colors = np.ascontiguousarray(data.colors[:, ::-1]) if len(data.colors) else np.zeros((0, 3), np.uint8)
        self.bar = turbo(np.linspace(0, 1, 140))[:, ::-1]

    def _upto(self, path: np.ndarray, times: np.ndarray, t: float) -> np.ndarray:
        return path[: int(np.searchsorted(times, t, side="right"))] if len(times) == len(path) else path

    def draw(self, t: float) -> np.ndarray:
        az = 2 * math.pi * t / self.period - math.pi / 2
        eye = self.center + self.dist * np.array(
            [math.cos(self.elev) * math.cos(az), math.cos(self.elev) * math.sin(az), math.sin(self.elev)])
        R = look_at(eye, self.center)
        img = np.zeros((self.H, self.W, 3), np.uint8)
        img[:] = BACKGROUND_BGR
        if len(self.data.points):
            _splat(img, self.data.points, self.colors, eye, R, self.f)
            img = np.maximum(img, cv2.dilate(img, np.ones((2, 2), np.uint8)))  # 3x3 dots: sparse maps read better
        for line in self.grid:
            _polyline(img, line, eye, R, self.f, (48, 40, 36), 1)
        for path, times, color in ((self.data.drone, self.data.drone_t, DRONE_BGR),
                                   (self.data.person, self.data.person_t, PERSON_BGR)):
            now = self._upto(path, times, t)
            _polyline(img, now, eye, R, self.f, color, 2)
            if len(now):
                Xc = (now[-1] - eye) @ R
                if Xc[2] > 0.1:
                    p = (int(self.f * Xc[0] / Xc[2] + self.W / 2), int(self.f * Xc[1] / Xc[2] + self.H / 2))
                    cv2.circle(img, p, 5, color, -1, cv2.LINE_AA)
        _label(img, "3D TRACK  //  COLMAP SfM", (12, 24), (230, 230, 230), 0.55)
        _label(img, "drone", (12, 46), DRONE_BGR, 0.45)
        _label(img, "person", (66, 46), PERSON_BGR, 0.45)
        if len(self.data.points):
            img[self.H - 30:self.H - 22, 12:12 + len(self.bar)] = self.bar[None, :, :]
            lo, hi = self.data.zrange
            _label(img, f"{lo:.1f} m", (12, self.H - 8), (200, 200, 200), 0.4)
            _label(img, f"{hi:.1f} m", (12 + len(self.bar) - 34, self.H - 8), (200, 200, 200), 0.4)
        return img


def no_map_panel(reason: str, size=PANEL) -> np.ndarray:
    img = np.zeros((size[1], size[0], 3), np.uint8)
    img[:] = BACKGROUND_BGR
    _label(img, "3D TRACK  //  NO MAP", (12, 24), (230, 230, 230), 0.55)
    for i, line in enumerate(reason.split("\n")):
        _label(img, line, (40, size[1] // 2 - 10 + 26 * i), (150, 170, 190), 0.55)
    return img


def _label(img, text, org, color, scale) -> None:
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)  # readable on sky
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def compose(run: Path, folder: Path, data: MapData | None, reason: str, out: Path) -> None:
    times = frame_times(folder)
    duration = times[-1] - times[0] if len(times) > 1 else 1.0
    fps = max(1.0, (len(times) - 1) / duration) if duration > 0 else 10.0
    panel = MapPanel(data, duration) if data is not None else None
    still = no_map_panel(reason) if data is None else None
    cap = cv2.VideoCapture(str(folder / "overlay.mp4"))
    with tempfile.TemporaryDirectory() as tmp:
        raw = Path(tmp) / "reel.mp4"
        writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, (PANEL[0] * 2, PANEL[1]))
        for t in times:
            ok, left = cap.read()
            if not ok:
                break
            left = cv2.resize(left, PANEL)
            stamp = f"{run.name.upper()}  //  T+{t:05.1f}s"
            (tw, _), _ = cv2.getTextSize(stamp, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            _label(left, stamp, (PANEL[0] - tw - 10, 20), (230, 230, 230), 0.45)
            writer.write(np.hstack([left, panel.draw(t) if panel else still]))
        writer.release()
        out.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(raw), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-crf", "20", "-movflags", "+faststart", str(out)], check=True)
    print(f"  wrote {out} ({len(times)} frames, {duration:.0f} s)")


def run_order(run: Path) -> tuple[int, str]:
    digits = re.sub(r"\D", "", run.name)
    return (int(digits) if digits else 0, run.name)


def recorded_at(run: Path) -> str:
    st = os.stat(run / "raw.mp4")
    when = time.localtime(getattr(st, "st_birthtime", st.st_mtime))
    return time.strftime("%b %-d, %-I:%M %p", when)


def write_index(flights: Path, reels: Path) -> Path:
    """index.md in the reels folder: one line per reel, from what's already on disk (no reprocessing)."""
    lines = ["# Flight reels", "",
             "Each video: the drone camera with the follow replay (left) and the 3D map of the flight (right).", "",
             "| Reel | Recorded | Length | Light | Face/body locked | 3D map |", "|---|---|---|---|---|---|"]
    for run in sorted((r for r in flights.iterdir() if (reels / f"{r.name}.mp4").exists()), key=run_order):
        folder = run / "replay"
        times = frame_times(folder) if (folder / "frames.csv").exists() else [0.0]
        light = footage_brightness(run / "raw.mp4")
        done = (folder / ".done").read_text().strip() if (folder / ".done").exists() else ""
        locked = int(done.split(":")[-1]) if done else 0
        found, data = load_map(folder / "map.npz")
        if data is not None:
            info = dict(data.info)
            mapped = f"{info.get('points', '?')} points, {info.get('keyframes', '')}"
        else:
            mapped = "none (too dark)" if light < DARK else ("none" if found else "not built")
        lines.append(f"| [{run.name}.mp4]({run.name}.mp4) | {recorded_at(run)} | {times[-1] - times[0]:.0f} s | "
                     f"{'night' if light < DARK else 'day'} ({light:.0f}/255) | {locked / max(len(times), 1):.0%} "
                     f"of frames | {mapped} |")
    out = reels / "index.md"
    out.write_text("\n".join(lines) + "\n")
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m mapping.reel", description=__doc__.split("\n\n")[0])
    p.add_argument("runs", nargs="*", type=Path, help="flights/<run>/ folders written by the recorder")
    p.add_argument("--all", action="store_true", help="every flight in flights/ that has no reel yet")
    p.add_argument("--force", action="store_true", help="with --all: remake reels that already exist")
    p.add_argument("--out", type=Path, default=ROOT / "video" / "reels")
    p.add_argument("--no-map", action="store_true", help="skip COLMAP (faster); the map panel says so")
    p.add_argument("--windows", type=float, nargs="+", default=[1 / 3, 0.5],
                   help="keyframe spacings to try for COLMAP, in seconds")
    args = p.parse_args(argv)
    flights = ROOT / "flights"
    runs = list(args.runs)
    if args.all:
        runs += [r for r in sorted(flights.iterdir(), key=run_order)
                 if (r / "raw.mp4").exists() and (args.force or not (args.out / f"{r.name}.mp4").exists())]
    if not runs and not args.all:
        p.error("name some flights/<run> folders, or use --all")
    for run in runs:
        if not (run / "raw.mp4").exists():
            print(f"{run}: no video, skipped")
            continue
        print(f"{run}:")
        folder, night = replay(run)
        data, reason = None, "skipped (--no-map)"
        if not args.no_map:
            data = build_map(folder, args.windows)
            reason = ("too dark to map:\nthe drone camera saw almost nothing" if night
                      else "COLMAP couldn't map this flight:\nthe drone barely moved, or the view was blurred")
        compose(run, folder, data, reason, args.out / f"{run.name}.mp4")
    if flights.is_dir() and args.out.is_dir():
        print(f"index: {write_index(flights, args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

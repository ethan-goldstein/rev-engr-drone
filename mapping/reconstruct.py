"""Post-flight 3D map from a recorded flight.

    python -m mapping.reconstruct flights/<run>/               # SfM on raw.mp4 (needs pycolmap)
    python -m mapping.reconstruct flights/<run>/ --from-track  # simulator flights: map from drone_x/drone_y

Writes map.html (interactive three.js viewer) and map.mp4 (fly-around) into the run folder. SfM working
files (keyframes, COLMAP database, sparse model) go in <run>/sfm/.
"""
from __future__ import annotations

import argparse
import csv
import math
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from mapping.geometry import (
    align,
    camera_from_heading,
    estimate_up,
    focal_px,
    ground_height,
    person_positions,
    scale_from_depth,
    scale_from_ground,
)
from mapping.render import MapData, height_colors, write_flyaround, write_html
from mapping.sfm import SfmResult, run_sfm  # pycolmap itself is only imported once SfM actually runs

NOMINAL_FPS = 15.0  # the F19's video rate, for when frames.csv is missing
FRAME_W, FRAME_H = 640, 360
FLIGHT_HEIGHT_M = 1.2  # --from-track has no altitude; about where the F19 follows from
EDGE = 0.005  # a box this close to the top/bottom edge is cut off, so its distance reads long
FOLLOWING = ("LOCKED", "COASTING")


@dataclass
class Track:
    """track.csv as arrays; blanks are NaN."""

    t: np.ndarray
    state: np.ndarray  # str
    cx: np.ndarray
    cy: np.ndarray
    w: np.ndarray
    h: np.ndarray
    dist: np.ndarray
    drone_x: np.ndarray
    drone_y: np.ndarray
    heading: np.ndarray  # radians

    @classmethod
    def empty(cls) -> Track:
        e = np.zeros(0)
        return cls(e, np.zeros(0, dtype=object), e, e, e, e, e, e, e, e)

    def box_ok(self, states=FOLLOWING) -> np.ndarray:
        """Rows with a whole person box and a distance worth trusting."""
        whole = (self.cy - self.h / 2 > EDGE) & (self.cy + self.h / 2 < 1 - EDGE)
        have = np.isfinite(self.cx) & np.isfinite(self.cy) & np.isfinite(self.dist) & (self.dist > 0)
        return np.isin(self.state, states) & have & whole

    def nearest(self, times: np.ndarray, max_dt: float = 0.2) -> np.ndarray:
        """Row index nearest each time, or -1 when no row is within max_dt."""
        times = np.asarray(times, float)
        if len(self.t) == 0:
            return np.full(len(times), -1)
        i = np.clip(np.searchsorted(self.t, times), 0, len(self.t) - 1)
        j = np.maximum(i - 1, 0)
        i = np.where(np.abs(self.t[j] - times) <= np.abs(self.t[i] - times), j, i)
        return np.where(np.abs(self.t[i] - times) <= max_dt, i, -1)


def _num(s: str) -> float:
    try:
        return float(s)
    except ValueError:
        return math.nan


def load_track(path: Path) -> Track:
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    col = lambda name: np.array([_num(r.get(name) or "") for r in rows], float)  # noqa: E731
    order = np.argsort(col("t"), kind="stable")
    return Track(
        t=col("t")[order],
        state=np.array([(r.get("state") or "").upper() for r in rows], dtype=object)[order],
        cx=col("cx")[order],
        cy=col("cy")[order],
        w=col("w")[order],
        h=col("h")[order],
        dist=col("distance_m")[order],
        drone_x=col("drone_x")[order],
        drone_y=col("drone_y")[order],
        heading=np.radians(col("drone_heading_deg"))[order],
    )


def load_frame_times(path: Path) -> dict[int, float]:
    with open(path, newline="") as fh:
        return {int(r["frame"]): float(r["t"]) for r in csv.DictReader(fh)}


# --- keyframes ---------------------------------------------------------------------------------------------

def sharpness(img: np.ndarray) -> float:
    """Variance of the Laplacian: high for crisp edges, low for motion blur and JPEG mush."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def pick_sharpest(times, scores, window_s: float = 1 / 3) -> list[int]:
    """Index of the sharpest frame in each window_s slice of time, in time order."""
    times = np.asarray(times, float)
    if len(times) == 0:
        return []
    slot = np.floor((times - times.min()) / window_s).astype(int)
    best: dict[int, int] = {}
    for i, (s, score) in enumerate(zip(slot, scores)):
        if s not in best or score > scores[best[s]]:
            best[s] = i
    return [best[s] for s in sorted(best)]


@dataclass
class Keyframe:
    frame: int
    t: float
    name: str


def extract_keyframes(video: Path, frame_times: dict[int, float], out_dir: Path, window_s: float) -> tuple[list[Keyframe], tuple[int, int]]:
    """Two passes over the video: score every frame, then save the winners. Returns (keyframes, (w, h))."""
    cap = cv2.VideoCapture(str(video))
    scores, size = [], (FRAME_W, FRAME_H)
    while True:
        ok, img = cap.read()
        if not ok:
            break
        size = (img.shape[1], img.shape[0])
        scores.append(sharpness(img))
    cap.release()
    if not scores:
        raise SystemExit(f"{video}: no frames could be read.")
    times = [frame_times.get(i, i / NOMINAL_FPS) for i in range(len(scores))]
    chosen = set(pick_sharpest(times, scores, window_s))

    out_dir.mkdir(parents=True, exist_ok=True)
    keys = []
    cap = cv2.VideoCapture(str(video))
    for i in range(len(scores)):
        ok, img = cap.read()
        if not ok:
            break
        if i in chosen:
            name = f"{i:06d}.jpg"
            cv2.imwrite(str(out_dir / name), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            keys.append(Keyframe(i, times[i], name))
    cap.release()
    return keys, size


# --- building the map --------------------------------------------------------------------------------------

def trim_far(points: np.ndarray, around: np.ndarray, pct: float = 98) -> np.ndarray:
    """Drop the farthest few points: SfM always triangulates a handful of near-parallel rays at huge range."""
    if len(points) < 50:
        return points
    d = np.linalg.norm(points - np.median(around, axis=0), axis=1)
    return points[d <= np.percentile(d, pct)]


def map_from_sfm(run: Path, window_s: float, seed: int = 0) -> MapData:
    frame_times = load_frame_times(run / "frames.csv") if (run / "frames.csv").exists() else {}
    track = load_track(run / "track.csv") if (run / "track.csv").exists() else Track.empty()
    work = run / "sfm"
    for old in ("images", "sparse"):  # leftovers from a previous run would mix into this one
        shutil.rmtree(work / old, ignore_errors=True)
    keys, (w, h) = extract_keyframes(run / "raw.mp4", frame_times, work / "images", window_s)
    print(f"{len(keys)} keyframes (sharpest per {window_s:.2f} s); running COLMAP...")
    sfm = run_sfm(work, work / "images", [k.name for k in keys], focal_px(w), w, h, seed)
    return map_from_reconstruction(sfm, keys, track, w, h)


def map_from_reconstruction(sfm: SfmResult, keys: list[Keyframe], track: Track, w: int, h: int) -> MapData:
    """SfM output + the follow log -> map frame (z up, metres), person path, height colors."""
    reg = [k for k in keys if k.name in sfm.cams]  # time order
    if len(reg) < 2:
        raise SystemExit("COLMAP registered fewer than 2 frames; nothing to map.")
    C = np.array([sfm.cams[k.name][0] for k in reg])
    R = np.array([sfm.cams[k.name][1] for k in reg])
    points = trim_far(sfm.points, C)

    row = track.nearest(np.array([k.t for k in reg]))
    has = row >= 0
    locked = np.zeros(len(reg), bool)
    usable = np.zeros(len(reg), bool)
    locked[has] = track.box_ok(("LOCKED",))[row[has]]  # fresh detections only for scale
    usable[has] = track.box_ok()[row[has]]
    r = np.where(has, row, 0)
    boxes = np.column_stack([track.cx[r], track.cy[r], track.w[r], track.h[r]])

    up, up_from = estimate_up(C, R, points)
    ground, found_ground = ground_height(points, C, up)
    seen = (C[locked], R[locked], boxes[locked], track.dist[r][locked], sfm.focal, w, h)
    scale, n_scale = scale_from_ground(*seen, up, ground) if found_ground else (None, 0)
    scale_from = f"person on the ground, {n_scale} frames"
    if scale is None:  # no ground plane: fall back to depth around the person (right only if they held still)
        scale, n_scale = scale_from_depth(points, *seen)
        scale_from = f"points around the person, {n_scale} frames"
    if scale is None:
        print("warning: nothing to measure scale from (no locked person, no ground); map is in SfM units")
        scale, scale_from = 1.0, "none (SfM units, not metres)"
    sim = align(C, R, up, scale, ground)
    points, C, R = sim.points(points), sim.points(C), sim.rotations(R)
    person = person_positions(C[usable], R[usable], track.cx[r][usable], track.cy[r][usable], track.dist[r][usable],
                              sfm.focal, w, h)
    person = smooth(person, 3)  # ~1 s at three keyframes a second; camera centers are already smooth
    colors, zrange = height_colors(points[:, 2])
    info = [
        ("source", "COLMAP SfM"),
        ("keyframes", f"{len(reg)} of {len(keys)} registered"),
        ("points", f"{len(points):,}"),
        ("up from", up_from),
        ("scale from", scale_from),
        ("path", f"{path_length(C):.1f} m"),
    ]
    if sfm.models > 1:
        info.append(("note", f"COLMAP split the flight into {sfm.models} pieces; showing the biggest"))
    t = np.array([k.t for k in reg])
    return MapData(points, colors, C, person, zrange, info, drone_t=t, person_t=t[usable])


def path_length(p: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum()) if len(p) > 1 else 0.0


def smooth(path: np.ndarray, k: int) -> np.ndarray:
    """Centered moving average over k samples (fewer at the ends). Display only: takes out the jitter of
    box-size distances and tracker poses."""
    if k <= 1 or len(path) < 3:
        return path
    c = np.vstack([np.zeros((1, path.shape[1])), np.cumsum(path, axis=0)])
    i = np.arange(len(path))
    lo, hi = np.maximum(i - k // 2, 0), np.minimum(i + k // 2 + 1, len(path))
    return (c[hi] - c[lo]) / (hi - lo)[:, None]


def samples_in(seconds: float, t: np.ndarray) -> int:
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 1.0
    return max(1, round(seconds / dt) if dt > 0 else 1)


def map_from_track(track: Track, w: int = FRAME_W, h: int = FRAME_H, height: float = FLIGHT_HEIGHT_M) -> MapData:
    """No SfM: the overhead tracker's drone_x/drone_y (simulator flights), at an assumed height."""
    pos = np.isfinite(track.drone_x) & np.isfinite(track.drone_y)
    if not pos.any():
        raise SystemExit("track.csv has no drone_x/drone_y; --from-track needs a flight with position tracking on.")
    drone = np.column_stack([track.drone_x[pos], track.drone_y[pos], np.full(pos.sum(), height)])
    seen = pos & track.box_ok() & np.isfinite(track.heading)
    R = np.array([camera_from_heading(a) for a in track.heading[seen]]).reshape(-1, 3, 3)
    C = np.column_stack([track.drone_x[seen], track.drone_y[seen], np.full(seen.sum(), height)])
    person = person_positions(C, R, track.cx[seen], track.cy[seen], track.dist[seen], focal_px(w), w, h)
    k = samples_in(0.5, track.t)
    drone, person = smooth(drone, k), smooth(person, k)
    info = [
        ("source", "track.csv (no SfM)"),
        ("drone samples", str(len(drone))),
        ("person samples", str(len(person))),
        ("height", f"assumed {height:.1f} m"),
        ("path", f"{path_length(drone):.1f} m"),
    ]
    return MapData(np.zeros((0, 3)), np.zeros((0, 3), np.uint8), drone, person, (0.0, 1.0), info)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m mapping.reconstruct", description=__doc__.split("\n\n")[0])
    p.add_argument("run", type=Path, help="a flights/<run>/ folder written by the recorder")
    p.add_argument("--from-track", action="store_true", help="skip SfM; map drone_x/drone_y from track.csv (sim flights)")
    p.add_argument("--window", type=float, default=1 / 3, help="keep the sharpest frame per this many seconds (default 1/3)")
    p.add_argument("--no-video", action="store_true", help="skip rendering map.mp4")
    args = p.parse_args(argv)

    run = args.run
    if not run.is_dir():
        raise SystemExit(f"{run} isn't a folder.")
    if args.from_track:
        data = map_from_track(load_track(run / "track.csv"))
    else:
        if not (run / "raw.mp4").exists():
            raise SystemExit(f"{run / 'raw.mp4'} not found (use --from-track for simulator flights).")
        data = map_from_sfm(run, args.window)

    title = f"{run.resolve().name} - 3D map"
    write_html(run / "map.html", data, title)
    print(f"wrote {run / 'map.html'}")
    if not args.no_video:
        write_flyaround(run / "map.mp4", data, title)
        print(f"wrote {run / 'map.mp4'}")
    for k, v in data.info:
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

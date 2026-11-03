"""Saves a flight for review and 3D mapping.

flights/<run>/
  raw.mp4      the drone's camera as received (input for: python -m mapping.reconstruct flights/<run>)
  frames.csv   frame,t: video frame index -> seconds since recording started (the drone's frame rate wobbles)
  overlay.mp4  the drone view with the lock box and telemetry, as shown on screen
  track.csv    t,state,id,cx,cy,w,h,distance_m,yaw_cmd,pitch_cmd,drone_x,drone_y,drone_heading_deg
               box normalized 0-1 to the frame; state is OFF while not following;
               drone_* blank unless position tracking is on; yaw/pitch are the sticks actually sent
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import cv2
import numpy as np

TRACK_COLUMNS = ["t", "state", "id", "cx", "cy", "w", "h", "distance_m", "yaw_cmd", "pitch_cmd",
                 "drone_x", "drone_y", "drone_heading_deg"]


class Recorder:
    FPS = 15.0  # the F19's nominal video rate; frames.csv has the real timing

    def __init__(self, folder: str | Path, start_time: float):
        self.dir = Path(folder)
        if (self.dir / "raw.mp4").exists():  # a run that never got video can be reused
            raise SystemExit(f"{self.dir} already has a recording in it; pick a new folder.")
        self.dir.mkdir(parents=True, exist_ok=True)
        self.t0 = start_time
        self.frames = 0
        self._videos: dict[str, cv2.VideoWriter] = {}
        self._sizes: dict[str, tuple[int, int]] = {}
        self._frames_file = open(self.dir / "frames.csv", "w", newline="")
        self._track_file = open(self.dir / "track.csv", "w", newline="")
        self._frames_csv = csv.writer(self._frames_file)
        self._track_csv = csv.writer(self._track_file)
        self._frames_csv.writerow(["frame", "t"])
        self._track_csv.writerow(TRACK_COLUMNS)

    def _write(self, name: str, frame: np.ndarray) -> None:
        if name not in self._videos:
            h, w = frame.shape[:2]
            self._sizes[name] = (w, h)
            self._videos[name] = cv2.VideoWriter(str(self.dir / name), cv2.VideoWriter_fourcc(*"mp4v"), self.FPS, (w, h))
        if (frame.shape[1], frame.shape[0]) != self._sizes[name]:
            frame = cv2.resize(frame, self._sizes[name])
        self._videos[name].write(frame)

    def add(self, t: float, raw: np.ndarray, overlay: np.ndarray | None, follow=None, axes=None, pose=None) -> None:
        """One new drone frame. `follow` is the FollowSystem (or None), `axes` the sticks sent, `pose` the tracked Pose."""
        rel = t - self.t0
        self._write("raw.mp4", raw)
        if overlay is not None:
            self._write("overlay.mp4", overlay)
        self._frames_csv.writerow([self.frames, f"{rel:.3f}"])
        self.frames += 1

        state, label, box, dist = "OFF", "", ("", "", "", ""), ""
        if follow is not None and follow.active:
            tr = follow.tracker
            state = tr.state.value.upper()
            if tr.target is not None:
                label = tr.target_label
                h, w = raw.shape[:2]
                x0, y0, x1, y1 = tr.target.box_at(t)
                box = (f"{(x0 + x1) / 2 / w:.4f}", f"{(y0 + y1) / 2 / h:.4f}", f"{(x1 - x0) / w:.4f}", f"{(y1 - y0) / h:.4f}")
            if follow.distance is not None:
                dist = f"{follow.distance:.2f}"
        sticks = (f"{axes.yaw:.3f}", f"{axes.pitch:.3f}") if axes is not None else ("", "")
        where = (f"{pose.x:.3f}", f"{pose.y:.3f}", f"{math.degrees(pose.heading):.1f}") if pose is not None else ("", "", "")
        self._track_csv.writerow([f"{rel:.3f}", state, label, *box, dist, *sticks, *where])

    def close(self) -> None:
        for v in self._videos.values():
            v.release()
        self._frames_file.close()
        self._track_file.close()

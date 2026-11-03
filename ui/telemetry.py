"""Strip charts under the drone view: distance, yaw and pitch over the last 20 s, shaded by lock state."""
from __future__ import annotations

import cv2
import numpy as np

from follow.controller import FollowConfig
from follow.system import STATE_COLORS, Sample

ROWS = (("DIST m", 0.0, 10.0), ("YAW", -1.0, 1.0), ("PITCH", -1.0, 1.0))


def render_strip(history, cfg: FollowConfig, width: int = 640, height: int = 150, span_s: float = 20.0) -> np.ndarray:
    img = np.full((height, width, 3), 22, np.uint8)
    samples: list[Sample] = list(history)
    label_w, row_h = 62, height // len(ROWS)
    plot_w = width - label_w - 6
    if not samples:
        cv2.putText(img, "telemetry appears once follow is on", (label_w, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (130, 130, 130), 1)
        return img
    t_end = samples[-1].t

    def x_of(t: float) -> int:
        return label_w + int(plot_w * (1 - (t_end - t) / span_s))

    for a, b in zip(samples, samples[1:]):  # lock-state bands behind everything
        if a.state is not None:
            band = tuple(int(c * 0.22) for c in STATE_COLORS[a.state])
            cv2.rectangle(img, (x_of(a.t), 0), (x_of(b.t), height), band, -1)

    for r, (name, lo, hi) in enumerate(ROWS):
        top = r * row_h

        def y_of(v: float) -> int:
            v = max(lo, min(hi, v))
            return top + 4 + int((row_h - 8) * (1 - (v - lo) / (hi - lo)))

        cv2.line(img, (label_w, top + row_h - 1), (width, top + row_h - 1), (55, 55, 55), 1)
        cv2.putText(img, name, (6, top + row_h // 2 + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (190, 190, 190), 1)
        if r == 0:
            for v, color in ((cfg.distance, (80, 220, 80)), (cfg.min_distance, (60, 60, 230))):
                cv2.line(img, (label_w, y_of(v)), (width, y_of(v)), color, 1)
        else:
            cv2.line(img, (label_w, y_of(0.0)), (width, y_of(0.0)), (70, 70, 70), 1)
        values = [s.distance if r == 0 else (s.yaw if r == 1 else s.pitch) for s in samples]
        seg: list[tuple[int, int]] = []
        for s, v in zip(samples, values):
            if v is None:
                if len(seg) > 1:
                    cv2.polylines(img, [np.int32(seg)], False, (235, 235, 235), 1, cv2.LINE_AA)
                seg = []
            else:
                seg.append((x_of(s.t), y_of(v)))
        if len(seg) > 1:
            cv2.polylines(img, [np.int32(seg)], False, (235, 235, 235), 1, cv2.LINE_AA)
        last = next((v for v in reversed(values) if v is not None), None)
        if last is not None:
            cv2.putText(img, f"{last:+.2f}" if r else f"{last:.1f}", (6, top + row_h // 2 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (235, 235, 235), 1)
    return img

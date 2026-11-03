"""Synthetic hand observations: every landmark sits at the palm center except the index tip."""
from __future__ import annotations

from gesture.tracker import HandObservation
from gesture.mapper import INDEX_TIP


def hand(gesture: str | None, palm=(0.5, 0.5), tip=None, score: float = 0.9) -> HandObservation:
    pts = [(palm[0], palm[1], 0.0)] * 21
    if tip is not None:
        pts[INDEX_TIP] = (tip[0], tip[1], 0.0)
    return HandObservation(gesture=gesture, score=score, landmarks=tuple(pts))

"""The FACE LOCK panel: an info card joined to the locked person's box, plus brackets around their face."""
from __future__ import annotations

import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX
SCALE, TITLE_SCALE = 0.36, 0.46
LINE_H, TITLE_H, PAD, GAP = 13, 18, 6, 16


def _ints(box) -> tuple[int, int, int, int]:
    return tuple(int(round(v)) for v in box)  # type: ignore[return-value]


def _text_w(text: str, scale: float = SCALE) -> int:
    return cv2.getTextSize(text, FONT, scale, 1)[0][0]


def draw_brackets(img: np.ndarray, box, color, frac: float = 0.3) -> None:
    """Four corner marks instead of a full rectangle."""
    x0, y0, x1, y1 = _ints(box)
    n = max(3, int(frac * min(x1 - x0, y1 - y0)))
    for x, y, dx, dy in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)):
        cv2.line(img, (x, y), (x + dx * n, y), color, 1, cv2.LINE_AA)
        cv2.line(img, (x, y), (x, y + dy * n), color, 1, cv2.LINE_AA)


def draw_target_card(
    img: np.ndarray,
    box,
    title: str,
    rows: list[tuple[str, str]],
    live: list[tuple[str, str]],
    color,
    face=None,
) -> tuple[int, int, int, int]:
    """Draws the card beside `box` (right side if it fits, else left) and returns where it went."""
    h, w = img.shape[:2]
    if face is not None:
        draw_brackets(img, face, color)
    every = rows + live
    label_w = max((_text_w(k) for k, _ in every), default=0)
    value_w = max((_text_w(v) for _, v in every), default=0)
    card_w = max(PAD + label_w + 10 + value_w + PAD, _text_w(title, TITLE_SCALE) + 2 * PAD)
    card_h = TITLE_H + 4 + LINE_H * len(every) + (5 if rows and live else 0) + PAD
    card_w, card_h = min(card_w, w - 8), min(card_h, h - 8)

    bx0, by0, bx1, by1 = _ints(box)
    on_right = bx1 + GAP + card_w <= w - 4 or bx0 - GAP - card_w < 4
    x = bx1 + GAP if on_right else bx0 - GAP - card_w
    x = max(4, min(x, w - 4 - card_w))
    y = max(4, min(by0, h - 4 - card_h))

    roi = img[y:y + card_h, x:x + card_w]
    cv2.addWeighted(roi, 0.35, np.full_like(roi, (18, 22, 18)), 0.65, 0, dst=roi)
    cv2.rectangle(img, (x, y), (x + card_w, y + card_h), color, 1)
    cv2.rectangle(img, (x, y), (x + card_w, y + TITLE_H), color, -1)
    cv2.putText(img, title, (x + PAD, y + TITLE_H - 5), FONT, TITLE_SCALE, (20, 20, 20), 1, cv2.LINE_AA)
    corner = (bx1, by0) if on_right else (bx0, by0)
    cv2.line(img, corner, (x if on_right else x + card_w, y + TITLE_H // 2), color, 1, cv2.LINE_AA)

    ty = y + TITLE_H + 4
    for i, (label, value) in enumerate(every):
        if rows and i == len(rows):
            cv2.line(img, (x + PAD, ty + 1), (x + card_w - PAD, ty + 1), color, 1)
            ty += 5
        ty += LINE_H
        if ty > y + card_h - 2:
            break
        cv2.putText(img, label, (x + PAD, ty - 3), FONT, SCALE, (170, 190, 170), 1, cv2.LINE_AA)
        cv2.putText(img, value, (x + PAD + label_w + 10, ty - 3), FONT, SCALE, (245, 245, 245), 1, cv2.LINE_AA)
    return x, y, x + card_w, y + card_h

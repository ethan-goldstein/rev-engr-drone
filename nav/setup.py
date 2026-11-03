"""One-time setup for position tracking (run from the project folder).

  python -m nav.setup marker                  # printable tag for the top of the drone
  python -m nav.setup arena --camera 1        # click the 4 corners of the flying area -> arena.json
  python -m nav.setup path --camera 1         # optional: click autopilot waypoints -> path.json

Add --image photo.jpg to `arena` or `path` to click on a still photo instead of the live camera.
"""
from __future__ import annotations

import argparse
import json
import struct
import zlib
from pathlib import Path

import cv2
import numpy as np

from nav.arena import Arena
from nav.marker_tracker import DICTIONARY, MARKER_ID, open_camera

WINDOW = "nav setup"
CORNERS = ("TOP-LEFT", "TOP-RIGHT", "BOTTOM-RIGHT", "BOTTOM-LEFT")
DPI = 300


# ---------------------------------------------------------------------- marker
def marker_image(size_cm: float = 5.0) -> np.ndarray:
    """The tag at `size_cm` (at 300 dpi), a 1 cm white border, and an arrow above it marking the nose."""
    px_cm = DPI / 2.54
    side, border = round(size_cm * px_cm), round(px_cm)
    tag = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(DICTIONARY), MARKER_ID, side)
    top = round(2.2 * px_cm)  # room for the arrow, which stays outside the tag's white border
    img = np.full((top + side + border, side + 2 * border), 255, np.uint8)
    img[top : top + side, border : border + side] = tag
    cx, tip, tail = img.shape[1] // 2, round(0.3 * px_cm), round(1.1 * px_cm)
    cv2.arrowedLine(img, (cx, tail), (cx, tip), 0, max(2, round(0.08 * px_cm)), tipLength=0.4)
    cv2.putText(img, "NOSE", (cx + round(0.3 * px_cm), tail), cv2.FONT_HERSHEY_SIMPLEX, px_cm / 60, 0, 2)
    return img


def png_with_dpi(img: np.ndarray, dpi: int = DPI) -> bytes:
    """PNG bytes with a pHYs chunk, so printing at 100% gives the real size."""
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("PNG encoding failed")
    data = bytes(buf)
    ppm = round(dpi / 0.0254)
    body = b"pHYs" + struct.pack(">IIB", ppm, ppm, 1)
    chunk = struct.pack(">I", 9) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)
    ihdr_end = 8 + 4 + 4 + 13 + 4  # signature, IHDR length, type, data, CRC
    return data[:ihdr_end] + chunk + data[ihdr_end:]


def cmd_marker(args: argparse.Namespace) -> None:
    out = Path(args.out)
    out.write_bytes(png_with_dpi(marker_image(args.size_cm)))
    print(f"Wrote {out}. Print at 100% (no 'fit to page'); the black square should measure {args.size_cm:g} cm.")
    print("Cut it out with its white border, and tape it flat on top of the drone with the arrow toward the nose.")


# ---------------------------------------------------------------------- clicking
def grab_frame(args: argparse.Namespace) -> np.ndarray:
    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            raise SystemExit(f"Could not read {args.image}")
        return frame
    cap = open_camera(args.camera)
    frame = None
    for _ in range(30):  # let auto-exposure settle
        ok, f = cap.read()
        if ok:
            frame = f
    cap.release()
    if frame is None:
        raise SystemExit("The tracking camera didn't return a picture.")
    return frame


def click_points(frame: np.ndarray, prompt, max_points: int | None, draw_extra=None) -> list[tuple[float, float]]:
    """Click points on `frame`. Enter accepts, Backspace undoes, R restarts, Esc cancels."""
    points: list[tuple[float, float]] = []

    def on_mouse(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN and (max_points is None or len(points) < max_points):
            points.append((float(x), float(y)))

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW, on_mouse)
    while True:
        view = frame.copy()
        if draw_extra:
            draw_extra(view)
        for i, p in enumerate(points):
            cv2.circle(view, (int(p[0]), int(p[1])), 8, (0, 220, 255), -1)
            cv2.putText(view, str(i + 1), (int(p[0]) + 10, int(p[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 220, 255), 2)
        if len(points) > 1:
            cv2.polylines(view, [np.int32(points).reshape(-1, 1, 2)], max_points == 4 and len(points) == 4, (0, 220, 255), 2)
        cv2.putText(view, prompt(points), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 5)
        cv2.putText(view, prompt(points), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.imshow(WINDOW, view)
        key = cv2.waitKey(30) & 0xFF
        if key in (13, 10) and (max_points is None or len(points) == max_points) and points:
            break
        if key in (8, 127) and points:
            points.pop()
        elif key in (ord("r"), ord("R")):
            points.clear()
        elif key == 27:
            cv2.destroyAllWindows()
            raise SystemExit("Cancelled.")
    cv2.destroyAllWindows()
    return points


def ask_meters(question: str) -> float:
    while True:
        try:
            value = float(input(question).strip())
            if value > 0:
                return value
        except ValueError:
            pass
        print("Please enter a positive number of meters, e.g. 2.5")


def cmd_arena(args: argparse.Namespace) -> None:
    frame = grab_frame(args)
    print("Click the 4 corners of the flying area on the picture, in order: top-left, top-right, bottom-right, "
          "bottom-left (as they appear on screen). Tape on the floor at the corners helps.")
    prompt = lambda pts: f"click the {CORNERS[len(pts)]} corner" if len(pts) < 4 else "Enter = save, R = redo"  # noqa: E731
    corners = click_points(frame, prompt, 4)
    width = ask_meters("Distance from the left edge to the right edge of that area, in meters: ")
    depth = ask_meters("Distance from the top edge to the bottom edge (as seen on screen), in meters: ")
    arena = Arena(width, depth, tuple(corners))
    arena.save(args.out)
    print(f"Wrote {args.out}: a {width:g} m x {depth:g} m flying area. The barrier sits 0.3 m inside its edges.")


def cmd_path(args: argparse.Namespace) -> None:
    arena = Arena.load(args.arena)
    frame = grab_frame(args)

    def outline(view):
        cv2.polylines(view, [np.int32(arena.image_corners).reshape(-1, 1, 2)], True, (60, 60, 230), 3)

    prompt = lambda pts: f"{len(pts)} waypoints - click to add, Backspace = undo, Enter = save"  # noqa: E731
    clicks = click_points(frame, prompt, None, outline)
    waypoints = [list(arena.to_world(p)) for p in clicks]
    Path(args.out).write_text(json.dumps(waypoints, indent=2))
    print(f"Wrote {args.out} with {len(waypoints)} waypoints. Fly it with: main.py --path {args.out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    m = sub.add_parser("marker", help="write a printable tag for the top of the drone")
    m.add_argument("--size-cm", type=float, default=5.0, help="side of the black square (default 5)")
    m.add_argument("--out", default="marker_0.png")
    m.set_defaults(func=cmd_marker)

    a = sub.add_parser("arena", help="click the 4 corners of the flying area")
    a.add_argument("--camera", type=int, default=1, help="tracking camera number (default 1)")
    a.add_argument("--image", help="click on a still photo instead of the live camera")
    a.add_argument("--out", default="arena.json")
    a.set_defaults(func=cmd_arena)

    p = sub.add_parser("path", help="click autopilot waypoints inside the flying area")
    p.add_argument("--camera", type=int, default=1)
    p.add_argument("--image")
    p.add_argument("--arena", default="arena.json")
    p.add_argument("--out", default="path.json")
    p.set_defaults(func=cmd_path)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

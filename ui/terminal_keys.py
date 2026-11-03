"""Keys typed in the terminal that started the app (VS Code's, say) work like keys in the drone window.

macOS usually leaves keyboard focus on the terminal rather than the OpenCV window, so F, SPACE and the
trim keys used to vanish into the terminal. This reads single key presses from it without waiting for
Enter. Ctrl+C still quits. Arrow keys and other escape sequences are ignored, so an arrow can't be read
as ESC (quit) or "[" (trim).
"""
from __future__ import annotations

import os
import queue
import select
import sys
import threading

try:
    import termios
    import tty
except ImportError:  # Windows: only the OpenCV window takes keys
    termios = tty = None


class TerminalKeys:
    def __init__(self, stream=None):
        self._keys: queue.Queue[int] = queue.Queue()
        self._fd: int | None = None
        self._saved = None
        stream = stream if stream is not None else sys.stdin
        if termios is None or not stream.isatty():
            return
        self._fd = stream.fileno()
        self._saved = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)  # one key at a time, no echo; Ctrl+C still interrupts
        threading.Thread(target=self._read, daemon=True).start()

    def _byte(self, timeout: float | None = None) -> bytes:
        if timeout is not None and not select.select([self._fd], [], [], timeout)[0]:
            return b""
        return os.read(self._fd, 1)

    def _read(self) -> None:
        try:
            while True:
                b = self._byte()
                if not b:
                    return
                if b != b"\x1b":
                    self._keys.put(b[0])
                    continue
                # Escape sequence (arrow keys etc.): drop it up to its final byte, keep what follows.
                if self._byte(0.05) in (b"[", b"O"):
                    while True:
                        c = self._byte(0.05)
                        if not c or 0x40 <= c[0] <= 0x7E:
                            break
        except OSError:
            return

    def get(self) -> int:
        """The next key pressed in the terminal, or -1."""
        try:
            return self._keys.get_nowait()
        except queue.Empty:
            return -1

    def close(self) -> None:
        if self._saved is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._saved)
            self._saved = None

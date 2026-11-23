"""Where frames come from: the game window, ADB, a video file.

Every source returns RGB numpy frames. The window source is what normal use
needs; ADB works when the emulator window is hidden or on another machine;
video is for testing the pipeline on a recording.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from typing import Optional, Tuple

import numpy as np

from . import winapi


class Source:
    name = "source"

    def read(self) -> Optional[np.ndarray]:
        raise NotImplementedError

    def close(self) -> None:
        pass

    # Where the overlay should dock: (left, top, width, height) on screen,
    # or None when the source has no window.
    def dock_rect(self) -> Optional[Tuple[int, int, int, int]]:
        return None

    def visible(self) -> bool:
        return True


def content_box(frame: np.ndarray, threshold: int = 12) -> Tuple[int, int, int, int]:
    """Bounding box of the non-black part of a frame (letterbox removal)."""
    grey = frame.max(axis=2) if frame.ndim == 3 else frame
    rows = np.flatnonzero(grey.max(axis=1) > threshold)
    cols = np.flatnonzero(grey.max(axis=0) > threshold)
    if rows.size == 0 or cols.size == 0:
        return 0, 0, frame.shape[1], frame.shape[0]
    return int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1


class WindowSource(Source):
    """The game window on this PC.

    PrintWindow first; if it keeps returning black (some GPU-rendered
    windows do), fall back to copying the window's area of the screen, which
    then needs the window to be uncovered.
    """

    name = "window"

    def __init__(self, title: Optional[str] = None, hwnd: Optional[int] = None,
                 method: str = "auto"):
        self.window = winapi.find(title, hwnd)
        self.method = method
        self._printer = winapi.PrintWindowGrabber(self.window.hwnd)
        self._black = 0

    def read(self) -> Optional[np.ndarray]:
        hwnd = self.window.hwnd
        if self.method in ("auto", "printwindow"):
            frame = self._printer.grab()
            if frame is not None and frame.max() > 8:
                self._black = 0
                return frame
            self._black += 1
            if self.method == "printwindow" or self._black < 5:
                return frame
        return winapi.grab_screen(hwnd)

    def dock_rect(self):
        return winapi.frame_rect(self.window.hwnd)

    def visible(self) -> bool:
        return winapi.alive(self.window.hwnd) and not winapi.minimized(self.window.hwnd)

    def close(self) -> None:
        self._printer.close()


class AdbSource(Source):
    """`adb exec-out screencap` - slower (~100 ms), but works with the
    emulator window minimised or behind everything."""

    name = "adb"

    def __init__(self, serial: Optional[str] = None, adb: str = "adb"):
        path = shutil.which(adb) or adb
        self.cmd = [path] + (["-s", serial] if serial else []) + ["exec-out", "screencap"]

    def read(self) -> Optional[np.ndarray]:
        try:
            raw = subprocess.run(self.cmd, capture_output=True, timeout=5).stdout
        except (OSError, subprocess.TimeoutExpired):
            return None
        if len(raw) < 16:
            return None
        width, height = int.from_bytes(raw[0:4], "little"), int.from_bytes(raw[4:8], "little")
        # Header is 12 bytes on older Android, 16 (with colour space) on newer.
        for header in (16, 12):
            body = raw[header:header + width * height * 4]
            if len(body) == width * height * 4:
                return np.frombuffer(body, np.uint8).reshape(height, width, 4)[:, :, :3].copy()
        return None


class VideoSource(Source):
    """A recording, played at its own speed (or as fast as possible)."""

    name = "video"

    def __init__(self, path: str, realtime: bool = True, start_s: float = 0.0):
        import cv2
        self.cv2 = cv2
        self.capture = cv2.VideoCapture(path)
        if not self.capture.isOpened():
            raise RuntimeError(f"cannot open video {path}")
        if start_s:
            self.capture.set(cv2.CAP_PROP_POS_MSEC, start_s * 1000.0)
        self.realtime = realtime
        self._t0 = time.monotonic()
        self._v0 = self.position()

    def position(self) -> float:
        return self.capture.get(self.cv2.CAP_PROP_POS_MSEC) / 1000.0

    def read(self) -> Optional[np.ndarray]:
        if self.realtime:
            # Skip ahead to "now" rather than queueing frames: a live overlay
            # wants the newest frame, and so does a test of one.
            target = self._v0 + (time.monotonic() - self._t0)
            while self.position() < target - 0.05:
                if not self.capture.grab():
                    return None
        ok, frame = self.capture.read()
        return frame[:, :, ::-1].copy() if ok else None

    def close(self) -> None:
        self.capture.release()

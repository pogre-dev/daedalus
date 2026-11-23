"""Find the Clash Royale window and copy its pixels (Windows only).

Adapted from Solar-CR's studio capture (`scripts/studio/capture.py`), which
measured the options against a live MuMu window:

* `PrintWindow(PW_RENDERFULLCONTENT)` asks the window to draw itself into our
  bitmap, so it keeps working when another window covers part of it - BitBlt
  from the desktop returned black in that case.
* An emulator renders Android into a *child* window (`nemuwin` for MuMu,
  `BlueStacksApp` for BlueStacks...). Capturing that child gives the game with
  no title bar or toolbar to crop.

Google Play Games for PC (and anything else) is found by window title instead,
and if PrintWindow hands back a black frame the screen region is grabbed.
"""

from __future__ import annotations

import ctypes
import sys
import time
from dataclasses import dataclass
from typing import Iterator, List, Optional, Tuple

import numpy as np

IS_WINDOWS = sys.platform == "win32"

# Child-window classes that hold the Android surface, by emulator.
SURFACE_CLASSES = ("nemuwin", "subWin", "BlueStacksApp", "AndroidSurface")
# Top-level titles that are the game itself (Google Play Games, Windows build).
GAME_TITLES = ("Clash Royale",)

if IS_WINDOWS:
    import ctypes.wintypes as wintypes

    user32 = ctypes.windll.user32
    gdi32 = ctypes.windll.gdi32
    dwmapi = ctypes.windll.dwmapi
    _ENUM = ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)

    class BITMAPINFOHEADER(ctypes.Structure):
        _fields_ = [
            ("biSize", wintypes.DWORD), ("biWidth", ctypes.c_long),
            ("biHeight", ctypes.c_long), ("biPlanes", wintypes.WORD),
            ("biBitCount", wintypes.WORD), ("biCompression", wintypes.DWORD),
            ("biSizeImage", wintypes.DWORD), ("biXPelsPerMeter", ctypes.c_long),
            ("biYPelsPerMeter", ctypes.c_long), ("biClrUsed", wintypes.DWORD),
            ("biClrImportant", wintypes.DWORD),
        ]

    class BITMAPINFO(ctypes.Structure):
        _fields_ = [("bmiHeader", BITMAPINFOHEADER), ("bmiColors", wintypes.DWORD * 3)]


PW_RENDERFULLCONTENT = 2
GA_ROOT = 2
DWMWA_EXTENDED_FRAME_BOUNDS = 9


def require_windows() -> None:
    if not IS_WINDOWS:
        raise RuntimeError("window capture needs Windows; use --source adb/video/tracks elsewhere")


def set_dpi_aware() -> None:
    """Use physical pixels everywhere, so window rects and Tk agree.

    Without this a 150% display scale makes Windows report scaled rects and
    the overlay lands in the wrong place.
    """
    if not IS_WINDOWS:
        return
    try:
        user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))   # per-monitor v2
    except Exception:
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            try:
                user32.SetProcessDPIAware()
            except Exception:
                pass


def _text(hwnd: int) -> str:
    n = user32.GetWindowTextLengthW(hwnd)
    buf = ctypes.create_unicode_buffer(n + 1)
    user32.GetWindowTextW(hwnd, buf, n + 1)
    return buf.value


def _class(hwnd: int) -> str:
    buf = ctypes.create_unicode_buffer(256)
    user32.GetClassNameW(hwnd, buf, 256)
    return buf.value


def _enum_top() -> List[int]:
    found: List[int] = []
    user32.EnumWindows(_ENUM(lambda h, _: (found.append(h), True)[1]), 0)
    return found


def _enum_children(hwnd: int) -> List[int]:
    found: List[int] = []
    user32.EnumChildWindows(hwnd, _ENUM(lambda h, _: (found.append(h), True)[1]), 0)
    return found


def client_rect(hwnd: int) -> Tuple[int, int, int, int]:
    """(left, top, width, height) of a window's client area, in screen pixels."""
    r = wintypes.RECT()
    user32.GetClientRect(hwnd, ctypes.byref(r))
    pt = wintypes.POINT(0, 0)
    user32.ClientToScreen(hwnd, ctypes.byref(pt))
    return pt.x, pt.y, r.right - r.left, r.bottom - r.top


def frame_rect(hwnd: int) -> Tuple[int, int, int, int]:
    """Visible bounds of the top-level window (no invisible resize border)."""
    root = user32.GetAncestor(hwnd, GA_ROOT) or hwnd
    r = wintypes.RECT()
    if dwmapi.DwmGetWindowAttribute(root, DWMWA_EXTENDED_FRAME_BOUNDS,
                                    ctypes.byref(r), ctypes.sizeof(r)) != 0:
        user32.GetWindowRect(root, ctypes.byref(r))
    return r.left, r.top, r.right - r.left, r.bottom - r.top


def alive(hwnd: int) -> bool:
    return bool(user32.IsWindow(hwnd))


def minimized(hwnd: int) -> bool:
    root = user32.GetAncestor(hwnd, GA_ROOT) or hwnd
    return bool(user32.IsIconic(root))


def foreground_is(hwnd: int, also: Tuple[int, ...] = ()) -> bool:
    fg = user32.GetForegroundWindow()
    root = user32.GetAncestor(hwnd, GA_ROOT) or hwnd
    return fg in (root, hwnd, *also)


@dataclass(frozen=True)
class GameWindow:
    hwnd: int          # what to capture (an emulator's render surface, or the game window)
    title: str         # owning top-level window's title
    kind: str          # "surface" or "window"

    def __str__(self) -> str:
        return f"{self.title!r} ({self.kind}, hwnd {self.hwnd})"


def candidates() -> Iterator[GameWindow]:
    """Every plausible game window, largest first."""
    found = []
    for top in _enum_top():
        if not user32.IsWindowVisible(top):
            continue
        title = _text(top) or _class(top)
        for child in _enum_children(top):
            if _class(child) in SURFACE_CLASSES and user32.IsWindowVisible(child):
                _, _, w, h = client_rect(child)
                if w >= 200 and h >= 200:
                    found.append((w * h, GameWindow(child, title, "surface")))
        if any(t.lower() in title.lower() for t in GAME_TITLES):
            _, _, w, h = client_rect(top)
            if w >= 200 and h >= 200:
                found.append((w * h, GameWindow(top, title, "window")))
    for _, win in sorted(found, key=lambda p: -p[0]):
        yield win


def find(title: Optional[str] = None, hwnd: Optional[int] = None) -> GameWindow:
    """The window to watch: by handle, by title substring, or the best guess.

    With several emulators open, pass `title` (matched against the owning
    window's title) - size is not identity.
    """
    require_windows()
    if hwnd:
        if not alive(hwnd):
            raise RuntimeError(f"no window with handle {hwnd}")
        return GameWindow(hwnd, _text(hwnd), "window")
    found = list(candidates())
    if title:
        wanted = title.lower()
        for win in found:
            if wanted in win.title.lower():
                return win
        for top in _enum_top():
            if user32.IsWindowVisible(top) and wanted in _text(top).lower():
                return GameWindow(top, _text(top), "window")
        raise RuntimeError(f"no visible window titled like {title!r}. Open: "
                           + ", ".join(repr(w.title) for w in found[:8]))
    if not found:
        raise RuntimeError("no Clash Royale window found: start the emulator (MuMu, "
                           "BlueStacks...) or Google Play Games, or pass --window TITLE")
    return found[0]


class PrintWindowGrabber:
    """Copy one window's pixels into a reused DIB (most of the cost is setup)."""

    def __init__(self, hwnd: int):
        require_windows()
        self.hwnd = hwnd
        self._dc = None
        self._bitmap = None
        self._buffer = None
        self._size = (0, 0)

    def _allocate(self, width: int, height: int) -> None:
        self.close()
        info = BITMAPINFO()
        info.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
        info.bmiHeader.biWidth = width
        info.bmiHeader.biHeight = -height          # top-down rows
        info.bmiHeader.biPlanes = 1
        info.bmiHeader.biBitCount = 32
        bits = ctypes.c_void_p()
        self._dc = gdi32.CreateCompatibleDC(None)
        self._bitmap = gdi32.CreateDIBSection(self._dc, ctypes.byref(info), 0,
                                              ctypes.byref(bits), None, 0)
        if not self._bitmap or not bits.value:
            raise RuntimeError("CreateDIBSection failed")
        gdi32.SelectObject(self._dc, self._bitmap)
        self._buffer = (ctypes.c_ubyte * (width * height * 4)).from_address(bits.value)
        self._size = (width, height)

    def grab(self) -> Optional[np.ndarray]:
        """RGB frame, or None when the window is gone or minimised."""
        if not alive(self.hwnd) or minimized(self.hwnd):
            return None
        _, _, width, height = client_rect(self.hwnd)
        if width <= 0 or height <= 0:
            return None
        if (width, height) != self._size:
            self._allocate(width, height)
        if not user32.PrintWindow(self.hwnd, self._dc, PW_RENDERFULLCONTENT):
            return None
        bgra = np.frombuffer(self._buffer, dtype=np.uint8).reshape(height, width, 4)
        return bgra[:, :, 2::-1].copy()            # BGRA -> RGB

    def close(self) -> None:
        if self._bitmap:
            gdi32.DeleteObject(self._bitmap)
        if self._dc:
            gdi32.DeleteDC(self._dc)
        self._dc = self._bitmap = self._buffer = None


def grab_screen(hwnd: int) -> Optional[np.ndarray]:
    """Fallback: copy the window's client area from the screen (must be visible)."""
    from PIL import ImageGrab
    if not alive(hwnd) or minimized(hwnd):
        return None
    left, top, width, height = client_rect(hwnd)
    if width <= 0 or height <= 0:
        return None
    image = ImageGrab.grab(bbox=(left, top, left + width, top + height), all_screens=True)
    return np.asarray(image.convert("RGB"))


def wait_for(title: Optional[str], hwnd: Optional[int], timeout: float = 0.0) -> GameWindow:
    """`find`, retrying until `timeout` seconds (0 = once)."""
    deadline = time.monotonic() + timeout
    while True:
        try:
            return find(title, hwnd)
        except RuntimeError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(1.0)

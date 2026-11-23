"""The overlay window: a borderless panel that sits beside the game window.

Every tick it re-reads where the game window is and moves to its right edge
(left edge if there is no room), matches its height, hides while the game is
minimised, and stays on top only while the game or the panel itself has focus,
so it does not float over other programs after you alt-tab away.

Clicks: a card logs a play the detector missed ("they just played this"),
"Undo last" removes the most recent play, "New battle" restarts the clock.
Ctrl+Z also undoes and Esc closes when the panel has focus. Drag to move it
when it is not docked to a window.
"""

from __future__ import annotations

import sys
from typing import Optional

from . import render
from .match import Session
from .pipeline import Pipeline

REFERENCE_HEIGHT = 996.0          # MuMu's default surface height; scale 1.0 there


def _hex(rgb) -> str:
    return "#%02x%02x%02x" % rgb


class Overlay:
    def __init__(self, session: Session, pipeline: Pipeline, art: render.CardArt,
                 width: int = 300, gap: int = 6, prefer_left: bool = False, dock: bool = True,
                 height: int = 900, alpha: float = 0.96, follow_focus: bool = True,
                 fps: float = 8.0):
        import tkinter as tk
        from PIL import ImageTk

        self.tk, self.ImageTk = tk, ImageTk
        self.session, self.pipeline, self.art = session, pipeline, art
        self.base_width, self.gap, self.prefer_left, self.dock = width, gap, prefer_left, dock
        self.height = height
        self.follow_focus = follow_focus
        self.interval = int(1000 / fps)
        self.scale = 1.0
        self.renderer = render.Renderer(width, 1.0, art)
        self.regions = []
        self.geometry = None
        self.hidden = False
        self._drag = None
        self._moved = False
        self._topmost = True

        self.root = tk.Tk()
        self.root.title("Clash Royale opponent overlay")
        self.root.configure(bg=_hex(render.BG))
        self.root.overrideredirect(True)
        try:
            self.root.attributes("-alpha", alpha)
        except tk.TclError:
            pass
        self.root.attributes("-topmost", True)
        self.label = tk.Label(self.root, bd=0, highlightthickness=0, bg=_hex(render.BG))
        self.label.pack(fill="both", expand=True)
        self.label.bind("<ButtonPress-1>", self._press)
        self.label.bind("<B1-Motion>", self._motion)
        self.label.bind("<ButtonRelease-1>", self._release)
        self.root.bind("<Escape>", lambda e: self.close())
        self.root.bind("<Control-z>", lambda e: self.session.undo())
        self.root.geometry(f"{width}x{height}+40+40")
        self.root.after(0, self._tick)

    # ------------------------------------------------------------ placement

    def _screen_right(self) -> int:
        if sys.platform == "win32":
            import ctypes
            m = ctypes.windll.user32.GetSystemMetrics
            return m(76) + m(78)                 # virtual screen left + width
        return self.root.winfo_screenwidth()

    def _place(self) -> None:
        feed = self.pipeline.feed
        rect = feed.dock_rect() if self.dock else None
        if rect is None:
            if self.geometry is None:
                self._set_scale(self.height)
                self.geometry = (self.renderer.L.width, self.height, 40, 40)
                self.root.geometry(f"{self.geometry[0]}x{self.geometry[1]}+40+40")
            return
        if not feed.visible():
            if not self.hidden:
                self.root.withdraw()
                self.hidden = True
            return
        if self.hidden:
            self.root.deiconify()
            self.hidden = False
        left, top, w, h = rect
        self._set_scale(h)
        pw = self.renderer.L.width
        x = left + w + self.gap
        if self.prefer_left or x + pw > self._screen_right():
            x = left - self.gap - pw
        geometry = (pw, h, x, top)
        if geometry != self.geometry:
            self.geometry = geometry
            self.root.geometry(f"{pw}x{h}+{x}+{top}")
        if self.follow_focus and sys.platform == "win32":
            from . import winapi
            hwnd = getattr(getattr(feed, "source", None), "window", None)
            if hwnd is not None:
                me = winapi.user32.GetAncestor(self.root.winfo_id(), winapi.GA_ROOT)
                wanted = winapi.foreground_is(hwnd.hwnd, (me,))
                if wanted != self._topmost:
                    self._topmost = wanted
                    self.root.attributes("-topmost", wanted)

    def _set_scale(self, h: int) -> None:
        scale = max(0.8, min(1.8, h / REFERENCE_HEIGHT))
        if abs(scale - self.scale) > 0.04:
            self.scale = scale
            self.renderer = render.Renderer(int(self.base_width * scale), scale, self.art)

    # ------------------------------------------------------------ drawing

    def _tick(self) -> None:
        try:
            self._place()
            if not self.hidden:
                height = self.geometry[1] if self.geometry else self.height
                snap = self.session.snapshot(self.pipeline.status())
                status = snap.status if snap else f"{self.session.status} · {self.pipeline.status()}"
                image, self.regions = self.renderer.render(snap, height, status)
                self._photo = self.ImageTk.PhotoImage(image)
                self.label.configure(image=self._photo)
        except Exception as exc:                      # never let the UI die silently
            print(f"overlay: {type(exc).__name__}: {exc}", file=sys.stderr)
        self.root.after(self.interval, self._tick)

    # ------------------------------------------------------------ input

    def _press(self, event) -> None:
        self._drag = (event.x_root, event.y_root, self.root.winfo_x(), self.root.winfo_y())
        self._moved = False

    def _motion(self, event) -> None:
        if self._drag is None or (self.dock and self.pipeline.feed.dock_rect() is not None):
            return
        x0, y0, wx, wy = self._drag
        dx, dy = event.x_root - x0, event.y_root - y0
        if abs(dx) + abs(dy) > 4:
            self._moved = True
            self.root.geometry(f"+{wx + dx}+{wy + dy}")

    def _release(self, event) -> None:
        if self._moved:
            self._moved = False
            return
        found = render.hit(self.regions, event.x, event.y)
        if not found:
            return
        action, arg = found
        if action == "play":
            self.session.manual_play(arg)
        elif action == "undo":
            self.session.undo()
        elif action == "reset":
            self.session.restart(self.pipeline.now())
        elif action == "close":
            self.close()

    def close(self) -> None:
        self.pipeline.stop()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()

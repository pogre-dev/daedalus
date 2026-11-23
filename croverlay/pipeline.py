"""The capture-and-detect loop, on its own thread.

Capture and detection run as fast as they can on a worker thread; the UI
polls `Session.snapshot()` on its own timer. Frames are never queued: each
pass takes the newest frame, so a slow detector makes updates less frequent
rather than later.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import List, Optional, Tuple

from . import geometry
from .match import Session
from .tracker import Detection


class Feed:
    """Something that yields (detections, towers_visible) once per step."""

    def step(self) -> Optional[Tuple[List[Detection], bool]]:
        raise NotImplementedError

    def status(self) -> str:
        return ""

    def close(self) -> None:
        pass

    def dock_rect(self):
        return None

    def visible(self) -> bool:
        return True


class LiveFeed(Feed):
    def __init__(self, source, detector):
        self.source = source
        self.detector = detector
        self.calibrator = geometry.Calibrator()
        self.frames = 0
        self.last_frame_shape = None

    def step(self):
        frame = self.source.read()
        if frame is None:
            return None
        self.last_frame_shape = frame.shape
        boxes = self.detector.boxes(frame)
        self.frames += 1
        return self.detector.detections(frame, boxes, self.calibrator)

    def status(self) -> str:
        cal = "towers fitted" if self.calibrator.fitted else "uncalibrated"
        side = " · side model" if self.detector.sides else ""
        return f"detect {self.detector.ms:.0f} ms · {cal}{side}"

    def dock_rect(self):
        return self.source.dock_rect()

    def visible(self) -> bool:
        return self.source.visible()

    def close(self) -> None:
        self.source.close()


class TracksFeed(Feed):
    """Replay Solar-CR's recorded detector tracks (`data/vod/tracks/*.jsonl`)
    in real time - a demo and a test of everything after the detector."""

    def __init__(self, path: Path, speed: float = 1.0):
        from .evaluate import load_tracks
        _, self.frames = load_tracks(Path(path))
        self.speed = speed
        self.index = 0
        self.t0 = time.monotonic()
        self.offset = self.frames[0][0] if self.frames else 0.0

    def step(self):
        if self.index >= len(self.frames):
            time.sleep(0.05)
            return [], False                     # towers gone: the battle ends
        t, detections = self.frames[self.index]
        wait = (t - self.offset) / self.speed - (time.monotonic() - self.t0)
        if wait > 0:
            time.sleep(min(wait, 0.1))
            return None
        self.index += 1
        towers = any(d.name in geometry.PRINCESS_CLASSES or d.name in geometry.KING_CLASSES
                     for d in detections)
        return detections, towers

    def status(self) -> str:
        if not self.frames:
            return "empty recording"
        done = self.index / len(self.frames)
        return f"replay {done:.0%} at {self.speed:g}x"


class Pipeline(threading.Thread):
    def __init__(self, feed: Feed, session: Session, max_fps: float = 30.0,
                 clock=time.monotonic):
        super().__init__(name="croverlay-pipeline", daemon=True)
        self.feed = feed
        self.session = session
        self.min_interval = 1.0 / max_fps
        self.clock = clock
        self.fps = 0.0
        self.error = ""
        self.speed = getattr(feed, "speed", 1.0)
        self._stop_event = threading.Event()
        self._t_start = clock()

    def now(self) -> float:
        """Session time; a sped-up replay runs the match clock faster too."""
        return (self.clock() - self._t_start) * self.speed

    def run(self) -> None:
        last = self.clock()
        while not self._stop_event.is_set():
            started = self.clock()
            try:
                result = self.feed.step()
            except Exception as exc:               # keep the overlay alive
                self.error = f"{type(exc).__name__}: {exc}"
                time.sleep(0.5)
                continue
            if result is None:
                time.sleep(0.02)
                continue
            detections, towers = result
            self.session.update(self.now(), detections, towers)
            now = self.clock()
            dt = now - last
            last = now
            if dt > 0:
                self.fps = 0.8 * self.fps + 0.2 * (1.0 / dt) if self.fps else 1.0 / dt
            spare = self.min_interval - (now - started)
            if spare > 0:
                self._stop_event.wait(spare)
        self.feed.close()

    def stop(self) -> None:
        self._stop_event.set()

    def status(self) -> str:
        if self.error:
            return f"error: {self.error}"
        extra = self.feed.status()
        return f"{self.fps:.0f} fps" + (f" · {extra}" if extra else "")

"""Run a YOLO detector on a frame and hand back tile-space detections.

The overlay has no model of its own: it uses the one you trained for the bot.
`find_weights` looks where Solar-CR keeps it - the newest
`tmp/yolo/runs/*/weights/best.pt`, then the vendored KataCR checkpoint - or
takes an explicit path. If the bot's ally/enemy side classifier
(`tmp/yolo/side_classifier.pt`) is there too, it is loaded and every unit
arrives with a side label, which is the strongest attribution cue there is.

Nothing here imports the bot's code; only its files are read.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from . import catalog, geometry
from .tracker import ALLY, ENEMY, Detection

BOT_DIR_NAMES = ("Solar-CR", "solar-cr", "Hasty-CR", "hasty-cr", "HastyCR")


@dataclass(frozen=True)
class Box:
    name: str
    conf: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def centre(self) -> Tuple[float, float]:
        return (self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0


def guess_bot_dir(start: Optional[Path] = None) -> Optional[Path]:
    """A Solar-CR checkout next to this one, in the home folder, or the cwd."""
    here = Path(start or Path(__file__).resolve().parents[2])
    places = [here, here.parent, Path.home(), Path.home() / "Desktop",
              Path.home() / "Documents", Path.cwd(), Path.cwd().parent]
    for place in places:
        for name in BOT_DIR_NAMES:
            candidate = place / name
            if (candidate / "scripts" / "brain").is_dir():
                return candidate
    return None


def find_weights(bot_dir: Optional[Path]) -> Optional[Path]:
    if bot_dir is None:
        return None
    runs = sorted((bot_dir / "tmp" / "yolo" / "runs").glob("*/weights/best.pt"),
                  key=lambda p: p.stat().st_mtime)
    if runs:
        return runs[-1]
    katacr = sorted((bot_dir / "vendor" / "KataCR" / "runs").glob("detector*.pt"))
    return katacr[-1] if katacr else None


def find_side_model(bot_dir: Optional[Path]) -> Optional[Path]:
    if bot_dir is None:
        return None
    path = bot_dir / "tmp" / "yolo" / "side_classifier.pt"
    return path if path.exists() else None


class SideClassifier:
    """Solar-CR's ally/enemy crop classifier (architecture copied, weights read)."""

    def __init__(self, path: Path, device: str = "cuda"):
        import torch
        import torch.nn as nn

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(),
                nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(),
                nn.MaxPool2d(2))

        model = nn.Sequential(block(3, 32), block(32, 64), block(64, 128),
                              nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.1),
                              nn.Linear(128, 2))
        blob = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(blob["state_dict"])
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = model.to(self.device).eval()
        self.crop = int(blob.get("crop", 32))
        self.accuracy = float(blob.get("accuracy", 0.0))
        self.torch = torch

    def classify(self, frame: np.ndarray, boxes: Sequence[Box]) -> List[Tuple[str, float]]:
        from PIL import Image
        if not boxes:
            return []
        height, width = frame.shape[:2]
        crops = []
        for b in boxes:
            pw, ph = (b.x2 - b.x1) * 0.125, (b.y2 - b.y1) * 0.125
            x1, y1 = max(0, int(b.x1 - pw)), max(0, int(b.y1 - ph))
            x2, y2 = min(width, int(b.x2 + pw)), min(height, int(b.y2 + ph))
            patch = frame[y1:y2, x1:x2]
            if patch.size == 0:
                patch = np.zeros((self.crop, self.crop, 3), np.uint8)
            crops.append(np.asarray(Image.fromarray(patch).resize((self.crop, self.crop),
                                                                  Image.BILINEAR)))
        batch = self.torch.from_numpy(np.stack(crops).astype(np.float32) / 255.0)
        batch = batch.permute(0, 3, 1, 2).to(self.device)
        with self.torch.no_grad():
            p = self.torch.softmax(self.model(batch), dim=1)
            conf, label = p.max(1)
        return [(ENEMY if int(l) == 1 else ALLY, float(c))
                for l, c in zip(label.tolist(), conf.tolist())]


class Detector:
    """YOLO over one frame -> boxes; boxes + calibration -> Detections."""

    def __init__(self, weights: Path, side_model: Optional[Path] = None,
                 confidence: float = 0.35, imgsz: int = 640, device: str = "",
                 side_floor: float = 0.7):
        self.weights = Path(weights)
        katacr = next((p for p in self.weights.parents if p.name == "KataCR"), None)
        if katacr is not None and str(katacr) not in sys.path:
            # KataCR pickles its own Ultralytics subclasses into the checkpoint.
            sys.path.insert(0, str(katacr))
        from ultralytics import YOLO
        self.model = YOLO(str(self.weights))
        self.names = dict(self.model.names)
        self.confidence = confidence
        self.imgsz = imgsz
        self.device = device or None
        self.side_floor = side_floor
        self.sides: Optional[SideClassifier] = None
        self.side_error = ""
        if side_model is not None:
            try:
                self.sides = SideClassifier(side_model)
            except Exception as exc:                  # torch missing, bad file...
                self.side_error = f"{type(exc).__name__}: {exc}"
        self.ms = 0.0
        unknown = [n for n in self.names.values() if catalog.classify(n) is None]
        self.unknown_classes = unknown

    def boxes(self, frame: np.ndarray) -> List[Box]:
        """Detect on an RGB frame. Box coordinates are in that frame's pixels."""
        started = time.perf_counter()
        height, width = frame.shape[:2]
        image, scale = frame, 1.0
        if max(height, width) > self.imgsz * 1.5:
            # Same picture, scaled once and cheaply here rather than letterboxed
            # inside predict: Solar-CR measured 6 ms vs 11.5 ms, same boxes.
            from PIL import Image
            scale = self.imgsz / float(max(height, width))
            image = np.asarray(Image.fromarray(frame).resize(
                (round(width * scale), round(height * scale)), Image.BILINEAR))
        result = self.model.predict(image[:, :, ::-1], imgsz=self.imgsz, conf=self.confidence,
                                    device=self.device, verbose=False, max_det=60)[0]
        out = []
        if result.boxes is not None and len(result.boxes):
            xyxy = result.boxes.xyxy.cpu().numpy() / scale
            confs = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)
            for (x1, y1, x2, y2), c, k in zip(xyxy, confs, classes):
                out.append(Box(self.names.get(int(k), str(k)), float(c), x1, y1, x2, y2))
        self.ms = (time.perf_counter() - started) * 1000.0
        return out

    def detections(self, frame: np.ndarray, boxes: List[Box],
                   calibrator: geometry.Calibrator) -> Tuple[List[Detection], bool]:
        """Map boxes into tiles; also report whether any crown tower is visible."""
        calibrator.resize(frame.shape[1], frame.shape[0])
        towers = [(b.name, *b.centre) for b in boxes
                  if b.name in geometry.PRINCESS_CLASSES or b.name in geometry.KING_CLASSES]
        calibrator.observe(towers)
        units = [b for b in boxes if (catalog.classify(b.name) or _NONE).role in ("card", "spell", "child")]
        sides = self.sides.classify(frame, units) if self.sides else []
        side_of = {id(b): s for b, s in zip(units, sides)}
        out = []
        for b in boxes:
            x, y = calibrator.to_tile(*b.centre)
            side, conf = side_of.get(id(b), (None, 0.0))
            out.append(Detection(b.name, x, y, b.conf, side if conf >= self.side_floor else None))
        return out, bool(towers)


class _NoInfo:
    role = "ignore"


_NONE = _NoInfo()

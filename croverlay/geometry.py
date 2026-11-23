"""Arena coordinates, and the pixel-to-tile mapping fitted from the towers.

Convention (the same one Solar-CR's VOD tracks use, so data recorded there can
be replayed here unchanged):

    x: 0 .. 18 tiles, left to right
    y: 0 .. 32 tiles, top to bottom - the opponent is at the top, you at the
       bottom, the river is the band around y = 16.

Positions are box centres mapped through an affine fit to the six crown towers.
Fitting per session rather than hard-coding one emulator's numbers is what lets
the overlay work on MuMu, BlueStacks, Google Play Games, a letterboxed window or
a resized one: the towers are always where the game puts them, and the detector
finds them at ~0.9 confidence every frame.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Iterable, Optional, Tuple

import numpy as np

WIDTH, HEIGHT = 18.0, 32.0
RIVER_Y = 16.0
CENTRE_X = 9.0

# Where a tower's detection box centre maps to. These are the anchors the VOD
# pipeline fitted to, kept identical so thresholds tuned on that data hold.
TOWERS: Dict[str, Tuple[float, float]] = {
    "enemy_left": (3.5, 7.5),
    "enemy_right": (14.5, 7.5),
    "enemy_king": (9.5, 3.5),
    "ally_left": (3.5, 24.5),
    "ally_right": (14.5, 24.5),
    "ally_king": (9.5, 28.5),
}
PRINCESS_CLASSES = frozenset({"queen-tower", "cannoneer-tower", "dagger-duchess-tower", "tower"})
KING_CLASSES = frozenset({"king-tower"})

# Furthest row each side may deploy troops on: their own half, extended by the
# "pocket" when the opposing princess tower in that lane has fallen.
ENEMY_DEPLOY_MAX_Y = 15.0
ENEMY_POCKET_MAX_Y = 21.5
ALLY_DEPLOY_MIN_Y = 17.0
ALLY_POCKET_MIN_Y = 10.5


def lane(x: float) -> str:
    return "left" if x < CENTRE_X else "right"


def half(y: float) -> str:
    """'enemy' above the river, 'ally' below, 'river' in between."""
    if y < RIVER_Y - 1.0:
        return "enemy"
    if y > RIVER_Y + 1.0:
        return "ally"
    return "river"


def tower_key(x: float, y: float, princess: bool) -> str:
    side = "enemy" if y < RIVER_Y else "ally"
    return f"{side}_{lane(x)}" if princess else f"{side}_king"


# ------------------------------------------------------------- pixel <-> tile

# Solar-CR's MuMu 1080x1920 measurement (scripts/brain/arena.py): tile centre
# x -> 103.5 + 51.0 * (x - 0.5) px. Used only until the towers have been seen.
_X0, _XS, _Y0, _YS = 103.5, 51.0, 171.9, 41.4


def default_transform(width: int, height: int) -> np.ndarray:
    """Affine (3x2) for a full-surface 9:16 portrait frame, before any fit."""
    sx, sy = 1080.0 / max(1, width), 1920.0 / max(1, height)
    return np.array([
        [sx / _XS, 0.0],
        [0.0, sy / _YS],
        [0.5 - _X0 / _XS, 0.5 - _Y0 / _YS],
    ])


def apply(matrix: np.ndarray, px: float, py: float) -> Tuple[float, float]:
    tx, ty = np.asarray([px, py, 1.0]) @ matrix
    return float(tx), float(ty)


def fit(anchors: Dict[str, Tuple[float, float]]) -> Tuple[Optional[np.ndarray], float]:
    """Least-squares affine pixels -> tiles from >= 3 tower anchors.

    Returns the matrix and its mean residual in tiles (inf if unfittable).
    """
    if len(anchors) < 3:
        return None, math.inf
    keys = list(anchors)
    src = np.asarray([[*anchors[k], 1.0] for k in keys], dtype=np.float64)
    dst = np.asarray([TOWERS[k] for k in keys], dtype=np.float64)
    # Three collinear-ish points (e.g. both kings and one princess) fit badly
    # in x; require both halves and both lanes before trusting it.
    lanes = {k.split("_")[1] for k in keys if not k.endswith("king")}
    halves = {k.split("_")[0] for k in keys}
    if len(halves) < 2 or not lanes:
        return None, math.inf
    matrix, *_ = np.linalg.lstsq(src, dst, rcond=None)
    errors = [math.dist(apply(matrix, *anchors[k]), TOWERS[k]) for k in keys]
    return matrix, float(np.mean(errors))


class Calibrator:
    """Keep a pixel->tile mapping fitted to recent tower sightings.

    Towers fall during a match, so the fit holds its last good answer rather
    than dropping back to the default when fewer than three remain.
    """

    MAX_RESIDUAL = 1.2        # tiles; a worse fit is ignored

    def __init__(self, width: int = 1080, height: int = 1920, history: int = 45):
        self.size = (width, height)
        self.matrix = default_transform(width, height)
        self.fitted = False
        self.residual = math.inf
        self._seen: Dict[str, Deque[Tuple[float, float]]] = {
            k: deque(maxlen=history) for k in TOWERS}

    def resize(self, width: int, height: int) -> None:
        if (width, height) != self.size:
            self.size = (width, height)
            self.matrix = default_transform(width, height)
            self.fitted = False
            for seen in self._seen.values():
                seen.clear()

    def observe(self, boxes: Iterable[Tuple[str, float, float]]) -> None:
        """Feed (class, centre px, centre py) for one frame's tower boxes."""
        width, height = self.size
        princesses = [(n, x, y) for n, x, y in boxes if n in PRINCESS_CLASSES or n in KING_CLASSES]
        if not princesses:
            return
        for name, px, py in princesses:
            # Identity from the frame geometry: which half, which side of centre.
            top = py < height * 0.45 if not self.fitted else apply(self.matrix, px, py)[1] < RIVER_Y
            if name in KING_CLASSES:
                key = "enemy_king" if top else "ally_king"
            else:
                left = px < width / 2 if not self.fitted else apply(self.matrix, px, py)[0] < CENTRE_X
                key = ("enemy_" if top else "ally_") + ("left" if left else "right")
            self._seen[key].append((px, py))
        anchors = {k: (float(np.median([p[0] for p in v])), float(np.median([p[1] for p in v])))
                   for k, v in self._seen.items() if len(v) >= 3}
        matrix, residual = fit(anchors)
        if matrix is not None and residual <= self.MAX_RESIDUAL:
            self.matrix, self.residual, self.fitted = matrix, residual, True

    def to_tile(self, px: float, py: float) -> Tuple[float, float]:
        return apply(self.matrix, px, py)


@dataclass
class TowerStatus:
    """Which princess towers are still standing, from what the detector sees."""
    last_seen: Dict[str, float]
    lost_after: float = 4.0

    @classmethod
    def new(cls) -> "TowerStatus":
        return cls({k: -1.0 for k in TOWERS if not k.endswith("king")})

    def see(self, key: str, t: float) -> None:
        if key in self.last_seen:
            self.last_seen[key] = t

    def standing(self, key: str, t: float) -> bool:
        seen = self.last_seen.get(key, -1.0)
        # Never seen yet: assume standing. Seen recently: standing.
        if seen < 0:
            return True
        # Down only if *other* towers are still being seen - an empty frame
        # (menu, loading) says nothing about this one.
        others = max((v for k, v in self.last_seen.items() if k != key), default=-1.0)
        return not (t - seen > self.lost_after and others >= 0 and t - others < 1.5)

    def enemy_deploy_max_y(self, x: float, t: float) -> float:
        """How far down the enemy may place troops at column x."""
        own = f"ally_{lane(x)}"
        return ENEMY_DEPLOY_MAX_Y if self.standing(own, t) else ENEMY_POCKET_MAX_Y

    def ally_deploy_min_y(self, x: float, t: float) -> float:
        theirs = f"enemy_{lane(x)}"
        return ALLY_DEPLOY_MIN_Y if self.standing(theirs, t) else ALLY_POCKET_MIN_Y

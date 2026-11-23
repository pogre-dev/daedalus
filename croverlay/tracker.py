"""Turn per-frame detections into card plays, and decide whose they were.

The detector names things on the board; it never sees a card being played. A
play has to be inferred from something *appearing*, and most appearances are
not plays:

* **re-acquisitions** - the detector loses a unit in a crowded fight and finds
  it again a second later. Tracks are matched by class only, never by which
  half of the arena they are in, so a Hog crossing the river stays one Hog. (A
  version keyed on the half, which Solar-CR's VOD tool uses, turns every
  river crossing into a new play for the other player.)
* **spawner and death output** - Tombstone skeletons, golemites, a Battle Ram's
  barbarians. A class that can come out of something is not a play when that
  something is (or just was) beside it.
* **one card, many units** - a Skeleton Army lands over several frames. Births
  of compatible classes close in space and time are folded into one group, and
  the group is only judged once it has settled, which is also when its count
  separates Skeletons from Skeleton Army and Minions from Minion Horde.

Whose play it was is scored from independent cues, each optional:

* the detector's own side label, when it has one;
* the floating card name / "-N" elixir marker the game draws at *your own*
  deploy point (detector classes `text` and `elixir`);
* where it appeared: troops can only be placed on their owner's half (plus the
  pocket once a princess tower falls);
* which way a thrown or rolled spell travels - away from whoever cast it;
* deck knowledge: a card that is already in one player's known deck and not the
  other's, and whether the cycle allows that player to hold it right now.

No single cue is trusted alone. Ties go to position.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Protocol, Sequence, Tuple

from . import catalog, geometry
from .catalog import SPELL, UnitInfo

ENEMY, ALLY = "enemy", "ally"


@dataclass
class Detection:
    name: str                      # detector class
    x: float                       # tile coordinates, see geometry.py
    y: float
    conf: float = 1.0
    side: Optional[str] = None     # ENEMY / ALLY if the detector knows


@dataclass
class Track:
    id: int
    unit: str
    info: UnitInfo
    x: float
    y: float
    first_t: float
    last_t: float
    first_x: float
    first_y: float
    hits: int = 1
    conf: float = 0.0
    side: Optional[str] = None
    side_votes: float = 0.0
    group: Optional[int] = None
    spawned: bool = False          # spawner/death output, never a play

    @property
    def is_spell(self) -> bool:
        return self.info.role == "spell"

    def velocity(self) -> Tuple[float, float]:
        dt = self.last_t - self.first_t
        if dt < 0.15:
            return 0.0, 0.0
        return (self.x - self.first_x) / dt, (self.y - self.first_y) / dt


@dataclass
class Group:
    id: int
    first_t: float
    x: float
    y: float
    tracks: List[int] = field(default_factory=list)
    units: Counter = field(default_factory=Counter)


@dataclass
class Play:
    t: float
    card: str
    side: str
    x: float
    y: float
    count: int
    score: float                   # > 0 enemy, < 0 ally
    confidence: float              # 0..1
    reasons: List[str] = field(default_factory=list)
    candidates: Tuple[str, ...] = ()

    @property
    def cost(self) -> float:
        return catalog.cost(self.card)


class Knowledge(Protocol):
    """What the rest of the overlay knows about both players' decks."""
    def deck(self, side: str) -> Sequence[str]: ...
    def deck_complete(self, side: str) -> bool: ...
    def feasible(self, side: str, card: str) -> bool: ...


class NoKnowledge:
    def deck(self, side: str) -> Sequence[str]:
        return ()

    def deck_complete(self, side: str) -> bool:
        return False

    def feasible(self, side: str, card: str) -> bool:
        return True


@dataclass
class Settings:
    # Low-recall footage loses a unit for seconds at a time. A same-class box
    # reachable from where it was last seen is that unit, not a new play.
    unit_keep_s: float = 10.0       # how long a lost unit can come back as itself
    spell_keep_s: float = 1.0
    building_keep_s: float = 30.0
    unit_speed: float = 2.2         # tiles/s used to grow the matching gate
    unit_gate: float = 1.5
    spell_gate: float = 4.0
    spell_speed: float = 15.0
    max_gate: float = 10.0
    flicker_tiles: float = 1.0      # a birth this close to another class's track is a misread
    join_s: float = 1.2             # births this close in time ...
    join_tiles: float = 3.0         # ... and space are one play
    settle_s: float = 1.0           # wait this long before judging a group
    parent_recent_s: float = 2.0
    marker_tiles: float = 2.5
    marker_s: float = 1.5
    markers_trusted: int = 20       # markers seen before their absence means anything
    min_hits: int = 2
    min_conf: float = 0.5           # best box of a new unit must reach this
    spell_min_conf: float = 0.45
    min_side_score: float = 1.0     # below this nobody can tell whose it was
    walk_speed: float = 2.2         # tiles/s: how far a play's units can have got
    memory_s: float = 25.0          # how long a play's units are remembered
    heading_tiles_s: float = 0.5    # |vy| that counts as walking one way
    deploy_tolerance: float = 1.5   # tiles past the deploy line still accepted


class PlayExtractor:
    def __init__(self, knowledge: Optional[Knowledge] = None,
                 settings: Optional[Settings] = None):
        self.knowledge: Knowledge = knowledge or NoKnowledge()
        self.cfg = settings or Settings()
        self.towers = geometry.TowerStatus.new()
        self.tracks: Dict[int, Track] = {}
        self.groups: Dict[int, Group] = {}
        self.markers: List[Tuple[float, float, float]] = []
        self.markers_seen = 0                   # proves the detector reports them
        self.rejected: Counter = Counter()      # why candidate plays were dropped
        self.recent: List[Play] = []            # plays emitted, for re-identification
        self._next = 0

    def reset(self) -> None:
        self.__init__(self.knowledge, self.cfg)

    # ------------------------------------------------------------- per frame

    def update(self, t: float, detections: Iterable[Detection]) -> List[Play]:
        by_unit: Dict[str, List[Tuple[Detection, UnitInfo]]] = {}
        for det in detections:
            info = catalog.classify(det.name)
            if info is None or info.role == "ignore":
                continue
            if info.role == "structure":
                self._tower(det, t)
            elif info.role == "marker":
                self.markers.append((t, det.x, det.y))
                self.markers_seen += 1
            else:
                by_unit.setdefault(info.unit, []).append((det, info))

        self.markers = [m for m in self.markers if t - m[0] <= 3.0]
        for unit, dets in by_unit.items():
            self._associate(unit, dets, t)

        plays = []
        for group in list(self.groups.values()):
            if t - group.first_t >= self.cfg.settle_s:
                del self.groups[group.id]
                play = self._judge(group, t)
                if play:
                    plays.append(play)
        self._expire(t)
        return plays

    def flush(self, t: float) -> List[Play]:
        """Judge every open group now (end of a replay)."""
        plays = []
        for group in list(self.groups.values()):
            del self.groups[group.id]
            play = self._judge(group, t)
            if play:
                plays.append(play)
        return plays

    # ------------------------------------------------------------- tracking

    def _keep(self, track: Track) -> float:
        if track.is_spell:
            return self.cfg.spell_keep_s
        if track.info.cards and catalog.CARDS[track.info.cards[0][0]].kind == catalog.BUILDING:
            return self.cfg.building_keep_s
        return self.cfg.unit_keep_s

    def _gate(self, track: Track, t: float) -> float:
        gap = max(0.0, t - track.last_t)
        if track.is_spell:
            return min(self.cfg.spell_gate + self.cfg.spell_speed * gap, 14.0)
        return min(self.cfg.unit_gate + self.cfg.unit_speed * gap, self.cfg.max_gate)

    def _associate(self, unit: str, dets: List[Tuple[Detection, UnitInfo]], t: float) -> None:
        live = [tr for tr in self.tracks.values()
                if tr.unit == unit and t - tr.last_t <= self._keep(tr) and tr.last_t < t]
        pairs = []
        for i, (det, _) in enumerate(dets):
            for tr in live:
                d = math.hypot(det.x - tr.x, det.y - tr.y)
                if d <= self._gate(tr, t):
                    pairs.append((d + 0.3 * (t - tr.last_t), i, tr.id))
        pairs.sort()
        used_det, used_track = set(), set()
        for _, i, tid in pairs:
            if i in used_det or tid in used_track:
                continue
            used_det.add(i)
            used_track.add(tid)
            det, _ = dets[i]
            tr = self.tracks[tid]
            tr.x, tr.y, tr.last_t = det.x, det.y, t
            tr.hits += 1
            tr.conf = max(tr.conf, det.conf)
            if det.side:
                tr.side_votes += 1.0 if det.side == ENEMY else -1.0
        for i, (det, info) in enumerate(dets):
            if i not in used_det:
                self._birth(det, info, t)

    def _birth(self, det: Detection, info: UnitInfo, t: float) -> None:
        self._next += 1
        tr = Track(self._next, info.unit, info, det.x, det.y, t, t, det.x, det.y,
                   conf=det.conf)
        if det.side:
            tr.side_votes = 1.0 if det.side == ENEMY else -1.0
        self.tracks[tr.id] = tr

        parent = self._parent(tr, t)
        if parent is not None or info.role == "child":
            tr.spawned = True
            tr.side = parent.side if parent is not None else None
            return
        if self._flicker(tr, t):
            tr.spawned = True
            self.rejected["class flicker"] += 1
            return

        group = self._join(tr, t)
        if group is None:
            self._next += 1
            group = Group(self._next, t, det.x, det.y)
            self.groups[group.id] = group
        n = len(group.tracks)
        group.x = (group.x * n + det.x) / (n + 1)
        group.y = (group.y * n + det.y) / (n + 1)
        group.tracks.append(tr.id)
        group.units[tr.unit] += 1
        tr.group = group.id

    def _parent(self, tr: Track, t: float) -> Optional[Track]:
        if not tr.info.parents:
            return None
        radius = max(tr.info.parent_radius, 2.0)
        best, best_d = None, radius
        for other in self.tracks.values():
            if other.id == tr.id or other.unit not in tr.info.parents:
                continue
            if t - other.last_t > self.cfg.parent_recent_s:
                continue
            d = math.hypot(other.x - tr.x, other.y - tr.y)
            if d <= best_d:
                best, best_d = other, d
        return best

    def _flicker(self, tr: Track, t: float) -> bool:
        """Is this 'new' unit just an established one read as another class?"""
        if tr.is_spell:
            return False
        for other in self.tracks.values():
            if (other.id != tr.id and other.unit != tr.unit and not other.is_spell
                    and other.hits >= 3 and t - other.last_t <= 0.3
                    and math.hypot(other.x - tr.x, other.y - tr.y) <= self.cfg.flicker_tiles):
                return True
        return False

    def _outvoted(self, tr: Track) -> bool:
        """Misread the other way round: the new unit appeared first, but a
        box of another class has been sitting on it more often since."""
        if tr.is_spell:
            return False
        for other in self.tracks.values():
            if (other.id != tr.id and other.unit != tr.unit and not other.is_spell
                    and not other.spawned and other.hits > tr.hits
                    and other.last_t >= tr.first_t and other.first_t <= tr.last_t
                    and math.hypot(other.x - tr.x, other.y - tr.y) <= self.cfg.flicker_tiles):
                return True
        return False

    def _join(self, tr: Track, t: float) -> Optional[Group]:
        best, best_d = None, self.cfg.join_tiles
        for group in self.groups.values():
            if t - group.first_t > self.cfg.join_s:
                continue
            if not _compatible(set(group.units) | {tr.unit}):
                continue
            d = math.hypot(group.x - tr.x, group.y - tr.y)
            if d <= best_d:
                best, best_d = group, d
        return best

    def _tower(self, det: Detection, t: float) -> None:
        name = catalog.classify(det.name).unit
        princess = name in geometry.PRINCESS_CLASSES
        key = geometry.tower_key(det.x, det.y, princess)
        anchor = geometry.TOWERS.get(key)
        if anchor and math.hypot(anchor[0] - det.x, anchor[1] - det.y) <= 2.5:
            self.towers.see(key, t)

    def _expire(self, t: float) -> None:
        for tid in [tid for tid, tr in self.tracks.items()
                    if t - tr.last_t > self._keep(tr) + 2.0]:
            del self.tracks[tid]

    # ------------------------------------------------------------- judging

    def _confirmed(self, tr: Track) -> bool:
        if tr.is_spell:
            return tr.conf >= self.cfg.spell_min_conf or tr.hits >= self.cfg.min_hits
        return tr.hits >= self.cfg.min_hits and tr.conf >= self.cfg.min_conf

    def _judge(self, group: Group, t: float) -> Optional[Play]:
        tracks = [self.tracks[i] for i in group.tracks if i in self.tracks]
        tracks = [tr for tr in tracks if not self._outvoted(tr)]
        if not tracks:
            self.rejected["class flicker"] += 1
            return None
        if not any(self._confirmed(tr) for tr in tracks):
            self.rejected["unconfirmed"] += 1
            return None
        units = Counter(tr.unit for tr in tracks)
        candidates = _cards_for(units)
        if not candidates:
            self.rejected["no card"] += 1
            return None

        card = _by_count(units, candidates)
        owner = self._reidentify(group, units, t)
        if owner is not None:
            self.rejected["re-acquired unit"] += 1
            for tr in tracks:
                tr.side = owner
            return None
        physical, knowledge, reasons = self._side_score(group, tracks, card, candidates, t)
        score = physical + knowledge
        side = ENEMY if score > 0 else ALLY
        is_spell = catalog.CARDS[card].kind == SPELL
        if abs(score) < self.cfg.min_side_score or (not is_spell and abs(physical) < 1.0):
            # A troop always lands on someone's side of the river; one first
            # seen *on* the river is nearly always a unit crossing it.
            self.rejected["side unclear"] += 1
            return None
        thrower = self._thrown_by(card, tracks)
        if thrower and thrower != side:
            # Deck cues outvoted the spell's own flight path. The path is
            # physics; the deck cue only says the name is wrong. A misread.
            self.rejected["spell misread"] += 1
            return None

        # Prefer a candidate this player is already known to own.
        owned = [c for c in candidates if c in self.knowledge.deck(side)]
        if len(owned) == 1 and owned[0] != card:
            reasons.append(f"{owned[0]} is in their deck")
            card = owned[0]

        if not self._placeable(card, side, group.x, group.y, t):
            self.rejected["outside deploy zone"] += 1
            for tr in tracks:          # it was a unit we lost track of
                tr.side = side
            return None

        for tr in tracks:
            tr.side = side
        # Confidence that this was a real play by `side` comes from what was
        # seen. Deck knowledge may pick the side, but it is not evidence that a
        # play happened, so it never raises confidence.
        hits = sum(tr.hits for tr in tracks)
        confidence = min(1.0, abs(physical) / 5.0) * (0.5 + 0.5 * min(1.0, hits / 4.0))
        if physical * score < 0:
            confidence *= 0.5
        play = Play(group.first_t, card, side, round(group.x, 2), round(group.y, 2),
                    sum(units.values()), round(score, 2), round(confidence, 2),
                    reasons, tuple(candidates))
        self.recent = [p for p in self.recent if t - p.t <= self.cfg.memory_s] + [play]
        return play

    def _reidentify(self, group: Group, units: Counter, t: float) -> Optional[str]:
        """Is this 'new' unit one a player deployed earlier, walked here, and
        the detector lost on the way?

        Only asked about appearances outside the owner's deploy zone: inside
        it a genuine second copy is likely, outside it nobody can have placed
        one. Units walk toward the other side's king, so an enemy unit is never
        found further up the board than where it was deployed (and vice versa).
        """
        x, y = group.x, group.y
        for play in reversed(self.recent):
            if catalog.CARDS[play.card].kind == SPELL:
                continue
            if not set(units) & {u for u, _ in catalog.CARDS[play.card].units}:
                continue
            gap = group.first_t - play.t
            if gap <= 0:
                continue
            if play.side == ENEMY:
                if y <= self.towers.enemy_deploy_max_y(x, t) + 0.5 or y < play.y - 1.5:
                    continue
            else:
                if y >= self.towers.ally_deploy_min_y(x, t) - 0.5 or y > play.y + 1.5:
                    continue
            if math.hypot(x - play.x, y - play.y) <= 2.0 + self.cfg.walk_speed * gap:
                return play.side
        return None

    def _thrown_by(self, card: str, tracks: List[Track]) -> Optional[str]:
        """Whose a thrown or rolled spell is, from which way it travelled."""
        if card not in catalog.DIRECTIONAL_SPELLS:
            return None
        vy = max((tr.velocity()[1] for tr in tracks), key=abs, default=0.0)
        if abs(vy) < 1.0:
            return None
        return ENEMY if vy > 0 else ALLY

    def _placeable(self, card: str, side: str, x: float, y: float, t: float) -> bool:
        info = catalog.CARDS[card]
        if info.kind == SPELL or card in catalog.DEPLOY_ANYWHERE:
            return True
        tol = self.cfg.deploy_tolerance
        if side == ENEMY:
            return y <= self.towers.enemy_deploy_max_y(x, t) + tol
        return y >= self.towers.ally_deploy_min_y(x, t) - tol

    def _side_score(self, group: Group, tracks: List[Track], card: str,
                    candidates: Sequence[str], t: float) -> Tuple[float, float, List[str]]:
        """(physical, knowledge, reasons); positive means enemy.

        Physical cues are things seen this frame: the detector's side label,
        deploy markers, where it landed, which way it moved. Knowledge cues
        come from the decks and cycles learned so far.
        """
        score, why = 0.0, []
        x, y = group.x, group.y

        votes = sum(tr.side_votes for tr in tracks)
        if votes:
            score += 4.0 * (1 if votes > 0 else -1)
            why.append(f"detector says {'enemy' if votes > 0 else 'ally'}")

        if any(abs(mt - group.first_t) <= self.cfg.marker_s
               and math.hypot(mx - x, my - y) <= self.cfg.marker_tiles
               for mt, mx, my in self.markers):
            score -= 3.0
            why.append("own deploy marker")
        elif (self.markers_seen >= self.cfg.markers_trusted
              and catalog.CARDS[card].kind == SPELL
              and t - group.first_t <= self.cfg.marker_s + self.cfg.settle_s):
            # This detector does report our deploy markers, and there is none
            # here, so it was not ours.
            score += 2.0
            why.append("no own deploy marker")

        kind = catalog.CARDS[card].kind
        where = geometry.half(y)
        if kind == SPELL:
            thrower = self._thrown_by(card, tracks)
            if thrower:
                score += 3.0 if thrower == ENEMY else -3.0
                why.append(f"travels {'down' if thrower == ENEMY else 'up'}")
            if where != "river":
                score += 0.75 if where == ALLY else -0.75
                why.append(f"lands in {where} half")
        elif card in catalog.DEPLOY_ANYWHERE:
            if where != "river":
                score += 1.5 if where == ALLY else -1.5
                why.append(f"{card} surfaced in {where} half")
        else:
            enemy_ok = y <= self.towers.enemy_deploy_max_y(x, t) + 0.5
            ally_ok = y >= self.towers.ally_deploy_min_y(x, t) - 0.5
            if enemy_ok and not ally_ok:
                score += 3.0
                why.append("enemy deploy zone")
            elif ally_ok and not enemy_ok:
                score -= 3.0
                why.append("ally deploy zone")
            moving = [tr.velocity()[1] for tr in tracks if tr.last_t - tr.first_t >= 0.4]
            if moving:
                vy = sum(moving) / len(moving)
                h = self.cfg.heading_tiles_s
                if where == ALLY and vy > h:
                    score += 2.0
                    why.append("walking at ally towers")
                elif where == ENEMY and vy < -h:
                    score -= 2.0
                    why.append("walking at enemy towers")

        physical, score = score, 0.0
        k = self.knowledge
        in_enemy = any(c in k.deck(ENEMY) for c in candidates)
        in_ally = any(c in k.deck(ALLY) for c in candidates)
        if in_enemy and not in_ally:
            score += 4.0 if k.deck_complete(ALLY) else 1.5
            why.append("in enemy deck")
        elif in_ally and not in_enemy:
            score -= 4.0 if k.deck_complete(ENEMY) else 1.5
            why.append("in ally deck")
        if not in_enemy and k.deck_complete(ENEMY):
            score -= 6.0
            why.append("not in enemy's full deck")
        if not in_ally and k.deck_complete(ALLY):
            score += 6.0
            why.append("not in ally's full deck")
        if not k.feasible(ENEMY, card):
            score -= 2.0
            why.append("enemy cycle says not in hand")
        if not k.feasible(ALLY, card):
            score += 2.0
            why.append("ally cycle says not in hand")
        return physical, score, why

    # ------------------------------------------------------------- queries

    def on_field(self, side: str, now: float, within: float = 1.0) -> Counter:
        """Units currently visible for a side, by class."""
        return Counter(tr.unit for tr in self.tracks.values()
                       if tr.side == side and now - tr.last_t <= within)


def _compatible(units: set) -> bool:
    return bool(_cards_for(Counter({u: 1 for u in units})))


def _cards_for(units: Counter) -> List[str]:
    """Cards whose deploy places every class in `units`."""
    out = []
    for key in catalog.candidates_for(units):
        produced = {u for u, _ in catalog.CARDS[key].units}
        if set(units) <= produced:
            out.append(key)
    return out


def _by_count(units: Counter, candidates: Sequence[str]) -> str:
    """The candidate whose unit counts best match what landed."""
    if len(candidates) == 1:
        return candidates[0]

    def misfit(key: str) -> float:
        expected = dict(catalog.CARDS[key].units)
        return sum(abs(math.log((units.get(u, 0) + 0.5) / (n + 0.5)))
                   for u, n in expected.items())
    return min(candidates, key=lambda k: (misfit(k), catalog.CARDS[k].cost))

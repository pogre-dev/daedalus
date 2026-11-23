"""One battle: detections in, everything the overlay draws out.

`Battle` owns a play extractor and, for each player, a cycle model and an
elixir model. It is also the `Knowledge` the extractor asks when deciding whose
a play was, so the two players' decks and cycles sharpen each other: once your
own deck is known (or given with --my-deck), anything outside it is theirs.

`Session` wraps battles with the lifecycle - a battle starts when the towers
appear and ends when they have been gone for a few seconds - so the overlay can
be left running across matches.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from . import catalog, predict
from .cycle import UNSEEN, CycleModel, CycleView
from .elixir import ElixirModel
from .tracker import ALLY, ENEMY, Detection, Play, PlayExtractor, Settings

# A play the cycle says is impossible is believed - and the cycle repaired by
# assuming plays were missed in between - only when the evidence for it is at
# least this strong. Below it the play is taken to be a misread.
REPAIR_CONFIDENCE = 0.8
# Plays below this confidence had no physical evidence for either side (a
# stationary spell with no markers, say) and are left out of the models: a
# missed play costs less than an invented one, because the cycle can absorb a
# gap but a phantom card pollutes the deck.
MIN_CONFIDENCE = 0.2
MIRROR_WINDOW_S = 2.5
# A play the elixir estimate says they could not afford by more than this is
# taken as a misread unless it is near-certain. The estimate drifts high, not
# low (missed plays leave it high), so this mostly catches invented plays.
AFFORD_SLACK = 1.0
CERTAIN = 0.9
# A card counts as part of a player's known deck for attributing *other* plays
# once it has this much accumulated play confidence (two decent sightings).
DECK_EVIDENCE = 1.2


@dataclass
class LoggedPlay:
    t: float
    card: str
    cost: float
    confidence: float
    manual: bool = False
    inferred_before: int = 0       # missed plays assumed to make it possible
    reasons: List[str] = field(default_factory=list)


@dataclass
class Player:
    cycle: CycleModel = field(default_factory=CycleModel)
    elixir: ElixirModel = field(default_factory=ElixirModel)
    log: List[LoggedPlay] = field(default_factory=list)
    rejected: int = 0
    collector_at: Optional[float] = None


@dataclass
class Snapshot:
    """Immutable-by-convention view handed to the UI thread."""
    in_battle: bool
    t: float                        # match seconds
    multiplier: int
    elixir: float
    to_next: float                  # seconds until the next whole elixir
    to_full: float
    spent_10s: float
    cycle: CycleView
    log: List[LoggedPlay]
    predicted: List[tuple]          # (card, p) for unseen cards
    champion_on_field: Optional[str]
    rejected: int
    my_deck: List[str]
    battles: int = 0
    status: str = ""


class Battle:
    def __init__(self, my_deck: Sequence[str] = (), settings: Optional[Settings] = None,
                 repair_confidence: float = REPAIR_CONFIDENCE):
        self.my_deck = [c for c in (catalog.from_any(x) for x in my_deck) if c]
        self.players: Dict[str, Player] = {ENEMY: Player(), ALLY: Player()}
        self.extractor = PlayExtractor(knowledge=self, settings=settings)
        self.repair_confidence = repair_confidence
        self.t = 0.0

    # ----------------------------------------------------- Knowledge protocol

    def deck(self, side: str) -> Sequence[str]:
        """Cards this player has shown convincingly (used to attribute plays)."""
        if side == ALLY and self.my_deck:
            return self.my_deck
        evidence: Dict[str, float] = {}
        for p in self.players[side].log:
            evidence[p.card] = evidence.get(p.card, 0.0) + (DECK_EVIDENCE if p.manual else p.confidence)
        shown = [c for c, e in evidence.items() if e >= DECK_EVIDENCE]
        return sorted(shown, key=lambda c: -evidence[c])[:8]

    def deck_complete(self, side: str) -> bool:
        return len(self.deck(side)) >= 8

    def feasible(self, side: str, card: str) -> bool:
        if side == ALLY and self.my_deck and card not in self.my_deck:
            return False
        return self.players[side].cycle.feasible(card)

    # ----------------------------------------------------- updates

    def update(self, t: float, detections: Sequence[Detection]) -> List[Play]:
        self.t = max(self.t, t)
        accepted = []
        for play in self.extractor.update(t, detections):
            if self.accept(play):
                accepted.append(play)
        self._collector(t)
        return accepted

    def accept(self, play: Play) -> bool:
        player = self.players[play.side]
        if play.confidence < MIN_CONFIDENCE:
            player.rejected += 1
            return False
        if play.side == ALLY and self.my_deck and play.card not in self.my_deck:
            player.rejected += 1
            return False

        if play.card == "mirror":
            return self._mirror(player, play)
        if self._is_mirrored_copy(player, play):
            return False

        if (play.confidence < CERTAIN
                and player.elixir.value(play.t) < play.cost - AFFORD_SLACK):
            player.rejected += 1
            return False

        inserted = 0
        if not player.cycle.feasible(play.card):
            if play.confidence < self.repair_confidence:
                player.rejected += 1
                return False
            inserted = player.cycle.play(play.card, play.t, repair=True)
        else:
            player.cycle.play(play.card, play.t)
        for _ in range(inserted):
            player.elixir.spend(play.t - 0.1, _mean_cost(player.cycle.deck), "missed", proven=False)
        player.elixir.spend(play.t, play.cost, play.card)
        if play.card == "elixir_collector":
            player.collector_at = play.t
            player.elixir.add_collector(play.t)
        player.log.append(LoggedPlay(play.t, play.card, play.cost, play.confidence,
                                     inferred_before=inserted, reasons=play.reasons))
        return True

    def _mirror(self, player: Player, play: Play) -> bool:
        source = player.log[-1].card if player.log else None
        cost = catalog.cost(source, 3.0) + 1.0 if source else 4.0
        player.cycle.play("mirror", play.t)
        player.elixir.spend(play.t, cost, "mirror")
        player.log.append(LoggedPlay(play.t, "mirror", cost, play.confidence, reasons=play.reasons))
        return True

    @staticmethod
    def _is_mirrored_copy(player: Player, play: Play) -> bool:
        """The unit a Mirror places is not a second play of that card."""
        if len(player.log) < 2 or player.log[-1].card != "mirror":
            return False
        return (play.t - player.log[-1].t <= MIRROR_WINDOW_S
                and player.log[-2].card == play.card)

    def _collector(self, t: float) -> None:
        enemy = self.players[ENEMY]
        if enemy.collector_at is None or t - enemy.collector_at < 3.0:
            return
        if "elixir-collector" not in self.extractor.on_field(ENEMY, t, within=3.0):
            enemy.elixir.collector_destroyed(enemy.collector_at, t - 3.0)
            enemy.collector_at = None

    # ----------------------------------------------------- corrections

    def manual_play(self, card: str, side: str = ENEMY) -> None:
        """The user says this was played just now (the detector missed it)."""
        player = self.players[side]
        inserted = player.cycle.play(card, self.t, repair=True)
        for _ in range(inserted):
            player.elixir.spend(self.t - 0.1, _mean_cost(player.cycle.deck), "missed", proven=False)
        player.elixir.spend(self.t, catalog.cost(card), card)
        player.log.append(LoggedPlay(self.t, card, catalog.cost(card), 1.0, manual=True,
                                     inferred_before=inserted))

    def undo(self, side: str = ENEMY) -> Optional[str]:
        """Take back the most recent play (wrong card, or not a play at all)."""
        player = self.players[side]
        if not player.log:
            return None
        last = player.log.pop()
        player.cycle.undo()
        player.elixir.remove_last(last.card)
        for _ in range(last.inferred_before):
            player.elixir.remove_last("missed")
        return last.card

    # ----------------------------------------------------- reading

    def snapshot(self, in_battle: bool = True, status: str = "", battles: int = 0) -> Snapshot:
        enemy = self.players[ENEMY]
        t = self.t
        view = enemy.cycle.view()
        value = enemy.elixir.value(t)
        to_next = 0.0 if value >= 10 else enemy.elixir.time_to(t, min(10.0, int(value) + 1))
        champions = [u for u in self.extractor.on_field(ENEMY, t)
                     if (catalog.from_any(u) or "") in catalog.CHAMPIONS]
        predicted = predict.predict(view.deck) if not view.deck_complete else []
        return Snapshot(
            in_battle=in_battle, t=t, multiplier=enemy.elixir.multiplier(t),
            elixir=value, to_next=to_next, to_full=enemy.elixir.time_to(t, 10.0),
            spent_10s=enemy.elixir.spent_since(t - 10.0, t), cycle=view,
            log=list(enemy.log[-12:]), predicted=predicted,
            champion_on_field=catalog.from_any(champions[0]) if champions else None,
            rejected=enemy.rejected, my_deck=list(self.my_deck), battles=battles,
            status=status,
        )


class Session:
    """Battles back to back, started and ended by what is on screen.

    Thread-safe: the capture thread calls `update`, the UI thread calls
    `snapshot` and the correction methods.
    """

    END_AFTER_ABSENT_S = 6.0        # no towers for this long: the battle is over

    def __init__(self, my_deck: Sequence[str] = (), settings: Optional[Settings] = None,
                 intro_delay_s: float = 0.0):
        self.my_deck = list(my_deck)
        self.settings = settings
        self.intro_delay_s = intro_delay_s
        self.battle: Optional[Battle] = None
        self.started_at: Optional[float] = None
        self.towers_seen_at: Optional[float] = None
        self.battles = 0
        self.status = "waiting for a battle"
        self.paused = False
        self._last: Optional[Battle] = None
        self._lock = threading.Lock()

    def _new_battle(self, now: float) -> None:
        self.battle = Battle(self.my_deck, self.settings)
        self.started_at = now + self.intro_delay_s
        self.battles += 1

    def update(self, now: float, detections: Sequence[Detection], towers_visible: bool) -> List[Play]:
        with self._lock:
            if towers_visible:
                if self.battle is None:
                    self._new_battle(now)
                self.towers_seen_at = now
            elif self.battle is not None and self._ended(now):
                self._last, self.battle = self.battle, None
                self.status = "battle over"
                return []
            if self.battle is None or self.paused:
                return []
            t = max(0.0, now - self.started_at)
            self.status = "tracking"
            return self.battle.update(t, detections)

    def _ended(self, now: float) -> bool:
        return self.towers_seen_at is not None and now - self.towers_seen_at > self.END_AFTER_ABSENT_S

    def restart(self, now: float) -> None:
        """User pressed reset: treat now as the start of a fresh battle."""
        with self._lock:
            self._new_battle(now)
            self.towers_seen_at = now

    def snapshot(self, status_extra: str = "") -> Optional[Snapshot]:
        with self._lock:
            battle = self.battle or self._last
            if battle is None:
                return None
            status = self.status + (f" · {status_extra}" if status_extra else "")
            return battle.snapshot(in_battle=self.battle is not None, status=status,
                                   battles=self.battles)

    def manual_play(self, card: str) -> None:
        with self._lock:
            if self.battle:
                self.battle.manual_play(card)

    def undo(self) -> Optional[str]:
        with self._lock:
            return self.battle.undo() if self.battle else None


def _mean_cost(deck: Sequence[str]) -> float:
    costs = [catalog.cost(c) for c in deck if c in catalog.CARDS]
    return sum(costs) / len(costs) if costs else 3.5


def unseen_label(card: str) -> str:
    return "?" if card == UNSEEN else catalog.CARDS[card].label

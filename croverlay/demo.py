"""A scripted opponent, for trying the overlay without the game.

The opponent holds a real deck, deals and cycles it the way the game does,
and only plays what its elixir allows. Its plays are rendered as detector
output - units walking down from their side, spells flying in - with missed
frames and jitter, and fed through the same pipeline as a live game. The true
hand is printed to the console so the panel can be checked against it.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

from . import catalog, geometry
from .elixir import ElixirModel
from .pipeline import Feed
from .tracker import Detection

DEFAULT_DECK = ["hog_rider", "musketeer", "cannon", "ice_golem",
                "skeletons", "ice_spirit", "the_log", "fireball"]
BATTLE_S = 180.0
BREAK_S = 8.0


@dataclass
class Sprite:
    name: str
    t0: float
    t1: float
    start: Tuple[float, float]
    end: Tuple[float, float]

    def at(self, t: float) -> Optional[Tuple[float, float]]:
        if not (self.t0 <= t <= self.t1):
            return None
        f = 0.0 if self.t1 == self.t0 else (t - self.t0) / (self.t1 - self.t0)
        return (self.start[0] + (self.end[0] - self.start[0]) * f,
                self.start[1] + (self.end[1] - self.start[1]) * f)


class DemoFeed(Feed):
    def __init__(self, deck: Optional[List[str]] = None, seed: int = 7, speed: float = 1.0,
                 recall: float = 0.85, verbose: bool = True, clock=time.monotonic):
        self.deck = list(deck or DEFAULT_DECK)
        self.rng = random.Random(seed)
        self.speed = speed
        self.recall = recall
        self.verbose = verbose
        self._clock = clock
        self.truth: List[Tuple[float, str]] = []
        self.t0 = clock()
        self.last_step = -1.0
        self._new_battle(0.0)

    def _new_battle(self, at: float) -> None:
        cards = self.deck[:]
        self.rng.shuffle(cards)
        self.hand, self.queue = cards[:4], cards[4:]
        self.elixir = ElixirModel()
        self.battle_start = at
        self.sprites: List[Sprite] = []
        self.next_at = at + self.rng.uniform(2.0, 6.0)
        self.played = 0
        if self.verbose:
            print(f"[demo] new battle - opponent deck: {', '.join(self.deck)}")

    def clock(self) -> float:
        return (self._clock() - self.t0) * self.speed

    def step(self):
        now = self.clock()
        if now - self.last_step < 0.1:
            if self._clock is time.monotonic:
                time.sleep(0.01)
            return None
        self.last_step = now
        t = now - self.battle_start
        if t > BATTLE_S + BREAK_S:
            self._new_battle(now)
            t = 0.0
        if t > BATTLE_S:
            return [], False                          # results screen: no arena
        if t >= self.next_at - self.battle_start:
            self._maybe_play(now, t)
        return self._detections(now), True

    def _maybe_play(self, now: float, t: float) -> None:
        card = self.rng.choice(self.hand)
        cost = catalog.cost(card)
        if self.elixir.value(t) < cost:
            self.next_at = now + 0.5
            return
        self.elixir.spend(t, cost, card)
        self.hand.remove(card)
        self.hand.append(self.queue.pop(0))
        self.queue.append(card)
        self.played += 1
        self.truth.append((t, card))
        self._sprites_for(card, now)
        if self.verbose:
            print(f"[demo] {t:5.1f}s  played {card:<14} hand now: {', '.join(sorted(self.hand))}"
                  f"  | next: {self.queue[0]}")
        self.next_at = now + self.rng.uniform(1.5, 6.0) / max(1, self.elixir.multiplier(t))

    def _sprites_for(self, card: str, now: float) -> None:
        info = catalog.CARDS[card]
        rng = self.rng
        lane_x = rng.choice([3.5, 14.5])
        if info.kind == catalog.SPELL:
            target = (lane_x + rng.uniform(-1.5, 1.5), rng.uniform(19.0, 25.0))
            if catalog.SPELL_INTENT.get(card) == -1:
                # Defensive spells land on their own side, on our attackers.
                target = (lane_x + rng.uniform(-1.5, 1.5), rng.uniform(6.0, 11.0))
            unit = info.units[0][0]
            if card in catalog.DIRECTIONAL_SPELLS:
                start = (9.5, 3.5) if card != "the_log" else (target[0], 15.0)
                self.sprites.append(Sprite(unit, now, now + 0.9, start, target))
            else:
                self.sprites.append(Sprite(unit, now, now + 0.7, target, target))
            return
        if info.kind == catalog.BUILDING:
            spot = (rng.uniform(7.0, 11.0), rng.uniform(9.0, 11.0))
            self.sprites.append(Sprite(info.units[0][0], now, now + 30.0, spot, spot))
            return
        spot = (lane_x + rng.uniform(-1.0, 1.0), rng.uniform(9.0, 14.0))
        speed = 1.2
        life = rng.uniform(6.0, 12.0)
        for unit, count in info.units:
            for i in range(count):
                offset = ((i % 3) - 1) * 0.6, (i // 3) * 0.6
                start = (spot[0] + offset[0], spot[1] + offset[1])
                end = (start[0], start[1] + speed * life)
                self.sprites.append(Sprite(unit, now, now + life, start, end))

    def _detections(self, now: float) -> List[Detection]:
        rng = self.rng
        out = [Detection("queen-tower" if "king" not in k else "king-tower", x, y, 0.9)
               for k, (x, y) in geometry.TOWERS.items()]
        self.sprites = [s for s in self.sprites if s.t1 >= now]
        for s in self.sprites:
            pos = s.at(now)
            if pos is None or rng.random() > self.recall:
                continue
            out.append(Detection(s.name, pos[0] + rng.gauss(0, 0.1), pos[1] + rng.gauss(0, 0.1),
                                 rng.uniform(0.55, 0.95)))
        return out

    def status(self) -> str:
        return f"demo opponent · {self.played} plays"


SCORE_DECKS = [
    None,
    ["golem", "night_witch", "baby_dragon", "lumberjack", "tornado", "lightning",
     "barbarian_barrel", "mega_minion"],
    ["x_bow", "tesla", "archers", "knight", "skeletons", "ice_spirit", "the_log", "fireball"],
    ["royal_giant", "fisherman", "hunter", "lightning", "the_log", "electro_spirit",
     "skeletons", "royal_ghost"],
]


def score(recall: float = 0.8, matches: int = 12, settle_s: float = 1.5) -> dict:
    """Play scripted matches through the real pipeline and compare with truth.

    Hand and next card are sampled every 5 s, skipping samples taken within
    `settle_s` of the opponent's latest play (a play takes ~1 s to confirm).
    """
    from .match import Session
    caught = total = invented = shown = right = nexts = next_ok = 0
    for seed in range(matches):
        now = [0.0]
        feed = DemoFeed(SCORE_DECKS[seed % len(SCORE_DECKS)], seed=seed, verbose=False,
                        recall=recall, clock=lambda: now[0])
        session = Session()
        for i in range(int(BATTLE_S / 0.1) - 5):
            now[0] += 0.1
            result = feed.step()
            if result:
                session.update(now[0], *result)
            settled = feed.truth and now[0] - feed.battle_start - feed.truth[-1][0] >= settle_s
            if i % 50 == 25 and session.battle and len(feed.truth) >= 4 and settled:
                view = session.battle.players["enemy"].cycle.view()
                known = [c for c in view.hand if c != "?"]
                shown += len(known)
                right += sum(1 for c in known if c in feed.hand)
                if view.next_card != "?":
                    nexts += 1
                    next_ok += view.next_card == feed.queue[0]
        seen = [p.card for p in session.battle.players["enemy"].log]
        truth = iter(c for _, c in feed.truth)
        in_order = sum(1 for c in seen if c in truth)
        caught += len(seen)
        invented += len(seen) - in_order
        total += len(feed.truth)
    return {"plays_caught": caught / total, "invented": invented,
            "hand_cards_right": right / max(1, shown), "next_right": next_ok / max(1, nexts)}


if __name__ == "__main__":
    import sys
    for r in [float(a) for a in sys.argv[1:]] or [1.0, 0.8, 0.6]:
        s = score(r)
        print(f"detector recall {r:.0%}: plays caught {s['plays_caught']:.0%} "
              f"({s['invented']} invented), hand cards shown right {s['hand_cards_right']:.0%}, "
              f"next card right {s['next_right']:.0%}")

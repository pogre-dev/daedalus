"""The opponent's hand and next card, worked out from the order they play.

How the game deals
------------------
A deck is eight distinct cards. Four are in hand; the other four wait in a
queue whose head is the "next" card. Playing a card moves the head of the queue
into the hand and puts the played card at the back. So:

* the queue is always the last four cards played (oldest first), once four have
  been played; before that its front is the shuffled starting queue,
* the hand is the deck minus the queue,
* the next card is the fourth most recent play.

Written as one sequence - the four starting-queue cards followed by every play -
the whole rule is: **any five consecutive entries are distinct.**

What is not observed
--------------------
The starting queue is unknown, the deck is learned one card at a time, and the
detector sometimes misses a play (a spell it did not see) or reports one that
did not happen. Rather than special-casing each of those, the model keeps a
probability distribution over the four queue slots and updates it play by play:

* a play we saw fixes its slot to that card,
* a play we know happened but could not identify (`UNKNOWN`) may be any card
  that was in hand at the time,
* cards never seen are interchangeable placeholders, so "one of their unseen
  cards" is a first-class answer.

There are only 8*7*6*5 = 1680 possible queues, so this is exact, not sampled.
A play that the distribution says is impossible (the card is certainly still in
the queue) is a contradiction: either the play is a misread, or plays were
missed in between. `play()` reports it and the caller decides which; `repair`
inserts the fewest missed plays that make it consistent.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from itertools import permutations
from typing import Dict, List, Optional, Sequence, Tuple

DECK_SIZE = 8
HAND_SIZE = 4
QUEUE_SIZE = DECK_SIZE - HAND_SIZE
UNKNOWN: Optional[str] = None
UNSEEN = "?"                       # stands for "a card we have not seen yet"

_START = tuple(permutations(range(DECK_SIZE), QUEUE_SIZE))


@dataclass
class Token:
    card: Optional[str]            # None: a play happened, card unidentified
    t: float = 0.0
    inferred: bool = False         # inserted to explain a contradiction


@dataclass
class CardStatus:
    card: str
    p_hand: float                  # probability it is in hand now
    p_next: float                  # probability it is the next card
    p_slot: Tuple[float, ...]      # probability of each queue slot (0 = next)

    @property
    def plays_until_hand(self) -> Optional[int]:
        """Most likely number of further plays before it is back in hand."""
        if self.p_hand >= max(self.p_slot):
            return 0
        return 1 + max(range(QUEUE_SIZE), key=lambda i: self.p_slot[i])

    @property
    def certain(self) -> bool:
        return self.p_hand > 0.999 or max(self.p_slot) > 0.999


@dataclass
class CycleView:
    """Everything the overlay draws about the cycle, computed once per change."""
    deck: List[str]                       # known cards, first seen first
    statuses: Dict[str, CardStatus]
    hand: List[str]                       # best guess, UNSEEN for unknown cards
    hand_p: float                         # probability of exactly that hand
    next_card: str                        # card key or UNSEEN
    next_p: float
    queue: List[str]                      # most likely queue, next first
    plays: int
    unknown_plays: int
    unseen_in_hand: float                 # expected unseen cards in hand

    @property
    def deck_complete(self) -> bool:
        return len(self.deck) >= DECK_SIZE


@dataclass
class CycleModel:
    tokens: List[Token] = field(default_factory=list)
    _symbols: Dict[str, int] = field(default_factory=dict)
    _cards: List[str] = field(default_factory=list)
    _dist: Dict[tuple, float] = field(default_factory=dict)
    _view: Optional[CycleView] = None

    def __post_init__(self) -> None:
        self._rebuild()

    # ------------------------------------------------------------ the DP

    def _deck_order(self) -> List[str]:
        """The eight cards the model believes in.

        Normally just the first eight distinct cards seen. If a detector misread
        has introduced a ninth, keep the eight seen most often - a misread is
        rarely repeated, a real card is played every cycle.
        """
        counts = Counter(t.card for t in self.tokens if t.card)
        order: List[str] = []
        for t in self.tokens:
            if t.card and t.card not in order:
                order.append(t.card)
        if len(order) <= DECK_SIZE:
            return order
        keep = sorted(order, key=lambda c: (-counts[c], order.index(c)))[:DECK_SIZE]
        return [c for c in order if c in keep]

    def _rebuild(self) -> None:
        self._cards = self._deck_order()
        self._symbols = {c: i for i, c in enumerate(self._cards)}
        self._dist = {state: 1.0 for state in _START}
        for token in self.tokens:
            self._dist = self._advance(self._dist, self._symbols.get(token.card or ""))
        self._view = None

    @staticmethod
    def _advance(dist: Dict[tuple, float], symbol: Optional[int]) -> Dict[tuple, float]:
        out: Dict[tuple, float] = {}
        if symbol is not None:
            for state, w in dist.items():
                if symbol not in state:
                    nxt = state[1:] + (symbol,)
                    out[nxt] = out.get(nxt, 0.0) + w
        else:
            for state, w in dist.items():
                for s in range(DECK_SIZE):
                    if s not in state:
                        nxt = state[1:] + (s,)
                        out[nxt] = out.get(nxt, 0.0) + w
        total = sum(out.values())
        return {k: v / total for k, v in out.items()} if total > 0 else {}

    def _symbol_for_new(self, card: str) -> Optional[int]:
        if card in self._symbols:
            return self._symbols[card]
        if len(self._cards) < DECK_SIZE:
            return len(self._cards)          # an unseen placeholder becomes it
        return None                          # a ninth card: cannot be placed

    # ------------------------------------------------------------ queries

    def feasible(self, card: str) -> bool:
        """Could `card` be in their hand right now?"""
        symbol = self._symbol_for_new(card)
        if symbol is None:
            return True                      # deck overflow: no opinion
        return any(symbol not in state for state in self._dist)

    def p_in_hand(self, card: str) -> float:
        symbol = self._symbol_for_new(card)
        if symbol is None:
            return 0.5
        return sum(w for state, w in self._dist.items() if symbol not in state)

    def missed_plays_needed(self, card: str) -> int:
        """Fewest unseen plays that would make `card` playable now (0-4)."""
        symbol = self._symbol_for_new(card)
        if symbol is None:
            return 0
        dist = self._dist
        for k in range(QUEUE_SIZE + 1):
            if any(symbol not in state for state in dist):
                return k
            dist = self._advance(dist, None)
        return QUEUE_SIZE

    @property
    def deck(self) -> List[str]:
        return list(self._cards)

    @property
    def plays(self) -> int:
        return len(self.tokens)

    # ------------------------------------------------------------ updates

    def play(self, card: Optional[str], t: float = 0.0, repair: bool = True) -> int:
        """Record a play. Returns the number of missed plays inserted.

        `card=None` records a play whose card is unknown. When `card` is
        impossible right now and `repair` is set, the fewest unknown plays that
        make it possible are inserted before it; with `repair` unset an
        impossible play raises `ValueError` and nothing changes.
        """
        if card is None:
            self.tokens.append(Token(None, t))
            self._dist = self._advance(self._dist, None)
            self._view = None
            return 0
        missing = 0 if self.feasible(card) else self.missed_plays_needed(card)
        if missing and not repair:
            raise ValueError(f"{card} cannot be in hand: it was played too recently")
        for _ in range(missing):
            self.tokens.append(Token(None, t, inferred=True))
        self.tokens.append(Token(card, t))
        if card not in self._symbols and len(self._cards) < DECK_SIZE and not missing:
            # Cheap path: extend in place rather than replaying history.
            self._cards.append(card)
            self._symbols[card] = len(self._cards) - 1
            self._dist = self._advance(self._dist, self._symbols[card])
            self._view = None
        else:
            self._rebuild()
        return missing

    def undo(self) -> Optional[Token]:
        """Remove the most recent play (and any plays inferred for it)."""
        if not self.tokens:
            return None
        last = self.tokens.pop()
        while self.tokens and self.tokens[-1].inferred:
            self.tokens.pop()
        self._rebuild()
        return last

    def remove_last(self, card: str) -> bool:
        """Forget the most recent play of `card` - a correction from the user."""
        for i in range(len(self.tokens) - 1, -1, -1):
            if self.tokens[i].card == card:
                del self.tokens[i]
                self._rebuild()
                return True
        return False

    def reset(self) -> None:
        self.tokens.clear()
        self._rebuild()

    # ------------------------------------------------------------ the view

    def view(self) -> CycleView:
        if self._view is None:
            self._view = self._compute_view()
        return self._view

    def _name(self, symbol: int) -> str:
        return self._cards[symbol] if symbol < len(self._cards) else UNSEEN

    def _compute_view(self) -> CycleView:
        n_known = len(self._cards)
        slot = [[0.0] * QUEUE_SIZE for _ in range(DECK_SIZE)]
        hands: Counter = Counter()
        queues: Counter = Counter()
        for state, w in self._dist.items():
            for i, s in enumerate(state):
                slot[s][i] += w
            hand = tuple(sorted(self._name(s) for s in range(DECK_SIZE) if s not in state))
            hands[hand] += w
            queues[tuple(self._name(s) for s in state)] += w

        statuses = {}
        for card, s in self._symbols.items():
            in_queue = sum(slot[s])
            statuses[card] = CardStatus(card, max(0.0, 1.0 - in_queue), slot[s][0],
                                        tuple(slot[s]))
        unseen_in_hand = sum(1.0 - sum(slot[s]) for s in range(n_known, DECK_SIZE))

        if hands:
            hand, hand_p = hands.most_common(1)[0]
            queue = list(queues.most_common(1)[0][0])
        else:                                     # cannot happen; be safe
            hand, hand_p, queue = (UNSEEN,) * HAND_SIZE, 0.0, [UNSEEN] * QUEUE_SIZE

        # Order the hand by certainty so the sure cards come first.
        def certainty(card: str) -> float:
            return statuses[card].p_hand if card in statuses else -1.0
        hand_list = sorted(hand, key=lambda c: (-certainty(c), c))

        nexts: Counter = Counter()
        for s in range(DECK_SIZE):
            nexts[self._name(s)] += slot[s][0]
        next_card, next_p = nexts.most_common(1)[0]

        return CycleView(
            deck=list(self._cards), statuses=statuses, hand=hand_list, hand_p=hand_p,
            next_card=next_card, next_p=next_p, queue=queue, plays=len(self.tokens),
            unknown_plays=sum(1 for t in self.tokens if t.card is None),
            unseen_in_hand=unseen_in_hand,
        )


def replay(cards: Sequence[Optional[str]]) -> CycleModel:
    """Build a model from a list of plays; handy in tests and tools."""
    model = CycleModel()
    for i, card in enumerate(cards):
        model.play(card, float(i))
    return model

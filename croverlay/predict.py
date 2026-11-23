"""Guess the opponent's unseen cards from the decks people actually play.

Once three or four cards are known, most ladder decks are identified: a Hog
Rider with a Cannon and an Ice Golem is 2.6 Hog Cycle about as often as not, and
that says Skeletons, Ice Spirit, The Log, Fireball and Musketeer are coming.

The source is the top 200 Path of Legends decks by usage (MIT-licensed battle
dump, see data/meta_decks.json). Each deck containing every known card votes
for its other cards with its usage count. A deck missing one known card still
votes at a tenth of its weight, so a single misread does not wipe the
prediction out, and evolutions are ignored because they do not change the deck.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Iterable, List, Tuple

from . import catalog

DATA = Path(__file__).with_name("data") / "meta_decks.json"


@lru_cache(maxsize=1)
def _decks() -> Tuple[Tuple[frozenset, int], ...]:
    try:
        raw = json.loads(DATA.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ()
    out = []
    for entry in raw.get("decks", []):
        cards = frozenset(filter(None, (catalog.from_meta(c) for c in entry["cards"])))
        out.append((cards, int(entry.get("uses", 1))))
    return tuple(out)


def predict(known: Iterable[str], limit: int = 8) -> List[Tuple[str, float]]:
    """(card, probability it is in their deck) for cards not yet seen."""
    known = frozenset(known)
    if not known:
        return []
    votes: dict = {}
    total = 0.0
    for deck, uses in _decks():
        missing = len(known - deck)
        if missing > 1:
            continue
        weight = uses * (1.0 if missing == 0 else 0.1)
        total += weight
        for card in deck - known:
            votes[card] = votes.get(card, 0.0) + weight
    if total <= 0:
        return []
    ranked = sorted(((c, v / total) for c, v in votes.items()), key=lambda cv: -cv[1])
    return ranked[:limit]


def matching_decks(known: Iterable[str]) -> int:
    known = frozenset(known)
    return sum(1 for deck, _ in _decks() if known <= deck)

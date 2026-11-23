"""Replay recorded detector output through the overlay and measure it.

    python -m croverlay.evaluate PATH/TO/data/vod/tracks [--my-deck ...] [-v]

Solar-CR's `data/vod/tracks/*.jsonl` hold the detector's output for 108 real
matches, ten frames a second, already in arena tiles. There is no ground-truth
play list, so the measures are the ones a correct extractor cannot fail:

* **deck size** - a deck is eight cards; more than eight distinct cards for one
  player means plays were invented or given to the wrong player;
* **cycle contradictions** - a card played again before four others have been
  played is impossible; each one the cycle had to repair (or reject) is a misread
  or a miss somewhere;
* **pace** - real players make a play every few seconds; a rate far above that
  is re-acquisitions being counted as plays.

`side` in those files is the half of the arena a box was in, not a real side
label, so it is ignored here.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterator, List, Sequence, Tuple

from . import catalog
from .match import Battle
from .tracker import ALLY, ENEMY, Detection


def load_tracks(path: Path) -> Tuple[dict, List[Tuple[float, List[Detection]]]]:
    meta: dict = {}
    frames: Dict[float, List[Detection]] = defaultdict(list)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("kind") != "det":
                meta = record
                continue
            if record.get("tile_x") is None or record["t"] < 0:
                continue
            frames[record["t"]].append(Detection(record["name"], record["tile_x"],
                                                 record["tile_y"], record.get("conf", 1.0)))
    return meta, sorted(frames.items())


def read_meta(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        first = json.loads(handle.readline() or "{}")
    return {} if first.get("kind") == "det" else first


def run(path: Path, my_deck: Sequence[str] = ()) -> Battle:
    _, frames = load_tracks(path)
    battle = Battle(my_deck)
    for t, detections in frames:
        battle.update(t, detections)
    for play in battle.extractor.flush(battle.t + 10):
        battle.accept(play)
    return battle


def describe(battle: Battle) -> Dict[str, dict]:
    out = {}
    minutes = max(battle.t, 1.0) / 60.0
    for side in (ENEMY, ALLY):
        player = battle.players[side]
        cards = [p.card for p in player.log]
        distinct = sorted(set(cards))
        out[side] = {
            "plays": len(cards),
            "per_min": round(len(cards) / minutes, 1),
            "distinct": len(distinct),
            "repaired": sum(1 for p in player.log if p.inferred_before),
            "overspend_per_min": round(player.elixir.overspend() / minutes, 2),
            "rejected": player.rejected,
            "deck": player.cycle.deck,
            "cards": distinct,
        }
    return out


def main(argv: Sequence[str] = ()) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tracks", type=Path, help="a .jsonl file or a directory of them")
    ap.add_argument("--my-deck", nargs="*", default=(), help="the recording player's 8 cards")
    ap.add_argument("--truth-ally", nargs="*", default=(),
                    help="the recording player's real deck: scores ally precision")
    ap.add_argument("--glob", default="*.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-seconds", type=float, default=320.0,
                    help="skip spans longer than one battle can be (default 320)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(list(argv) or None)

    files = [args.tracks] if args.tracks.is_file() else sorted(args.tracks.glob(args.glob))
    if args.limit:
        files = files[:args.limit]
    rows = []
    for path in files:
        if args.max_seconds and read_meta(path).get("duration_s", 0) > args.max_seconds:
            continue
        battle = run(path, args.my_deck)
        stats = describe(battle)
        rows.append((path.stem, battle.t, stats, battle))
        if args.verbose:
            e, a = stats[ENEMY], stats[ALLY]
            print(f"{path.stem:<22} {battle.t:6.0f}s  enemy {e['plays']:>3} plays "
                  f"{e['distinct']:>2} cards {e['repaired']:>2} rep {e['rejected']:>2} rej | "
                  f"ally {a['plays']:>3} plays {a['distinct']:>2} cards")
            print(f"{'':<22} enemy deck: {', '.join(e['deck'])}")

    truth = set(filter(None, (catalog.from_any(c) for c in args.truth_ally)))
    if truth:
        plays = [c for r in rows for c in r[3].players[ALLY].log]
        good = sum(1 for p in plays if p.card in truth)
        top8 = [len(truth & {c for c, _ in Counter(p.card for p in r[3].players[ALLY].log).most_common(8)})
                for r in rows]
        print(f"ALLY vs TRUE DECK  precision {good}/{len(plays)} = {good / max(1, len(plays)):.0%}"
              f"   true cards among top-8 by count: mean {statistics.mean(top8):.1f}/8")
        for card in sorted(truth):
            print(f"  {card:<14} {sum(1 for p in plays if p.card == card):>5} plays")
        wrong = Counter(p.card for p in plays if p.card not in truth)
        print("  most common wrong:", ", ".join(f"{c} {n}" for c, n in wrong.most_common(8)))

    for side in (ENEMY, ALLY):
        s = [r[2][side] for r in rows]
        if not s:
            continue
        over = sum(1 for x in s if x["distinct"] > 8)
        print(f"\n{side.upper()}  ({len(s)} matches)")
        print(f"  plays per minute     median {statistics.median(x['per_min'] for x in s):5.1f}")
        print(f"  distinct cards       median {statistics.median(x['distinct'] for x in s):5.1f}"
              f"   matches over 8: {over} ({over / len(s):.0%})")
        print(f"  cycle repairs/match  mean   {statistics.mean(x['repaired'] for x in s):5.2f}")
        print(f"  rejected/match       mean   {statistics.mean(x['rejected'] for x in s):5.2f}")
        print(f"  overspend/min        median {statistics.median(x['overspend_per_min'] for x in s):5.2f}"
              "   (elixir the plays needed that the opponent could not have had)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

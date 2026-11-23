"""The opponent's elixir, as a timeline rather than a running total.

Nothing on screen reports enemy elixir, so it is rebuilt from what is public:

* both players start at 5 and cap at 10,
* one elixir every 2.8 s, every 1.4 s from 2:00 elapsed (the last minute of
  regulation and the first of overtime), every 0.933 s from 4:00,
* each play costs its card's elixir,
* an Elixir Collector adds 1 every 9 s for its 65 s life and 1 when destroyed.

(Those constants are the ones Solar-CR's simulator uses, which were taken from
the game's tables: `sim/match.py`, `cards_stats_building.json`.)

Two properties make the estimate self-correcting in the direction that matters:

* **A play proves they could afford it.** If the estimate says 2 and they play a
  4-cost card, the estimate was low, so it is floored at 0 after the purchase
  rather than going negative. A spurious play therefore costs at most one
  card's worth of error, and the next real play clears it.
* **The bar cannot exceed 10.** A missed play leaves the estimate high, but only
  until the bar would have filled anyway.

Events are stored with their match time and the value is recomputed from the
start on every query, so a correction (undo, a late-inferred missed play) is
just an edit to the list.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

START = 5.0
CAP = 10.0
SINGLE_S, DOUBLE_S, TRIPLE_S = 2.8, 1.4, 2.8 / 3
DOUBLE_AT, TRIPLE_AT = 120.0, 240.0

COLLECTOR_EVERY = 9.0
COLLECTOR_LIFE = 65.0


@dataclass(order=True)
class Event:
    t: float
    delta: float = field(compare=False)          # negative = spend
    label: str = field(compare=False, default="")
    floor: bool = field(compare=False, default=True)  # spends prove affordability


@dataclass
class ElixirModel:
    start: float = START
    double_at: float = DOUBLE_AT
    triple_at: float = TRIPLE_AT
    events: List[Event] = field(default_factory=list)

    # ---------------------------------------------------------------- rates

    def seconds_per_elixir(self, t: float) -> float:
        if t >= self.triple_at:
            return TRIPLE_S
        if t >= self.double_at:
            return DOUBLE_S
        return SINGLE_S

    def multiplier(self, t: float) -> int:
        return 3 if t >= self.triple_at else 2 if t >= self.double_at else 1

    def _regen(self, t0: float, t1: float) -> float:
        """Elixir generated between two match times, crossing phase changes."""
        if t1 <= t0:
            return 0.0
        total, t = 0.0, t0
        for boundary in (self.double_at, self.triple_at, float("inf")):
            if t >= boundary:
                continue
            end = min(t1, boundary)
            total += (end - t) / self.seconds_per_elixir(t)
            t = end
            if t >= t1:
                break
        return total

    def _time_to_gain(self, t0: float, amount: float) -> float:
        """Seconds from t0 until `amount` more elixir has been generated."""
        t, left = t0, amount
        for boundary in (self.double_at, self.triple_at, float("inf")):
            if t >= boundary:
                continue
            rate = 1.0 / self.seconds_per_elixir(t)
            span = boundary - t
            if span * rate >= left:
                return t + left / rate - t0
            left -= span * rate
            t = boundary
        return t - t0

    # ---------------------------------------------------------------- events

    def spend(self, t: float, cost: float, label: str = "", proven: bool = True) -> None:
        """A purchase. `proven=False` for a play inferred rather than seen,
        whose cost is a guess and so says nothing about what they could afford."""
        bisect.insort(self.events, Event(t, -abs(cost), label, floor=proven))

    def gain(self, t: float, amount: float, label: str = "") -> None:
        bisect.insort(self.events, Event(t, abs(amount), label, floor=False))

    def add_collector(self, t: float, destroyed_at: Optional[float] = None) -> None:
        """Their Elixir Collector landed at `t`; schedule what it pays out."""
        end = t + COLLECTOR_LIFE if destroyed_at is None else min(destroyed_at, t + COLLECTOR_LIFE)
        tick = t + COLLECTOR_EVERY
        while tick <= end + 1e-6:
            self.gain(tick, 1.0, "collector")
            tick += COLLECTOR_EVERY
        self.gain(end, 1.0, "collector destroyed")

    def collector_destroyed(self, placed_at: float, at: float) -> None:
        """Cut a collector's schedule short when it is seen to die early."""
        self.events = [e for e in self.events
                       if not (e.label.startswith("collector") and placed_at < e.t)]
        self.add_collector(placed_at, destroyed_at=at)

    def remove_last(self, label: str) -> bool:
        for i in range(len(self.events) - 1, -1, -1):
            if self.events[i].label == label:
                del self.events[i]
                return True
        return False

    def reset(self) -> None:
        self.events.clear()

    # ---------------------------------------------------------------- reading

    def value(self, t: float) -> float:
        """Estimated elixir at match time `t`."""
        v, now = self.start, 0.0
        for event in self.events:
            if event.t > t:
                break
            v = min(CAP, v + self._regen(now, event.t))
            now = event.t
            if event.delta < 0 and event.floor:
                v = max(v, -event.delta)          # they had at least this much
            v = min(CAP, max(0.0, v + event.delta))
        return min(CAP, v + self._regen(now, t))

    def overspend(self) -> float:
        """Total elixir the recorded plays needed beyond what the model allowed.

        Zero for a correct play list (missed plays only make the estimate
        high). Every invented play pushes it up, so it is the evaluation's
        measure of false positives.
        """
        v, now, short = self.start, 0.0, 0.0
        for event in self.events:
            v = min(CAP, v + self._regen(now, event.t))
            now = event.t
            if event.delta < 0 and event.floor:
                short += max(0.0, -event.delta - v)
                v = max(v, -event.delta)
            v = min(CAP, max(0.0, v + event.delta))
        return short

    def time_to(self, t: float, target: float) -> float:
        """Seconds until the estimate reaches `target` with no further plays."""
        v = self.value(t)
        if v >= target:
            return 0.0
        return self._time_to_gain(t, target - v)

    def spent_since(self, t0: float, t1: float) -> float:
        return -sum(e.delta for e in self.events if t0 <= e.t <= t1 and e.delta < 0)

    def timeline(self, t: float, step: float = 0.5) -> List[Tuple[float, float]]:
        """(time, value) samples up to `t`, for a sparkline."""
        out, s = [], 0.0
        while s <= t:
            out.append((s, self.value(s)))
            s += step
        return out

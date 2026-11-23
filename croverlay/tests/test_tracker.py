"""Play extraction on synthetic detector output: one test per failure mode."""

from typing import Dict, List, Sequence, Tuple

import pytest

from croverlay.match import Battle
from croverlay.tracker import ALLY, ENEMY, Detection, PlayExtractor

FPS = 10


class Script:
    """Units moving on straight lines, rendered as detector frames."""

    def __init__(self):
        self.items: List[Tuple[str, float, float, Tuple[float, float], Tuple[float, float], float]] = []
        self.gaps: List[Tuple[str, float, float]] = []

    def unit(self, name, t0, t1, start, end=None, conf=0.8):
        self.items.append((name, t0, t1, start, end or start, conf))
        return self

    def hide(self, name, t0, t1):
        """The detector misses `name` between t0 and t1."""
        self.gaps.append((name, t0, t1))
        return self

    def frames(self, until: float):
        n = int(until * FPS) + 1
        for i in range(n):
            t = round(i / FPS, 3)
            dets = []
            for name, t0, t1, (x0, y0), (x1, y1), conf in self.items:
                if not (t0 <= t <= t1):
                    continue
                if any(g == name and a <= t <= b for g, a, b in self.gaps):
                    continue
                f = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
                dets.append(Detection(name, x0 + (x1 - x0) * f, y0 + (y1 - y0) * f, conf))
            yield t, dets


def run(script: Script, until: float, extractor=None):
    ex = extractor or PlayExtractor()
    plays = []
    for t, dets in script.frames(until):
        plays += ex.update(t, dets)
    plays += ex.flush(until + 5)
    return plays, ex


def summary(plays) -> List[Tuple[str, str]]:
    return [(p.side, p.card) for p in plays]


def test_enemy_troop_deployed_in_their_half():
    plays, _ = run(Script().unit("hog-rider", 1.0, 6.0, (3.5, 13.0), (3.5, 24.0)), 8)
    assert summary(plays) == [(ENEMY, "hog_rider")]


def test_crossing_the_river_is_not_a_new_play():
    # Our Hog walks from our bridge up into their half, continuously tracked.
    plays, _ = run(Script().unit("hog-rider", 1.0, 6.0, (14.5, 18.0), (14.5, 8.0)), 8)
    assert summary(plays) == [(ALLY, "hog_rider")]


def test_our_unit_lost_and_refound_in_their_half_is_not_their_play():
    s = (Script().unit("hog-rider", 1.0, 9.0, (14.5, 18.0), (14.5, 8.5))
         .hide("hog-rider", 2.5, 8.0))          # lost for 5.5 s, found at their tower
    plays, _ = run(s, 10)
    assert summary(plays) == [(ALLY, "hog_rider")]


def test_unit_count_separates_skeletons_from_skeleton_army():
    army = Script()
    for i in range(15):
        army.unit("skeleton", 2.0 + 0.02 * i, 6.0, (8 + (i % 5) * 0.5, 12 + (i // 5) * 0.5))
    plays, _ = run(army, 8)
    assert summary(plays) == [(ENEMY, "skeleton_army")]

    trio = Script()
    for i in range(3):
        trio.unit("skeleton", 2.0, 6.0, (8 + i * 0.6, 12.0))
    plays, _ = run(trio, 8)
    assert summary(plays) == [(ENEMY, "skeletons")]


def test_spawner_output_is_not_a_play():
    s = Script().unit("tombstone", 1.0, 20.0, (9.0, 10.0))
    for k in range(3):
        s.unit("skeleton", 5.0 + 4 * k, 7.0 + 4 * k, (9.0 + 0.5 * k, 11.0))
    plays, _ = run(s, 20)
    assert summary(plays) == [(ENEMY, "tombstone")]


def test_two_classes_one_card():
    s = Script()
    for i in range(3):
        s.unit("goblin", 2.0, 6.0, (4 + i * 0.6, 13.0))
        s.unit("spear-goblin", 2.0, 6.0, (4 + i * 0.6, 11.8))
    plays, _ = run(s, 8)
    assert summary(plays) == [(ENEMY, "goblin_gang")]


def test_spell_owner_from_direction_of_travel():
    theirs = Script().unit("fireball", 3.0, 3.8, (9.0, 4.0), (4.0, 23.0))
    plays, _ = run(theirs, 6)
    assert summary(plays) == [(ENEMY, "fireball")]

    ours = Script().unit("the-log", 3.0, 4.5, (14.0, 18.0), (14.0, 8.0))
    plays, _ = run(ours, 6)
    assert summary(plays) == [(ALLY, "the_log")]


def test_own_deploy_marker_claims_the_play():
    s = (Script().unit("zap", 3.0, 3.4, (5.0, 10.0))
         .unit("elixir", 3.0, 4.0, (5.2, 10.3))
         .unit("text", 3.0, 4.0, (5.0, 9.8)))
    plays, _ = run(s, 6)
    assert summary(plays) == [(ALLY, "zap")]


def test_misread_on_an_established_unit_is_ignored():
    s = (Script().unit("knight", 1.0, 8.0, (5.0, 12.0), (5.0, 18.0))
         .unit("valkyrie", 4.0, 4.3, (5.0, 14.1)))
    plays, _ = run(s, 10)
    assert summary(plays) == [(ENEMY, "knight")]


def test_troop_appearing_deep_in_the_wrong_half_is_dropped():
    # First seen at our king tower with no history: a unit we lost, not a play
    # anyone could have made there - unless it is ours.
    plays, _ = run(Script().unit("pekka", 1.0, 3.0, (9.0, 26.0), (9.0, 26.0)), 5)
    assert summary(plays) == [(ALLY, "pekka")]
    b = Battle(my_deck=["hog_rider", "musketeer", "cannon", "ice_golem",
                        "skeletons", "ice_spirit", "the_log", "fireball"])
    plays, _ = run(Script().unit("pekka", 1.0, 3.0, (9.0, 26.0)), 5, b.extractor)
    assert plays == []


def test_battle_builds_hand_and_elixir():
    b = Battle()
    s = Script()
    deck = [("hog-rider", "hog_rider"), ("musketeer", "musketeer"), ("cannon", "cannon"),
            ("ice-golem", "ice_golem"), ("knight", "knight"), ("ice-spirit", "ice_spirit"),
            ("valkyrie", "valkyrie"), ("bomber", "bomber")]
    for i, (cls, _) in enumerate(deck):
        t = 2.0 + 6.0 * i
        s.unit(cls, t, t + 3.0, (3.0 + 1.5 * i, 11.0))
    for t, dets in s.frames(55):
        b.update(t, dets)
    log = [p.card for p in b.players[ENEMY].log]
    assert log == [card for _, card in deck]
    view = b.players[ENEMY].cycle.view()
    # The queue is the last four plays, so the next card is the oldest of them.
    assert view.queue == ["knight", "ice_spirit", "valkyrie", "bomber"]
    assert view.next_card == "knight" and view.next_p == pytest.approx(1.0)
    assert sorted(view.hand) == ["cannon", "hog_rider", "ice_golem", "musketeer"]
    snap = b.snapshot()
    assert 0.0 <= snap.elixir <= 10.0
    assert snap.cycle.deck_complete


def test_reidentified_after_a_gap_longer_than_track_memory():
    # Our Hog reaches their tower and sits there hitting it; the detector loses
    # it for 13 s - longer than tracks are kept - then finds it again.
    s = (Script().unit("hog-rider", 1.0, 6.0, (14.5, 18.0), (14.5, 8.5))
         .unit("hog-rider", 6.0, 20.0, (14.5, 8.5))
         .hide("hog-rider", 2.0, 15.0))
    plays, ex = run(s, 21)
    assert summary(plays) == [(ALLY, "hog_rider")]
    assert ex.rejected["re-acquired unit"] == 1

    # Their Prince crosses into our half and is lost the same way.
    s = (Script().unit("prince", 1.0, 6.0, (3.5, 12.0), (3.5, 23.0))
         .unit("prince", 6.0, 20.0, (3.5, 23.0))
         .hide("prince", 2.0, 15.0))
    plays, _ = run(s, 21)
    assert summary(plays) == [(ENEMY, "prince")]

import pytest

from croverlay.cycle import UNSEEN, CycleModel, replay

DECK = ["hog_rider", "musketeer", "cannon", "ice_golem",
        "skeletons", "ice_spirit", "the_log", "fireball"]


def test_queue_is_last_four_plays_once_four_are_played():
    model = replay(DECK[:4])
    view = model.view()
    # Four distinct plays: the queue is exactly those four, oldest first,
    # so the hand is the four cards not yet seen.
    assert view.queue == DECK[:4]
    assert view.next_card == "hog_rider" and view.next_p == pytest.approx(1.0)
    assert view.hand == [UNSEEN] * 4
    assert view.hand_p == pytest.approx(1.0)


def test_full_deck_gives_exact_hand_and_next():
    model = replay(DECK + ["hog_rider"])   # hog back after 4 others: legal
    view = model.view()
    assert view.deck_complete
    assert view.queue == ["ice_spirit", "the_log", "fireball", "hog_rider"]
    assert view.next_card == "ice_spirit"
    assert sorted(view.hand) == sorted(["musketeer", "cannon", "ice_golem", "skeletons"])
    assert view.hand_p == pytest.approx(1.0)
    assert model.view().statuses["hog_rider"].plays_until_hand == 4
    assert model.view().statuses["ice_spirit"].plays_until_hand == 1


def test_early_game_probabilities_follow_the_unknown_starting_queue():
    model = replay(["hog_rider"])
    view = model.view()
    # After one play the hog is at the back of the queue for certain.
    assert view.statuses["hog_rider"].p_slot[3] == pytest.approx(1.0)
    assert view.statuses["hog_rider"].p_hand == pytest.approx(0.0)
    # The next card is one of the starting queue: an unseen card.
    assert view.next_card == UNSEEN
    # 7 unknown cards, 4 in hand, 3 of the starting queue still waiting.
    assert view.unseen_in_hand == pytest.approx(4.0)


def test_replaying_a_card_too_soon_is_infeasible():
    model = replay(["hog_rider", "musketeer"])
    assert not model.feasible("hog_rider")
    assert model.missed_plays_needed("hog_rider") == 3
    with pytest.raises(ValueError):
        model.play("hog_rider", repair=False)
    assert model.plays == 2


def test_repair_inserts_the_fewest_missed_plays():
    model = replay(["hog_rider", "musketeer"])
    inserted = model.play("hog_rider", 10.0)
    assert inserted == 3
    assert model.plays == 6
    assert model.view().unknown_plays == 3
    # Undo removes the play and the plays inferred for it.
    model.undo()
    assert model.plays == 2


def test_unknown_play_spreads_probability_but_stays_consistent():
    model = replay(DECK + [None])
    view = model.view()
    # The unknown play was one of musketeer..? (the hand at the time was
    # the first four cards), so each of them is now in hand with p = 3/4.
    for card in DECK[:4]:
        assert view.statuses[card].p_hand == pytest.approx(0.75)
    assert view.next_card == "ice_spirit"


def test_a_ninth_card_does_not_break_the_model():
    model = replay(DECK + ["hog_rider", "musketeer", "cannon", "ice_golem"])
    model.play("wizard", 99.0)           # a misread
    view = model.view()
    assert len(view.deck) == 8 and "wizard" not in view.deck
    assert view.plays == 13


def test_remove_last_forgets_a_wrong_play():
    model = replay(["hog_rider", "musketeer", "cannon"])
    assert model.remove_last("musketeer")
    assert model.deck == ["hog_rider", "cannon"]

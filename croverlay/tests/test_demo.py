"""End to end: scripted opponent -> noisy detections -> session -> hand."""

from croverlay.demo import DemoFeed
from croverlay.match import Session


def test_demo_opponent_is_reconstructed():
    now = [0.0]
    feed = DemoFeed(seed=3, verbose=False, recall=0.85, clock=lambda: now[0])
    session = Session()
    for _ in range(int(170 / 0.1)):
        now[0] += 0.1
        result = feed.step()
        if result:
            session.update(now[0], *result)
    battle = session.battle
    seen = [p.card for p in battle.players["enemy"].log]
    truth = [c for _, c in feed.truth]
    assert len(truth) > 25
    # Every logged play is a real one, in order; most real plays are caught.
    it = iter(truth)
    assert all(card in it for card in seen), (seen, truth)
    assert len(seen) >= 0.85 * len(truth), (len(seen), len(truth))
    view = battle.players["enemy"].cycle.view()
    assert sorted(view.deck) == sorted(feed.deck)
    if view.unknown_plays == 0:
        assert sorted(view.hand) == sorted(feed.hand)
        assert view.next_card == feed.queue[0]
    assert battle.players["ally"].log == []

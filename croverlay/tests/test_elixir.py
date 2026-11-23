import pytest

from croverlay.elixir import ElixirModel


def test_starts_at_five_and_regenerates_at_single_rate():
    m = ElixirModel()
    assert m.value(0) == pytest.approx(5.0)
    assert m.value(2.8) == pytest.approx(6.0)
    assert m.value(100) == pytest.approx(10.0)          # capped


def test_regen_crosses_into_double_and_triple():
    m = ElixirModel()
    m.spend(119.0, 10)                                  # drain to 0
    assert m.value(119.0) == pytest.approx(0.0)
    # 1 s at single (1/2.8) then 1.4 s at double (1.0)
    assert m.value(121.4) == pytest.approx(1 / 2.8 + 1.0)
    m.spend(239.0, 10)
    assert m.value(240.0 + 2.8 / 3) == pytest.approx(1 / 1.4 + 1.0)
    assert m.multiplier(250) == 3


def test_a_play_proves_affordability():
    m = ElixirModel()
    m.spend(1.0, 5)
    m.spend(2.0, 4)        # estimate said ~0.36: they must have had 4
    assert m.value(2.0) == pytest.approx(0.0)


def test_inferred_spend_does_not_floor():
    m = ElixirModel()
    m.spend(0.0, 5)
    m.spend(1.0, 3.5, proven=False)
    assert m.value(1.0) == pytest.approx(0.0)
    m2 = ElixirModel()
    m2.spend(0.0, 2)
    m2.spend(0.0, 3.5, proven=False)   # 3 left, guessed spend clamps at 0
    assert m2.value(0.0) == pytest.approx(0.0)


def test_leaking_at_ten_is_not_banked():
    m = ElixirModel()
    m.spend(60.0, 4)
    assert m.value(60.0) == pytest.approx(6.0)


def test_time_to_full_and_collector():
    m = ElixirModel()
    m.spend(0.0, 5)
    assert m.time_to(0.0, 10) == pytest.approx(28.0)
    assert m.time_to(0.0, 4) == pytest.approx(4 * 2.8)
    m.add_collector(0.0)
    assert m.value(9.0) == pytest.approx(9 / 2.8 + 1)
    m.collector_destroyed(0.0, 10.0)
    gains = [e for e in m.events if e.delta > 0]
    assert len(gains) == 2                    # one tick at 9 s, one on death

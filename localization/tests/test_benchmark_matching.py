import pytest

from localization.benchmark.matching import circular_error_deg, match_predictions


def test_circular_error_wraparound():
    assert circular_error_deg(359.0, 1.0) == pytest.approx(2.0)
    assert circular_error_deg(10.0, 350.0) == pytest.approx(20.0)


def test_matching_balanced():
    true = [10.0, 200.0]
    pred = [12.0, 198.0]
    m = match_predictions(true, pred)
    assert len(m.matched_errors_deg) == 2
    assert m.misses == 0
    assert m.false_alarms == 0


def test_matching_unbalanced_counts():
    true = [10.0, 200.0, 300.0]
    pred = [9.0, 195.0]
    m = match_predictions(true, pred)
    assert len(m.matched_errors_deg) == 2
    assert m.misses == 1
    assert m.false_alarms == 0

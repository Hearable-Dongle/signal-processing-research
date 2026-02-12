import pytest

from localization.benchmark.matching import MatchResult
from localization.benchmark.metrics import compute_scene_metrics


def test_scene_metrics_basic():
    match = MatchResult(
        matched_true_indices=[0, 1],
        matched_pred_indices=[0, 1],
        matched_errors_deg=[2.0, 8.0],
        misses=0,
        false_alarms=0,
    )
    m = compute_scene_metrics(match, n_true=2, n_pred=2)
    assert m.mae_deg_matched == pytest.approx(5.0)
    assert m.acc_within_5deg == pytest.approx(0.5)
    assert m.acc_within_10deg == pytest.approx(1.0)
    assert m.recall == pytest.approx(1.0)
    assert m.precision == pytest.approx(1.0)
    assert m.f1 == pytest.approx(1.0)


def test_scene_metrics_precision_recall_edge_cases():
    match = MatchResult([], [], [], misses=2, false_alarms=0)
    m = compute_scene_metrics(match, n_true=2, n_pred=0)
    assert m.recall == 0.0
    assert m.precision is None
    assert m.f1 is None

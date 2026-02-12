from __future__ import annotations

from dataclasses import dataclass

from localization.benchmark.matching import MatchResult, safe_mean, safe_median, safe_rmse


@dataclass(frozen=True)
class SceneMetrics:
    mae_deg_matched: float | None
    rmse_deg_matched: float | None
    median_ae_deg: float | None
    acc_within_5deg: float | None
    acc_within_10deg: float | None
    acc_within_15deg: float | None
    recall: float | None
    precision: float | None
    f1: float | None
    n_true: int
    n_pred: int
    n_matched: int
    misses: int
    false_alarms: int


def _acc_within(errors_deg: list[float], threshold: float) -> float | None:
    if not errors_deg:
        return None
    hits = sum(1 for e in errors_deg if e <= threshold)
    return hits / len(errors_deg)


def _safe_div(num: float, den: float) -> float | None:
    if den == 0:
        return None
    return num / den


def compute_scene_metrics(match: MatchResult, n_true: int, n_pred: int) -> SceneMetrics:
    n_matched = len(match.matched_errors_deg)
    recall = _safe_div(n_matched, n_true)
    precision = _safe_div(n_matched, n_pred)
    if recall is None or precision is None or recall + precision == 0:
        f1 = None
    else:
        f1 = 2.0 * recall * precision / (recall + precision)

    return SceneMetrics(
        mae_deg_matched=safe_mean(match.matched_errors_deg),
        rmse_deg_matched=safe_rmse(match.matched_errors_deg),
        median_ae_deg=safe_median(match.matched_errors_deg),
        acc_within_5deg=_acc_within(match.matched_errors_deg, 5.0),
        acc_within_10deg=_acc_within(match.matched_errors_deg, 10.0),
        acc_within_15deg=_acc_within(match.matched_errors_deg, 15.0),
        recall=recall,
        precision=precision,
        f1=f1,
        n_true=n_true,
        n_pred=n_pred,
        n_matched=n_matched,
        misses=match.misses,
        false_alarms=match.false_alarms,
    )

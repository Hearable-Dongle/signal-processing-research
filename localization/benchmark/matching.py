from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class MatchResult:
    matched_true_indices: list[int]
    matched_pred_indices: list[int]
    matched_errors_deg: list[float]
    misses: int
    false_alarms: int


def circular_error_deg(a_deg: float, b_deg: float) -> float:
    diff = abs((a_deg - b_deg) % 360.0)
    return min(diff, 360.0 - diff)


def _cost_matrix(true_angles_deg: list[float], pred_angles_deg: list[float]) -> np.ndarray:
    mat = np.zeros((len(true_angles_deg), len(pred_angles_deg)), dtype=float)
    for i, t in enumerate(true_angles_deg):
        for j, p in enumerate(pred_angles_deg):
            mat[i, j] = circular_error_deg(t, p)
    return mat


def match_predictions(true_angles_deg: list[float], pred_angles_deg: list[float]) -> MatchResult:
    n_true = len(true_angles_deg)
    n_pred = len(pred_angles_deg)

    if n_true == 0 and n_pred == 0:
        return MatchResult([], [], [], 0, 0)
    if n_true == 0:
        return MatchResult([], [], [], 0, n_pred)
    if n_pred == 0:
        return MatchResult([], [], [], n_true, 0)

    cost = _cost_matrix(true_angles_deg, pred_angles_deg)
    row_ind, col_ind = linear_sum_assignment(cost)

    matched_errors = [float(cost[r, c]) for r, c in zip(row_ind, col_ind)]

    return MatchResult(
        matched_true_indices=[int(x) for x in row_ind],
        matched_pred_indices=[int(x) for x in col_ind],
        matched_errors_deg=matched_errors,
        misses=max(0, n_true - len(row_ind)),
        false_alarms=max(0, n_pred - len(col_ind)),
    )


def safe_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(values))


def safe_rmse(values: list[float]) -> float | None:
    if not values:
        return None
    arr = np.asarray(values, dtype=float)
    return float(math.sqrt(np.mean(arr * arr)))


def safe_median(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.median(np.asarray(values, dtype=float)))

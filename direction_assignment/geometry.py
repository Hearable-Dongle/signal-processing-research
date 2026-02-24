from __future__ import annotations

import numpy as np


def normalize_angle_deg(angle_deg: float) -> float:
    return float(angle_deg % 360.0)


def circular_diff_deg(a_deg: float, b_deg: float) -> float:
    return float((a_deg - b_deg + 180.0) % 360.0 - 180.0)


def circular_distance_deg(a_deg: float, b_deg: float) -> float:
    return abs(circular_diff_deg(a_deg, b_deg))


def circular_mean_deg(angles_deg: np.ndarray, weights: np.ndarray | None = None) -> float:
    if angles_deg.size == 0:
        raise ValueError("angles_deg must not be empty")

    angles_rad = np.deg2rad(angles_deg)
    if weights is None:
        weights = np.ones_like(angles_rad, dtype=float)
    else:
        weights = np.asarray(weights, dtype=float)

    s = float(np.sum(weights * np.sin(angles_rad)))
    c = float(np.sum(weights * np.cos(angles_rad)))
    return normalize_angle_deg(np.rad2deg(np.arctan2(s, c)))


def build_mic_pairs(mic_geometry: np.ndarray, min_baseline_m: float) -> list[tuple[int, int]]:
    # Only XY is used for azimuth estimation.
    xy = np.asarray(mic_geometry, dtype=float)
    if xy.ndim != 2:
        raise ValueError("mic_geometry must be 2D array")
    if xy.shape[1] > 2:
        xy = xy[:, :2]

    n_mics = xy.shape[0]
    pairs: list[tuple[int, int]] = []
    for i in range(n_mics):
        for j in range(i + 1, n_mics):
            if float(np.linalg.norm(xy[j] - xy[i])) >= min_baseline_m:
                pairs.append((i, j))
    return pairs

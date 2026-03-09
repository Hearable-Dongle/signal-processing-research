from __future__ import annotations

import numpy as np


def mic_positions_xyz(profile: str) -> np.ndarray:
    if profile == "respeaker_cross_0640":
        r = 0.032  # 64.0 mm across
        return np.array(
            [
                [r, 0.0, 0.0],
                [0.0, r, 0.0],
                [-r, 0.0, 0.0],
                [0.0, -r, 0.0],
            ],
            dtype=np.float64,
        )
    if profile == "respeaker_v3_0457":
        r = 0.0457 / 2.0  # ReSpeaker Mic Array v3.0 ~45.7 mm across.
        return np.array(
            [
                [r, 0.0, 0.0],
                [0.0, r, 0.0],
                [-r, 0.0, 0.0],
                [0.0, -r, 0.0],
            ],
            dtype=np.float64,
        )
    raise ValueError(f"Unsupported mic array profile: {profile}")

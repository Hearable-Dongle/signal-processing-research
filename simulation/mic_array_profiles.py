from __future__ import annotations

import numpy as np


SUPPORTED_MIC_ARRAY_PROFILES = (
    "respeaker_v3_0457",
    "respeaker_xvf3800_0650",
)


def mic_positions_xyz(profile: str) -> np.ndarray:
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
    if profile == "respeaker_xvf3800_0650":
        r = 0.065 / 2.0  # 65.0 mm across, square geometry.
        return np.array(
            [
                [r, r, 0.0],
                [-r, r, 0.0],
                [-r, -r, 0.0],
                [r, -r, 0.0],
            ],
            dtype=np.float64,
        )
    raise ValueError(f"Unsupported mic array profile: {profile}")

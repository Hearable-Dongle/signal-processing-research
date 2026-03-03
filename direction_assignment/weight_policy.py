from __future__ import annotations

import math

import numpy as np

from .config import DirectionAssignmentConfig
from .geometry import circular_distance_deg
from .types import SpeakerDirectionState


def db_to_linear(db: float) -> float:
    db_val = float(db)
    if not math.isfinite(db_val):
        return 1.0
    linear = 10.0 ** (db_val / 20.0)
    if not math.isfinite(linear):
        return 1.0
    return float(max(linear, 0.0))


def build_target_weights(
    states: dict[int, SpeakerDirectionState],
    candidate_ids: list[int],
    focus_speaker_ids: set[int] | None,
    focus_direction_deg: float | None,
    user_boost_db: float | None,
    cfg: DirectionAssignmentConfig,
) -> list[float]:
    if not candidate_ids:
        return []

    w = np.ones(len(candidate_ids), dtype=float)
    focus_gain = db_to_linear(float(cfg.focus_gain_db) + float(user_boost_db or 0.0))
    non_focus_gain = db_to_linear(cfg.non_focus_attenuation_db)

    if focus_speaker_ids:
        for i, sid in enumerate(candidate_ids):
            w[i] = focus_gain if sid in focus_speaker_ids else non_focus_gain

    if focus_direction_deg is not None and not focus_speaker_ids:
        doas = [states[sid].direction_deg for sid in candidate_ids]
        nearest_idx = int(np.argmin([circular_distance_deg(d, focus_direction_deg) for d in doas]))
        w[:] = non_focus_gain
        w[nearest_idx] = focus_gain

    w = np.maximum(w, 0.0)
    if not np.all(np.isfinite(w)) or float(np.sum(w)) <= 1e-12:
        w[:] = 1.0

    return [float(v) for v in w]

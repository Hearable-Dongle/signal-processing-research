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

    if str(cfg.control_mode) == "speaker_tracking_mode":
        for i, sid in enumerate(candidate_ids):
            st = states[sid]
            confidence_gate = float(
                np.clip(
                    0.55 * float(getattr(st, "confidence", 0.0))
                    + 0.3 * float(getattr(st, "identity_confidence", 0.0))
                    + 0.15 * float(getattr(st, "activity_confidence", 0.0)),
                    0.0,
                    1.0,
                )
            )
            maturity_scale = 1.0 if str(getattr(st, "identity_maturity", "unknown")) == "stable" else 0.75
            w[i] *= max(0.1, confidence_gate * maturity_scale)

    w = np.maximum(w, 0.0)
    if not np.all(np.isfinite(w)) or float(np.sum(w)) <= 1e-12:
        w[:] = 1.0

    return [float(v) for v in w]

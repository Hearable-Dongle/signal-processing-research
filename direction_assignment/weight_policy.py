from __future__ import annotations

import numpy as np

from .config import DirectionAssignmentConfig
from .geometry import circular_distance_deg
from .types import SpeakerDirectionState


def build_target_weights(
    states: dict[int, SpeakerDirectionState],
    candidate_ids: list[int],
    focus_speaker_ids: set[int] | None,
    focus_direction_deg: float | None,
    cfg: DirectionAssignmentConfig,
) -> list[float]:
    if not candidate_ids:
        return []

    w = np.ones(len(candidate_ids), dtype=float)

    if focus_speaker_ids:
        for i, sid in enumerate(candidate_ids):
            w[i] = 1.0 if sid in focus_speaker_ids else cfg.non_focus_weight

    if focus_direction_deg is not None and not focus_speaker_ids:
        doas = [states[sid].direction_deg for sid in candidate_ids]
        nearest_idx = int(np.argmin([circular_distance_deg(d, focus_direction_deg) for d in doas]))
        w[:] = cfg.non_focus_weight
        w[nearest_idx] = 1.0

    w = np.maximum(w, 0.0)
    if float(np.sum(w)) <= 1e-12:
        w[:] = 1.0

    # beamforming.normalize_target_weights normalizes again, but we emit normalized too.
    w = w / np.max(w)
    return [float(v) for v in w]

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class DirectionAssignmentInput:
    chunk_id: int
    timestamp_ms: float
    raw_mic_chunk: np.ndarray  # shape: (samples, n_mics)
    separated_streams: list[np.ndarray]  # each shape: (samples,)
    stream_to_speaker: dict[int, int | None]
    active_speakers: list[int]
    srp_doa_peaks_deg: list[float]
    srp_peak_scores: list[float] | None = None


@dataclass
class SpeakerDirectionState:
    speaker_id: int
    direction_deg: float
    confidence: float
    last_update_ms: float
    updates: int = 0


@dataclass
class DirectionAssignmentOutput:
    chunk_id: int
    timestamp_ms: float
    speaker_directions_deg: dict[int, float]
    speaker_confidence: dict[int, float]
    target_speaker_ids: list[int]
    target_doas_deg: list[float]
    target_weights: list[float]
    debug: dict = field(default_factory=dict)

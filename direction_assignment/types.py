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
    control_mode: str = "spatial_peak_mode"
    per_stream_identity_confidence: dict[int, float] = field(default_factory=dict)
    speaker_identity_metadata: dict[int, dict[str, object]] = field(default_factory=dict)
    per_speaker_activity_confidence: dict[int, float] = field(default_factory=dict)


@dataclass
class SpeakerDirectionState:
    speaker_id: int
    direction_deg: float
    confidence: float
    last_update_ms: float
    updates: int = 0
    last_observed_ms: float = 0.0
    hold_count: int = 0
    stale_updates: int = 0
    last_raw_direction_deg: float | None = None
    velocity_deg_per_chunk: float = 0.0
    recent_direction_history_deg: tuple[float, ...] = ()
    predicted_direction_deg: float | None = None
    identity_confidence: float = 0.0
    identity_maturity: str = "unknown"
    activity_confidence: float = 0.0
    last_separator_stream_index: int | None = None
    large_deviation_count: int = 0
    pending_large_deviation_deg: float | None = None
    anchor_direction_deg: float | None = None
    anchor_confidence: float = 0.0
    anchor_last_confirmed_ms: float = 0.0
    anchor_observation_count: int = 0
    anchor_locked: bool = False
    anchor_recent_observations_deg: tuple[float, ...] = ()
    anchor_relock_candidate_deg: float | None = None
    anchor_relock_count: int = 0


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

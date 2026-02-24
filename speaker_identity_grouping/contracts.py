from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(slots=True)
class IdentityConfig:
    sample_rate_hz: int = 16000
    chunk_duration_ms: int = 200
    vad_rms_threshold: float = 0.01
    match_threshold: float = 0.82
    ema_alpha: float = 0.1
    max_speakers: int = 8
    retire_after_chunks: int = 25
    new_speaker_confidence: float = 0.5

    # Embedding controls
    n_mfcc: int = 13
    n_mels: int = 26
    n_fft: int = 512
    frame_length_ms: float = 25.0
    frame_hop_ms: float = 10.0
    preemphasis: float = 0.97
    eps: float = 1e-8


@dataclass(slots=True)
class StreamObservation:
    stream_index: int
    rms: float
    embedding: np.ndarray | None
    active: bool


@dataclass(slots=True)
class SpeakerState:
    speaker_id: int
    centroid: np.ndarray
    sample_count: int
    last_seen_chunk: int
    last_seen_timestamp_ms: float


@dataclass(slots=True)
class IdentityChunkInput:
    chunk_id: int
    timestamp_ms: float
    sample_rate_hz: int
    streams: list[np.ndarray]


@dataclass(slots=True)
class IdentityChunkOutput:
    chunk_id: int
    timestamp_ms: float
    stream_to_speaker: dict[int, int | None]
    active_speakers: list[int]
    new_speakers: list[int]
    retired_speakers: list[int]
    per_stream_confidence: dict[int, float]
    debug: dict[str, object] = field(default_factory=dict)

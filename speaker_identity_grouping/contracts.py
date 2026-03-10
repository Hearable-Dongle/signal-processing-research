from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(slots=True)
class IdentityConfig:
    sample_rate_hz: int = 16000
    chunk_duration_ms: int = 200
    backend: str = "mfcc_legacy"
    vad_rms_threshold: float = 0.01
    match_threshold: float = 0.82
    ema_alpha: float = 0.1
    max_speakers: int = 8
    retire_after_chunks: int = 25
    new_speaker_confidence: float = 0.5
    continuity_bonus: float = 0.04
    switch_penalty: float = 0.06
    hold_similarity_threshold: float = 0.6
    carry_forward_chunks: int = 1
    confidence_decay: float = 0.92

    # Embedding controls
    n_mfcc: int = 13
    n_mels: int = 26
    n_fft: int = 512
    frame_length_ms: float = 25.0
    frame_hop_ms: float = 10.0
    preemphasis: float = 0.97
    eps: float = 1e-8
    # Session-local voiceprint lookup controls
    speaker_embedding_model: str = "wavlm_base_plus_sv"
    speaker_embedding_device: str = "cpu"
    speaker_embedding_min_speech_ms: float = 600.0
    speaker_embedding_buffer_ms: float = 1000.0
    speaker_embedding_update_interval_chunks: int = 2
    speaker_embedding_match_threshold: float = 0.72
    speaker_embedding_merge_threshold: float = 0.82
    speaker_embedding_margin: float = 0.05
    provisional_speaker_timeout_chunks: int = 6


@dataclass(slots=True)
class StreamObservation:
    stream_index: int
    rms: float
    embedding: np.ndarray | None
    active: bool
    voiceprint: np.ndarray | None = None


@dataclass(slots=True)
class SpeakerState:
    speaker_id: int
    centroid: np.ndarray
    sample_count: int
    last_seen_chunk: int
    last_seen_timestamp_ms: float
    last_confidence: float = 0.0
    hold_count: int = 0
    voiceprint: np.ndarray | None = None
    voiceprint_updates: int = 0
    speech_support_ms: float = 0.0
    provisional: bool = True
    last_stream_index: int | None = None
    last_voiceprint_chunk: int = -1


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

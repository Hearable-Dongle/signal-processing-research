from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SpeakerGainDirection:
    speaker_id: int
    direction_degrees: float
    gain_weight: float
    confidence: float
    updated_at_ms: float


@dataclass(frozen=True, slots=True)
class SRPPeakSnapshot:
    timestamp_ms: float
    peaks_deg: tuple[float, ...]
    peak_scores: tuple[float, ...] | None = None


@dataclass(slots=True)
class PipelineConfig:
    sample_rate_hz: int = 16000
    fast_frame_ms: int = 10
    slow_chunk_ms: int = 200
    srp_window_ms: int = 40
    srp_nfft: int = 512
    srp_overlap: float = 0.5
    srp_freq_min_hz: int = 200
    srp_freq_max_hz: int = 3000
    srp_max_sources: int = 8
    slow_queue_max_frames: int = 256
    sound_speed_m_s: float = 343.0
    process_partial_chunk: bool = True
    max_speakers_hint: int = 8
    prefer_multispeaker_module: bool = True
    multispeaker_model_dir: str = "multispeaker_separation/models"
    multispeaker_backend: str = "pytorch"
    # Asteroid backend defaults
    convtasnet_model_name: str = "JorisCos/ConvTasNet_Libri2Mix_sepnoisy_16k"
    convtasnet_device: str = "cpu"

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SpeakerGainDirection:
    speaker_id: int
    direction_degrees: float
    gain_weight: float
    confidence: float
    active: bool
    activity_confidence: float
    updated_at_ms: float


@dataclass(frozen=True, slots=True)
class SRPPeakSnapshot:
    timestamp_ms: float
    peaks_deg: tuple[float, ...]
    peak_scores: tuple[float, ...] | None = None
    raw_peaks_deg: tuple[float, ...] = ()
    raw_peak_scores: tuple[float, ...] | None = None


@dataclass(frozen=True, slots=True)
class FocusControlSnapshot:
    focused_speaker_ids: tuple[int, ...] | None = None
    focused_direction_deg: float | None = None
    user_boost_db: float = 0.0


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
    srp_prior_enabled: bool = True
    srp_peak_min_score: float = 0.05
    srp_peak_ema_alpha: float = 0.35
    srp_peak_hysteresis_margin: float = 0.08
    srp_peak_match_tolerance_deg: float = 20.0
    srp_peak_hold_frames: int = 8
    srp_peak_max_step_deg: float = 12.0
    srp_peak_score_decay: float = 0.9
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
    direction_focus_gain_db: float = 0.0
    direction_non_focus_attenuation_db: float = -14.0
    max_user_boost_db: float = 12.0
    identity_continuity_bonus: float = 0.12
    identity_switch_penalty: float = 0.2
    identity_hold_similarity_threshold: float = 0.45
    identity_carry_forward_chunks: int = 3
    identity_confidence_decay: float = 0.85
    direction_transition_penalty_deg: float = 35.0
    direction_min_confidence_for_switch: float = 0.55
    direction_hold_confidence_decay: float = 0.9
    direction_stale_confidence_decay: float = 0.96
    direction_min_persist_confidence: float = 0.05
    speaker_map_min_confidence_for_refresh: float = 0.2
    speaker_map_hold_ms: float = 800.0
    speaker_map_confidence_decay: float = 0.9
    speaker_map_activity_decay: float = 0.92
    # Beamformer mode
    beamforming_mode: str = "mvdr_fd"  # one of: mvdr_fd, gsc_fd, delay_sum
    # DOA/gain smoothing to reduce steering chatter
    doa_ema_alpha: float = 0.2
    gain_ema_alpha: float = 0.2
    doa_max_step_deg_per_frame: float = 10.0
    # Frequency-domain covariance smoothing
    fd_cov_ema_alpha: float = 0.08
    fd_diag_load: float = 1e-3
    fd_speech_cov_update_scale: float = 0.25
    # Optional postfilter (mild, speech-preserving)
    postfilter_enabled: bool = True
    postfilter_noise_ema_alpha: float = 0.08
    postfilter_speech_ema_alpha: float = 0.12
    postfilter_gain_floor: float = 0.18
    postfilter_gain_ema_alpha: float = 0.2
    # Fast-path safety
    output_soft_clip_enabled: bool = True
    output_soft_clip_drive: float = 1.2
    output_target_rms: float | None = 0.08
    output_rms_ema_alpha: float = 0.2
    output_rms_max_gain_db: float = 6.0
    output_normalization_enabled: bool = True
    output_allow_amplification: bool = False

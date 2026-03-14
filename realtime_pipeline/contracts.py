from __future__ import annotations

from dataclasses import dataclass

from .tracking_modes import SUPPORTED_TRACKING_MODE


@dataclass(frozen=True, slots=True)
class SpeakerGainDirection:
    speaker_id: int
    direction_degrees: float
    gain_weight: float
    confidence: float
    active: bool
    activity_confidence: float
    updated_at_ms: float
    identity_confidence: float = 0.0
    identity_maturity: str = "unknown"
    predicted_direction_deg: float | None = None
    angular_velocity_deg_per_chunk: float = 0.0
    last_separator_stream_index: int | None = None
    anchor_direction_deg: float | None = None
    anchor_confidence: float = 0.0
    anchor_locked: bool = False
    anchor_last_confirmed_ms: float = 0.0


@dataclass(frozen=True, slots=True)
class SRPPeakSnapshot:
    timestamp_ms: float
    peaks_deg: tuple[float, ...]
    peak_scores: tuple[float, ...] | None = None
    raw_peaks_deg: tuple[float, ...] = ()
    raw_peak_scores: tuple[float, ...] | None = None
    debug: dict | None = None


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
    slow_chunk_hop_ms: int | None = None
    slow_path_enabled: bool = True
    fast_path_reference_mode: str = "speaker_map"  # one of: speaker_map, srp_peak
    localization_backend: str = "srp_phat_localization"  # one of: srp_phat_legacy, srp_phat_localization, srp_phat_mvdr_refine, capon_1src, capon_multisrc, capon_mvdr_refine_1src, music_1src
    tracking_mode: str = SUPPORTED_TRACKING_MODE
    control_mode: str = "spatial_peak_mode"  # one of: spatial_peak_mode, speaker_tracking_mode
    localization_window_ms: int = 160
    localization_hop_ms: int = 50
    localization_grid_size: int = 72
    localization_min_peak_separation_deg: float = 15.0
    localization_min_relative_peak_score: float = 0.28
    localization_min_peak_contrast: float = 0.08
    localization_small_aperture_bias: bool = True
    localization_fusion_enabled: bool = True
    localization_model_path: str | None = None
    localization_use_hailo: bool = False
    localization_max_tracks: int = 3
    localization_max_assoc_distance_deg: float = 20.0
    localization_track_hold_frames: int = 5
    localization_track_kill_frames: int = 9
    localization_new_track_min_confidence: float = 0.42
    localization_track_confidence_decay: float = 0.88
    localization_velocity_alpha: float = 0.35
    localization_angle_alpha: float = 0.30
    dominant_lock_acquire_min_score: float = 0.45
    dominant_lock_acquire_confirm_frames: int = 2
    dominant_lock_stay_radius_deg: float = 12.0
    dominant_lock_update_alpha: float = 0.15
    dominant_lock_max_step_deg: float = 6.0
    dominant_lock_hold_missing_frames: int = 20
    dominant_lock_unlock_after_missing_frames: int = 80
    dominant_lock_challenger_min_score: float = 0.55
    dominant_lock_challenger_margin: float = 0.08
    dominant_lock_challenger_consistency_deg: float = 10.0
    dominant_lock_switch_confirm_frames: int = 4
    dominant_lock_switch_min_confidence: float = 0.60
    single_source_mode_enabled: bool = False
    single_source_window_ms: int = 80
    single_source_hop_ms: int = 20
    single_source_freq_min_hz: int = 300
    single_source_freq_max_hz: int = 3000
    single_source_grid_size: int = 72
    single_source_motion_filter_enabled: bool = True
    srp_window_ms: int = 40
    srp_nfft: int = 512
    srp_overlap: float = 0.5
    srp_freq_min_hz: int = 200
    srp_freq_max_hz: int = 3000
    localization_pair_selection_mode: str = "all"
    localization_vad_enabled: bool = True
    capon_spectrum_ema_alpha: float = 0.78
    capon_peak_min_sharpness: float = 0.12
    capon_peak_min_margin: float = 0.04
    capon_hold_frames: int = 2
    speaker_match_window_deg: float = 25.0
    centroid_association_mode: str = "hard_window"  # one of: hard_window, gaussian
    centroid_association_sigma_deg: float = 10.0
    centroid_association_min_score: float = 0.15
    srp_max_sources: int = 8
    srp_prior_enabled: bool = True
    srp_peak_min_score: float = 0.05
    srp_peak_ema_alpha: float = 0.35
    srp_peak_hysteresis_margin: float = 0.08
    srp_peak_match_tolerance_deg: float = 12.0
    srp_peak_hold_frames: int = 4
    srp_peak_max_step_deg: float = 12.0
    srp_peak_score_decay: float = 0.9
    slow_queue_max_frames: int = 256
    sound_speed_m_s: float = 343.0
    process_partial_chunk: bool = True
    max_speakers_hint: int = 8
    assume_single_speaker: bool = False
    prefer_multispeaker_module: bool = True
    multispeaker_model_dir: str = "multispeaker_separation/models"
    multispeaker_backend: str = "pytorch"
    # Asteroid backend defaults
    convtasnet_model_name: str = "JorisCos/ConvTasNet_Libri2Mix_sepnoisy_16k"
    convtasnet_device: str = "cpu"
    convtasnet_model_sample_rate_hz: int = 16000
    convtasnet_input_sample_rate_hz: int = 16000
    convtasnet_resample_mode: str = "polyphase"
    convtasnet_expected_num_sources: int | None = None
    direction_focus_gain_db: float = 0.0
    direction_non_focus_attenuation_db: float = -14.0
    max_user_boost_db: float = 12.0
    identity_continuity_bonus: float = 0.04
    identity_switch_penalty: float = 0.06
    identity_hold_similarity_threshold: float = 0.6
    identity_carry_forward_chunks: int = 1
    identity_confidence_decay: float = 0.92
    identity_retire_after_chunks: int = 25
    identity_speech_likelihood_threshold: float = 0.45
    identity_match_weight: float = 0.7
    identity_direction_match_weight: float = 0.3
    identity_combined_match_threshold: float = 0.58
    identity_new_speaker_max_existing_score: float = 0.32
    identity_direction_match_max_distance_deg: float = 35.0
    identity_direction_mismatch_block_deg: float = 60.0
    identity_direction_gate_confidence: float = 0.3
    identity_backend: str = "mfcc_legacy"
    identity_speaker_embedding_model: str = "wavlm_base_plus_sv"
    identity_speaker_embedding_device: str = "cpu"
    identity_speaker_embedding_min_speech_ms: float = 600.0
    identity_speaker_embedding_buffer_ms: float = 1000.0
    identity_speaker_embedding_update_interval_chunks: int = 2
    identity_speaker_embedding_match_threshold: float = 0.72
    identity_speaker_embedding_merge_threshold: float = 0.82
    identity_speaker_embedding_margin: float = 0.05
    identity_provisional_speaker_timeout_chunks: int = 6
    direction_transition_penalty_deg: float = 22.0
    direction_min_confidence_for_switch: float = 0.35
    direction_hold_confidence_decay: float = 0.9
    direction_stale_confidence_decay: float = 0.96
    direction_min_persist_confidence: float = 0.05
    direction_small_change_deg: float = 12.0
    direction_medium_change_deg: float = 30.0
    direction_large_change_persist_chunks: int = 3
    direction_identity_hold_margin: float = 0.08
    direction_stable_confidence_threshold: float = 0.55
    direction_history_window_chunks: int = 4
    direction_long_memory_enabled: bool = True
    direction_long_memory_window_ms: float = 60000.0
    direction_long_memory_min_observations: int = 4
    direction_long_memory_anchor_lock_confidence: float = 0.6
    direction_long_memory_max_anchor_spread_deg: float = 20.0
    direction_long_memory_soft_prior_margin_deg: float = 18.0
    direction_long_memory_relock_persist_chunks: int = 4
    direction_long_memory_decay: float = 0.995
    direction_long_memory_stale_timeout_ms: float = 60000.0
    direction_speaker_stale_timeout_ms: float = 2000.0
    direction_speaker_forget_timeout_ms: float = 8000.0
    speaker_map_min_confidence_for_refresh: float = 0.2
    speaker_map_hold_ms: float = 300.0
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
    postfilter_gain_floor: float = 0.22
    postfilter_gain_ema_alpha: float = 0.2
    # Fast-path safety
    output_soft_clip_enabled: bool = True
    output_soft_clip_drive: float = 1.2
    output_target_rms: float | None = 0.08
    output_rms_ema_alpha: float = 0.2
    output_rms_max_gain_db: float = 6.0
    output_normalization_enabled: bool = True
    output_allow_amplification: bool = False

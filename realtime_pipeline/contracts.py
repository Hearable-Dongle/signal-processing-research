from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

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


@dataclass(frozen=True, slots=True)
class NoiseModelUpdateSnapshot:
    timestamp_ms: float = 0.0
    active: bool = False
    sources: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()
    debug: dict | None = None


@dataclass(slots=True)
class FastPathAudioPacket:
    frame_index: int
    start_sample: int
    end_sample: int
    sample_rate_hz: int
    frame_samples: int
    beamformed_audio: np.ndarray
    postfilter_wiener_audio: np.ndarray | None = None
    postfilter_rnnoise_audio: np.ndarray | None = None
    postfilter_inverse_rnnoise_audio: np.ndarray | None = None
    postfilter_bandpass_audio: np.ndarray | None = None
    postfilter_output_audio: np.ndarray | None = None
    beamformer_output_noise_psd: np.ndarray | None = None
    frame_mc: np.ndarray | None = None
    target_doa_deg: float | None = None
    target_activity_score: float = 0.0
    target_activity_state: bool = False
    speech_activity: float = 0.0
    beamforming_mode: str = "delay_sum"
    postfilter_method: str = "rnnoise"
    capture_t_monotonic: float = 0.0
    beamform_start_t: float = 0.0
    beamform_end_t: float = 0.0
    postfilter_start_t: float = 0.0
    postfilter_end_t: float = 0.0
    queue_enqueue_t: float | None = None
    queue_wait_ms: float = 0.0
    weights_reused: bool = False
    queue_overflow_dropped: bool = False
    target_doa_missing: bool = False
    noise_model_update_active: bool = False
    noise_model_update_sources: tuple[str, ...] = ()
    noise_model_update_reasons: tuple[str, ...] = ()
    noise_model_update_debug: dict | None = None

    @classmethod
    def create(
        cls,
        *,
        frame_index: int,
        start_sample: int,
        end_sample: int,
        sample_rate_hz: int,
        frame_samples: int,
        beamformed_audio: np.ndarray,
        beamformer_output_noise_psd: np.ndarray | None,
        frame_mc: np.ndarray | None,
        target_doa_deg: float | None,
        target_activity_score: float,
        target_activity_state: bool,
        speech_activity: float,
        beamforming_mode: str,
        postfilter_method: str,
        beamform_start_t: float,
        beamform_end_t: float,
        capture_t_monotonic: float | None = None,
        weights_reused: bool = False,
        noise_model_update_active: bool = False,
        noise_model_update_sources: tuple[str, ...] = (),
        noise_model_update_reasons: tuple[str, ...] = (),
        noise_model_update_debug: dict | None = None,
    ) -> "FastPathAudioPacket":
        capture_ts = perf_counter() if capture_t_monotonic is None else float(capture_t_monotonic)
        return cls(
            frame_index=int(frame_index),
            start_sample=int(start_sample),
            end_sample=int(end_sample),
            sample_rate_hz=int(sample_rate_hz),
            frame_samples=int(frame_samples),
            beamformed_audio=np.asarray(beamformed_audio, dtype=np.float32).reshape(-1).copy(),
            postfilter_wiener_audio=None,
            postfilter_rnnoise_audio=None,
            postfilter_inverse_rnnoise_audio=None,
            postfilter_bandpass_audio=None,
            postfilter_output_audio=None,
            beamformer_output_noise_psd=(
                None
                if beamformer_output_noise_psd is None
                else np.asarray(beamformer_output_noise_psd, dtype=np.float32).reshape(-1).copy()
            ),
            frame_mc=None if frame_mc is None else np.asarray(frame_mc, dtype=np.float32).copy(),
            target_doa_deg=None if target_doa_deg is None else float(target_doa_deg),
            target_activity_score=float(target_activity_score),
            target_activity_state=bool(target_activity_state),
            speech_activity=float(speech_activity),
            beamforming_mode=str(beamforming_mode),
            postfilter_method=str(postfilter_method),
            capture_t_monotonic=float(capture_ts),
            beamform_start_t=float(beamform_start_t),
            beamform_end_t=float(beamform_end_t),
            weights_reused=bool(weights_reused),
            target_doa_missing=bool(target_doa_deg is None),
            noise_model_update_active=bool(noise_model_update_active),
            noise_model_update_sources=tuple(str(v) for v in noise_model_update_sources),
            noise_model_update_reasons=tuple(str(v) for v in noise_model_update_reasons),
            noise_model_update_debug=None if noise_model_update_debug is None else dict(noise_model_update_debug),
        )


@dataclass(slots=True)
class PipelineConfig:
    sample_rate_hz: int = 16000
    input_sample_rate_hz: int = 16000
    input_downsample_rate_hz: int | None = None
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
    capon_freq_bin_subsample_stride: int = 1
    capon_freq_bin_min_hz: int | None = None
    capon_freq_bin_max_hz: int | None = None
    capon_use_cholesky_solve: bool = False
    capon_covariance_ema_alpha: float = 0.0
    capon_full_scan_every_n_updates: int = 1
    capon_local_refine_enabled: bool = False
    capon_local_refine_half_width_deg: float = 30.0
    own_voice_suppression_mode: str = "off"  # one of: off, lcmv_null_hysteresis, soft_output_gate
    suppressed_user_voice_doa_deg: float | None = None
    suppressed_user_match_window_deg: float = 33.0
    suppressed_user_null_on_frames: int = 3
    suppressed_user_null_off_frames: int = 30
    suppressed_user_gate_attenuation_db: float = 18.0
    suppressed_user_target_conflict_deg: float = 30.0
    speaker_match_window_deg: float = 25.0
    single_active_min_observation_score: float = 0.65
    focus_direction_match_window_deg: float = 30.0
    focus_target_hold_frames: int = 16
    multi_target_max_speakers: int = 2
    multi_target_hold_frames: int = 12
    multi_target_min_confidence: float = 0.2
    multi_target_min_activity: float = 0.15
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
    split_runtime_mode: str = "monolithic"  # one of: monolithic, pipelined, beamforming_only, postfilter_only
    postfilter_queue_max_frames: int = 4
    postfilter_queue_drop_oldest: bool = False
    sound_speed_m_s: float = 343.0
    process_partial_chunk: bool = True
    max_speakers_hint: int = 8
    assume_single_speaker: bool = False
    single_active: bool = False
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
    beamforming_mode: str = "delay_sum"  # one of: mvdr_fd, gsc_fd, delay_sum, delay_sum_subtractive, delay_sum_subtractive_multi, delay_sum_differential, lcmv_top2_tracked, lcmv_target_band
    mvdr_hop_ms: int | None = None
    # DOA/gain smoothing to reduce steering chatter
    doa_ema_alpha: float = 0.2
    gain_ema_alpha: float = 0.2
    doa_max_step_deg_per_frame: float = 10.0
    delay_sum_update_min_delta_deg: float = 3.0
    delay_sum_crossfade_frames: int = 1
    delay_sum_use_smoothed_doa: bool = True
    delay_sum_subtractive_alpha: float = 0.5
    delay_sum_subtractive_interferer_doa_deg: float | None = None
    delay_sum_subtractive_multi_offset_deg: float = 10.0
    delay_sum_subtractive_use_suppressed_user_doa: bool = True
    delay_sum_subtractive_output_clip_guard: bool = True
    delay_sum_subtractive_silence_guard_enabled: bool = True
    delay_sum_subtractive_silence_guard_ratio_threshold: float = 0.15
    delay_sum_subtractive_silence_guard_target_rms_floor: float = 0.005
    delay_sum_subtractive_spike_guard_enabled: bool = True
    delay_sum_subtractive_spike_guard_rms_ratio_threshold: float = 2.0
    delay_sum_subtractive_spike_guard_peak_ratio_threshold: float = 3.0
    delay_sum_subtractive_spike_guard_sample_jump_threshold: float = 0.15
    delay_sum_subtractive_output_crossfade_enabled: bool = False
    delay_sum_subtractive_output_crossfade_samples: int = 16
    delay_sum_subtractive_declick_enabled: bool = False
    delay_sum_subtractive_declick_alpha: float = 0.9
    delay_sum_subtractive_declick_spike_threshold: float = 0.08
    delay_sum_subtractive_interferer_ema_enabled: bool = False
    delay_sum_subtractive_interferer_ema_alpha: float = 0.7
    delay_sum_subtractive_adaptive_alpha_enabled: bool = False
    delay_sum_subtractive_adaptive_alpha_min: float = 0.2
    delay_sum_subtractive_adaptive_alpha_delta_scale: float = 1.0
    # Frequency-domain covariance smoothing
    fd_analysis_window_ms: float = 20.0
    # Defaults track the sensitivity-tuned Silero preset from
    # `beamforming/benchmark/run_optuna_babble_bootstrap_mvdr.py`
    # (`beamforming/benchmark/_sens_tune_silero/best_params.json`).
    fd_cov_ema_alpha: float = 0.2965906035161345
    fd_diag_load: float = 0.012141307774357374
    fd_trace_diagonal_loading_factor: float = 0.0
    fd_identity_blend_alpha: float = 0.0
    beamformer_rnn_skip_refresh_when_clean: bool = False
    beamformer_rnn_dirty_threshold: float = 0.0
    beamformer_rnn_dirty_eps: float = 1e-8
    beamformer_rnn_dirty_stat: str = "max"  # one of: max, mean
    beamformer_sparse_solve_enabled: bool = False
    beamformer_sparse_solve_stride: int = 1
    beamformer_sparse_solve_min_freq_hz: float = 200.0
    beamformer_sparse_solve_interp: str = "linear_complex"  # one of: linear_complex
    beamformer_weight_reuse_enabled: bool = True
    beamformer_weight_smoothing_alpha: float = 1.0
    beamformer_doa_refresh_tolerance_deg: float = 5.0
    fd_noise_covariance_mode: str = "estimated_target_subtractive"  # one of: estimated_target_subtractive, estimated_target_subtractive_frozen, oracle_non_target_residual
    target_activity_rnn_update_mode: str | None = None  # one of: oracle_target_activity, estimated_target_activity
    target_activity_low_threshold: float = 0.10544774305969414
    target_activity_high_threshold: float = 0.6508335197763335
    target_activity_enter_frames: int = 1
    target_activity_exit_frames: int = 7
    fd_cov_update_scale_target_active: float = 0.4241144063085703
    fd_cov_update_scale_target_inactive: float = 1.2561064512368887
    target_activity_detector_mode: str = "target_blocker_calibrated"
    target_activity_detector_backend: str = "silero_fused"
    target_activity_update_every_n_fast_frames: int = 1
    target_activity_blocker_offset_deg: float = 120.0
    target_activity_bootstrap_only_calibration: bool = True
    target_activity_ratio_floor_db: float = -1.5557320895954578
    target_activity_ratio_active_db: float = 3.1884929640820445
    target_activity_target_rms_floor_scale: float = 1.3476071785753891
    target_activity_blocker_rms_floor_scale: float = 2.008344796225831
    target_activity_speech_weight: float = 0.6051437824379127
    target_activity_ratio_weight: float = 0.26508371615422194
    target_activity_blocker_weight: float = 0.02224542260010827
    target_activity_vad_mode: int = 1
    target_activity_vad_hangover_frames: int = 2
    target_activity_noise_floor_rise_alpha: float = 0.024462802690520202
    target_activity_noise_floor_fall_alpha: float = 0.16301379312525116
    target_activity_noise_floor_margin_scale: float = 2.33081911386449
    target_activity_rms_scale: float = 4.305504476133645
    target_activity_score_exponent: float = 0.15763482134447154
    fd_speech_cov_update_scale: float = 0.25
    # Optional postfilter (mild, speech-preserving)
    postfilter_enabled: bool = True
    postfilter_method: str = "rnnoise"  # one of: off, wiener_dd, log_mmse, rnnoise, coherence_wiener, wiener_then_rnnoise, voice_bandpass, rnnoise_then_voice_bandpass, wiener_then_voice_bandpass
    postfilter_noise_source: str = "tracked_mono"  # one of: tracked_mono, beamformer_rnn_output
    postfilter_input_source: str = "beamformed_mono"  # one of: beamformed_mono, raw_mix_mono
    postfilter_noise_ema_alpha: float = 0.02
    postfilter_speech_ema_alpha: float = 0.01
    postfilter_gain_floor: float = 0.22
    postfilter_gain_ema_alpha: float = 0.2
    postfilter_dd_alpha: float = 0.92
    postfilter_noise_update_speech_scale: float = 0.0
    postfilter_oversubtraction_alpha: float = 1.0
    postfilter_spectral_floor_beta: float = 0.01
    postfilter_freq_smoothing_bins: int = 2
    postfilter_gain_max_step_db: float = 2.5
    rnnoise_wet_mix: float = 0.9
    rnnoise_input_gain_db: float = 0.0
    rnnoise_input_highpass_enabled: bool = True
    rnnoise_input_highpass_cutoff_hz: float = 80.0
    rnnoise_output_highpass_enabled: bool = True
    rnnoise_output_highpass_cutoff_hz: float = 70.0
    rnnoise_output_lowpass_cutoff_hz: float = 7500.0
    rnnoise_output_notch_freq_hz: float = 500.0
    rnnoise_output_notch_q: float = 20.0
    rnnoise_vad_adaptive_blend_enabled: bool = False
    rnnoise_vad_blend_gamma: float = 0.5
    rnnoise_vad_min_speech_preserve: float = 0.15
    rnnoise_vad_max_speech_preserve: float = 0.95
    rnnoise_startup_warmup_enabled: bool = False
    rnnoise_startup_warmup_frames: int = 10
    rnnoise_chunk_crossfade_enabled: bool = False
    rnnoise_chunk_crossfade_samples: int = 16
    rnnoise_declick_enabled: bool = False
    rnnoise_declick_alpha: float = 0.92
    rnnoise_declick_conditional: bool = True
    rnnoise_declick_spike_threshold: float = 0.03
    rnnoise_output_clip_guard_enabled: bool = False
    rnnoise_output_clip_guard_abs_max: float = 0.95
    rnnoise_corruption_guard_enabled: bool = True
    rnnoise_corruption_guard_rms_ratio_threshold: float = 2.0
    rnnoise_corruption_guard_peak_ratio_threshold: float = 3.0
    rnnoise_corruption_guard_mode: str = "hold_previous"  # one of: hold_previous, use_input, mute
    rnnoise_voice_eq_enabled: bool = True
    rnnoise_voice_eq_presence_gain_db: float = 2.5
    rnnoise_voice_eq_presence_center_hz: float = 3000.0
    rnnoise_voice_eq_presence_q: float = 0.9
    rnnoise_voice_eq_lowmid_gain_db: float = 0.0
    rnnoise_voice_eq_lowmid_center_hz: float = 300.0
    rnnoise_voice_eq_lowmid_q: float = 0.8
    rnnoise_residual_highband_enabled: bool = False
    rnnoise_residual_highband_cutoff_hz: float = 3000.0
    rnnoise_residual_highband_gain: float = 0.5
    rnnoise_residual_jump_limit_enabled: bool = False
    rnnoise_residual_jump_limit_band_low_hz: float = 3000.0
    rnnoise_residual_jump_limit_rise_db_per_frame: float = 4.0
    rnnoise_residual_ema_enabled: bool = False
    rnnoise_residual_ema_alpha: float = 0.0
    coherence_wiener_gain_floor: float = 0.12
    coherence_wiener_coherence_exponent: float = 1.5
    coherence_wiener_temporal_alpha: float = 0.65
    # Fast-path safety
    output_soft_clip_enabled: bool = True
    output_soft_clip_drive: float = 1.2
    output_target_rms: float | None = 0.08
    output_rms_ema_alpha: float = 0.2
    output_rms_max_gain_db: float = 6.0
    output_normalization_enabled: bool = True
    output_allow_amplification: bool = False
    robust_target_band_width_deg: float = 10.0
    robust_target_band_conditioning_enabled: bool = False
    robust_target_band_max_freq_hz: float = 0.0
    robust_target_band_condition_limit: float = 1e3
    beamformer_snapshot_frame_indices: tuple[int, ...] = ()

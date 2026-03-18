from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from realtime_pipeline.tracking_modes import SUPPORTED_TRACKING_MODE, validate_tracking_mode

SCHEMA_VERSION = "v1"


class FastPathConfig(BaseModel):
    split_runtime_mode: Literal["monolithic", "pipelined", "beamforming_only", "postfilter_only"] = "monolithic"
    postfilter_queue_max_frames: int = 4
    postfilter_queue_drop_oldest: bool = False
    localization_hop_ms: int = 10
    localization_window_ms: int = 160
    input_downsample_rate_hz: int | None = None
    overlap: float = 0.2
    freq_low_hz: int = 200
    freq_high_hz: int = 3000
    localization_pair_selection_mode: Literal["all", "adjacent_only"] = "all"
    localization_vad_enabled: bool = True
    localization_backend: Literal[
        "srp_phat_legacy",
        "srp_phat_localization",
        "srp_phat_mvdr_refine",
        "capon_1src",
        "capon_multisrc",
        "capon_mvdr_refine_1src",
        "music_1src",
    ] = "srp_phat_localization"
    beamforming_mode: Literal["mvdr_fd", "sd_mvdr_fd", "gsc_fd", "delay_sum", "lcmv_top2_tracked", "lcmv_target_band"] = "mvdr_fd"
    mvdr_hop_ms: int | None = None
    fd_analysis_window_ms: float = 20.0
    # Defaults track the sensitivity-tuned Silero preset from
    # `beamforming/benchmark/run_optuna_babble_bootstrap_mvdr.py`
    # (`beamforming/benchmark/_sens_tune_silero/best_params.json`).
    fd_cov_ema_alpha: float = 0.2965906035161345
    fd_diag_load: float = 0.012141307774357374
    fd_trace_diagonal_loading_factor: float = 0.0
    fd_identity_blend_alpha: float = 0.0
    fd_noise_covariance_mode: Literal["estimated_target_subtractive", "estimated_target_subtractive_frozen", "oracle_non_target_residual"] = "estimated_target_subtractive"
    target_activity_rnn_update_mode: Literal["oracle_target_activity", "estimated_target_activity"] | None = (
        "estimated_target_activity"
    )
    target_activity_low_threshold: float = 0.10544774305969414
    target_activity_high_threshold: float = 0.6508335197763335
    target_activity_enter_frames: int = 1
    target_activity_exit_frames: int = 7
    fd_cov_update_scale_target_active: float = 0.4241144063085703
    fd_cov_update_scale_target_inactive: float = 1.2561064512368887
    target_activity_detector_mode: Literal["target_blocker_calibrated", "localization_peak_confidence"] = "target_blocker_calibrated"
    target_activity_detector_backend: Literal["webrtc_fused", "silero_fused"] = "silero_fused"
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
    assume_single_speaker: bool = False
    capon_spectrum_ema_alpha: float = 0.78
    capon_peak_min_sharpness: float = 0.12
    capon_peak_min_margin: float = 0.04
    capon_hold_frames: int = 2
    enhancement_tier: Literal["custom", "baseline_pi", "classical_plus", "quality_cpu", "quality_heavy"] = "custom"
    output_enhancer_mode: Literal["off", "wiener"] = "off"
    postfilter_method: Literal["off", "wiener_dd", "log_mmse", "rnnoise", "coherence_wiener", "wiener_then_rnnoise", "voice_bandpass", "rnnoise_then_voice_bandpass", "wiener_then_voice_bandpass"] = "off"
    postfilter_enabled: bool = True
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
    rnnoise_wet_mix: float = 0.8
    rnnoise_input_gain_db: float = 0.0
    rnnoise_residual_ema_enabled: bool = False
    rnnoise_residual_ema_alpha: float = 0.0
    coherence_wiener_gain_floor: float = 0.12
    coherence_wiener_coherence_exponent: float = 1.5
    coherence_wiener_temporal_alpha: float = 0.65
    own_voice_suppression_mode: Literal["off", "lcmv_null_hysteresis", "soft_output_gate"] = "off"
    suppressed_user_voice_doa_deg: float | None = None
    suppressed_user_match_window_deg: float = 33.0
    suppressed_user_null_on_frames: int = 3
    suppressed_user_null_off_frames: int = 10
    suppressed_user_gate_attenuation_db: float = 18.0
    suppressed_user_target_conflict_deg: float = 30.0
    focus_direction_match_window_deg: float = 30.0
    focus_target_hold_frames: int = 8
    multi_target_max_speakers: int = 2
    multi_target_hold_frames: int = 12
    multi_target_min_confidence: float = 0.2
    multi_target_min_activity: float = 0.15
    output_normalization_enabled: bool = True
    output_allow_amplification: bool = False
    robust_target_band_width_deg: float = 10.0
    robust_target_band_conditioning_enabled: bool = False
    robust_target_band_max_freq_hz: float = 0.0
    robust_target_band_condition_limit: float = 1e3


class SlowPathConfig(BaseModel):
    enabled: bool = False
    tracking_mode: str = SUPPORTED_TRACKING_MODE
    single_active: bool = False
    speaker_history_size: int = 8
    speaker_activation_min_predictions: int = 3
    speaker_match_window_deg: float = 25.0
    centroid_association_mode: Literal["hard_window", "gaussian"] = "hard_window"
    centroid_association_sigma_deg: float = 10.0
    centroid_association_min_score: float = 0.15
    slow_chunk_ms: int = 300
    long_memory_enabled: bool = False
    long_memory_window_ms: float = 60000.0

    @field_validator("tracking_mode")
    @classmethod
    def _validate_tracking_mode(cls, value: str) -> str:
        return validate_tracking_mode(value)


class SessionStartRequest(BaseModel):
    scene_config_path: str = ""
    input_source: Literal["simulation", "respeaker_live"] = "simulation"
    audio_device_query: str | None = None
    channel_count: int = 4
    sample_rate_hz: int = 48000
    monitor_source: Literal["processed", "raw_mixed"] = "processed"
    mic_array_profile: Literal["respeaker_v3_0457", "respeaker_xvf3800_0650"] = "respeaker_xvf3800_0650"
    channel_map: list[int] | None = None
    background_noise_audio_path: str | None = None
    background_noise_gain: float = 0.15
    use_ground_truth_location: bool = False
    use_ground_truth_speaker_sources: bool = False
    fast_path: FastPathConfig = Field(default_factory=FastPathConfig)
    slow_path: SlowPathConfig = Field(default_factory=SlowPathConfig)
    focus_ratio: float = 2.0
    max_speakers_hint: int = 4
    separation_mode: Literal["auto", "mock", "single_dominant_no_separator"] = "auto"
    convtasnet_model_name: str = "JorisCos/ConvTasNet_Libri2Mix_sepnoisy_16k"
    convtasnet_model_sample_rate_hz: int = 16000
    convtasnet_input_sample_rate_hz: int = 16000
    convtasnet_resample_mode: Literal["polyphase"] = "polyphase"
    convtasnet_expected_num_sources: int | None = None
    identity_backend: Literal["mfcc_legacy", "speaker_embed_session"] = "mfcc_legacy"
    identity_speaker_embedding_model: Literal["ecapa_voxceleb", "wavlm_base_sv", "wavlm_base_plus_sv"] = "wavlm_base_plus_sv"
    processing_mode: Literal[
        "specific_speaker_enhancement",
        "localize_and_beamform",
        "beamform_from_ground_truth",
    ] = "specific_speaker_enhancement"

    @property
    def localization_hop_ms(self) -> int:
        return int(self.fast_path.localization_hop_ms)

    @property
    def split_runtime_mode(self) -> str:
        return str(self.fast_path.split_runtime_mode)

    @property
    def postfilter_queue_max_frames(self) -> int:
        return int(self.fast_path.postfilter_queue_max_frames)

    @property
    def postfilter_queue_drop_oldest(self) -> bool:
        return bool(self.fast_path.postfilter_queue_drop_oldest)

    @property
    def localization_window_ms(self) -> int:
        return int(self.fast_path.localization_window_ms)

    @property
    def input_downsample_rate_hz(self) -> int | None:
        value = self.fast_path.input_downsample_rate_hz
        return None if value is None else int(value)

    @property
    def overlap(self) -> float:
        return float(self.fast_path.overlap)

    @property
    def freq_low_hz(self) -> int:
        return int(self.fast_path.freq_low_hz)

    @property
    def freq_high_hz(self) -> int:
        return int(self.fast_path.freq_high_hz)

    @property
    def localization_pair_selection_mode(self) -> str:
        return str(self.fast_path.localization_pair_selection_mode)

    @property
    def localization_vad_enabled(self) -> bool:
        return bool(self.fast_path.localization_vad_enabled)

    @property
    def localization_backend(self) -> str:
        return str(self.fast_path.localization_backend)

    @property
    def beamforming_mode(self) -> str:
        return str(self.fast_path.beamforming_mode)

    @property
    def fd_analysis_window_ms(self) -> float:
        return float(self.fast_path.fd_analysis_window_ms)

    @property
    def mvdr_hop_ms(self) -> int | None:
        value = self.fast_path.mvdr_hop_ms
        return None if value is None else int(value)

    @property
    def fd_cov_ema_alpha(self) -> float:
        return float(self.fast_path.fd_cov_ema_alpha)

    @property
    def fd_diag_load(self) -> float:
        return float(self.fast_path.fd_diag_load)

    @property
    def fd_trace_diagonal_loading_factor(self) -> float:
        return float(self.fast_path.fd_trace_diagonal_loading_factor)

    @property
    def fd_identity_blend_alpha(self) -> float:
        return float(self.fast_path.fd_identity_blend_alpha)

    @property
    def fd_noise_covariance_mode(self) -> str:
        return str(self.fast_path.fd_noise_covariance_mode)

    @property
    def target_activity_rnn_update_mode(self) -> str | None:
        value = self.fast_path.target_activity_rnn_update_mode
        return None if value is None else str(value)

    @property
    def target_activity_low_threshold(self) -> float:
        return float(self.fast_path.target_activity_low_threshold)

    @property
    def target_activity_high_threshold(self) -> float:
        return float(self.fast_path.target_activity_high_threshold)

    @property
    def target_activity_enter_frames(self) -> int:
        return int(self.fast_path.target_activity_enter_frames)

    @property
    def target_activity_exit_frames(self) -> int:
        return int(self.fast_path.target_activity_exit_frames)

    @property
    def fd_cov_update_scale_target_active(self) -> float:
        return float(self.fast_path.fd_cov_update_scale_target_active)

    @property
    def fd_cov_update_scale_target_inactive(self) -> float:
        return float(self.fast_path.fd_cov_update_scale_target_inactive)

    @property
    def target_activity_detector_mode(self) -> str:
        return str(self.fast_path.target_activity_detector_mode)

    @property
    def target_activity_detector_backend(self) -> str:
        return str(self.fast_path.target_activity_detector_backend)

    @property
    def target_activity_update_every_n_fast_frames(self) -> int:
        return int(self.fast_path.target_activity_update_every_n_fast_frames)

    @property
    def target_activity_blocker_offset_deg(self) -> float:
        return float(self.fast_path.target_activity_blocker_offset_deg)

    @property
    def target_activity_bootstrap_only_calibration(self) -> bool:
        return bool(self.fast_path.target_activity_bootstrap_only_calibration)

    @property
    def target_activity_ratio_floor_db(self) -> float:
        return float(self.fast_path.target_activity_ratio_floor_db)

    @property
    def target_activity_ratio_active_db(self) -> float:
        return float(self.fast_path.target_activity_ratio_active_db)

    @property
    def target_activity_target_rms_floor_scale(self) -> float:
        return float(self.fast_path.target_activity_target_rms_floor_scale)

    @property
    def target_activity_blocker_rms_floor_scale(self) -> float:
        return float(self.fast_path.target_activity_blocker_rms_floor_scale)

    @property
    def target_activity_speech_weight(self) -> float:
        return float(self.fast_path.target_activity_speech_weight)

    @property
    def target_activity_ratio_weight(self) -> float:
        return float(self.fast_path.target_activity_ratio_weight)

    @property
    def target_activity_blocker_weight(self) -> float:
        return float(self.fast_path.target_activity_blocker_weight)

    @property
    def target_activity_vad_mode(self) -> int:
        return int(self.fast_path.target_activity_vad_mode)

    @property
    def target_activity_vad_hangover_frames(self) -> int:
        return int(self.fast_path.target_activity_vad_hangover_frames)

    @property
    def target_activity_noise_floor_rise_alpha(self) -> float:
        return float(self.fast_path.target_activity_noise_floor_rise_alpha)

    @property
    def target_activity_noise_floor_fall_alpha(self) -> float:
        return float(self.fast_path.target_activity_noise_floor_fall_alpha)

    @property
    def target_activity_noise_floor_margin_scale(self) -> float:
        return float(self.fast_path.target_activity_noise_floor_margin_scale)

    @property
    def target_activity_rms_scale(self) -> float:
        return float(self.fast_path.target_activity_rms_scale)

    @property
    def target_activity_score_exponent(self) -> float:
        return float(self.fast_path.target_activity_score_exponent)

    @property
    def assume_single_speaker(self) -> bool:
        return bool(self.fast_path.assume_single_speaker)

    @property
    def capon_spectrum_ema_alpha(self) -> float:
        return float(self.fast_path.capon_spectrum_ema_alpha)

    @property
    def capon_peak_min_sharpness(self) -> float:
        return float(self.fast_path.capon_peak_min_sharpness)

    @property
    def capon_peak_min_margin(self) -> float:
        return float(self.fast_path.capon_peak_min_margin)

    @property
    def capon_hold_frames(self) -> int:
        return int(self.fast_path.capon_hold_frames)

    @property
    def enhancement_tier(self) -> str:
        return str(self.fast_path.enhancement_tier)

    @property
    def output_enhancer_mode(self) -> str:
        return str(self.fast_path.output_enhancer_mode)

    @property
    def postfilter_enabled(self) -> bool:
        return bool(self.fast_path.postfilter_enabled)

    @property
    def postfilter_method(self) -> str:
        return str(self.fast_path.postfilter_method)

    @property
    def postfilter_noise_ema_alpha(self) -> float:
        return float(self.fast_path.postfilter_noise_ema_alpha)

    @property
    def postfilter_speech_ema_alpha(self) -> float:
        return float(self.fast_path.postfilter_speech_ema_alpha)

    @property
    def postfilter_gain_floor(self) -> float:
        return float(self.fast_path.postfilter_gain_floor)

    @property
    def postfilter_gain_ema_alpha(self) -> float:
        return float(self.fast_path.postfilter_gain_ema_alpha)

    @property
    def postfilter_dd_alpha(self) -> float:
        return float(self.fast_path.postfilter_dd_alpha)

    @property
    def postfilter_noise_update_speech_scale(self) -> float:
        return float(self.fast_path.postfilter_noise_update_speech_scale)

    @property
    def postfilter_oversubtraction_alpha(self) -> float:
        return float(self.fast_path.postfilter_oversubtraction_alpha)

    @property
    def postfilter_spectral_floor_beta(self) -> float:
        return float(self.fast_path.postfilter_spectral_floor_beta)

    @property
    def postfilter_freq_smoothing_bins(self) -> int:
        return int(self.fast_path.postfilter_freq_smoothing_bins)

    @property
    def postfilter_gain_max_step_db(self) -> float:
        return float(self.fast_path.postfilter_gain_max_step_db)

    @property
    def rnnoise_wet_mix(self) -> float:
        return float(self.fast_path.rnnoise_wet_mix)

    @property
    def rnnoise_input_gain_db(self) -> float:
        return float(self.fast_path.rnnoise_input_gain_db)

    @property
    def rnnoise_residual_ema_enabled(self) -> bool:
        return bool(self.fast_path.rnnoise_residual_ema_enabled)

    @property
    def rnnoise_residual_ema_alpha(self) -> float:
        return float(self.fast_path.rnnoise_residual_ema_alpha)

    @property
    def coherence_wiener_gain_floor(self) -> float:
        return float(self.fast_path.coherence_wiener_gain_floor)

    @property
    def coherence_wiener_coherence_exponent(self) -> float:
        return float(self.fast_path.coherence_wiener_coherence_exponent)

    @property
    def coherence_wiener_temporal_alpha(self) -> float:
        return float(self.fast_path.coherence_wiener_temporal_alpha)

    @property
    def own_voice_suppression_mode(self) -> str:
        return str(self.fast_path.own_voice_suppression_mode)

    @property
    def suppressed_user_voice_doa_deg(self) -> float | None:
        return None if self.fast_path.suppressed_user_voice_doa_deg is None else float(self.fast_path.suppressed_user_voice_doa_deg)

    @property
    def suppressed_user_match_window_deg(self) -> float:
        return float(self.fast_path.suppressed_user_match_window_deg)

    @property
    def suppressed_user_null_on_frames(self) -> int:
        return int(self.fast_path.suppressed_user_null_on_frames)

    @property
    def suppressed_user_null_off_frames(self) -> int:
        return int(self.fast_path.suppressed_user_null_off_frames)

    @property
    def suppressed_user_gate_attenuation_db(self) -> float:
        return float(self.fast_path.suppressed_user_gate_attenuation_db)

    @property
    def suppressed_user_target_conflict_deg(self) -> float:
        return float(self.fast_path.suppressed_user_target_conflict_deg)

    @property
    def focus_direction_match_window_deg(self) -> float:
        return float(self.fast_path.focus_direction_match_window_deg)

    @property
    def focus_target_hold_frames(self) -> int:
        return int(self.fast_path.focus_target_hold_frames)

    @property
    def multi_target_max_speakers(self) -> int:
        return int(self.fast_path.multi_target_max_speakers)

    @property
    def multi_target_hold_frames(self) -> int:
        return int(self.fast_path.multi_target_hold_frames)

    @property
    def multi_target_min_confidence(self) -> float:
        return float(self.fast_path.multi_target_min_confidence)

    @property
    def multi_target_min_activity(self) -> float:
        return float(self.fast_path.multi_target_min_activity)

    @property
    def output_normalization_enabled(self) -> bool:
        return bool(self.fast_path.output_normalization_enabled)

    @property
    def output_allow_amplification(self) -> bool:
        return bool(self.fast_path.output_allow_amplification)

    @property
    def robust_target_band_width_deg(self) -> float:
        return float(self.fast_path.robust_target_band_width_deg)

    @property
    def robust_target_band_conditioning_enabled(self) -> bool:
        return bool(self.fast_path.robust_target_band_conditioning_enabled)

    @property
    def robust_target_band_max_freq_hz(self) -> float:
        return float(self.fast_path.robust_target_band_max_freq_hz)

    @property
    def robust_target_band_condition_limit(self) -> float:
        return float(self.fast_path.robust_target_band_condition_limit)

    @property
    def tracking_mode(self) -> str:
        return str(self.slow_path.tracking_mode)

    @property
    def slow_path_enabled(self) -> bool:
        return bool(self.slow_path.enabled)

    @property
    def single_active(self) -> bool:
        return bool(self.slow_path.single_active)

    @property
    def speaker_history_size(self) -> int:
        return int(self.slow_path.speaker_history_size)

    @property
    def speaker_activation_min_predictions(self) -> int:
        return int(self.slow_path.speaker_activation_min_predictions)

    @property
    def speaker_match_window_deg(self) -> float:
        return float(self.slow_path.speaker_match_window_deg)

    @property
    def centroid_association_mode(self) -> str:
        return str(self.slow_path.centroid_association_mode)

    @property
    def centroid_association_sigma_deg(self) -> float:
        return float(self.slow_path.centroid_association_sigma_deg)

    @property
    def centroid_association_min_score(self) -> float:
        return float(self.slow_path.centroid_association_min_score)

    @property
    def slow_chunk_ms(self) -> int:
        return int(self.slow_path.slow_chunk_ms)

    @property
    def direction_long_memory_enabled(self) -> bool:
        return bool(self.slow_path.long_memory_enabled)

    @property
    def direction_long_memory_window_ms(self) -> float:
        return float(self.slow_path.long_memory_window_ms)


class SessionStartResponse(BaseModel):
    session_id: str
    status: Literal["starting", "running"]
    config_echo: SessionStartRequest


class SessionStatusResponse(BaseModel):
    session_id: str
    status: Literal["starting", "running", "stopping", "stopped", "error"]
    started_at_ms: float
    uptime_ms: float
    selected_speaker_id: int | None
    speaker_gain_delta_db: dict[int, float]
    last_metrics: dict | None = None


class SessionStopResponse(BaseModel):
    session_id: str
    status: Literal["stopped"]


class RawChannelDescriptor(BaseModel):
    channel_index: int
    filename: str


class RawChannelsResponse(BaseModel):
    session_id: str
    sample_rate_hz: int
    channel_count: int
    channels: list[RawChannelDescriptor]


class SpeakerStateItem(BaseModel):
    speaker_id: int
    direction_degrees: float
    confidence: float
    active: bool
    activity_confidence: float
    gain_weight: float


class GroundTruthSpeakerItem(BaseModel):
    source_id: int
    direction_degrees: float


class SpeakerStateMessage(BaseModel):
    schema_version: Literal["v1"] = SCHEMA_VERSION
    type: Literal["speaker_state"] = "speaker_state"
    timestamp_ms: float
    speakers: list[SpeakerStateItem]
    ground_truth: list[GroundTruthSpeakerItem] = []
    noise_model_update_active: bool | None = None
    noise_model_update_sources: list[str] | None = None


class MetricsMessage(BaseModel):
    schema_version: Literal["v1"] = SCHEMA_VERSION
    type: Literal["metrics"] = "metrics"
    timestamp_ms: float
    fast_rtf: float
    slow_rtf: float
    fast_stage_avg_ms: dict[str, float]
    slow_stage_avg_ms: dict[str, float]
    startup_lock_ms: float = 0.0
    reacquire_catchup_ms_median: float = 0.0
    nearest_change_catchup_ms_median: float = 0.0
    noise_model_update_active: bool | None = None
    noise_model_update_sources: list[str] | None = None


class SessionEventMessage(BaseModel):
    schema_version: Literal["v1"] = SCHEMA_VERSION
    type: Literal["session_event"] = "session_event"
    event: Literal["started", "stopped", "error", "device_connected"]
    detail: str = ""
    timestamp_ms: float


class ErrorMessage(BaseModel):
    schema_version: Literal["v1"] = SCHEMA_VERSION
    type: Literal["error"] = "error"
    error: str
    timestamp_ms: float


class SelectSpeakerMessage(BaseModel):
    schema_version: Literal["v1"] = SCHEMA_VERSION
    type: Literal["select_speaker"] = "select_speaker"
    speaker_id: int


class AdjustSpeakerGainMessage(BaseModel):
    schema_version: Literal["v1"] = SCHEMA_VERSION
    type: Literal["adjust_speaker_gain"] = "adjust_speaker_gain"
    speaker_id: int
    delta_db_step: Literal[-1, 1] = Field(description="Relative adjustment step in dB")


class ClearFocusMessage(BaseModel):
    schema_version: Literal["v1"] = SCHEMA_VERSION
    type: Literal["clear_focus"] = "clear_focus"


class SetMonitorSourceMessage(BaseModel):
    schema_version: Literal["v1"] = SCHEMA_VERSION
    type: Literal["set_monitor_source"] = "set_monitor_source"
    monitor_source: Literal["processed", "raw_mixed"]


class StopSessionMessage(BaseModel):
    schema_version: Literal["v1"] = SCHEMA_VERSION
    type: Literal["stop_session"] = "stop_session"


ClientMessage = (
    SelectSpeakerMessage | AdjustSpeakerGainMessage | ClearFocusMessage | SetMonitorSourceMessage | StopSessionMessage
)

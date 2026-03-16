from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from realtime_pipeline.tracking_modes import SUPPORTED_TRACKING_MODE, validate_tracking_mode

SCHEMA_VERSION = "v1"


class FastPathConfig(BaseModel):
    localization_hop_ms: int = 10
    localization_window_ms: int = 160
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
    beamforming_mode: Literal["mvdr_fd", "sd_mvdr_fd", "gsc_fd", "delay_sum"] = "mvdr_fd"
    fd_analysis_window_ms: float = 20.0
    target_activity_rnn_update_mode: Literal["oracle_target_activity", "estimated_target_activity"] | None = None
    target_activity_low_threshold: float = 0.25
    target_activity_high_threshold: float = 0.45
    target_activity_enter_frames: int = 2
    target_activity_exit_frames: int = 4
    fd_cov_update_scale_target_active: float = 0.0
    fd_cov_update_scale_target_inactive: float = 1.0
    assume_single_speaker: bool = False
    capon_spectrum_ema_alpha: float = 0.78
    capon_peak_min_sharpness: float = 0.12
    capon_peak_min_margin: float = 0.04
    capon_hold_frames: int = 2
    enhancement_tier: Literal["custom", "baseline_pi", "classical_plus", "quality_cpu", "quality_heavy"] = "custom"
    output_enhancer_mode: Literal["off", "wiener"] = "off"
    postfilter_enabled: bool = True
    own_voice_suppression_mode: Literal["off", "lcmv_null_hysteresis", "soft_output_gate"] = "off"
    suppressed_user_voice_doa_deg: float | None = None
    suppressed_user_match_window_deg: float = 33.0
    suppressed_user_null_on_frames: int = 3
    suppressed_user_null_off_frames: int = 10
    suppressed_user_gate_attenuation_db: float = 18.0
    suppressed_user_target_conflict_deg: float = 30.0
    output_normalization_enabled: bool = True
    output_allow_amplification: bool = False


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
    def localization_window_ms(self) -> int:
        return int(self.fast_path.localization_window_ms)

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
    def output_normalization_enabled(self) -> bool:
        return bool(self.fast_path.output_normalization_enabled)

    @property
    def output_allow_amplification(self) -> bool:
        return bool(self.fast_path.output_allow_amplification)

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


class SessionEventMessage(BaseModel):
    schema_version: Literal["v1"] = SCHEMA_VERSION
    type: Literal["session_event"] = "session_event"
    event: Literal["started", "stopped", "error"]
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

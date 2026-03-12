from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

SCHEMA_VERSION = "v1"


class SessionStartRequest(BaseModel):
    algorithm_mode: Literal[
        "localization_only",
        "spatial_baseline",
        "speaker_tracking",
        "speaker_tracking_long_memory",
        "speaker_tracking_single_active",
        "single_dominant_no_separator",
    ] = "single_dominant_no_separator"
    scene_config_path: str = ""
    input_source: Literal["simulation", "respeaker_live"] = "simulation"
    audio_device_query: str | None = None
    channel_count: int = 4
    sample_rate_hz: int = 48000
    monitor_source: Literal["processed", "raw_mixed"] = "processed"
    mic_array_profile: Literal["respeaker_v3_0457", "respeaker_cross_0640"] = "respeaker_v3_0457"
    channel_map: list[int] | None = None
    background_noise_audio_path: str | None = None
    background_noise_gain: float = 0.15
    use_ground_truth_location: bool = False
    use_ground_truth_speaker_sources: bool = False
    localization_hop_ms: int = 10
    localization_window_ms: int = 160
    overlap: float = 0.2
    freq_low_hz: int = 200
    freq_high_hz: int = 3000
    speaker_history_size: int = 8
    speaker_activation_min_predictions: int = 3
    speaker_match_window_deg: float = 30.0
    localization_vad_enabled: bool = True
    localization_vad_rms_floor: float = 5e-4
    localization_vad_speech_ratio_threshold: float = 0.62
    localization_vad_rms_ratio_threshold: float = 1.2
    localization_vad_flux_threshold: float = 0.12
    localization_snr_gating_enabled: bool = True
    localization_snr_threshold_db: float = 3.0
    localization_snr_soft_range_db: float = 12.0
    localization_snr_weight_exponent: float = 1.0
    localization_noise_floor_alpha_fast: float = 0.35
    localization_noise_floor_alpha_slow: float = 0.97
    localization_msc_variance_enabled: bool = True
    localization_msc_history_frames: int = 6
    localization_msc_variance_floor: float = 0.002
    localization_msc_weight_exponent: float = 1.0
    localization_hsda_enabled: bool = True
    localization_hsda_window_frames: int = 5
    focus_ratio: float = 2.0
    slow_chunk_ms: int = 300
    max_speakers_hint: int = 4
    separation_mode: Literal["auto", "mock", "single_dominant_no_separator"] = "auto"
    convtasnet_model_name: str = "JorisCos/ConvTasNet_Libri2Mix_sepnoisy_16k"
    convtasnet_model_sample_rate_hz: int = 16000
    convtasnet_input_sample_rate_hz: int = 16000
    convtasnet_resample_mode: Literal["polyphase"] = "polyphase"
    convtasnet_expected_num_sources: int | None = None
    identity_backend: Literal["mfcc_legacy", "speaker_embed_session"] = "mfcc_legacy"
    identity_speaker_embedding_model: Literal["ecapa_voxceleb", "wavlm_base_sv", "wavlm_base_plus_sv"] = "wavlm_base_plus_sv"
    beamforming_mode: Literal["mvdr_fd", "gsc_fd", "delay_sum"] = "mvdr_fd"
    localization_backend: Literal["srp_phat_legacy", "srp_phat_localization", "music_1src"] = "srp_phat_localization"
    tracking_mode: Literal["legacy", "multi_peak_v2"] = "multi_peak_v2"
    output_normalization_enabled: bool = True
    output_allow_amplification: bool = False
    processing_mode: Literal[
        "specific_speaker_enhancement",
        "localize_and_beamform",
        "beamform_from_ground_truth",
    ] = "specific_speaker_enhancement"


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

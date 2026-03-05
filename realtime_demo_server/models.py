from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

SCHEMA_VERSION = "v1"


class SessionStartRequest(BaseModel):
    scene_config_path: str
    background_noise_audio_path: str | None = None
    background_noise_gain: float = 0.15
    focus_ratio: float = 2.0
    slow_chunk_ms: int = 300
    max_speakers_hint: int = 4
    separation_mode: Literal["auto", "mock"] = "auto"
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


class StopSessionMessage(BaseModel):
    schema_version: Literal["v1"] = SCHEMA_VERSION
    type: Literal["stop_session"] = "stop_session"


ClientMessage = SelectSpeakerMessage | AdjustSpeakerGainMessage | ClearFocusMessage | StopSessionMessage

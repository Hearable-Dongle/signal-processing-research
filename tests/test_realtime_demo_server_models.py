from mic_array_forwarder.models import (
    AdjustSpeakerGainMessage,
    MetricsMessage,
    RawChannelsResponse,
    SCHEMA_VERSION,
    SelectSpeakerMessage,
    SessionStartRequest,
    SpeakerStateItem,
    SpeakerStateMessage,
)


def test_models_roundtrip_and_schema_version() -> None:
    req = SessionStartRequest(scene_config_path="x.json")
    assert req.background_noise_audio_path is None
    assert req.background_noise_gain == 0.15
    assert req.focus_ratio == 2.0
    assert req.separation_mode == "auto"
    assert req.localization_backend == "tiny_dp_ipd"
    assert req.tracking_mode == "multi_peak_v2"
    assert req.processing_mode == "specific_speaker_enhancement"
    assert req.monitor_source == "processed"
    assert req.sample_rate_hz == 48000
    req2 = SessionStartRequest(localization_backend="music_1src")
    assert req2.localization_backend == "music_1src"

    state = SpeakerStateMessage(
        timestamp_ms=1.0,
        speakers=[
            SpeakerStateItem(
                speaker_id=1,
                direction_degrees=12.0,
                confidence=0.9,
                active=True,
                activity_confidence=0.8,
                gain_weight=1.2,
            )
        ],
    )
    payload = state.model_dump()
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["type"] == "speaker_state"

    metrics = MetricsMessage(
        timestamp_ms=2.0,
        fast_rtf=0.6,
        slow_rtf=0.7,
        fast_stage_avg_ms={"srp": 1.0},
        slow_stage_avg_ms={"separation": 2.0},
    )
    assert metrics.schema_version == SCHEMA_VERSION


def test_adjust_message_allows_only_unit_steps() -> None:
    msg = AdjustSpeakerGainMessage(schema_version="v1", type="adjust_speaker_gain", speaker_id=2, delta_db_step=1)
    assert msg.delta_db_step == 1


def test_select_message_schema() -> None:
    msg = SelectSpeakerMessage(schema_version="v1", type="select_speaker", speaker_id=3)
    assert msg.schema_version == "v1"


def test_raw_channels_response_schema() -> None:
    resp = RawChannelsResponse.model_validate(
        {
            "session_id": "abc123",
            "sample_rate_hz": 16000,
            "channel_count": 2,
            "channels": [
                {"channel_index": 0, "filename": "channel_000.wav"},
                {"channel_index": 1, "filename": "channel_001.wav"},
            ],
        }
    )
    assert resp.channel_count == 2
    assert resp.channels[1].filename == "channel_001.wav"

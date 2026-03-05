from realtime_demo_server.models import (
    AdjustSpeakerGainMessage,
    MetricsMessage,
    SCHEMA_VERSION,
    SelectSpeakerMessage,
    SessionStartRequest,
    SpeakerStateItem,
    SpeakerStateMessage,
)


def test_models_roundtrip_and_schema_version() -> None:
    req = SessionStartRequest(scene_config_path="x.json")
    assert req.focus_ratio == 2.0
    assert req.separation_mode == "auto"
    assert req.processing_mode == "specific_speaker_enhancement"

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

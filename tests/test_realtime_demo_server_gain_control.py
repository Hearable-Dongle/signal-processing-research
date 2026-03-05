from realtime_demo_server.models import SessionStartRequest
from realtime_demo_server.session import DemoSession


def test_gain_clamp_bounds() -> None:
    session = DemoSession(
        SessionStartRequest(
            scene_config_path="simulation/simulations/configs/library_scene/library_k1_scene00.json",
            separation_mode="mock",
        )
    )
    for _ in range(30):
        v = session.adjust_speaker_gain(7, 1)
    assert v == 12.0

    for _ in range(40):
        v = session.adjust_speaker_gain(7, -1)
    assert v == -12.0


def test_non_speaker_mode_ignores_speaker_gain_adjustments() -> None:
    session = DemoSession(
        SessionStartRequest(
            scene_config_path="simulation/simulations/configs/library_scene/library_k1_scene00.json",
            separation_mode="mock",
            processing_mode="beamform_from_ground_truth",
        )
    )
    session.select_speaker(7)
    v = session.adjust_speaker_gain(7, 1)
    assert v == 0.0
    assert session.get_status().selected_speaker_id is None
    assert session.get_status().speaker_gain_delta_db == {}

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

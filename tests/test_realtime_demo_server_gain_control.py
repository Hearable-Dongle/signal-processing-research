from realtime_demo_server.models import SessionStartRequest
from realtime_demo_server.session import DemoSession, _inject_background_noise_source
from simulation.simulation_config import SimulationConfig


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


def test_background_noise_source_injection() -> None:
    cfg = SimulationConfig.from_file("simulation/simulations/configs/library_scene/library_k1_scene00.json")
    before = len(cfg.audio.sources)
    _inject_background_noise_source(
        cfg,
        audio_path="wham_noise/tr/01dc0215_0.22439_01fc0207_-0.22439sp12.wav",
        gain=0.2,
    )
    assert len(cfg.audio.sources) == before + 1
    extra = cfg.audio.sources[-1]
    assert extra.classification == "noise"
    assert extra.audio_path.endswith(".wav")

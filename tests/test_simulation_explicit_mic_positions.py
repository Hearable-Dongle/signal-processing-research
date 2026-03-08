import numpy as np

from simulation.mic_array_profiles import mic_positions_xyz
from simulation.simulation_config import MicrophoneArray, Room, SimulationAudio, SimulationConfig, SimulationSource
from simulation.simulator import run_simulation


def test_simulation_supports_explicit_mic_positions() -> None:
    rel_positions = mic_positions_xyz("respeaker_v3_0457")
    cfg = SimulationConfig(
        room=Room(dimensions=[5.0, 4.0, 3.0], absorption=0.4),
        microphone_array=MicrophoneArray(
            mic_center=[2.5, 2.0, 1.5],
            mic_radius=0.02285,
            mic_count=4,
            mic_positions=rel_positions.tolist(),
        ),
        audio=SimulationAudio(
            sources=[
                SimulationSource(
                    loc=[3.5, 2.0, 1.5],
                    audio_path="LibriSpeech/train-clean-100/19/198/19-198-0000.flac",
                    gain=1.0,
                )
            ],
            duration=0.25,
            fs=16000,
        ),
    )

    mic_audio, mic_pos, source_signals = run_simulation(cfg)

    assert mic_audio.shape[1] == 4
    assert mic_pos.shape == (3, 4)
    assert len(source_signals) == 1
    center = np.asarray(cfg.microphone_array.mic_center, dtype=float).reshape(3, 1)
    np.testing.assert_allclose(mic_pos - center, rel_positions.T, atol=1e-6)

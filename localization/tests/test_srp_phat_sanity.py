from __future__ import annotations

from pathlib import Path

import numpy as np

from localization.algo import SRPPHATLocalization
from simulation.simulator import run_simulation
from simulation.create_testing_specific_angles_scene import _room_position
from simulation.mic_array_profiles import mic_positions_xyz
from simulation.simulation_config import MicrophoneArray, Room, SimulationAudio, SimulationConfig, SimulationSource


FS = 16000
ROOM_DIMS = (8.0, 6.4, 3.1)
MIC_CENTER = (4.0, 3.2, 1.45)
SOURCE_AUDIO = Path(
    "simulation/simulations/assets/testing_specific_angles/testing_specific_angles_k2_scene00/render_assets/speaker_0_seg000.wav"
)
NOISE_AUDIO = Path(
    "simulation/simulations/assets/testing_specific_angles/testing_specific_angles_k2_scene00/render_assets/noise_0_seg000.wav"
)


def _build_scene(profile: str, angle_deg: float, *, absorption: float, noise_gain: float) -> SimulationConfig:
    rel_positions = mic_positions_xyz(profile)
    center = np.asarray(MIC_CENTER, dtype=float)
    src_loc = center + np.asarray(
        [np.cos(np.deg2rad(angle_deg)), np.sin(np.deg2rad(angle_deg)), 0.0],
        dtype=float,
    )
    sources = [
        SimulationSource(
            loc=src_loc.tolist(),
            audio_path=str(SOURCE_AUDIO.resolve()),
            gain=1.0,
            classification="speech",
        )
    ]
    if noise_gain > 0.0:
        noise_loc = center + np.asarray([0.0, 1.0, 0.0], dtype=float)
        sources.append(
            SimulationSource(
                loc=noise_loc.tolist(),
                audio_path=str(NOISE_AUDIO.resolve()),
                gain=float(noise_gain),
                classification="noise",
            )
        )
    return SimulationConfig(
        room=Room(dimensions=list(ROOM_DIMS), absorption=float(absorption)),
        microphone_array=MicrophoneArray(
            mic_center=list(MIC_CENTER),
            mic_radius=float(np.max(np.linalg.norm(rel_positions[:, :2], axis=1))),
            mic_count=int(rel_positions.shape[0]),
            mic_positions=rel_positions.tolist(),
        ),
        audio=SimulationAudio(sources=sources, duration=4.0, fs=FS),
    )


def _pred_deg(profile: str, angle_deg: float, *, absorption: float, noise_gain: float) -> float:
    cfg = _build_scene(profile, angle_deg, absorption=absorption, noise_gain=noise_gain)
    mic_audio, mic_pos_abs, _ = run_simulation(cfg)
    center = np.array(cfg.microphone_array.mic_center).reshape(3, 1)
    mic_pos_rel = mic_pos_abs - center
    algo = SRPPHATLocalization(mic_pos=mic_pos_rel, fs=FS, max_sources=1)
    pred_rad, _hist, _history = algo.process(mic_audio.T)
    assert pred_rad
    return float(np.degrees(pred_rad[0]) % 360.0)


def _circular_error(pred_deg: float, truth_deg: float) -> float:
    return float(abs((pred_deg - truth_deg + 180.0) % 360.0 - 180.0))


def test_srp_phat_dry_sanity_keeps_basic_angles() -> None:
    for angle_deg in (0.0, 90.0, 225.0, 270.0):
        pred_deg = _pred_deg("respeaker_v3_0457", angle_deg, absorption=0.99, noise_gain=0.0)
        assert _circular_error(pred_deg, angle_deg) <= 1.0


def test_srp_phat_mild_noise_sanity_matches_debug_winner_behavior() -> None:
    checks = {
        0.0: 20.0,
        225.0: 15.0,
        270.0: 5.0,
    }
    for angle_deg, max_err in checks.items():
        pred_deg = _pred_deg("respeaker_v3_0457", angle_deg, absorption=0.99, noise_gain=0.2)
        assert _circular_error(pred_deg, angle_deg) <= max_err

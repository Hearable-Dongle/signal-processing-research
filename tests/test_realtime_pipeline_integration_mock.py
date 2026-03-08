from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from realtime_pipeline.contracts import PipelineConfig
from realtime_pipeline.orchestrator import RealtimeSpeakerPipeline
from realtime_pipeline.separation_backends import MockSeparationBackend
from realtime_pipeline.simulation_runner import run_simulation_pipeline
from simulation.simulation_config import SimulationConfig
from simulation.simulator import run_simulation


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_realtime_pipeline_simulation_mock_end_to_end(tmp_path: Path) -> None:
    pytest.importorskip("pyroomacoustics")

    repo = Path(__file__).resolve().parents[1]
    input_dir = repo / "beamforming" / "input"

    scene = {
        "room": {"dimensions": [5.0, 4.0, 3.0], "absorption": 0.25},
        "microphone_array": {"mic_center": [2.5, 2.0, 1.5], "mic_radius": 0.05, "mic_count": 4},
        "audio": {
            "sources": [
                {
                    "loc": [1.5, 1.5, 1.5],
                    "audio": str((input_dir / "brit_talking.wav").resolve()),
                    "gain": 1.0,
                    "classification": "signal",
                },
                {
                    "loc": [3.7, 2.8, 1.5],
                    "audio": str((input_dir / "matthew_talking.wav").resolve()),
                    "gain": 1.0,
                    "classification": "signal",
                },
                {
                    "loc": [2.0, 3.3, 1.5],
                    "audio": str((input_dir / "babble_10dB.wav").resolve()),
                    "gain": 0.1,
                    "classification": "noise",
                },
            ],
            "duration": 0.6,
            "fs": 16000,
        },
    }

    scene_path = tmp_path / "scene.json"
    scene_path.write_text(json.dumps(scene), encoding="utf-8")

    out_dir = tmp_path / "out"
    summary = run_simulation_pipeline(scene_config_path=scene_path, out_dir=out_dir, use_mock_separation=True)

    assert summary["fast_frames"] > 0
    assert summary["slow_chunks"] > 0
    assert summary["speaker_map_updates"] > 0
    assert summary["localization_backend"] == "tiny_dp_ipd"
    assert summary["tracking_mode"] == "multi_peak_v2"
    assert (out_dir / "summary.json").exists()
    assert (out_dir / "enhanced_fast_path.wav").exists()


def test_realtime_pipeline_focus_boost_increases_output_rms(tmp_path: Path) -> None:
    pytest.importorskip("pyroomacoustics")

    repo = Path(__file__).resolve().parents[1]
    input_dir = repo / "beamforming" / "input"

    scene = {
        "room": {"dimensions": [5.0, 4.0, 3.0], "absorption": 0.25},
        "microphone_array": {"mic_center": [2.5, 2.0, 1.5], "mic_radius": 0.05, "mic_count": 4},
        "audio": {
            "sources": [
                {
                    "loc": [1.4, 1.4, 1.5],
                    "audio": str((input_dir / "brit_talking.wav").resolve()),
                    "gain": 1.0,
                    "classification": "signal",
                },
                {
                    "loc": [3.8, 2.9, 1.5],
                    "audio": str((input_dir / "matthew_talking.wav").resolve()),
                    "gain": 1.0,
                    "classification": "signal",
                },
            ],
            "duration": 1.0,
            "fs": 16000,
        },
    }
    scene_path = tmp_path / "scene_focus.json"
    scene_path.write_text(json.dumps(scene), encoding="utf-8")

    sim_cfg = SimulationConfig.from_file(scene_path)
    mic_audio, mic_pos, _ = run_simulation(sim_cfg)
    frame_samples = int(sim_cfg.audio.fs * 10 / 1000)

    def frame_iter():
        total = mic_audio.shape[0]
        for start in range(0, total, frame_samples):
            frame = mic_audio[start : start + frame_samples, :]
            if frame.shape[0] < frame_samples:
                frame = np.pad(frame, ((0, frame_samples - frame.shape[0]), (0, 0)))
            yield frame.astype(np.float32, copy=False)

    def run_once(boost_db: float) -> tuple[float, dict]:
        out_parts: list[np.ndarray] = []

        def sink(x: np.ndarray) -> None:
            out_parts.append(np.asarray(x, dtype=np.float32).reshape(-1))

        cfg = PipelineConfig(sample_rate_hz=sim_cfg.audio.fs, fast_frame_ms=10, slow_chunk_ms=200, max_speakers_hint=2)
        pipe = RealtimeSpeakerPipeline(
            config=cfg,
            mic_geometry_xyz=np.asarray(mic_pos, dtype=float),
            mic_geometry_xy=np.asarray(mic_pos, dtype=float)[:2, :].T,
            frame_iterator=frame_iter(),
            frame_sink=sink,
            separation_backend=MockSeparationBackend(n_streams=2),
        )
        pipe.set_focus_control(focused_speaker_ids=[0], user_boost_db=boost_db)
        pipe.run_blocking()
        y = np.concatenate(out_parts) if out_parts else np.zeros(1, dtype=np.float32)
        rms = float(np.sqrt(np.mean(y.astype(np.float64) ** 2)))
        smap = dict(pipe.shared_state.get_speaker_map_snapshot())
        return rms, smap

    rms_base, smap_base = run_once(boost_db=0.0)
    rms_boost, smap_boost = run_once(boost_db=10.0)
    assert rms_boost > rms_base
    if 0 in smap_base and 0 in smap_boost:
        assert smap_boost[0].gain_weight > smap_base[0].gain_weight


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize("method", ["mvdr_fd", "gsc_fd", "delay_sum"])
def test_realtime_pipeline_methods_execute(tmp_path: Path, method: str) -> None:
    pytest.importorskip("pyroomacoustics")

    repo = Path(__file__).resolve().parents[1]
    input_dir = repo / "beamforming" / "input"
    scene = {
        "room": {"dimensions": [5.0, 4.0, 3.0], "absorption": 0.25},
        "microphone_array": {"mic_center": [2.5, 2.0, 1.5], "mic_radius": 0.05, "mic_count": 4},
        "audio": {
            "sources": [
                {
                    "loc": [1.5, 1.5, 1.5],
                    "audio": str((input_dir / "brit_talking.wav").resolve()),
                    "gain": 1.0,
                    "classification": "signal",
                },
                {
                    "loc": [3.5, 2.8, 1.5],
                    "audio": str((input_dir / "babble_10dB.wav").resolve()),
                    "gain": 0.25,
                    "classification": "noise",
                },
            ],
            "duration": 0.5,
            "fs": 16000,
        },
    }
    scene_path = tmp_path / f"scene_{method}.json"
    scene_path.write_text(json.dumps(scene), encoding="utf-8")
    out_dir = tmp_path / f"out_{method}"
    summary = run_simulation_pipeline(
        scene_config_path=scene_path,
        out_dir=out_dir,
        use_mock_separation=True,
        beamforming_mode=method,
        output_normalization_enabled=True,
        output_allow_amplification=False,
    )
    assert summary["fast_frames"] > 0
    assert summary["slow_chunks"] > 0
    assert summary["beamforming_mode"] == method
    assert (out_dir / "enhanced_fast_path.wav").exists()

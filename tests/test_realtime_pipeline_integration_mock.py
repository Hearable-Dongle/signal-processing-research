from __future__ import annotations

import json
from pathlib import Path

import pytest

from realtime_pipeline.simulation_runner import run_simulation_pipeline


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
    assert (out_dir / "summary.json").exists()
    assert (out_dir / "enhanced_fast_path.wav").exists()

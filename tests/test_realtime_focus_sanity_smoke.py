from __future__ import annotations

import json
from pathlib import Path

import pytest

from realtime_pipeline.focus_sanity_check import run_focus_sanity_check


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_focus_sanity_smoke_single_scene(tmp_path: Path) -> None:
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
            ],
            "duration": 0.8,
            "fs": 16000,
        },
    }
    scene_path = tmp_path / "scene.json"
    scene_path.write_text(json.dumps(scene), encoding="utf-8")

    out_dir = tmp_path / "out"
    summary = run_focus_sanity_check(
        out_dir=out_dir,
        scene_paths=[scene_path],
        focus_ratio=2.0,
        use_mock_separation=True,
    )

    assert summary["num_scenes"] == 1
    assert summary["num_runs"] == 1
    assert (out_dir / "summary.json").exists()
    assert (out_dir / "per_scene_metrics.json").exists()
    scene_result = summary["scenes"][0]
    run_dir = out_dir / scene_result["scene_run_id"]
    assert (run_dir / "enhanced_fast_path.wav").exists()
    assert (run_dir / "selection_trace.csv").exists()
    assert scene_result["fast_frames"] > 0
    assert scene_result["slow_chunks"] > 0
    assert scene_result["speaker_map_updates"] > 0
    assert scene_result["events_count"] >= 1

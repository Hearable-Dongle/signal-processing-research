from __future__ import annotations

import json
from pathlib import Path

from realtime_pipeline.compare_realtime_methods import (
    METHOD_LOCALIZATION_ONLY,
    METHOD_SPEAKER_TRACKING,
    METHOD_SPEAKER_TRACKING_LONG_MEMORY,
    METHOD_SPATIAL_BASELINE,
    build_active_speaker_ground_truth,
    get_method_preset,
)


def test_get_method_preset_returns_stable_public_variants() -> None:
    localization = get_method_preset(METHOD_LOCALIZATION_ONLY)
    assert localization.method_id == METHOD_LOCALIZATION_ONLY
    assert localization.fast_path_reference_mode == "srp_peak"
    assert localization.direction_long_memory_enabled is False

    spatial = get_method_preset(METHOD_SPATIAL_BASELINE)
    assert spatial.control_mode == "spatial_peak_mode"
    assert spatial.fast_path_reference_mode == "speaker_map"
    assert spatial.direction_long_memory_enabled is False

    tracking = get_method_preset(METHOD_SPEAKER_TRACKING)
    assert tracking.control_mode == "speaker_tracking_mode"
    assert tracking.direction_long_memory_enabled is False

    long_memory = get_method_preset(METHOD_SPEAKER_TRACKING_LONG_MEMORY)
    assert long_memory.control_mode == "speaker_tracking_mode"
    assert long_memory.direction_long_memory_enabled is True


def test_build_active_speaker_ground_truth_uses_latest_active_speaker(tmp_path: Path) -> None:
    scene = {
        "room": {"dimensions": [5.0, 4.0, 3.0], "absorption": 0.25},
        "microphone_array": {"mic_center": [2.5, 2.0, 1.5], "mic_radius": 0.05, "mic_count": 4, "mic_positions": None},
        "audio": {
            "sources": [
                {"loc": [3.5, 2.0, 1.5], "audio": "spk0.wav", "gain": 1.0, "classification": "speech"},
                {"loc": [1.5, 2.0, 1.5], "audio": "spk1.wav", "gain": 1.0, "classification": "speech"},
            ],
            "duration": 3.0,
            "fs": 16000,
        },
    }
    scene_path = tmp_path / "scene.json"
    scene_path.write_text(json.dumps(scene), encoding="utf-8")

    metadata = {
        "config": {"render": {"duration_sec": 3.0}},
        "assets": {
            "render_segments": [
                {"classification": "speech", "speaker_id": 0, "position_m": [3.5, 2.0, 1.5]},
                {"classification": "speech", "speaker_id": 1, "position_m": [1.5, 2.0, 1.5]},
            ],
            "speech_events": [
                {"speaker_id": 0, "start_sec": 0.0, "end_sec": 2.0},
                {"speaker_id": 1, "start_sec": 1.0, "end_sec": 3.0},
            ],
        },
    }
    metadata_path = tmp_path / "scenario_metadata.json"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    out = build_active_speaker_ground_truth(
        scene_config_path=scene_path,
        scenario_metadata_path=metadata_path,
        frame_step_ms=500.0,
    )
    assert out["active_speaker_ids"][:2] == [0, 0]
    assert out["active_speaker_ids"][2:] == [1, 1, 1, 1]
    assert out["doa_by_speaker"]["0"] == 0.0
    assert out["doa_by_speaker"]["1"] == 180.0


def test_build_active_speaker_ground_truth_supports_testing_specific_angles_metadata(tmp_path: Path) -> None:
    scene = {
        "room": {"dimensions": [8.0, 6.4, 3.1], "absorption": 0.5},
        "microphone_array": {"mic_center": [4.0, 3.2, 1.45], "mic_radius": 0.1, "mic_count": 4, "mic_positions": None},
        "audio": {
            "sources": [
                {"loc": [5.792, 3.2, 1.45], "audio": "spk0.wav", "gain": 1.0, "classification": "speech"},
                {"loc": [5.2671, 4.4671, 1.45], "audio": "spk1.wav", "gain": 1.0, "classification": "speech"},
            ],
            "duration": 10.0,
            "fs": 16000,
        },
    }
    scene_path = tmp_path / "scene.json"
    scene_path.write_text(json.dumps(scene), encoding="utf-8")

    metadata = {
        "scene_name": "testing_specific_angles_k2_scene00",
        "main_angle_deg": 0,
        "secondary_angle_deg": 45,
        "room": {"dimensions_m": [8.0, 6.4, 3.1], "absorption": 0.5},
        "mic_array": {"mic_center_m": [4.0, 3.2, 1.45], "mic_radius_m": 0.1, "mic_count": 4},
        "assets": {
            "speech": [
                {
                    "speaker_id": 0,
                    "active_window_sec": [0.0, 5.0],
                    "position_m": [5.792, 3.2, 1.45],
                    "angle_deg": 0,
                },
                {
                    "speaker_id": 1,
                    "active_window_sec": [5.0, 10.0],
                    "position_m": [5.2671, 4.4671, 1.45],
                    "angle_deg": 45,
                },
            ]
        },
    }
    metadata_path = tmp_path / "scenario_metadata.json"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    out = build_active_speaker_ground_truth(
        scene_config_path=scene_path,
        scenario_metadata_path=metadata_path,
        frame_step_ms=1000.0,
    )
    assert out["active_speaker_ids"][:5] == [0, 0, 0, 0, 0]
    assert out["active_speaker_ids"][5:] == [1, 1, 1, 1, 1]
    assert out["active_directions_deg"][:5] == [0.0, 0.0, 0.0, 0.0, 0.0]
    assert out["active_directions_deg"][5:] == [45.0, 45.0, 45.0, 45.0, 45.0]
    assert out["doa_by_speaker"]["0"] == 0.0
    assert out["doa_by_speaker"]["1"] == 45.0

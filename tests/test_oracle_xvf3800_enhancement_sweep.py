from __future__ import annotations

import json
import importlib.util
from pathlib import Path

import numpy as np
import soundfile as sf

try:
    import pytest
except ModuleNotFoundError:  # pragma: no cover - local fallback for direct execution in lighter envs
    class _PytestFallback:
        class mark:
            @staticmethod
            def skipif(condition: bool, reason: str = ""):
                def _decorator(fn):
                    return fn

                return _decorator

    pytest = _PytestFallback()

HAS_PYROOM = importlib.util.find_spec("pyroomacoustics") is not None


from beamforming.benchmark.oracle_xvf3800_enhancement_sweep import (
    _build_active_target_schedule,
    _build_oracle_frame_states,
    _load_scene_metadata,
    _run_job,
    _scale_noise_gains,
)
from simulation.simulation_config import SimulationConfig


def _write_wave(path: Path, audio: np.ndarray, sample_rate: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.asarray(audio, dtype=np.float32), sample_rate)


def test_scale_noise_gains_only_updates_noise_sources(tmp_path: Path) -> None:
    scene = {
        "room": {"dimensions": [4.0, 3.0, 2.8], "absorption": 0.3},
        "microphone_array": {"mic_center": [2.0, 1.5, 1.4], "mic_radius": 0.05, "mic_count": 4, "mic_positions": None},
        "audio": {
            "sources": [
                {"loc": [3.0, 1.5, 1.4], "audio": "spk0.wav", "gain": 1.0, "classification": "speech"},
                {"loc": [1.0, 1.5, 1.4], "audio": "spk1.wav", "gain": 1.0, "classification": "speech"},
                {"loc": [2.0, 2.5, 1.2], "audio": "noise.wav", "gain": 0.5, "classification": "noise"},
            ],
            "duration": 1.0,
            "fs": 16000,
        },
    }
    path = tmp_path / "scene.json"
    path.write_text(json.dumps(scene), encoding="utf-8")
    cfg = SimulationConfig.from_file(path)
    scaled = _scale_noise_gains(cfg, 0.2)
    gains = [float(source.gain) for source in scaled.audio.sources]
    assert gains == [1.0, 1.0, 0.1]


def test_build_active_target_schedule_prefers_latest_active_speaker() -> None:
    metadata = {
        "assets": {
            "speech": [
                {"speaker_id": 0, "active_window_sec": [0.0, 0.6], "angle_deg": 0.0},
                {"speaker_id": 1, "active_window_sec": [0.3, 1.0], "angle_deg": 90.0},
            ]
        }
    }
    speaker_ids, doa_deg = _build_active_target_schedule(metadata, sample_rate=10, n_samples=10)
    assert speaker_ids.tolist()[:3] == [0, 0, 0]
    assert speaker_ids.tolist()[3:] == [1, 1, 1, 1, 1, 1, 1]
    assert doa_deg.tolist()[:3] == [0.0, 0.0, 0.0]
    assert doa_deg.tolist()[3:] == [90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0]


def test_load_scene_metadata_derives_missing_top_level_fields(tmp_path: Path) -> None:
    meta = {
        "assets": {
            "speech": [
                {"speaker_id": 0, "active_window_sec": [0.0, 0.5], "angle_deg": 15.0},
                {"speaker_id": 1, "active_window_sec": [0.5, 1.0], "angle_deg": 75.0},
            ],
            "noise": [
                {"noise_id": 0, "active_window_sec": [0.0, 1.0], "angle_deg": 0.0},
                {"noise_id": 1, "active_window_sec": [0.0, 1.0], "angle_deg": 180.0},
            ],
        }
    }
    path = tmp_path / "scenario_metadata.json"
    path.write_text(json.dumps(meta), encoding="utf-8")
    out = _load_scene_metadata("testing_specific_angles_k2_scene00", path)
    assert out["main_angle_deg"] == 15.0
    assert out["secondary_angle_deg"] == 75.0
    assert out["noise_layout_type"] == "opposite_pair"
    assert out["noise_angles_deg"] == [0.0, 180.0]


def test_build_oracle_frame_states_falls_back_when_nulled_user_is_active_target() -> None:
    active_speaker_ids = np.asarray([0, 0, 1, 1], dtype=np.int32)
    active_doa_deg = np.asarray([10.0, 10.0, 90.0, 90.0], dtype=np.float64)
    states = _build_oracle_frame_states(
        active_speaker_ids=active_speaker_ids,
        active_doa_deg=active_doa_deg,
        sample_rate=100,
        fast_frame_ms=10,
        null_user_speaker_id=0,
        null_user_doa_deg=10.0,
        null_conflict_deg=30.0,
    )
    assert states[0].target_speaker_id == 0
    assert states[0].force_suppression_active is False
    assert states[0].null_fallback is True
    assert states[0].peaks_deg == (10.0,)
    assert states[2].target_speaker_id == 1
    assert states[2].force_suppression_active is True
    assert states[2].null_candidate is True
    assert states[2].peaks_deg == (90.0, 10.0)


@pytest.mark.skipif(not HAS_PYROOM, reason="pyroomacoustics is required for simulation-backed benchmark smoke test")
def test_oracle_xvf3800_enhancement_smoke(tmp_path: Path) -> None:
    sample_rate = 16000
    duration_sec = 0.5
    n_samples = int(sample_rate * duration_sec)
    t = np.arange(n_samples, dtype=np.float64) / sample_rate
    speaker0 = 0.1 * np.sin(2.0 * np.pi * 220.0 * t)
    speaker1 = 0.1 * np.sin(2.0 * np.pi * 330.0 * t)
    noise = 0.02 * np.random.default_rng(0).standard_normal(n_samples)

    assets_root = tmp_path / "assets"
    scene_id = "testing_specific_angles_k2_scene00"
    scene_asset_dir = assets_root / scene_id / "render_assets"
    _write_wave(scene_asset_dir / "speaker_0.wav", speaker0, sample_rate)
    _write_wave(scene_asset_dir / "speaker_1.wav", speaker1, sample_rate)
    _write_wave(scene_asset_dir / "noise_0.wav", noise, sample_rate)

    scenes_root = tmp_path / "configs"
    scenes_root.mkdir(parents=True, exist_ok=True)
    scene_path = scenes_root / f"{scene_id}.json"
    scene = {
        "room": {"dimensions": [5.0, 4.0, 3.0], "absorption": 0.25},
        "microphone_array": {"mic_center": [2.5, 2.0, 1.4], "mic_radius": 0.05, "mic_count": 4, "mic_positions": None},
        "audio": {
            "sources": [
                {"loc": [3.5, 2.0, 1.4], "audio": str((scene_asset_dir / "speaker_0.wav").resolve()), "gain": 1.0, "classification": "speech"},
                {"loc": [2.5, 3.0, 1.4], "audio": str((scene_asset_dir / "speaker_1.wav").resolve()), "gain": 1.0, "classification": "speech"},
                {"loc": [1.5, 2.0, 1.2], "audio": str((scene_asset_dir / "noise_0.wav").resolve()), "gain": 1.0, "classification": "noise"},
            ],
            "duration": duration_sec,
            "fs": sample_rate,
        },
    }
    scene_path.write_text(json.dumps(scene), encoding="utf-8")

    metadata_dir = assets_root / scene_id
    metadata = {
        "main_angle_deg": 0.0,
        "secondary_angle_deg": 90.0,
        "noise_layout_type": "single",
        "noise_angles_deg": [180.0],
        "assets": {
            "speech": [
                {
                    "speaker_id": 0,
                    "speaker_label": "s0",
                    "render_asset_path": str((scene_asset_dir / "speaker_0.wav").resolve()),
                    "active_window_sec": [0.0, duration_sec / 2.0],
                    "position_m": [3.5, 2.0, 1.4],
                    "angle_deg": 0.0,
                },
                {
                    "speaker_id": 1,
                    "speaker_label": "s1",
                    "render_asset_path": str((scene_asset_dir / "speaker_1.wav").resolve()),
                    "active_window_sec": [duration_sec / 2.0, duration_sec],
                    "position_m": [2.5, 3.0, 1.4],
                    "angle_deg": 90.0,
                },
            ],
            "noise": [
                {
                    "noise_id": 0,
                    "render_asset_path": str((scene_asset_dir / "noise_0.wav").resolve()),
                    "active_window_sec": [0.0, duration_sec],
                    "position_m": [1.5, 2.0, 1.2],
                    "angle_deg": 180.0,
                }
            ],
        },
    }
    (metadata_dir / "scenario_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    out_root = tmp_path / "out"
    delay_sum_result = _run_job(
        scene_path=str(scene_path),
        assets_root=str(assets_root),
        out_root=str(out_root),
        profile="respeaker_xvf3800_0650",
        noise_gain_scale=0.4,
        method="delay_sum",
    )
    mvdr_result = _run_job(
        scene_path=str(scene_path),
        assets_root=str(assets_root),
        out_root=str(out_root),
        profile="respeaker_xvf3800_0650",
        noise_gain_scale=0.4,
        method="mvdr_fd",
    )
    lcmv_result = _run_job(
        scene_path=str(scene_path),
        assets_root=str(assets_root),
        out_root=str(out_root),
        profile="respeaker_xvf3800_0650",
        noise_gain_scale=0.4,
        method="lcmv_null",
    )

    for result in (delay_sum_result, mvdr_result, lcmv_result):
        row = result.row
        assert row["scene"] == scene_id
        assert row["status"] == "ok"
        assert np.isfinite(float(row["snr_db_raw"]))
        assert np.isfinite(float(row["snr_db_processed"]))
        assert np.isfinite(float(row["sii_raw"]))
        assert np.isfinite(float(row["sii_processed"]))
        run_dir = Path(str(row["run_dir"]))
        assert (run_dir / "summary.json").exists()
        assert (run_dir / "raw_mix_mean.wav").exists()
        assert (run_dir / "clean_active_target.wav").exists()
        assert (run_dir / "enhanced.wav").exists()
    assert float(lcmv_result.row["null_fallback_fraction"]) > 0.0

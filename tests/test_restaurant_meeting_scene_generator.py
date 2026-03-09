from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf
from simulation.create_restaurant_meeting_scene import generate_restaurant_meeting_dataset
from simulation.simulation_config import SimulationConfig
from sim.realistic_conversations.evaluate import summarize_scene


def _write_tone(path: Path, freq: float, sr: int = 16000, duration_sec: float = 8.0, noise: float = 0.01) -> None:
    t = np.arange(int(sr * duration_sec), dtype=np.float32) / sr
    y = 0.6 * np.sin(2 * np.pi * freq * t) + noise * np.random.default_rng(0).standard_normal(t.shape[0]).astype(np.float32)
    sf.write(path, y.astype(np.float32), sr)


def _make_manifest(tmp_path: Path) -> Path:
    assets_dir = tmp_path / "manifest_assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    speech_entries = []
    for idx, freq in enumerate([170.0, 220.0, 260.0, 320.0, 380.0, 440.0]):
        path = assets_dir / f"speaker_{idx}.wav"
        _write_tone(path, freq=freq)
        speech_entries.append({"speaker_id": f"spk_{idx}", "path": str(path.resolve())})
    noise_path = assets_dir / "noise.wav"
    _write_tone(noise_path, freq=90.0, noise=0.08)
    manifest = {
        "speech": speech_entries,
        "noise": [{"category": "noise", "path": str(noise_path.resolve())}],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def test_restaurant_meeting_generator_emits_configs_and_scene_assets(tmp_path: Path) -> None:
    manifest_path = _make_manifest(tmp_path)
    config_root = tmp_path / "configs"
    asset_root = tmp_path / "assets"

    generated = generate_restaurant_meeting_dataset(
        config_root=config_root,
        asset_root=asset_root,
        seed=17,
        scenes_per_k=1,
        k_values=(2, 5),
        duration_sec=8.0,
        frame_ms=20,
        manifest_path=manifest_path,
        export_audio=False,
        max_attempts_per_scene=5,
    )

    assert len(generated) == 2
    for row in generated:
        scene_name = row["scene_name"]
        config_path = Path(row["scene_config_path"])
        scene_dir = Path(row["scene_dir"])
        assert config_path.exists()
        assert config_path.parent == config_root
        assert scene_name.startswith(f"restaurant_meeting_k{row['speaker_count']}_scene")
        cfg = SimulationConfig.from_file(config_path)
        speech_sources = [src for src in cfg.audio.sources if src.classification == "speech"]
        noise_sources = [src for src in cfg.audio.sources if src.classification == "noise"]
        assert noise_sources

        metadata_path = scene_dir / "scenario_metadata.json"
        metrics_path = scene_dir / "metrics_summary.json"
        assert metadata_path.exists()
        assert metrics_path.exists()
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        realized_speakers = {
            int(item["speaker_id"])
            for item in metadata["assets"]["render_segments"]
            if item.get("classification") == "speech" and "speaker_id" in item
        }
        assert len(realized_speakers) == row["speaker_count"]
        assert len(speech_sources) >= len(realized_speakers)
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        assert float(metrics["overlap_ratio"]) <= 0.18
        assert float(metrics["snr_distribution_db"]["mean"]) <= 9.0
        summary = summarize_scene(scene_dir)
        assert summary["preset"] == "restaurant_meeting"

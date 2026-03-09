from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sim.realistic_conversations.evaluate import summarize_scene
from sim.realistic_conversations.generator import generate_scenario
from sim.realistic_conversations.scheduler import build_conversation_plan
from sim.realistic_conversations.config import build_preset


def _write_tone(path: Path, freq: float, sr: int = 16000, duration_sec: float = 8.0, noise: float = 0.01) -> None:
    t = np.arange(int(sr * duration_sec), dtype=np.float32) / sr
    y = 0.6 * np.sin(2 * np.pi * freq * t) + noise * np.random.default_rng(0).standard_normal(t.shape[0]).astype(np.float32)
    sf.write(path, y.astype(np.float32), sr)


def _make_manifest(tmp_path: Path) -> Path:
    assets_dir = tmp_path / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    speech_entries = []
    for idx, freq in enumerate([170.0, 220.0, 260.0, 320.0]):
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


def test_scheduler_sparse_overlap_bias() -> None:
    cfg = build_preset("office")
    cfg.render.duration_sec = 10.0
    plan = build_conversation_plan(cfg, np.random.default_rng(4))
    assert 2 <= len(plan.speaker_ids) <= 4
    assert plan.utterances
    backchannels = [item for item in plan.utterances if item.kind == "backchannel"]
    overlaps = [item for item in plan.utterances if item.kind in {"turn_overlap", "interruption"}]
    assert len(backchannels) <= max(2, len(plan.utterances) // 3)
    assert len(overlaps) <= len(plan.utterances)


def test_generate_scenario_outputs_metadata_and_metrics(tmp_path: Path) -> None:
    manifest_path = _make_manifest(tmp_path)
    result = generate_scenario(
        preset="quiet_room",
        out_dir=tmp_path / "out",
        seed=11,
        duration_sec=4.0,
        manifest_path=manifest_path,
        export_audio=False,
    )
    assert result.scene_config_path.exists()
    assert result.metadata_path.exists()
    assert result.frame_truth_path.exists()
    assert result.metrics_path.exists()

    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    metrics = json.loads(result.metrics_path.read_text(encoding="utf-8"))
    assert metadata["preset"] == "quiet_room"
    assert 2 <= metadata["speaker_count"] <= 4
    assert 0.0 <= metrics["overlap_ratio"] <= 1.0
    assert "snr_distribution_db" in metrics


def test_evaluate_scene_summary(tmp_path: Path) -> None:
    manifest_path = _make_manifest(tmp_path)
    result = generate_scenario(
        preset="moving_speaker",
        out_dir=tmp_path / "out",
        seed=21,
        duration_sec=4.0,
        manifest_path=manifest_path,
        export_audio=False,
    )
    summary = summarize_scene(result.scene_dir)
    assert summary["preset"] == "moving_speaker"
    assert 0.0 <= summary["overlap_ratio"] <= 1.0
    assert "snr_distribution_db" in summary

    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    moving = [speaker for speaker in metadata["speakers"] if speaker["moving"]]
    assert len(moving) == 1


def test_restaurant_meeting_preset_matches_requested_style() -> None:
    cfg = build_preset("restaurant_meeting")

    assert cfg.turn_taking.min_speakers == 2
    assert cfg.turn_taking.max_speakers == 5
    assert cfg.turn_taking.overlap_probability <= 0.15
    assert cfg.turn_taking.interruption_probability <= 0.05
    assert cfg.turn_taking.backchannel_probability <= 0.06
    assert cfg.noise.base_snr_db_range[1] <= 8.0
    assert "distant_chatter" in cfg.noise.ambience_layers

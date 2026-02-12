import json
from pathlib import Path

from localization.benchmark.scene_loader import discover_scenes, scene_targets_count
from simulation.simulation_config import SimulationConfig


def test_discover_scenes_parses_k(tmp_path: Path):
    lib = tmp_path / "library_scene"
    res = tmp_path / "restaurant_scene"
    lib.mkdir()
    res.mkdir()
    (lib / "library_k3_scene01.json").write_text("{}", encoding="utf-8")
    (res / "restaurant_k2_scene09.json").write_text("{}", encoding="utf-8")

    cases = discover_scenes({"library": lib, "restaurant": res})
    ks = sorted((c.scene_type, c.k) for c in cases)
    assert ks == [("library", 3), ("restaurant", 2)]


def test_scene_targets_count_speech_only():
    cfg_dict = {
        "room": {"dimensions": [5, 5, 3], "absorption": 0.2},
        "microphone_array": {"mic_center": [2.5, 2.5, 1.5], "mic_radius": 0.1, "mic_count": 4},
        "audio": {
            "sources": [
                {"loc": [1, 1, 1.5], "audio": "LibriSpeech/train-clean-100/a.flac", "gain": 1.0},
                {"loc": [2, 2, 1.5], "audio": "wham_noise/tr/n.wav", "gain": 0.2},
            ],
            "duration": 1.0,
            "fs": 16000,
        },
    }
    sim_cfg = SimulationConfig.from_dict(json.loads(json.dumps(cfg_dict)))
    assert scene_targets_count(sim_cfg) == 1

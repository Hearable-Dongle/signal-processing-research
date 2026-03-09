from __future__ import annotations

from pathlib import Path

from realtime_pipeline.focus_sanity_check import discover_sanity_scenes


DEFAULT_BENCHMARK_CONFIG = Path("beamforming/benchmark/configs/default.json")


def list_sample_scenes(
    *,
    beamforming_config: str | Path = DEFAULT_BENCHMARK_CONFIG,
    scene_types: list[str] | None = None,
    scenes_per_type: int = 3,
    seed: int = 7,
) -> list[str]:
    scene_types = scene_types or ["library", "restaurant"]
    picked = discover_sanity_scenes(
        beamforming_config=str(beamforming_config),
        scene_types=scene_types,
        scenes_per_type=int(scenes_per_type),
        seed=int(seed),
    )
    return [str(Path(p).resolve()) for p in picked]

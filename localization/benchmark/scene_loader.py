from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from localization.target_policy import iter_speech_target_indices
from simulation.simulation_config import SimulationConfig


SCENE_NAME_RE = re.compile(r"_k(?P<k>\d+)_scene\d+\.json$")


@dataclass(frozen=True)
class SceneCase:
    path: Path
    scene_type: str
    k: int
    scene_id: str


def _parse_k(path: Path) -> int:
    match = SCENE_NAME_RE.search(path.name)
    if not match:
        raise ValueError(f"Could not parse k from scene file: {path}")
    return int(match.group("k"))


def discover_scenes(scene_roots: dict[str, str | Path]) -> list[SceneCase]:
    cases: list[SceneCase] = []
    for scene_type, root in scene_roots.items():
        scene_dir = Path(root)
        if not scene_dir.exists():
            raise FileNotFoundError(f"Scene root not found: {scene_dir}")
        for path in sorted(scene_dir.glob("*.json")):
            cases.append(
                SceneCase(
                    path=path,
                    scene_type=scene_type,
                    k=_parse_k(path),
                    scene_id=path.stem,
                )
            )
    return cases


def load_simulation_config(path: Path) -> SimulationConfig:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return SimulationConfig.from_dict(data)


def iter_target_source_indices(sim_config: SimulationConfig) -> Iterable[int]:
    # Shared policy with manual flow: speech-only targets.
    yield from iter_speech_target_indices(sim_config)


def scene_targets_count(sim_config: SimulationConfig) -> int:
    return sum(1 for _ in iter_target_source_indices(sim_config))

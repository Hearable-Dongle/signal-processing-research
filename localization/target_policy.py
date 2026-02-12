from __future__ import annotations

import math
from typing import Iterable

from simulation.simulation_config import SimulationConfig, SimulationSource


def _normalize_deg(angle_deg: float) -> float:
    return float(angle_deg % 360.0)


def is_speech_target(source: SimulationSource) -> bool:
    """Benchmark/manual shared policy: speech targets are LibriSpeech sources."""
    audio_path = source.audio_path.replace("\\", "/").lower()
    return "librispeech/" in audio_path


def iter_speech_target_indices(sim_config: SimulationConfig) -> Iterable[int]:
    for idx, source in enumerate(sim_config.audio.sources):
        if is_speech_target(source):
            yield idx


def true_target_doas_deg(sim_config: SimulationConfig) -> list[float]:
    center = sim_config.microphone_array.mic_center
    doas_deg: list[float] = []
    for source in sim_config.audio.sources:
        if not is_speech_target(source):
            continue
        dx = source.loc[0] - center[0]
        dy = source.loc[1] - center[1]
        doas_deg.append(_normalize_deg(math.degrees(math.atan2(dy, dx))))
    return doas_deg

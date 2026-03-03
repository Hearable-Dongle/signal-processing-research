from __future__ import annotations

from simulation.simulation_config import SimulationConfig, SimulationSource


def is_speech_target(source: SimulationSource) -> bool:
    """
    Determine whether a source should be treated as a target speech source.

    Policy:
    - If classification is explicitly provided, trust it.
    - Otherwise, fall back to known dataset path conventions.
    """
    cls = (source.classification or "").strip().lower()
    if cls:
        if cls in {"noise", "interference", "background"}:
            return False
        if cls in {"signal", "speech", "target"}:
            return True

    audio_path = source.audio_path.replace("\\", "/").lower()
    if "librispeech/" in audio_path:
        return True
    if "wham_noise/" in audio_path:
        return False
    return cls != "noise"


def iter_target_source_indices(sim_config: SimulationConfig):
    for idx, source in enumerate(sim_config.audio.sources):
        if is_speech_target(source):
            yield idx


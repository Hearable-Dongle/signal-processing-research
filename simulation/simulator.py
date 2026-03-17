import numpy as np
import pyroomacoustics as pra
import librosa
from pathlib import Path
from typing import Tuple, List

from simulation.simulation_config import SimulationConfig


def _build_room_and_mics(config: SimulationConfig) -> tuple[pra.ShoeBox, np.ndarray, int]:
    room = pra.ShoeBox(
        config.room.dimensions,
        fs=config.audio.fs,
        materials=pra.Material(config.room.absorption),
        max_order=17,
    )
    center = np.array(config.microphone_array.mic_center, dtype=float)
    explicit_positions = config.microphone_array.mic_positions
    if explicit_positions:
        mic_pos_rel = np.asarray(explicit_positions, dtype=float).T
        if mic_pos_rel.shape[0] != 3:
            raise ValueError("microphone_array.mic_positions must have shape (n_mics, 3)")
        mic_count = mic_pos_rel.shape[1]
    else:
        radius = config.microphone_array.mic_radius
        mic_count = config.microphone_array.mic_count
        phi = np.linspace(0, 2 * np.pi, mic_count, endpoint=False)
        mic_pos_rel = np.array([
            radius * np.cos(phi),
            radius * np.sin(phi),
            np.zeros(mic_count),
        ])
    mic_pos = mic_pos_rel + center.reshape(3, 1)
    room.add_microphone_array(pra.MicrophoneArray(mic_pos, fs=config.audio.fs))
    return room, mic_pos, mic_count


def _load_source_signals(config: SimulationConfig) -> tuple[list[np.ndarray], int]:
    target_samples = int(config.audio.duration * config.audio.fs)
    source_signals: list[np.ndarray] = []
    for source_conf in config.audio.sources:
        path = source_conf.get_absolute_path()
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")
        audio, _ = librosa.load(path, sr=config.audio.fs)
        if len(audio) < target_samples:
            audio = np.pad(audio, (0, target_samples - len(audio)))
        else:
            audio = audio[:target_samples]
        audio = audio * source_conf.gain
        source_signals.append(audio.astype(np.float32, copy=False))
    return source_signals, target_samples


def _simulate_room_signals(
    config: SimulationConfig,
    source_signals: list[np.ndarray],
    *,
    source_indices: list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    room, mic_pos, _ = _build_room_and_mics(config)
    target_samples = int(config.audio.duration * config.audio.fs)
    indices = list(range(len(config.audio.sources))) if source_indices is None else [int(v) for v in source_indices]
    for source_idx in indices:
        source_conf = config.audio.sources[source_idx]
        room.add_source(source_conf.loc, signal=np.asarray(source_signals[source_idx], dtype=np.float32))
    room.simulate()
    mic_audio = room.mic_array.signals.T
    if mic_audio.shape[0] > target_samples:
        mic_audio = mic_audio[:target_samples, :]
    elif mic_audio.shape[0] < target_samples:
        mic_audio = np.pad(mic_audio, ((0, target_samples - mic_audio.shape[0]), (0, 0)))
    return mic_audio, mic_pos

def run_simulation(config: SimulationConfig) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    """
    Runs the simulation based on the provided configuration.

    Args:
        config: The simulation configuration object.

    Returns:
        mic_audio: Simulated audio from microphones (samples, channels)
        mic_pos: Microphone positions (3, n_mics)
        source_signals: Original source signals (resampled and truncated) (n_sources, samples)
    """
    source_signals, _target_samples = _load_source_signals(config)
    mic_audio, mic_pos = _simulate_room_signals(config, source_signals)
    return mic_audio, mic_pos, source_signals


def run_simulation_with_source_contributions(config: SimulationConfig) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], List[np.ndarray]]:
    """
    Runs the simulation and also returns per-source microphone contributions.

    Returns:
        mic_audio: Simulated mixed microphone audio (samples, channels)
        mic_pos: Microphone positions (3, n_mics)
        source_signals: Original source signals (resampled and truncated)
        source_mic_audio: Per-source microphone contributions [(samples, channels), ...]
    """
    source_signals, _target_samples = _load_source_signals(config)
    mic_audio, mic_pos = _simulate_room_signals(config, source_signals)
    source_mic_audio = []
    for source_idx in range(len(source_signals)):
        source_audio, _ = _simulate_room_signals(config, source_signals, source_indices=[source_idx])
        source_mic_audio.append(np.asarray(source_audio, dtype=np.float32))
    return mic_audio, mic_pos, source_signals, source_mic_audio

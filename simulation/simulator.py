import numpy as np
import pyroomacoustics as pra
import librosa
from pathlib import Path
from typing import Tuple, List

from simulation.simulation_config import SimulationConfig

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
    # Setup Room
    room = pra.ShoeBox(
        config.room.dimensions,
        fs=config.audio.fs,
        materials=pra.Material(config.room.absorption),
        max_order=17 
    )

    # Setup Microphones
    R = config.microphone_array.mic_radius
    M = config.microphone_array.mic_count
    center = np.array(config.microphone_array.mic_center)
    
    # Assuming center[2] provides the height, and the circle is on that plane.
    phi = np.linspace(0, 2 * np.pi, M, endpoint=False)
    mic_pos_rel = np.array([
        R * np.cos(phi),
        R * np.sin(phi),
        np.zeros(M)
    ])
    
    # Absolute positions
    mic_pos = mic_pos_rel + center.reshape(3, 1)
    
    room.add_microphone_array(pra.MicrophoneArray(mic_pos, fs=config.audio.fs))

    # Add Sources
    target_samples = int(config.audio.duration * config.audio.fs)
    source_signals = []

    for source_conf in config.audio.sources:
        path = source_conf.get_absolute_path()
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")
            
        audio, _ = librosa.load(path, sr=config.audio.fs)
        
        # Handle duration (loop/truncate/pad)
        if len(audio) < target_samples:
            # Pad with zeros
            padding = target_samples - len(audio)
            audio = np.pad(audio, (0, padding))
        else:
            audio = audio[:target_samples]
            
        # Normalize
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))
            
        room.add_source(source_conf.loc, signal=audio)
        source_signals.append(audio)

    # Simulate
    room.simulate()
    
    mic_audio = room.mic_array.signals.T  # (samples, channels)
    
    if mic_audio.shape[0] > target_samples:
        mic_audio = mic_audio[:target_samples, :]
    elif mic_audio.shape[0] < target_samples:
         padding = target_samples - mic_audio.shape[0]
         mic_audio = np.pad(mic_audio, ((0, padding), (0, 0)))

    return mic_audio, mic_pos, source_signals

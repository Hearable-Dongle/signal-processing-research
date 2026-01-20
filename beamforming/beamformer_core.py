from dataclasses import dataclass
from typing import Dict, Tuple, Callable, Optional
import numpy as np
from scipy.signal import stft
from numpy.typing import NDArray

from algo.beamformer import (
    compute_steering_vector,
    wng_mvdr_newton,
    wng_mvdr_steepest,
)

@dataclass
class BeamformerConfig:
    """DSP Hyperparameters independent of simulation."""
    fs: int
    frame_duration_ms: float
    sound_speed: float
    gamma_dB: float
    iterations: int
    mic_array_center: NDArray  # config.mic_loc
    mic_geometry: NDArray      # mic_pos

def compute_spectral_features(
    audio_multichannel: NDArray, 
    fs: int, 
    window_size: int, 
    hop: int
) -> Tuple[NDArray, NDArray]:
    """
    Converts time-domain audio to STFT tensor.
    Returns:
        stft_tensor: Shape (freq_bins, time_frames, channels)
        fvec: Frequency vector
    """
    window = np.hanning(window_size)
    
    stft_results = [
        stft(
            audio_multichannel[:, i],
            fs=fs,
            nperseg=window_size,
            noverlap=window_size - hop,
            window=window,
            padded=True,
            return_onesided=True
        ) for i in range(audio_multichannel.shape[1])
    ]
    
    # Extract fvec from the first channel (it's the same for all)
    fvec = stft_results[0][0]
    
    # Stack the Zxx matrices (3rd element of tuple) into a 3D Tensor
    # Shape: (freq_bins, time_frames, channels)
    stft_tensor = np.stack(
        [res[2].astype(np.complex128) for res in stft_results], 
        axis=-1
    )
    
    return stft_tensor, fvec

def solve_weights_per_bin(
    solver_fn: Callable, 
    Rnn: NDArray, 
    steering_vecs: NDArray, 
    gamma: float, 
    mu: float, 
    iterations: int
) -> Tuple[NDArray, list]:
    """Functional wrapper to apply a solver across all frequency bins."""
    freq_bin_count = steering_vecs[0].shape[0]
    
    def process_bin(kf):
        # Extract specific frequency slice
        a_vecs_bin = [sv[kf, :].reshape(-1, 1) for sv in steering_vecs]
        # Run solver
        w_bin, p_hist = solver_fn(Rnn, a_vecs_bin, gamma, mu, iterations)
        return w_bin[:, 0], p_hist

    results = [process_bin(kf) for kf in range(freq_bin_count)]
    weights_list, power_histories = zip(*results)
    
    return np.array(weights_list, dtype=complex), list(power_histories)

def compute_beamforming_weights(
    audio_input: NDArray,
    source_locations: NDArray,
    noise_covariance: NDArray,
    config: BeamformerConfig
) -> Dict[str, Tuple[NDArray, list]]:
    """
    Main Interface: Calculates beamforming weights using available methods.
    
    Args:
        audio_input: (samples, channels)
        source_locations: (n_sources, 3)
        noise_covariance: Rnn matrix
        config: BeamformerConfig object
    
    Returns:
        Dictionary containing weights and histories for 'steepest' and 'newton',
        plus the STFT tensor and frequency vector used for computation.
    """
    # 1. Derive STFT Parameters
    stft_window_size = int(config.fs * config.frame_duration_ms / 1000)
    hop = stft_window_size // 2
    
    # 2. Compute Spectral Features
    stft_tensor, fvec = compute_spectral_features(
        audio_input, config.fs, stft_window_size, hop
    )

    # 3. Compute Steering Vectors
    steering_vecs = compute_steering_vector(
        config.mic_geometry,
        config.mic_array_center,
        fvec,
        source_locations,
        config.sound_speed,
    )

    # 4. Prepare Solver Params
    gamma = 10 ** (config.gamma_dB / 10)
    mu = 0.01 / np.trace(noise_covariance.real)
    
    # 5. Compute Weights (Functional Strategy Pattern)
    weights_steepest, hist_steepest = solve_weights_per_bin(
        wng_mvdr_steepest, noise_covariance, steering_vecs, gamma, mu, config.iterations
    )
    
    weights_newton, hist_newton = solve_weights_per_bin(
        wng_mvdr_newton, noise_covariance, steering_vecs, gamma, mu, config.iterations
    )

    return {
        "stft_tensor": stft_tensor,
        "fvec": fvec,
        "steepest": (weights_steepest, hist_steepest),
        "newton": (weights_newton, hist_newton),
        "params": (stft_window_size, hop, np.hanning(stft_window_size)) # needed for ISTFT
    }
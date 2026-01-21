from dataclasses import dataclass
from typing import Dict, Tuple, Callable
import numpy as np
from scipy.signal import stft
from numpy.typing import NDArray

from beamforming.algo.beamformer import (
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

def compute_spatial_covariance_matrix(
    audio_multichannel: NDArray,
    fs: int,
    window_size: int,
    hop: int
) -> NDArray:
    """
    Computes the Spatial Covariance Matrix (Rnn) for each frequency bin.
    
    Returns:
        Rnn_stft: Shape (n_freq, n_mics, n_mics)
    """
    # 1. Get STFT of the noise (re-use existing logic)
    # We don't need the fvec here, just the tensor
    stft_tensor, _ = compute_spectral_features(audio_multichannel, fs, window_size, hop)
    
    # stft_tensor shape: (n_freq, n_frames, n_mics)
    n_freq, n_frames, n_mics = stft_tensor.shape
    
    # 2. Compute Rnn for each frequency bin: E[x * x^H]
    # We want to average over time frames.
    # Result shape: (n_freq, n_mics, n_mics)
    
    # Efficient einsum: for each freq f, matmul(frame vector, frame vector conjugate)
    # 'ftm,ftn->fmn' means:
    # f: freq (keep)
    # t: time (sum/contract)
    # m, n: mics (outer product)
    Rnn_stft = np.einsum('ftm,ftn->fmn', stft_tensor, stft_tensor.conj())
    
    # Normalize by number of frames to get average
    Rnn_stft /= n_frames
    
    # Optional: Add slight diagonal loading (regularization) to prevent singularity
    diagonal_loading = 1e-6 * np.eye(n_mics)[None, :, :]
    Rnn_stft += diagonal_loading
    
    return Rnn_stft


def solve_weights_per_bin(
    solver_fn: Callable, 
    Rnn_tensor: NDArray, 
    steering_vecs: NDArray, 
    gamma: float, 
    iterations: int
) -> Tuple[NDArray, list]:
    
    freq_bin_count = steering_vecs[0].shape[0]
    
    def process_bin(kf):
        # 1. Extract Steering Vector for this bin
        a_vecs_bin = [sv[kf, :].reshape(-1, 1) for sv in steering_vecs]
        
        # 2. Extract Covariance Matrix for THIS bin <<-- CRITICAL FIX
        Rnn_bin = Rnn_tensor[kf, :, :] 

        # 3. Calculate dynamic step size mu based on THIS bin's power
        # (Standard MVDR often uses a fixed mu or normalized mu per bin)
        bin_power = np.trace(Rnn_bin.real)
        if solver_fn.__name__ == "wng_mvdr_newton":
            mu_bin = 0.5  # Fixed step size for Newton
        else:
            mu_bin = 0.01 / (bin_power + 1e-10) # Dynamic for Steepest
        
        
        w_bin, p_hist = solver_fn(Rnn_bin, a_vecs_bin, gamma, mu_bin, iterations)
        return w_bin[:, 0], p_hist

    results = [process_bin(kf) for kf in range(freq_bin_count)]
    weights_list, power_histories = zip(*results)
    
    return np.array(weights_list, dtype=complex), list(power_histories)


def compute_beamforming_weights(
    audio_input: NDArray,
    source_locations: NDArray,
    noise_audio: NDArray, 
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
    
    # 1. Derive STFT Params
    stft_window_size = int(config.fs * config.frame_duration_ms / 1000)
    hop = stft_window_size // 2

    # 2. Compute Rnn per frequency bin <<-- NEW STEP
    # This ensures Rnn matches the STFT properties exactly
    Rnn_tensor = compute_spatial_covariance_matrix(
        noise_audio, config.fs, stft_window_size, hop
    )

    # 3. Compute Spectral Features for Signal
    stft_tensor, fvec = compute_spectral_features(
        audio_input, config.fs, stft_window_size, hop
    )

    # 4. Compute Steering Vectors
    steering_vecs = compute_steering_vector(
        config.mic_geometry,
        config.mic_array_center,
        fvec,
        source_locations,
        config.sound_speed,
    )

    # 4. Prepare Solver Params
    gamma = 10 ** (config.gamma_dB / 10)
    
    # 5. Compute Weights (Functional Strategy Pattern)
    weights_steepest, hist_steepest = solve_weights_per_bin(
        wng_mvdr_steepest, Rnn_tensor, steering_vecs, gamma, config.iterations
    )
    
    weights_newton, hist_newton = solve_weights_per_bin(
        wng_mvdr_newton, Rnn_tensor, steering_vecs, gamma, config.iterations
    )

    return {
        "stft_tensor": stft_tensor,
        "fvec": fvec,
        "steepest": (weights_steepest, hist_steepest),
        "newton": (weights_newton, hist_newton),
        "params": (stft_window_size, hop, np.hanning(stft_window_size)) # needed for ISTFT
    }
from dataclasses import dataclass
from typing import Dict, Tuple, Callable
import numpy as np
from scipy.signal import stft
from numpy.typing import NDArray

import torch

from beamforming.algo.beamformer import (
    compute_steering_vector,
    wng_mvdr_newton,
    wng_mvdr_steepest,
)
from beamforming.nn.mask_nn import MaskEstimationNetwork

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
    stft_tensor: NDArray,
    mask: NDArray | None = None
) -> NDArray:
    """
    Computes the Spatial Covariance Matrix (Rnn) for each frequency bin, optionally weighted by a mask.
    
    Args:
        stft_tensor: Shape (n_freq, n_frames, n_mics)
        mask: Shape (n_freq, n_frames) - Optional weighting mask [0, 1]
    
    Returns:
        Rnn_stft: Shape (n_freq, n_mics, n_mics)
    """
    n_freq, n_frames, n_mics = stft_tensor.shape
    
    if mask is None:
        mask = np.ones((n_freq, n_frames), dtype=stft_tensor.real.dtype)
        
    # Numerator: Sum_t ( M(t,f) * y(t,f) * y(t,f)^H )
    # einsum: mask 'ft', stft 'ftm', stft_conj 'ftn' -> 'fmn'
    numerator = np.einsum('ft,ftm,ftn->fmn', mask, stft_tensor, stft_tensor.conj())
    
    # Denominator: Sum_t ( M(t,f) )
    denominator = np.sum(mask, axis=1) # Shape (n_freq,)
    
    # Avoid division by zero
    denominator = np.maximum(denominator, 1e-6)
    
    # Broadcast denominator: (n_freq, 1, 1)
    Rnn_stft = numerator / denominator[:, None, None]
    
    # Optional: Add slight diagonal loading (regularization) to prevent singularity
    diagonal_loading = 1e-6 * np.eye(n_mics)[None, :, :]
    Rnn_stft += diagonal_loading
    
    return Rnn_stft

def compute_principal_eigenvector(R_ss: NDArray) -> NDArray:
    """
    Extracts the principal eigenvector (corresponding to the largest eigenvalue)
    for each frequency bin to serve as the steering vector.
    
    Args:
        R_ss: Weighted Speech Covariance Matrix (n_freq, n_mics, n_mics)
        
    Returns:
        steering_vecs: Shape (n_freq, n_mics)
    """
    n_freq, n_mics, _ = R_ss.shape
    steering_vecs = np.zeros((n_freq, n_mics), dtype=np.complex128)
    
    for f in range(n_freq):
        # eigh returns eigenvalues in ascending order
        _, vecs = np.linalg.eigh(R_ss[f])
        steering_vecs[f] = vecs[:, -1] # Last column is principal eigenvector
        
    return steering_vecs


def solve_weights_per_bin(
    solver_fn: Callable, 
    Rnn_tensor: NDArray, 
    steering_vecs: NDArray, 
    gamma: float, 
    iterations: int
) -> Tuple[NDArray, list]:
    
    freq_bin_count = steering_vecs[0].shape[0]
    
    def process_bin(kf):
        a_vecs_bin = [sv[kf, :].reshape(-1, 1) for sv in steering_vecs]
        Rnn_bin = Rnn_tensor[kf, :, :] 

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

def compute_beamforming_weights_classical(
        fvec, 
        stft_noise,
        source_locations, 
        mic_geometry,
        mic_array_center,
        sound_speed,
        gamma_dB,
        num_iterations=10,
    ):
 
    Rnn_tensor_classical = compute_spatial_covariance_matrix(stft_noise)

    steering_vecs_classical = compute_steering_vector(
        mic_geometry,
        mic_array_center,
        fvec,
        source_locations,
        sound_speed,
    )

    gamma = 10 ** (gamma_dB / 10)
    
    weights_steepest, hist_steepest = solve_weights_per_bin(
        wng_mvdr_steepest, Rnn_tensor_classical, steering_vecs_classical, gamma, num_iterations
    )
    
    weights_newton, hist_newton = solve_weights_per_bin(
        wng_mvdr_newton, Rnn_tensor_classical, steering_vecs_classical, gamma, num_iterations
    )

    
    return {
        "weights_steepest": weights_steepest,
        "hist_steepest": hist_steepest,
        "weights_newton": weights_newton,
        "hist_newton": hist_newton,
    }

    
def compute_beamforming_weights_mvdr(stft_tensor: NDArray) -> NDArray:
    mag_tensor = np.abs(stft_tensor).astype(np.float32)
    nn_input = torch.from_numpy(mag_tensor).unsqueeze(0)
    
    n_freq, _n_frames, n_mics = stft_tensor.shape
    model = MaskEstimationNetwork(input_channels=n_mics, freq_bins=n_freq)
    model.eval()
    
    with torch.no_grad():
        masks = model(nn_input)
        
    masks_np = masks.squeeze(0).numpy()
    mask_speech = masks_np[0, :, :]
    mask_noise = masks_np[1, :, :]
    
    R_nn_neural = compute_spatial_covariance_matrix(stft_tensor, mask_noise)
    R_ss_neural = compute_spatial_covariance_matrix(stft_tensor, mask_speech)
    
    steering_vecs_neural = compute_principal_eigenvector(R_ss_neural) 

    def compute_wieight_for_frequency_bin(f):
        R_inv = np.linalg.pinv(R_nn_neural[f])
        d = steering_vecs_neural[f].reshape(-1, 1)
        
        num = R_inv @ d
        denom = d.conj().T @ num
        w = num / (denom + 1e-10)
        return w.reshape(-1)    

    weights_mvdr = np.array(list(map(compute_wieight_for_frequency_bin, range(n_freq))), dtype=np.complex128)

    return weights_mvdr


def compute_beamforming_weights(
    audio_input: NDArray,
    source_locations: NDArray,
    noise_audio: NDArray, 
    config: BeamformerConfig
) -> Dict[str, Tuple[NDArray, list]]:
    """
    Main Interface: Calculates beamforming weights using both classical and neural methods.
    
    Args:
        audio_input: (samples, channels)
        source_locations: (n_sources, 3)
        noise_audio: Used for classical noise estimation
        config: BeamformerConfig object
    
    Returns:
        Dictionary containing weights for 'steepest', 'newton', and 'mvdr'.
    """
    
    stft_window_size = int(config.fs * config.frame_duration_ms / 1000)
    hop = stft_window_size // 2

    stft_noise, _ = compute_spectral_features(
        noise_audio, config.fs, stft_window_size, hop
    )

    stft_tensor, fvec = compute_spectral_features(
        audio_input, config.fs, stft_window_size, hop
    )
    
    stft_window_size = int(config.fs * config.frame_duration_ms / 1000)
    hop = stft_window_size // 2

    stft_noise, _ = compute_spectral_features(
        noise_audio, config.fs, stft_window_size, hop
    )

    stft_tensor, fvec = compute_spectral_features(
        audio_input, config.fs, stft_window_size, hop
    )
    
    classical_beamforming_weights = compute_beamforming_weights_classical(
        fvec=fvec,
        stft_noise=stft_noise,
        source_locations=source_locations, 
        mic_geometry=config.mic_geometry,
        mic_array_center=config.mic_array_center,
        sound_speed=config.sound_speed,
        gamma_dB=config.gamma_dB,
    )

    weights_steepest = classical_beamforming_weights["weights_steepest"]
    hist_steepest = classical_beamforming_weights["hist_steepest"]
    weights_newton = classical_beamforming_weights["weights_newton"]
    hist_newton = classical_beamforming_weights["hist_newton"]

    weights_mvdr = compute_beamforming_weights_mvdr(stft_tensor=stft_tensor)

    return {
        "stft_tensor": stft_tensor,
        "fvec": fvec,
        "steepest": (weights_steepest, hist_steepest),
        "newton": (weights_newton, hist_newton),
        "mvdr": (weights_mvdr, []),
        "params": (stft_window_size, hop, np.hanning(stft_window_size))
    }
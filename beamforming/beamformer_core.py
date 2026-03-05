from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.signal import stft

from beamforming.algo.beamformer import (
    compute_steering_vector,
    compute_steering_vector_from_azimuths,
    wng_mvdr_newton,
    wng_mvdr_steepest,
)


@dataclass
class BeamformerConfig:
    """DSP hyperparameters independent of simulation."""

    fs: int
    frame_duration_ms: float
    sound_speed: float
    gamma_dB: float
    iterations: int
    output_dir: Path | str | None = None
    mic_array_center: NDArray | None = None
    mic_geometry: NDArray | None = None
    steering_mode: str = "locations"
    doa_update_hop_frames: int = 10
    localization_default_method: str = "SSZ"
    localization_fallback_methods: list[str] | None = None

    @classmethod
    def from_dict(cls, data: dict, fs: int) -> "BeamformerConfig":
        bf_data = data.get("beamforming", data)
        loc_cfg = bf_data.get("localization", {})
        fallback = loc_cfg.get("fallback_methods", bf_data.get("localization_fallback_methods", ["SSZ", "GMDA"]))

        return cls(
            fs=fs,
            frame_duration_ms=bf_data.get("frame_duration", 10.0),
            sound_speed=bf_data.get("sound_speed", 343.0),
            gamma_dB=bf_data.get("gamma_dB", 15.0),
            iterations=bf_data.get("iterations", 10),
            output_dir=bf_data.get("output_dir"),
            steering_mode=bf_data.get("steering_mode", "locations"),
            doa_update_hop_frames=bf_data.get("doa_update_hop_frames", 10),
            localization_default_method=loc_cfg.get("default_method", bf_data.get("localization_default_method", "SSZ")),
            localization_fallback_methods=list(fallback) if fallback else None,
        )


def _compute_steering_vectors(
    *,
    fvec: NDArray,
    mic_geometry: NDArray,
    mic_array_center: NDArray,
    sound_speed: float,
    source_locations: NDArray | None = None,
    source_azimuths_deg: NDArray | None = None,
) -> NDArray:
    if source_azimuths_deg is not None and len(source_azimuths_deg) > 0:
        return compute_steering_vector_from_azimuths(
            mic_pos=mic_geometry,
            fvec=fvec,
            azimuths_deg=source_azimuths_deg,
            sound_speed=sound_speed,
        )

    if source_locations is None or len(source_locations) == 0:
        raise ValueError("No steering targets provided: source_locations/source_azimuths_deg are empty.")

    return compute_steering_vector(
        mic_pos=mic_geometry,
        mic_loc=mic_array_center,
        fvec=fvec,
        signal_loc=source_locations,
        sound_speed=sound_speed,
    )


def compute_spectral_features(audio_multichannel: NDArray, fs: int, window_size: int, hop: int) -> tuple[NDArray, NDArray]:
    """Convert time-domain multichannel signal into STFT tensor."""
    window = np.hanning(window_size)

    stft_results = [
        stft(
            audio_multichannel[:, i],
            fs=fs,
            nperseg=window_size,
            noverlap=window_size - hop,
            window=window,
            padded=True,
            return_onesided=True,
        )
        for i in range(audio_multichannel.shape[1])
    ]

    fvec = stft_results[0][0]
    stft_tensor = np.stack([res[2].astype(np.complex128) for res in stft_results], axis=-1)
    return stft_tensor, fvec


def compute_spatial_covariance_matrix(stft_tensor: NDArray, mask: NDArray | None = None) -> NDArray:
    """Compute frequency-wise spatial covariance matrix."""
    n_freq, n_frames, n_mics = stft_tensor.shape

    if mask is None:
        mask = np.ones((n_freq, n_frames), dtype=stft_tensor.real.dtype)

    numerator = np.einsum("ft,ftm,ftn->fmn", mask, stft_tensor, stft_tensor.conj())
    denominator = np.sum(mask, axis=1)
    denominator = np.maximum(denominator, 1e-6)

    rnn_stft = numerator / denominator[:, None, None]
    diagonal_loading = 1e-6 * np.eye(n_mics)[None, :, :]
    rnn_stft += diagonal_loading
    return rnn_stft


def solve_weights_per_bin(
    solver_fn: Callable,
    rnn_tensor: NDArray,
    steering_vecs: NDArray,
    gamma: float,
    iterations: int,
) -> tuple[NDArray, list]:
    freq_bin_count = steering_vecs[0].shape[0]

    def process_bin(kf: int):
        a_vecs_bin = [sv[kf, :].reshape(-1, 1) for sv in steering_vecs]
        rnn_bin = rnn_tensor[kf, :, :]

        bin_power = np.trace(rnn_bin.real)
        if solver_fn.__name__ == "wng_mvdr_newton":
            mu_bin = 0.5
        else:
            mu_bin = 0.01 / (bin_power + 1e-10)

        w_bin, p_hist = solver_fn(rnn_bin, a_vecs_bin, gamma, mu_bin, iterations)
        return w_bin[:, 0], p_hist

    results = [process_bin(kf) for kf in range(freq_bin_count)]
    weights_list, power_histories = zip(*results)
    return np.array(weights_list, dtype=complex), list(power_histories)


def compute_beamforming_weights_mvdr_classical(
    *,
    fvec: NDArray,
    stft_noise: NDArray,
    source_locations: NDArray | None,
    mic_geometry: NDArray,
    mic_array_center: NDArray,
    sound_speed: float,
    gamma_dB: float,
    source_azimuths_deg: NDArray | None = None,
    num_iterations: int = 10,
) -> dict:
    rnn_tensor = compute_spatial_covariance_matrix(stft_noise)

    steering_vecs = _compute_steering_vectors(
        fvec=fvec,
        mic_geometry=mic_geometry,
        mic_array_center=mic_array_center,
        sound_speed=sound_speed,
        source_locations=source_locations,
        source_azimuths_deg=source_azimuths_deg,
    )

    gamma = 10 ** (gamma_dB / 10)
    weights_steepest, hist_steepest = solve_weights_per_bin(
        wng_mvdr_steepest, rnn_tensor, steering_vecs, gamma, num_iterations
    )
    weights_newton, hist_newton = solve_weights_per_bin(
        wng_mvdr_newton, rnn_tensor, steering_vecs, gamma, num_iterations
    )

    return {
        "weights_steepest": weights_steepest,
        "hist_steepest": hist_steepest,
        "weights_newton": weights_newton,
        "hist_newton": hist_newton,
    }


def compute_beamforming_weights(
    *,
    audio_input: NDArray,
    source_locations: NDArray | None,
    noise_audio: NDArray,
    config: BeamformerConfig,
    source_azimuths_deg: NDArray | None = None,
    target_weights: NDArray | None = None,
) -> dict:
    """Main interface: calculate top-performing MVDR beamforming weights."""
    stft_window_size = int(config.fs * config.frame_duration_ms / 1000)
    hop = stft_window_size // 2

    stft_noise, _ = compute_spectral_features(noise_audio, config.fs, stft_window_size, hop)
    stft_tensor, fvec = compute_spectral_features(audio_input, config.fs, stft_window_size, hop)

    classical = compute_beamforming_weights_mvdr_classical(
        fvec=fvec,
        stft_noise=stft_noise,
        source_locations=source_locations,
        mic_geometry=config.mic_geometry,
        mic_array_center=config.mic_array_center,
        sound_speed=config.sound_speed,
        gamma_dB=config.gamma_dB,
        source_azimuths_deg=source_azimuths_deg,
        num_iterations=config.iterations,
    )

    return {
        "stft_tensor": stft_tensor,
        "fvec": fvec,
        "steepest": (classical["weights_steepest"], classical["hist_steepest"]),
        "newton": (classical["weights_newton"], classical["hist_newton"]),
        "target_weights": [] if target_weights is None else [float(v) for v in np.asarray(target_weights).reshape(-1)],
        "params": (stft_window_size, hop, np.hanning(stft_window_size)),
    }

from __future__ import annotations

import numpy as np
from scipy.signal import istft, stft

from .config import DirectionAssignmentConfig


def _stft_1d(x: np.ndarray, cfg: DirectionAssignmentConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return stft(
        x,
        fs=cfg.sample_rate,
        nperseg=cfg.win_length,
        noverlap=cfg.win_length - cfg.hop_length,
        nfft=cfg.n_fft,
        boundary="zeros",
        padded=True,
        return_onesided=True,
    )


def _istft_1d(z: np.ndarray, cfg: DirectionAssignmentConfig, target_len: int) -> np.ndarray:
    _, x = istft(
        z,
        fs=cfg.sample_rate,
        nperseg=cfg.win_length,
        noverlap=cfg.win_length - cfg.hop_length,
        nfft=cfg.n_fft,
        input_onesided=True,
        boundary=True,
    )
    x = np.real(x)
    if len(x) > target_len:
        return x[:target_len]
    if len(x) < target_len:
        return np.pad(x, (0, target_len - len(x)))
    return x


def backproject_streams_to_multichannel(
    raw_mic_chunk: np.ndarray,
    separated_streams: list[np.ndarray],
    cfg: DirectionAssignmentConfig,
) -> tuple[list[np.ndarray], dict]:
    """
    Returns per-stream multichannel signals as list of arrays shaped (n_mics, samples).
    """
    mix = np.asarray(raw_mic_chunk, dtype=float)
    if mix.ndim != 2:
        raise ValueError("raw_mic_chunk must have shape (samples, n_mics)")

    n_samples, n_mics = mix.shape
    if not separated_streams:
        return [], {"reason": "no_streams"}

    mix_spec_list: list[np.ndarray] = []
    for m in range(n_mics):
        _f, _t, z = _stft_1d(mix[:, m], cfg)
        mix_spec_list.append(z)

    stream_specs: list[np.ndarray] = []
    for s in separated_streams:
        stream = np.asarray(s, dtype=float).reshape(-1)
        if len(stream) < n_samples:
            stream = np.pad(stream, (0, n_samples - len(stream)))
        elif len(stream) > n_samples:
            stream = stream[:n_samples]
        _f, _t, z = _stft_1d(stream, cfg)
        stream_specs.append(z)

    # Align to min time bins in case of boundary differences.
    t_bins = min(z.shape[1] for z in mix_spec_list + stream_specs)
    mix_spec_list = [z[:, :t_bins] for z in mix_spec_list]
    stream_specs = [z[:, :t_bins] for z in stream_specs]

    stream_mag = np.stack([np.abs(z) for z in stream_specs], axis=0)  # (K, F, T)
    denom = np.sum(stream_mag, axis=0) + 1e-8

    if cfg.mask_power != 1.0:
        stream_mag = np.power(stream_mag, cfg.mask_power)
        denom = np.sum(stream_mag, axis=0) + 1e-8

    masks = stream_mag / denom[None, :, :]
    masks = np.clip(masks, cfg.mask_floor, 1.0)

    out_streams_mc: list[np.ndarray] = []
    for k in range(len(separated_streams)):
        s_mc = np.zeros((n_mics, n_samples), dtype=float)
        for m in range(n_mics):
            z_masked = masks[k] * mix_spec_list[m]
            s_mc[m] = _istft_1d(z_masked, cfg, n_samples)
        out_streams_mc.append(s_mc)

    debug = {
        "n_streams": len(separated_streams),
        "n_mics": n_mics,
        "n_samples": n_samples,
        "t_bins": t_bins,
    }
    return out_streams_mc, debug

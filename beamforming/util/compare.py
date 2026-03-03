from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.signal import correlate


def calc_rmse(ref: NDArray[Any], out: NDArray[Any]) -> tuple[float, float]:
    mse = np.mean((ref - out) ** 2, dtype=float)
    rmse = np.sqrt(mse)

    return (rmse, mse)


def rms(sig: NDArray[Any], eps: float = 1e-12) -> float:
    return float(np.sqrt(np.mean(np.asarray(sig, dtype=float) ** 2) + eps))


def match_rms_to_reference(
    pred: NDArray[Any],
    ref: NDArray[Any],
    peak_guard: float = 0.98,
    eps: float = 1e-12,
) -> tuple[NDArray[np.float64], float]:
    """
    Gain-match prediction RMS to reference RMS with a peak guard.

    Returns:
        scaled_pred, applied_gain_db
    """
    pred_f = np.asarray(pred, dtype=float)
    ref_f = np.asarray(ref, dtype=float)

    pred_rms = rms(pred_f, eps=eps)
    ref_rms = rms(ref_f, eps=eps)
    if pred_rms <= eps:
        return np.zeros_like(pred_f), 0.0

    gain = ref_rms / pred_rms
    scaled = pred_f * gain

    peak = float(np.max(np.abs(scaled)) + eps)
    if peak > peak_guard:
        limiter = peak_guard / peak
        scaled = scaled * limiter
        gain *= limiter

    gain_db = float(20.0 * np.log10(max(gain, eps)))
    return scaled.astype(np.float64), gain_db


def calc_snr(ref: NDArray[Any], out: NDArray[Any]) -> float:
    noise = ref - out

    return 10 * np.log10(np.sum(ref**2) / (np.sum(noise**2) + 1e-8))


def calc_si_sdr(ref: NDArray[Any], out: NDArray[Any]) -> float:
    # Remove DC offset 
    ref_ac = ref - np.mean(ref)
    out_ac = out - np.mean(out)

    # Project AC output audio onto AC reference audio to compute optimal scaling factor
    alpha = np.dot(out_ac, ref_ac) / (np.dot(ref_ac, ref_ac) + 1e-8)
    s_target = alpha * ref_ac
    out_noise = out_ac - s_target

    return 10 * np.log10(np.sum(s_target**2) / (np.sum(out_noise**2) + 1e-8))


def align_signals(ref, pred):
    # FFT-based cross-correlation is much faster than direct O(N^2) correlation
    # for long waveforms used in simulation benchmarking.
    corr = correlate(ref, pred, mode="full", method="fft")
    delay = np.argmax(corr) - (len(pred) - 1)
    
    # Shift reference to match prediction
    if delay > 0:
        ref_aligned = np.pad(ref, (0, delay))[:len(pred)]
    else:
        ref_aligned = ref[-delay:]
        # pad end to match length
        ref_aligned = np.pad(ref_aligned, (0, len(pred)-len(ref_aligned)))
        
    return ref_aligned

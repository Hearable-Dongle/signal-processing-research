from typing import Any

import numpy as np
from numpy.typing import NDArray


def calc_rmse(ref: NDArray[Any], out: NDArray[Any]) -> tuple[float, float]:
    # Compute mean squared error
    mse = np.mean((ref - out) ** 2, dtype=float)

    # Compute root mean squared error
    rmse = np.sqrt(mse)

    #  Return both error metrics
    return (rmse, mse)


def calc_snr(ref: NDArray[Any], out: NDArray[Any]) -> float:
    # Compute noise aspect of audio
    noise = ref - out

    # Compute and return signal to noise ratio in dB
    return 10 * np.log10(np.sum(ref**2) / (np.sum(noise**2) + 1e-8))


def calc_si_sdr(ref: NDArray[Any], out: NDArray[Any]) -> float:
    # Remove DC offset from audio to focus on AC part of signals
    ref_ac = ref - np.mean(ref)
    out_ac = out - np.mean(out)

    # Project AC output audio onto AC reference audio to compute optimal scaling factor
    alpha = np.dot(out_ac, ref_ac) / (np.dot(ref_ac, ref_ac) + 1e-8)

    # Apply scaling factor to AC reference audio
    s_target = alpha * ref_ac

    # Determine noise in AC output audio by subtracting AC reference audio
    out_noise = out_ac - s_target

    # Compute and return scale invariant signal to distortion ratio in dB
    return 10 * np.log10(np.sum(s_target**2) / (np.sum(out_noise**2) + 1e-8))

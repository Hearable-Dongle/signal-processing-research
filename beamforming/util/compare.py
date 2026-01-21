from typing import Any

import numpy as np
from numpy.typing import NDArray


def calc_rmse(ref: NDArray[Any], out: NDArray[Any]) -> tuple[float, float]:
    mse = np.mean((ref - out) ** 2, dtype=float)
    rmse = np.sqrt(mse)

    return (rmse, mse)


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
    # Cross-correlate to find delay
    corr = np.correlate(ref, pred, mode='full')
    delay = np.argmax(corr) - (len(pred) - 1)
    
    # Shift reference to match prediction
    if delay > 0:
        ref_aligned = np.pad(ref, (0, delay))[:len(pred)]
    else:
        ref_aligned = ref[-delay:]
        # pad end to match length
        ref_aligned = np.pad(ref_aligned, (0, len(pred)-len(ref_aligned)))
        
    return ref_aligned


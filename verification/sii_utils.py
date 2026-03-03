from __future__ import annotations

import numpy as np

from speech_separation.postprocessing.calculate_sii import sii

# One-third octave bands approximating existing in-repo usage.
FREQ_BANDS = [
    (141, 178),
    (178, 224),
    (224, 281),
    (281, 355),
    (355, 447),
    (447, 562),
    (562, 708),
    (708, 891),
    (891, 1122),
    (1122, 1413),
    (1413, 1778),
    (1778, 2239),
    (2239, 2818),
    (2818, 3548),
    (3548, 4467),
    (4467, 5623),
    (5623, 7079),
    (7079, 8913),
]


def _mono(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 1:
        return arr
    return np.mean(arr, axis=1 if arr.shape[1] < arr.shape[0] else 0)


def _align_pair(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = min(len(a), len(b))
    if n <= 0:
        return np.zeros(1, dtype=float), np.zeros(1, dtype=float)
    return a[:n], b[:n]


def audio_to_18_band_spectrum_level(audio: np.ndarray, sample_rate: int, n_fft: int = 4096) -> np.ndarray:
    x = _mono(audio)
    if x.size < 4:
        return np.full(18, 1e-12, dtype=float)

    win = np.hanning(min(n_fft, x.size))
    hop = max(1, win.size // 4)

    # Frame-wise average power spectrum.
    psd_acc = None
    n_frames = 0
    for start in range(0, max(1, x.size - win.size + 1), hop):
        frame = x[start : start + win.size]
        if frame.size < win.size:
            frame = np.pad(frame, (0, win.size - frame.size))
        frame = frame * win
        spec = np.fft.rfft(frame, n=n_fft)
        p = (np.abs(spec) ** 2).astype(float)
        psd_acc = p if psd_acc is None else (psd_acc + p)
        n_frames += 1

    if psd_acc is None or n_frames == 0:
        return np.full(18, 1e-12, dtype=float)

    avg_power = psd_acc / float(n_frames)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / float(sample_rate))

    band_powers: list[float] = []
    for low_f, high_f in FREQ_BANDS:
        m = (freqs >= low_f) & (freqs < high_f)
        if np.any(m):
            band_powers.append(float(np.sum(avg_power[m])))
        else:
            band_powers.append(1e-12)

    band_powers_arr = np.asarray(band_powers, dtype=float)
    band_widths = np.asarray([h - l for l, h in FREQ_BANDS], dtype=float)
    return band_powers_arr / np.maximum(band_widths, 1.0)


def compute_sii(clean_ref: np.ndarray, degraded: np.ndarray, sample_rate: int) -> float:
    ref, deg = _align_pair(_mono(clean_ref), _mono(degraded))

    ssl_level = audio_to_18_band_spectrum_level(ref, sample_rate)
    # Treat degradation residual as effective noise seen by the target speech.
    residual = deg - ref
    nsl_level = audio_to_18_band_spectrum_level(residual, sample_rate)

    eps = 1e-20
    ssl_db = 10.0 * np.log10(ssl_level + eps)
    nsl_db = 10.0 * np.log10(nsl_level + eps)

    # Calibrate peak speech to 65 dB SPL style convention used in in-repo scripts.
    peak_ssl_db = float(np.max(ssl_db))
    offset = 65.0 - peak_ssl_db

    ssl_cal = ssl_db + offset
    nsl_cal = nsl_db + offset
    hearing_threshold = np.zeros(18, dtype=float)

    val = float(sii(ssl_cal, nsl_cal, hearing_threshold))
    return float(np.clip(val, 0.0, 1.0))


def compute_delta_sii(clean_ref: np.ndarray, raw: np.ndarray, processed: np.ndarray, sample_rate: int) -> dict[str, float]:
    sii_raw = compute_sii(clean_ref, raw, sample_rate)
    sii_proc = compute_sii(clean_ref, processed, sample_rate)
    return {
        "sii_raw": float(sii_raw),
        "sii_processed": float(sii_proc),
        "delta_sii": float(sii_proc - sii_raw),
    }

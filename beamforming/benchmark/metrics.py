from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import soundfile as sf

from beamforming.util.compare import align_signals, calc_si_sdr, calc_snr
from verification.sii_utils import compute_delta_sii

try:
    from pystoi import stoi as stoi_fn
except Exception:  # pragma: no cover - optional dependency fallback
    stoi_fn = None


@dataclass(frozen=True)
class MetricBundle:
    sii_raw: float
    sii_processed: float
    delta_sii: float
    stoi_raw: float
    stoi_processed: float
    delta_stoi: float
    snr_db_raw: float
    snr_db_processed: float
    delta_snr_db: float
    si_sdr_db_raw: float
    si_sdr_db_processed: float
    delta_si_sdr_db: float


def load_audio_mono(path: str) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    return np.asarray(audio, dtype=np.float64), int(sr)


def _safe_stoi(ref: np.ndarray, deg: np.ndarray, sr: int) -> float:
    if stoi_fn is None:
        raise RuntimeError(
            "STOI metric requested but `pystoi` is not installed. "
            "Install dependencies from beamforming/requirements.txt."
        )
    if len(ref) < 256 or len(deg) < 256:
        return 0.0
    if float(np.mean(ref**2)) <= 1e-12 or float(np.mean(deg**2)) <= 1e-12:
        return 0.0
    try:
        return float(stoi_fn(ref, deg, sr, extended=False))
    except Exception:
        try:
            return float(stoi_fn(ref, deg, sr, extended=True))
        except Exception:
            return 0.0


def compute_metric_bundle(
    *,
    clean_ref: np.ndarray,
    raw_audio: np.ndarray,
    processed_audio: np.ndarray,
    sample_rate: int,
) -> MetricBundle:
    aligned_ref_raw = align_signals(clean_ref, raw_audio)
    aligned_ref_proc = align_signals(clean_ref, processed_audio)

    n_raw = min(len(aligned_ref_raw), len(raw_audio))
    n_proc = min(len(aligned_ref_proc), len(processed_audio))
    n = min(n_raw, n_proc)
    if n <= 0:
        z = 0.0
        return MetricBundle(z, z, z, z, z, z, z, z, z, z, z, z)

    ref = aligned_ref_proc[:n]
    raw = np.asarray(raw_audio[:n], dtype=np.float64)
    proc = np.asarray(processed_audio[:n], dtype=np.float64)

    sii = compute_delta_sii(ref, raw, proc, sample_rate)

    stoi_raw = _safe_stoi(ref, raw, sample_rate)
    stoi_proc = _safe_stoi(ref, proc, sample_rate)

    snr_raw = float(calc_snr(ref, raw))
    snr_proc = float(calc_snr(ref, proc))
    sdr_raw = float(calc_si_sdr(ref, raw))
    sdr_proc = float(calc_si_sdr(ref, proc))

    return MetricBundle(
        sii_raw=float(sii["sii_raw"]),
        sii_processed=float(sii["sii_processed"]),
        delta_sii=float(sii["delta_sii"]),
        stoi_raw=stoi_raw,
        stoi_processed=stoi_proc,
        delta_stoi=float(stoi_proc - stoi_raw) if np.isfinite(stoi_raw) and np.isfinite(stoi_proc) else float("nan"),
        snr_db_raw=snr_raw,
        snr_db_processed=snr_proc,
        delta_snr_db=float(snr_proc - snr_raw),
        si_sdr_db_raw=sdr_raw,
        si_sdr_db_processed=sdr_proc,
        delta_si_sdr_db=float(sdr_proc - sdr_raw),
    )

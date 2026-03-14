from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from beamforming.localization_bridge import normalize_doa_list
from localization.algo import CaponLocalization, CaponMVDRRefineLocalization, MUSICLocalization, SRPPHATLocalization


SUPPORTED_LOCALIZATION_BACKENDS = (
    "srp_phat_legacy",
    "srp_phat_localization",
    "capon_1src",
    "capon_mvdr_refine_1src",
    "music_1src",
)


def _pair_indices(n_mics: int) -> list[tuple[int, int]]:
    return [(i, j) for i in range(n_mics) for j in range(i + 1, n_mics)]


def _local_maxima(values: np.ndarray, min_separation_bins: int, max_peaks: int) -> tuple[list[int], list[float]]:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        return [], []
    peaks: list[tuple[float, int]] = []
    for idx in range(arr.size):
        prev_v = arr[(idx - 1) % arr.size]
        cur_v = arr[idx]
        next_v = arr[(idx + 1) % arr.size]
        if cur_v > prev_v and cur_v >= next_v:
            peaks.append((float(cur_v), int(idx)))
    peaks.sort(key=lambda item: item[0], reverse=True)

    picked: list[int] = []
    picked_scores: list[float] = []
    for score, idx in peaks:
        if any(min(abs(idx - p), arr.size - abs(idx - p)) < min_separation_bins for p in picked):
            continue
        picked.append(int(idx))
        picked_scores.append(float(score))
        if len(picked) >= max_peaks:
            break
    return picked, picked_scores


def _normalize_spectrum(values: np.ndarray | None) -> np.ndarray | None:
    if values is None:
        return None
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return arr
    arr = np.clip(arr, 0.0, None)
    vmax = float(np.max(arr))
    if vmax <= 0.0:
        return np.zeros_like(arr)
    return arr / vmax


@dataclass(frozen=True, slots=True)
class LocalizationBackendResult:
    peaks_deg: list[float]
    peak_scores: list[float]
    score_spectrum: np.ndarray | None
    debug: dict


class LocalizationBackendBase:
    def __init__(
        self,
        *,
        mic_pos: np.ndarray,
        fs: int,
        nfft: int,
        overlap: float,
        freq_range: tuple[int, int],
        max_sources: int,
        sound_speed_m_s: float = 343.0,
        grid_size: int = 72,
        min_separation_deg: float = 15.0,
        small_aperture_bias: bool = True,
        pair_selection_mode: str = "all",
        vad_enabled: bool = True,
        capon_spectrum_ema_alpha: float = 0.78,
        capon_peak_min_sharpness: float = 0.12,
        capon_peak_min_margin: float = 0.04,
        capon_hold_frames: int = 2,
    ):
        self.mic_pos = np.asarray(mic_pos, dtype=np.float64)
        self.fs = int(fs)
        self.nfft = int(nfft)
        self.overlap = float(overlap)
        self.freq_range = tuple(int(v) for v in freq_range)
        self.max_sources = int(max_sources)
        self.sound_speed_m_s = float(sound_speed_m_s)
        self.grid_size = int(grid_size)
        self.min_separation_deg = float(min_separation_deg)
        self.small_aperture_bias = bool(small_aperture_bias)
        self.pair_selection_mode = str(pair_selection_mode)
        self.vad_enabled = bool(vad_enabled)
        self.capon_spectrum_ema_alpha = float(capon_spectrum_ema_alpha)
        self.capon_peak_min_sharpness = float(capon_peak_min_sharpness)
        self.capon_peak_min_margin = float(capon_peak_min_margin)
        self.capon_hold_frames = int(capon_hold_frames)

    def process(self, audio: np.ndarray) -> LocalizationBackendResult:
        raise NotImplementedError

    def _peaks_from_spectrum(self, spectrum: np.ndarray) -> tuple[list[float], list[float]]:
        spec = np.asarray(spectrum, dtype=np.float64).reshape(-1)
        if spec.size != self.grid_size:
            x_old = np.linspace(0.0, 1.0, spec.size, endpoint=False)
            x_new = np.linspace(0.0, 1.0, self.grid_size, endpoint=False)
            spec = np.interp(x_new, x_old, spec)
        spec = _normalize_spectrum(spec)
        if spec is None:
            return [], []
        peak_idx, peak_scores = _local_maxima(
            spec,
            min_separation_bins=max(1, int(round(self.min_separation_deg / (360.0 / self.grid_size)))),
            max_peaks=max(1, self.max_sources),
        )
        angles = np.linspace(0.0, 360.0, self.grid_size, endpoint=False, dtype=np.float64)
        peaks_deg = normalize_doa_list([float(angles[idx]) for idx in peak_idx], max_targets=self.max_sources)
        return peaks_deg, [float(v) for v in peak_scores[: len(peaks_deg)]]


class _LocalizationAlgoAdapter(LocalizationBackendBase):
    def __init__(self, *, backend_name: str, algo_cls, **kwargs):
        super().__init__(**kwargs)
        self.backend_name = str(backend_name)
        self._localizer = algo_cls(
            mic_pos=self.mic_pos,
            fs=self.fs,
            nfft=self.nfft,
            overlap=self.overlap,
            freq_range=self.freq_range,
            max_sources=self.max_sources,
            pair_selection_mode=self.pair_selection_mode,
            vad_enabled=self.vad_enabled,
            capon_spectrum_ema_alpha=self.capon_spectrum_ema_alpha,
            capon_peak_min_sharpness=self.capon_peak_min_sharpness,
            capon_peak_min_margin=self.capon_peak_min_margin,
            capon_hold_frames=self.capon_hold_frames,
        )

    def process(self, audio: np.ndarray) -> LocalizationBackendResult:
        doas_rad, p_theta, history = self._localizer.process(audio)
        peaks_deg = normalize_doa_list([float(np.degrees(v) % 360.0) for v in doas_rad], max_targets=self.max_sources)
        spec = _normalize_spectrum(p_theta)
        explicit_peak_scores = list(getattr(self._localizer, "last_peak_scores", []) or [])
        peak_scores: list[float] = []
        if explicit_peak_scores:
            peak_scores = [float(v) for v in explicit_peak_scores[: len(peaks_deg)]]
        elif spec is not None and spec.size > 0:
            for doa in peaks_deg:
                idx = int(round((float(doa) % 360.0) / 360.0 * (spec.size - 1)))
                idx = max(0, min(spec.size - 1, idx))
                peak_scores.append(float(spec[idx]))
        return LocalizationBackendResult(
            peaks_deg=peaks_deg,
            peak_scores=peak_scores,
            score_spectrum=spec,
            debug={
                "backend": self.backend_name,
                "spectrum_bins": int(np.asarray(p_theta).size if p_theta is not None else 0),
                "history_len": int(len(history) if history is not None else 0),
                "localization_source": "localization.algo",
                **(dict(getattr(self._localizer, "last_debug", {})) if getattr(self._localizer, "last_debug", None) else {}),
            },
        )


class LegacySRPBackend(_LocalizationAlgoAdapter):
    def __init__(self, *, backend_name: str = "srp_phat_legacy", **kwargs):
        super().__init__(backend_name=backend_name, algo_cls=SRPPHATLocalization, **kwargs)


class Music1SrcBackend(_LocalizationAlgoAdapter):
    def __init__(self, *, backend_name: str = "music_1src", **kwargs):
        super().__init__(backend_name=backend_name, algo_cls=MUSICLocalization, **kwargs)


class Capon1SrcBackend(_LocalizationAlgoAdapter):
    def __init__(self, *, backend_name: str = "capon_1src", **kwargs):
        super().__init__(backend_name=backend_name, algo_cls=CaponLocalization, **kwargs)


class CaponMVDRRefine1SrcBackend(_LocalizationAlgoAdapter):
    def __init__(self, *, backend_name: str = "capon_mvdr_refine_1src", **kwargs):
        super().__init__(backend_name=backend_name, algo_cls=CaponMVDRRefineLocalization, **kwargs)


def build_localization_backend(
    name: str,
    *,
    mic_pos: np.ndarray,
    fs: int,
    nfft: int,
    overlap: float,
    freq_range: tuple[int, int],
    max_sources: int,
    sound_speed_m_s: float = 343.0,
    grid_size: int = 72,
    min_separation_deg: float = 15.0,
    small_aperture_bias: bool = True,
    pair_selection_mode: str = "all",
    vad_enabled: bool = True,
    capon_spectrum_ema_alpha: float = 0.78,
    capon_peak_min_sharpness: float = 0.12,
    capon_peak_min_margin: float = 0.04,
    capon_hold_frames: int = 2,
) -> LocalizationBackendBase:
    common = dict(
        mic_pos=mic_pos,
        fs=fs,
        nfft=nfft,
        overlap=overlap,
        freq_range=freq_range,
        max_sources=max_sources,
        sound_speed_m_s=sound_speed_m_s,
        grid_size=grid_size,
        min_separation_deg=min_separation_deg,
        small_aperture_bias=small_aperture_bias,
        pair_selection_mode=pair_selection_mode,
        vad_enabled=vad_enabled,
        capon_spectrum_ema_alpha=capon_spectrum_ema_alpha,
        capon_peak_min_sharpness=capon_peak_min_sharpness,
        capon_peak_min_margin=capon_peak_min_margin,
        capon_hold_frames=capon_hold_frames,
    )
    name = str(name)
    if name in {"srp_phat_legacy", "srp_phat_localization"}:
        return LegacySRPBackend(backend_name=name, **common)
    if name == "capon_1src":
        return Capon1SrcBackend(backend_name=name, **common)
    if name == "capon_mvdr_refine_1src":
        return CaponMVDRRefine1SrcBackend(backend_name=name, **common)
    if name == "music_1src":
        return Music1SrcBackend(backend_name=name, **common)
    supported = ", ".join(SUPPORTED_LOCALIZATION_BACKENDS)
    raise ValueError(
        f"Unsupported localization backend '{name}'. "
        f"realtime_pipeline now delegates localization to localization/. "
        f"Supported backends: {supported}."
    )

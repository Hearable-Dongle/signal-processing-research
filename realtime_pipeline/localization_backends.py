from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal

from beamforming.localization_bridge import normalize_doa_list
from localization.algo import SRPPHATLocalization


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
        if any(min((abs(idx - p), arr.size - abs(idx - p))) < min_separation_bins for p in picked):
            continue
        picked.append(int(idx))
        picked_scores.append(float(score))
        if len(picked) >= max_peaks:
            break
    return picked, picked_scores


def _grid_angles_deg(grid_size: int) -> np.ndarray:
    return np.linspace(0.0, 360.0, int(grid_size), endpoint=False, dtype=np.float64)


def _angular_error_deg(a: float, b: float) -> float:
    return float(abs((float(a) - float(b) + 180.0) % 360.0 - 180.0))


def _stft_roi(audio: np.ndarray, fs: int, nfft: int, overlap: float, freq_range: tuple[int, int]) -> tuple[np.ndarray, np.ndarray] | None:
    f_vec, _t_vec, zxx = signal.stft(
        audio,
        fs=fs,
        nperseg=nfft,
        noverlap=int(nfft * overlap),
        boundary=None,
        padded=False,
    )
    f_min, f_max = freq_range
    f_mask = (f_vec >= f_min) & (f_vec <= f_max)
    if not np.any(f_mask):
        return None
    z = zxx[:, f_mask, :]
    if z.shape[1] == 0 or z.shape[2] == 0:
        return None
    return f_vec[f_mask].astype(np.float64), z.astype(np.complex128)


def _speech_frame_mask(zxx_roi: np.ndarray) -> np.ndarray:
    frame_energy = np.sum(np.sum(np.abs(zxx_roi) ** 2, axis=0), axis=0)
    if frame_energy.size == 0:
        return np.zeros(0, dtype=bool)
    thresh = np.percentile(frame_energy, 55.0)
    active = frame_energy >= thresh
    if not np.any(active):
        active = frame_energy >= np.percentile(frame_energy, 25.0)
    return active


def _pair_indices(n_mics: int) -> list[tuple[int, int]]:
    return [(i, j) for i in range(n_mics) for j in range(i + 1, n_mics)]


def _pair_delays(mic_pos: np.ndarray, freqs: np.ndarray, grid_size: int, sound_speed_m_s: float) -> tuple[list[tuple[int, int]], np.ndarray]:
    pairs = _pair_indices(mic_pos.shape[1])
    angles_rad = np.deg2rad(_grid_angles_deg(grid_size))
    delays = np.zeros((len(pairs), freqs.size, grid_size), dtype=np.float64)
    for p_idx, (i, j) in enumerate(pairs):
        diff = mic_pos[:, j] - mic_pos[:, i]
        for a_idx, ang in enumerate(angles_rad):
            u = np.array([np.cos(ang), np.sin(ang), 0.0], dtype=np.float64)
            tau = float(np.dot(diff, u) / sound_speed_m_s)
            delays[p_idx, :, a_idx] = 2.0 * np.pi * freqs * tau
    return pairs, delays


def _band_weight(freqs: np.ndarray, small_aperture_bias: bool) -> np.ndarray:
    f = np.asarray(freqs, dtype=np.float64)
    w = np.ones_like(f)
    if small_aperture_bias:
        # Favor lower/mid speech bands for small ReSpeaker apertures.
        w *= np.exp(-np.maximum(0.0, f - 1800.0) / 1500.0)
        w *= np.clip((f - 150.0) / 250.0, 0.0, 1.0)
    else:
        w *= np.sqrt(np.maximum(f, 1.0))
    if np.max(w) > 0:
        w /= np.max(w)
    return w


@dataclass(frozen=True, slots=True)
class LocalizationBackendResult:
    peaks_deg: list[float]
    peak_scores: list[float]
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

    def process(self, audio: np.ndarray) -> LocalizationBackendResult:
        raise NotImplementedError

    def _peaks_from_spectrum(self, spectrum: np.ndarray) -> tuple[list[float], list[float]]:
        spec = np.asarray(spectrum, dtype=np.float64).reshape(-1)
        if spec.size != self.grid_size:
            x_old = np.linspace(0.0, 1.0, spec.size, endpoint=False)
            x_new = np.linspace(0.0, 1.0, self.grid_size, endpoint=False)
            spec = np.interp(x_new, x_old, spec)
        spec[spec < 0] = 0.0
        if np.max(spec) > 0:
            spec = spec / np.max(spec)
        peak_idx, peak_scores = _local_maxima(
            spec,
            min_separation_bins=max(1, int(round(self.min_separation_deg / (360.0 / self.grid_size)))),
            max_peaks=max(1, self.max_sources),
        )
        angles = _grid_angles_deg(self.grid_size)
        peaks_deg = normalize_doa_list([float(angles[idx]) for idx in peak_idx], max_targets=self.max_sources)
        return peaks_deg, [float(v) for v in peak_scores[: len(peaks_deg)]]


class LegacySRPBackend(LocalizationBackendBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._localizer = SRPPHATLocalization(
            mic_pos=self.mic_pos,
            fs=self.fs,
            nfft=self.nfft,
            overlap=self.overlap,
            freq_range=self.freq_range,
            max_sources=self.max_sources,
        )

    def process(self, audio: np.ndarray) -> LocalizationBackendResult:
        doas_rad, p_theta, _ = self._localizer.process(audio)
        peaks_deg = normalize_doa_list([float(np.degrees(v) % 360.0) for v in doas_rad], max_targets=self.max_sources)
        scores = []
        if p_theta is not None and len(p_theta) > 0:
            spec = np.asarray(p_theta, dtype=np.float64).reshape(-1)
            if np.max(spec) > 0:
                spec = spec / np.max(spec)
            for d in peaks_deg:
                idx = int(round((d % 360.0) / 360.0 * (spec.size - 1)))
                scores.append(float(spec[idx]))
        return LocalizationBackendResult(
            peaks_deg=peaks_deg,
            peak_scores=scores,
            debug={"backend": "srp_phat_legacy", "spectrum_bins": int(np.asarray(p_theta).size if p_theta is not None else 0)},
        )


class WeightedSRPDPBackend(LocalizationBackendBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._legacy = LegacySRPBackend(**kwargs)

    def _resolve_hemisphere(self, peaks_deg: list[float], peak_scores: list[float], debug: dict) -> tuple[list[float], dict]:
        if not peaks_deg:
            return peaks_deg, debug
        legacy = self._legacy.process(self._last_audio)
        if not legacy.peaks_deg:
            return peaks_deg, debug
        anchor = float(legacy.peaks_deg[0])
        primary = float(peaks_deg[0])
        flipped = float((primary + 180.0) % 360.0)
        if _angular_error_deg(flipped, anchor) + 5.0 < _angular_error_deg(primary, anchor):
            debug["hemisphere_resolved_from_legacy"] = True
            debug["legacy_anchor_deg"] = anchor
            return [float((p + 180.0) % 360.0) for p in peaks_deg], debug
        debug["hemisphere_resolved_from_legacy"] = False
        debug["legacy_anchor_deg"] = anchor
        return peaks_deg, debug

    def process(self, audio: np.ndarray) -> LocalizationBackendResult:
        self._last_audio = np.asarray(audio, dtype=np.float64)
        roi = _stft_roi(audio, self.fs, self.nfft, self.overlap, self.freq_range)
        if roi is None:
            return LocalizationBackendResult(peaks_deg=[], peak_scores=[], debug={"backend": "weighted_srp_dp", "reason": "empty_roi"})
        freqs, zxx_roi = roi
        active_mask = _speech_frame_mask(zxx_roi)
        if active_mask.size == 0 or not np.any(active_mask):
            active_mask = np.ones(zxx_roi.shape[2], dtype=bool)
        z_active = zxx_roi[:, :, active_mask]
        pairs, delays = _pair_delays(self.mic_pos, freqs, self.grid_size, self.sound_speed_m_s)
        band_weight = _band_weight(freqs, self.small_aperture_bias)

        spectrum = np.zeros(self.grid_size, dtype=np.float64)
        pair_reliability: list[float] = []
        for p_idx, (i, j) in enumerate(pairs):
            prod = z_active[i] * np.conj(z_active[j])  # (F, T)
            denom = np.abs(prod)
            norm_prod = prod / np.maximum(denom, 1e-10)
            mean_vec = np.mean(norm_prod, axis=1)
            coherence = np.clip(np.abs(mean_vec), 0.0, 1.0)
            phase_var = np.mean(np.abs(norm_prod - mean_vec[:, None]), axis=1)
            direct_weight = coherence * np.exp(-phase_var)
            weight = direct_weight * band_weight
            pair_reliability.append(float(np.mean(weight)))
            phase = np.angle(mean_vec)[:, None]
            score_pf = np.cos(phase - delays[p_idx]) * weight[:, None]
            spectrum += np.sum(np.maximum(score_pf, 0.0), axis=0)

        peaks_deg, peak_scores = self._peaks_from_spectrum(spectrum)
        debug = {
            "backend": "weighted_srp_dp",
            "active_frames": int(np.sum(active_mask)),
            "pair_reliability_mean": float(np.mean(pair_reliability)) if pair_reliability else 0.0,
            "grid_size": int(self.grid_size),
        }
        peaks_deg, debug = self._resolve_hemisphere(peaks_deg, peak_scores, debug)
        return LocalizationBackendResult(
            peaks_deg=peaks_deg,
            peak_scores=peak_scores,
            debug=debug,
        )


class TinyDPIPDBackend(LocalizationBackendBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._srp = WeightedSRPDPBackend(**kwargs)
        self._legacy = LegacySRPBackend(**kwargs)

    def process(self, audio: np.ndarray) -> LocalizationBackendResult:
        roi = _stft_roi(audio, self.fs, self.nfft, self.overlap, self.freq_range)
        if roi is None:
            return LocalizationBackendResult(peaks_deg=[], peak_scores=[], debug={"backend": "tiny_dp_ipd", "reason": "empty_roi"})
        freqs, zxx_roi = roi
        active_mask = _speech_frame_mask(zxx_roi)
        if active_mask.size == 0 or not np.any(active_mask):
            active_mask = np.ones(zxx_roi.shape[2], dtype=bool)
        z_active = zxx_roi[:, :, active_mask]
        pairs, delays = _pair_delays(self.mic_pos, freqs, self.grid_size, self.sound_speed_m_s)
        band_weight = _band_weight(freqs, self.small_aperture_bias)

        dp_spectrum = np.zeros(self.grid_size, dtype=np.float64)
        reliability_trace: list[float] = []
        for p_idx, (i, j) in enumerate(pairs):
            prod = z_active[i] * np.conj(z_active[j])
            denom = np.abs(prod)
            norm_prod = prod / np.maximum(denom, 1e-10)
            mean_vec = np.mean(norm_prod, axis=1)
            phase = np.angle(mean_vec)[:, None]
            coherence = np.clip(np.abs(mean_vec), 0.0, 1.0)
            magnitude = np.mean(denom, axis=1)
            magnitude /= np.max(magnitude) + 1e-10
            # Tiny "front-end": direct-path reliability from phase coherence, speech-band energy, and small-aperture prior.
            direct_reliability = np.sqrt(coherence) * np.sqrt(np.clip(magnitude, 0.0, 1.0)) * band_weight
            reliability_trace.append(float(np.mean(direct_reliability)))
            pair_score = 0.5 * (1.0 + np.cos(phase - delays[p_idx]))
            dp_spectrum += np.sum(pair_score * direct_reliability[:, None], axis=0)

        srp_result = self._srp.process(audio)
        srp_debug = dict(srp_result.debug)
        roi2 = _stft_roi(audio, self.fs, self.nfft, self.overlap, self.freq_range)
        if roi2 is None:
            fused = dp_spectrum
        else:
            freqs2, zxx_roi2 = roi2
            active_mask2 = _speech_frame_mask(zxx_roi2)
            if active_mask2.size == 0 or not np.any(active_mask2):
                active_mask2 = np.ones(zxx_roi2.shape[2], dtype=bool)
            z_active2 = zxx_roi2[:, :, active_mask2]
            pairs2, delays2 = _pair_delays(self.mic_pos, freqs2, self.grid_size, self.sound_speed_m_s)
            band_weight2 = _band_weight(freqs2, self.small_aperture_bias)
            srp_spectrum = np.zeros(self.grid_size, dtype=np.float64)
            for p_idx, (i, j) in enumerate(pairs2):
                prod = z_active2[i] * np.conj(z_active2[j])
                denom = np.abs(prod)
                norm_prod = prod / np.maximum(denom, 1e-10)
                mean_vec = np.mean(norm_prod, axis=1)
                coherence = np.clip(np.abs(mean_vec), 0.0, 1.0)
                weight = coherence * band_weight2
                score_pf = np.cos(np.angle(mean_vec)[:, None] - delays2[p_idx]) * weight[:, None]
                srp_spectrum += np.sum(np.maximum(score_pf, 0.0), axis=0)
            if np.max(dp_spectrum) > 0:
                dp_norm = dp_spectrum / np.max(dp_spectrum)
            else:
                dp_norm = dp_spectrum
            if np.max(srp_spectrum) > 0:
                srp_norm = srp_spectrum / np.max(srp_spectrum)
            else:
                srp_norm = srp_spectrum
            fused = (0.65 * dp_norm) + (0.35 * srp_norm)

        peaks_deg, peak_scores = self._peaks_from_spectrum(fused)
        debug = {
            "backend": "tiny_dp_ipd",
            "active_frames": int(np.sum(active_mask)),
            "direct_path_reliability_mean": float(np.mean(reliability_trace)) if reliability_trace else 0.0,
            "srp_debug": srp_debug,
            "grid_size": int(self.grid_size),
        }
        legacy = self._legacy.process(audio)
        if peaks_deg and legacy.peaks_deg:
            anchor = float(legacy.peaks_deg[0])
            primary = float(peaks_deg[0])
            flipped = float((primary + 180.0) % 360.0)
            if _angular_error_deg(flipped, anchor) + 5.0 < _angular_error_deg(primary, anchor):
                peaks_deg = [float((p + 180.0) % 360.0) for p in peaks_deg]
                debug["hemisphere_resolved_from_legacy"] = True
            else:
                debug["hemisphere_resolved_from_legacy"] = False
            debug["legacy_anchor_deg"] = anchor
        return LocalizationBackendResult(
            peaks_deg=peaks_deg,
            peak_scores=peak_scores,
            debug=debug,
        )


def build_localization_backend(
    backend: str,
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
) -> LocalizationBackendBase:
    common = dict(
        mic_pos=np.asarray(mic_pos, dtype=np.float64),
        fs=int(fs),
        nfft=int(nfft),
        overlap=float(overlap),
        freq_range=tuple(int(v) for v in freq_range),
        max_sources=int(max_sources),
        sound_speed_m_s=float(sound_speed_m_s),
        grid_size=int(grid_size),
        min_separation_deg=float(min_separation_deg),
        small_aperture_bias=bool(small_aperture_bias),
    )
    name = str(backend).strip().lower()
    if name == "srp_phat_legacy":
        return LegacySRPBackend(**common)
    if name == "weighted_srp_dp":
        return WeightedSRPDPBackend(**common)
    if name == "tiny_dp_ipd":
        return TinyDPIPDBackend(**common)
    raise ValueError(f"Unsupported localization backend: {backend}")

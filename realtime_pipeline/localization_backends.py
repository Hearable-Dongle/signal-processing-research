from __future__ import annotations

from collections import deque
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


def _normalize_spectrum(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return arr
    arr = np.clip(arr, 0.0, None)
    vmax = float(np.max(arr))
    if vmax <= 0.0:
        return np.zeros_like(arr)
    return arr / vmax


def _dominant_peak_idx(values: np.ndarray) -> int | None:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0 or float(np.max(arr)) <= 0.0:
        return None
    return int(np.argmax(arr))


def _peak_contrasts(values: np.ndarray, min_separation_bins: int) -> tuple[float, float]:
    arr = _normalize_spectrum(values)
    if arr.size == 0:
        return 0.0, 0.0
    peak_idx, peak_scores = _local_maxima(arr, min_separation_bins=max(1, min_separation_bins), max_peaks=2)
    if not peak_idx:
        return 0.0, 0.0
    base = float(np.median(arr))
    dominant = float(max(0.0, peak_scores[0] - base))
    secondary = float(max(0.0, peak_scores[1] - base)) if len(peak_scores) > 1 else 0.0
    return dominant, secondary


def _circular_smooth(values: np.ndarray, radius_bins: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0 or radius_bins <= 0:
        return arr.copy()
    offsets = np.arange(-int(radius_bins), int(radius_bins) + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * np.square(offsets / max(1.0, float(radius_bins) / 2.0)))
    kernel /= np.sum(kernel)
    out = np.zeros_like(arr)
    for idx, weight in enumerate(kernel):
        shift = int(idx - radius_bins)
        out += float(weight) * np.roll(arr, shift)
    return out


def _angular_bin_distance(a_idx: int, b_idx: int, n_bins: int) -> int:
    diff = abs(int(a_idx) - int(b_idx))
    return int(min(diff, int(n_bins) - diff))


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
        spec = None
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
            score_spectrum=spec,
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
            return LocalizationBackendResult(peaks_deg=[], peak_scores=[], score_spectrum=None, debug={"backend": "weighted_srp_dp", "reason": "empty_roi"})
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
            score_spectrum=np.asarray(spectrum, dtype=np.float64),
            debug=debug,
        )


class TinyDPIPDBackend(LocalizationBackendBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._srp = WeightedSRPDPBackend(**kwargs)
        self._legacy = LegacySRPBackend(**kwargs)
        self._dominant_history: deque[np.ndarray] = deque(maxlen=4)
        self._residual_history: deque[np.ndarray] = deque(maxlen=4)
        self._fused_history: deque[np.ndarray] = deque(maxlen=4)
        self._history_weights = np.asarray([1.0, 0.7, 0.45, 0.25], dtype=np.float64)
        self._secondary_min_score = 0.32
        self._secondary_min_contrast = 0.10
        self._secondary_min_separation_deg = 20.0
        self._secondary_temporal_support_min = 0.18
        self._secondary_consistency_window = 3
        self._secondary_consistency_min_hits = 2
        self._secondary_consistency_tolerance_deg = 18.0
        self._secondary_history_score_floor = 0.22
        self._primary_exclusion_radius_deg = 18.0
        self._final_history_blend = 0.28

    def _build_temporal_prior(self, current_primary_idx: int | None) -> np.ndarray:
        if current_primary_idx is None or not self._fused_history:
            return np.zeros(self.grid_size, dtype=np.float64)

        accum = np.zeros(self.grid_size, dtype=np.float64)
        weight_sum = 0.0
        hist_len = min(len(self._fused_history), len(self._history_weights))
        blur_radius = max(1, int(round(10.0 / (360.0 / self.grid_size))))
        for hist_idx in range(hist_len):
            weight = float(self._history_weights[hist_idx])
            dom_hist = np.asarray(self._dominant_history[-1 - hist_idx], dtype=np.float64)
            res_hist = np.asarray(self._residual_history[-1 - hist_idx], dtype=np.float64)
            fused_hist = np.asarray(self._fused_history[-1 - hist_idx], dtype=np.float64)
            hist_primary_idx = _dominant_peak_idx(dom_hist)
            if hist_primary_idx is None:
                shift = 0
            else:
                shift = int(current_primary_idx - hist_primary_idx)
            hist_map = (0.55 * fused_hist) + (0.25 * res_hist) + (0.20 * dom_hist)
            hist_map = _normalize_spectrum(np.roll(hist_map, shift))
            hist_map = _circular_smooth(hist_map, blur_radius)
            accum += weight * hist_map
            weight_sum += weight
        if weight_sum <= 0.0:
            return np.zeros(self.grid_size, dtype=np.float64)
        return _normalize_spectrum(accum / weight_sum)

    def _promote_secondary_peak(
        self,
        dominant_map: np.ndarray,
        residual_map: np.ndarray,
        temporal_prior: np.ndarray,
    ) -> tuple[np.ndarray, dict]:
        dominant_map = _normalize_spectrum(dominant_map)
        residual_map = _normalize_spectrum(residual_map)
        temporal_prior = _normalize_spectrum(temporal_prior)
        primary_idx = _dominant_peak_idx(dominant_map)
        if primary_idx is None:
            return np.zeros(self.grid_size, dtype=np.float64), {
                "secondary_candidate_angle_deg": None,
                "secondary_gate_passed": False,
                "secondary_gate_reasons": ["no_primary"],
                "promoted_secondary_gain": 0.0,
                "secondary_consistency_hits": 0,
                "temporal_prior_spectrum": [float(v) for v in temporal_prior],
            }

        min_sep_bins = max(1, int(round(self._secondary_min_separation_deg / (360.0 / self.grid_size))))
        primary_exclusion_bins = max(1, int(round(self._primary_exclusion_radius_deg / (360.0 / self.grid_size))))
        peaks, _scores = _local_maxima(residual_map, min_separation_bins=min_sep_bins, max_peaks=max(3, self.max_sources + 1))
        base = float(np.median(residual_map))
        candidate_idx = None
        candidate_score = 0.0
        candidate_contrast = 0.0
        candidate_temporal_support = 0.0
        candidate_ratio = 0.0
        candidate_consistency_hits = 0
        candidate_consistency_support = 0.0
        gate_reasons: list[str] = []
        for idx in peaks:
            if _angular_bin_distance(idx, primary_idx, self.grid_size) < primary_exclusion_bins:
                continue
            score = float(residual_map[idx])
            contrast = float(max(0.0, score - base))
            temporal_support = float(temporal_prior[idx])
            support_ratio = float(score / max(1e-6, dominant_map[idx]))
            outside_primary_basin = float(dominant_map[idx]) < 0.78
            strong_current = score >= (self._secondary_min_score + 0.08) and contrast >= (self._secondary_min_contrast + 0.03)
            if score < self._secondary_min_score:
                continue
            if contrast < self._secondary_min_contrast:
                continue
            if not outside_primary_basin:
                continue
            if temporal_support < self._secondary_temporal_support_min and not strong_current:
                continue
            consistency_hits, consistency_support = self._secondary_consistency_support(
                candidate_idx=int(idx),
                current_primary_idx=primary_idx,
            )
            if consistency_hits < self._secondary_consistency_min_hits:
                continue
            candidate_idx = int(idx)
            candidate_score = score
            candidate_contrast = contrast
            candidate_temporal_support = temporal_support
            candidate_ratio = support_ratio
            candidate_consistency_hits = consistency_hits
            candidate_consistency_support = consistency_support
            gate_reasons = [f"consistency_hits={consistency_hits}", f"consistency_support={consistency_support:.3f}"]
            break

        if candidate_idx is None:
            gate_reasons = ["weak_or_unstable_secondary"]
            return np.zeros(self.grid_size, dtype=np.float64), {
                "secondary_candidate_angle_deg": None,
                "secondary_gate_passed": False,
                "secondary_gate_reasons": gate_reasons,
                "promoted_secondary_gain": 0.0,
                "secondary_consistency_hits": 0,
                "temporal_prior_spectrum": [float(v) for v in temporal_prior],
            }

        promotion_gain = 0.25
        if candidate_temporal_support >= 0.28:
            promotion_gain = 0.45
        elif candidate_temporal_support >= self._secondary_temporal_support_min:
            promotion_gain = 0.35

        width_bins = max(2, int(round(13.0 / (360.0 / self.grid_size))))
        promotion_mask = np.zeros(self.grid_size, dtype=np.float64)
        for idx in range(self.grid_size):
            dist = _angular_bin_distance(idx, candidate_idx, self.grid_size)
            if dist > width_bins:
                continue
            promotion_mask[idx] = np.exp(-0.5 * np.square(dist / max(1.0, width_bins / 2.0)))
            if _angular_bin_distance(idx, primary_idx, self.grid_size) < primary_exclusion_bins:
                promotion_mask[idx] = 0.0

        promoted = promotion_gain * promotion_mask * residual_map
        primary_level = float(dominant_map[primary_idx])
        max_secondary = float(np.max(promoted))
        if max_secondary > 0.82 * primary_level:
            promoted *= float((0.82 * primary_level) / max_secondary)
        return promoted, {
            "secondary_candidate_angle_deg": float(_grid_angles_deg(self.grid_size)[candidate_idx]),
            "secondary_gate_passed": True,
            "secondary_gate_reasons": [f"score={candidate_score:.3f}", f"contrast={candidate_contrast:.3f}", f"temporal={candidate_temporal_support:.3f}", f"ratio={candidate_ratio:.3f}", *gate_reasons],
            "promoted_secondary_gain": float(promotion_gain),
            "secondary_consistency_hits": int(candidate_consistency_hits),
            "secondary_consistency_support": float(candidate_consistency_support),
            "temporal_prior_spectrum": [float(v) for v in temporal_prior],
        }

    def _secondary_consistency_support(self, *, candidate_idx: int, current_primary_idx: int) -> tuple[int, float]:
        if not self._residual_history or not self._dominant_history:
            return 0, 0.0
        tolerance_bins = max(1, int(round(self._secondary_consistency_tolerance_deg / (360.0 / self.grid_size))))
        hist_len = min(len(self._residual_history), len(self._dominant_history), self._secondary_consistency_window)
        hits = 0
        support_values: list[float] = []
        for hist_idx in range(hist_len):
            hist_res = np.asarray(self._residual_history[-1 - hist_idx], dtype=np.float64)
            hist_dom = np.asarray(self._dominant_history[-1 - hist_idx], dtype=np.float64)
            hist_primary_idx = _dominant_peak_idx(hist_dom)
            shift = 0 if hist_primary_idx is None else int(current_primary_idx - hist_primary_idx)
            aligned_res = _normalize_spectrum(np.roll(hist_res, shift))
            peaks, _scores = _local_maxima(aligned_res, min_separation_bins=tolerance_bins, max_peaks=max(3, self.max_sources + 1))
            matched = False
            for peak_idx in peaks:
                if _angular_bin_distance(peak_idx, candidate_idx, self.grid_size) > tolerance_bins:
                    continue
                peak_score = float(aligned_res[peak_idx])
                if peak_score < self._secondary_history_score_floor:
                    continue
                hits += 1
                support_values.append(peak_score)
                matched = True
                break
            if not matched:
                support_values.append(float(aligned_res[candidate_idx]))
        mean_support = float(np.mean(support_values)) if support_values else 0.0
        return int(hits), mean_support

    def _accumulate_direct_path_spectra(
        self,
        z_active: np.ndarray,
        delays: np.ndarray,
        band_weight: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, list[float]]:
        dp_spectrum = np.zeros(self.grid_size, dtype=np.float64)
        residual_spectrum = np.zeros(self.grid_size, dtype=np.float64)
        pair_reliability: list[float] = []

        pair_count = delays.shape[0]
        for p_idx, (i, j) in enumerate(_pair_indices(z_active.shape[0])):
            prod = z_active[i] * np.conj(z_active[j])  # (F, T)
            denom = np.abs(prod)
            if denom.size == 0:
                pair_reliability.append(0.0)
                continue
            norm_prod = prod / np.maximum(denom, 1e-10)
            mean_vec = np.mean(norm_prod, axis=1)
            coherence_f = np.clip(np.abs(mean_vec), 0.0, 1.0)[:, None]

            mag_ft = denom / (np.max(denom, axis=1, keepdims=True) + 1e-10)
            phase_ft = np.angle(norm_prod)  # (F, T)
            score_fta = 0.5 * (1.0 + np.cos(phase_ft[:, :, None] - delays[p_idx][:, None, :]))
            score_fta = np.clip(score_fta, 0.0, 1.0)

            # Reward cells with a clear directional preference instead of broad lobes.
            sharpness_ft = np.max(score_fta, axis=2) - np.mean(score_fta, axis=2)
            sharpness_ft = np.clip(sharpness_ft, 0.0, 1.0)

            direct_weight_ft = (
                np.sqrt(coherence_f)
                * np.sqrt(np.clip(mag_ft, 0.0, 1.0))
                * np.power(np.clip(sharpness_ft, 0.0, 1.0), 0.8)
                * band_weight[:, None]
            )
            pair_reliability.append(float(np.mean(direct_weight_ft)))

            # Compress very dominant cells so weak secondary evidence survives aggregation.
            comp_score_fta = np.sqrt(score_fta)
            dp_spectrum += np.sum(comp_score_fta * direct_weight_ft[:, :, None], axis=(0, 1))

            primary_idx = _dominant_peak_idx(np.sum(comp_score_fta * direct_weight_ft[:, :, None], axis=(0, 1)))
            if primary_idx is None:
                continue
            primary_support_ft = score_fta[:, :, primary_idx]
            residual_gate_ft = np.power(np.clip(1.0 - primary_support_ft, 0.0, 1.0), 1.5)
            residual_spectrum += np.sum(comp_score_fta * (direct_weight_ft * residual_gate_ft)[:, :, None], axis=(0, 1))

        if pair_count <= 0:
            return dp_spectrum, residual_spectrum, pair_reliability
        return dp_spectrum, residual_spectrum, pair_reliability

    def process(self, audio: np.ndarray) -> LocalizationBackendResult:
        roi = _stft_roi(audio, self.fs, self.nfft, self.overlap, self.freq_range)
        if roi is None:
            return LocalizationBackendResult(peaks_deg=[], peak_scores=[], score_spectrum=None, debug={"backend": "tiny_dp_ipd", "reason": "empty_roi"})
        freqs, zxx_roi = roi
        active_mask = _speech_frame_mask(zxx_roi)
        if active_mask.size == 0 or not np.any(active_mask):
            active_mask = np.ones(zxx_roi.shape[2], dtype=bool)
        z_active = zxx_roi[:, :, active_mask]
        pairs, delays = _pair_delays(self.mic_pos, freqs, self.grid_size, self.sound_speed_m_s)
        band_weight = _band_weight(freqs, self.small_aperture_bias)
        _ = pairs
        dp_spectrum, residual_spectrum, pair_reliability = self._accumulate_direct_path_spectra(z_active, delays, band_weight)

        srp_result = self._srp.process(audio)
        srp_debug = dict(srp_result.debug)
        srp_spectrum = np.asarray(srp_result.score_spectrum, dtype=np.float64) if srp_result.score_spectrum is not None else np.zeros(self.grid_size, dtype=np.float64)
        dp_norm = _normalize_spectrum(dp_spectrum)
        residual_norm = _normalize_spectrum(residual_spectrum)
        srp_norm = _normalize_spectrum(srp_spectrum)
        dominant_map = (0.65 * dp_norm) + (0.35 * srp_norm)
        current_primary_idx = _dominant_peak_idx(dominant_map)
        temporal_prior = self._build_temporal_prior(current_primary_idx)
        residual_leak = 0.06 * residual_norm * np.power(np.clip(1.0 - dominant_map, 0.0, 1.0), 1.35)
        promoted_secondary, promotion_debug = self._promote_secondary_peak(dominant_map, residual_norm, temporal_prior)
        current_fused = np.clip(dominant_map + residual_leak + promoted_secondary, 0.0, None)
        if np.max(temporal_prior) > 0.0:
            fused = ((1.0 - self._final_history_blend) * current_fused) + (self._final_history_blend * temporal_prior)
        else:
            fused = current_fused
        fused = np.clip(fused, 0.0, None)

        self._dominant_history.append(_normalize_spectrum(dominant_map))
        self._residual_history.append(_normalize_spectrum(residual_norm))
        self._fused_history.append(_normalize_spectrum(current_fused))

        peaks_deg, peak_scores = self._peaks_from_spectrum(fused)
        peak_bin_separation = max(1, int(round(self.min_separation_deg / (360.0 / self.grid_size))))
        dominant_contrast, secondary_contrast = _peak_contrasts(fused, peak_bin_separation)
        debug = {
            "backend": "tiny_dp_ipd",
            "active_frames": int(np.sum(active_mask)),
            "direct_path_reliability_mean": float(np.mean(pair_reliability)) if pair_reliability else 0.0,
            "pair_reliability_by_pair": [float(v) for v in pair_reliability],
            "dp_spectrum": [float(v) for v in _normalize_spectrum(dp_spectrum)],
            "srp_spectrum": [float(v) for v in _normalize_spectrum(srp_spectrum)],
            "residual_spectrum": [float(v) for v in _normalize_spectrum(residual_spectrum)],
            "dominant_spectrum": [float(v) for v in _normalize_spectrum(dominant_map)],
            "current_fused_spectrum": [float(v) for v in _normalize_spectrum(current_fused)],
            "fused_spectrum": [float(v) for v in _normalize_spectrum(fused)],
            "dominant_peak_contrast": float(dominant_contrast),
            "secondary_peak_contrast": float(secondary_contrast),
            "srp_debug": srp_debug,
            "grid_size": int(self.grid_size),
            **promotion_debug,
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
            score_spectrum=np.asarray(fused, dtype=np.float64),
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

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from mic_array_forwarder.mode_presets import (
    METHOD_LOCALIZATION_ONLY,
    METHOD_SINGLE_DOMINANT_NO_SEPARATOR,
    METHOD_SPEAKER_TRACKING_SINGLE_ACTIVE,
    get_live_algorithm_preset,
)
from mic_array_forwarder.models import SessionStartRequest, SpeakerStateItem
from realtime_pipeline.contracts import SpeakerGainDirection
from realtime_pipeline.fast_path import delay_and_sum_frame
from realtime_pipeline.srp_tracker import SRPPeakTracker


_UI_PRESENCE_HOLD_MS = 1000.0


def _normalize_angle_deg(v: float) -> float:
    return float(v % 360.0)


def _angle_error_deg(a: float, b: float) -> float:
    return float(abs((float(a) - float(b) + 180.0) % 360.0 - 180.0))


def _ratio_to_db(ratio: float) -> float:
    return float(20.0 * math.log10(max(float(ratio), 1e-6)))


def _clip01(v: float) -> float:
    return float(np.clip(float(v), 0.0, 1.0))


def _circular_mean_deg(values: list[float] | deque[float], weights: list[float] | deque[float] | None = None) -> float:
    if not values:
        return 0.0
    angles = np.deg2rad(np.asarray(list(values), dtype=np.float64))
    if weights is None:
        weights_arr = np.ones_like(angles, dtype=np.float64)
    else:
        weights_arr = np.asarray(list(weights), dtype=np.float64)
        if weights_arr.shape != angles.shape:
            raise ValueError("weights must match angle history length")
    sin_sum = float(np.sum(np.sin(angles) * weights_arr))
    cos_sum = float(np.sum(np.cos(angles) * weights_arr))
    if abs(sin_sum) < 1e-9 and abs(cos_sum) < 1e-9:
        return float(_normalize_angle_deg(list(values)[-1]))
    return float(_normalize_angle_deg(np.rad2deg(math.atan2(sin_sum, cos_sum))))


def _serialize_item(item: SpeakerGainDirection) -> dict[str, Any]:
    return {
        "speaker_id": int(item.speaker_id),
        "direction_degrees": float(item.direction_degrees),
        "gain_weight": float(item.gain_weight),
        "confidence": float(item.confidence),
        "active": bool(item.active),
        "activity_confidence": float(item.activity_confidence),
        "updated_at_ms": float(item.updated_at_ms),
        "identity_confidence": float(item.identity_confidence),
        "identity_maturity": str(item.identity_maturity),
    }


@dataclass(slots=True)
class _TrackState:
    speaker_id: int
    direction_degrees: float
    updated_at_ms: float
    angle_history_deg: deque[float]
    score_history: deque[float]
    support_count: int = 0
    confirmed: bool = False


class LiveCausalProcessor:
    def __init__(self, req: SessionStartRequest, mic_geometry_xyz: np.ndarray):
        self.req = req
        self.mic_geometry_xyz = np.asarray(mic_geometry_xyz, dtype=np.float64)
        self.algorithm = get_live_algorithm_preset(req.algorithm_mode)
        self.speaker_track_states: dict[int, _TrackState] = {}
        self.speaker_tracks: dict[int, SpeakerGainDirection] = {}
        self.next_speaker_id = 1
        self._preferred_active_speaker_id: int | None = None
        self.selected_speaker_id: int | None = None
        self.speaker_gain_delta_db: dict[int, float] = {}

        channel_count = int(req.channel_count)
        self.tracker = SRPPeakTracker(
            mic_pos=self.mic_geometry_xyz.T,
            fs=int(req.sample_rate_hz),
            window_ms=max(int(req.localization_window_ms), max(10, int(req.localization_hop_ms))),
            nfft=512,
            overlap=float(req.overlap),
            freq_range=(int(req.freq_low_hz), int(req.freq_high_hz)),
            max_sources=(
                1
                if req.algorithm_mode in {METHOD_LOCALIZATION_ONLY, METHOD_SINGLE_DOMINANT_NO_SEPARATOR}
                or str(req.localization_backend) == "music_1src"
                else max(1, min(int(self.algorithm.max_sources), channel_count))
            ),
            prior_enabled=True,
            min_score=0.03,
            ema_alpha=0.35,
            hysteresis_margin=0.05,
            match_tolerance_deg=20.0,
            hold_frames=6,
            max_step_deg=12.0,
            score_decay=0.9,
            backend=str(req.localization_backend),
            grid_size=72,
            min_peak_separation_deg=15.0,
            small_aperture_bias=True,
            tracking_mode=str(req.tracking_mode),
            max_tracks=max(1, min(int(self.algorithm.max_sources), 3)),
            single_source_motion_filter_enabled=True,
        )

    def process_frame(self, frame_mc: np.ndarray, timestamp_ms: float) -> tuple[np.ndarray, np.ndarray, list[SpeakerStateItem], dict[str, Any]]:
        frame = np.asarray(frame_mc, dtype=np.float32)
        raw_mono = np.mean(frame, axis=1).astype(np.float32, copy=False)
        peaks, scores, tracker_debug = self.tracker.update(frame)
        items = self._build_speaker_items(peaks, scores, timestamp_ms)
        output = self._render_output(frame, items)
        rms_by_channel = np.sqrt(np.mean(np.square(frame), axis=0)).astype(np.float32, copy=False)
        debug = dict(tracker_debug)
        debug["last_frame_shape"] = [int(frame.shape[0]), int(frame.shape[1])]
        debug["rms_by_channel"] = [float(v) for v in rms_by_channel]
        return output, raw_mono, items, debug

    def set_selected_speaker(self, speaker_id: int | None) -> None:
        self.selected_speaker_id = None if speaker_id is None else int(speaker_id)
        if speaker_id is not None:
            self.speaker_gain_delta_db.setdefault(int(speaker_id), 0.0)

    def adjust_speaker_gain(self, speaker_id: int, delta_db_step: int) -> float:
        sid = int(speaker_id)
        prev = float(self.speaker_gain_delta_db.get(sid, 0.0))
        curr = float(np.clip(prev + int(delta_db_step), -12.0, 12.0))
        self.speaker_gain_delta_db[sid] = curr
        return curr

    def current_speaker_items(self) -> list[SpeakerStateItem]:
        return [
            SpeakerStateItem(
                speaker_id=int(item.speaker_id),
                direction_degrees=float(item.direction_degrees),
                confidence=float(item.confidence),
                active=bool(item.active),
                activity_confidence=float(item.activity_confidence),
                gain_weight=float(item.gain_weight),
            )
            for item in self._sorted_tracks()
        ]

    def current_speaker_rows(self) -> list[dict[str, Any]]:
        return [_serialize_item(item) for item in self._sorted_tracks()]

    def _sorted_tracks(self) -> list[SpeakerGainDirection]:
        return sorted(
            self.speaker_tracks.values(),
            key=lambda item: (-float(item.active), -float(item.confidence), int(item.speaker_id)),
        )

    def _make_track_state(self, speaker_id: int, angle_deg: float, score: float, now_ms: float) -> _TrackState:
        history_size = max(1, int(self.req.speaker_history_size))
        angle_history_deg: deque[float] = deque(maxlen=history_size)
        score_history: deque[float] = deque(maxlen=history_size)
        angle_history_deg.append(float(_normalize_angle_deg(angle_deg)))
        score_history.append(float(_clip01(score)))
        return _TrackState(
            speaker_id=int(speaker_id),
            direction_degrees=float(_normalize_angle_deg(angle_deg)),
            updated_at_ms=float(now_ms),
            angle_history_deg=angle_history_deg,
            score_history=score_history,
            support_count=1,
            confirmed=bool(max(1, int(self.req.speaker_activation_min_predictions)) <= 1),
        )

    def _track_centroid_deg(self, track: _TrackState) -> float:
        return _circular_mean_deg(track.angle_history_deg, track.score_history)

    def _track_confidence(self, track: _TrackState) -> float:
        if not track.score_history:
            return 0.0
        return _clip01(float(np.mean(np.asarray(list(track.score_history), dtype=np.float64))))

    def _build_public_track(self, track: _TrackState, now_ms: float, observed_now: bool) -> SpeakerGainDirection:
        age_ms = max(0.0, float(now_ms) - float(track.updated_at_ms))
        active_hold_ms = min(float(self.algorithm.inactive_hold_ms), _UI_PRESENCE_HOLD_MS)
        still_present = age_ms <= active_hold_ms
        confidence = self._track_confidence(track)
        if not observed_now:
            confidence = _clip01(confidence * 0.85)
        activity_confidence = confidence if track.confirmed else confidence * 0.6
        gain_floor = 0.2 if observed_now or still_present else 0.05
        return SpeakerGainDirection(
            speaker_id=int(track.speaker_id),
            direction_degrees=float(track.direction_degrees),
            gain_weight=float(max(gain_floor, confidence)),
            confidence=float(confidence),
            active=bool(track.confirmed and still_present),
            activity_confidence=float(activity_confidence),
            updated_at_ms=float(track.updated_at_ms),
        )

    def _single_active_mode_enabled(self) -> bool:
        return str(self.req.algorithm_mode) == METHOD_SPEAKER_TRACKING_SINGLE_ACTIVE

    def _track_rank_key(self, track: _TrackState, now_ms: float) -> tuple[float, float, float, float]:
        age_ms = max(0.0, float(now_ms) - float(track.updated_at_ms))
        return (
            float(track.support_count),
            float(len(track.angle_history_deg)),
            float(self._track_confidence(track)),
            -float(age_ms),
        )

    def _apply_single_active_constraint(
        self,
        next_states: dict[int, _TrackState],
        next_tracks: dict[int, SpeakerGainDirection],
        now_ms: float,
    ) -> None:
        if not self._single_active_mode_enabled():
            return
        active_hold_ms = min(float(self.algorithm.inactive_hold_ms), _UI_PRESENCE_HOLD_MS)
        candidate_ids = [
            sid
            for sid, track in next_states.items()
            if track.confirmed and (float(now_ms) - float(track.updated_at_ms)) <= active_hold_ms
        ]
        if not candidate_ids:
            self._preferred_active_speaker_id = None
            return
        best_sid = max(candidate_ids, key=lambda sid: self._track_rank_key(next_states[sid], now_ms))
        preferred_sid = self._preferred_active_speaker_id
        if preferred_sid in candidate_ids:
            preferred_key = self._track_rank_key(next_states[preferred_sid], now_ms)
            best_key = self._track_rank_key(next_states[best_sid], now_ms)
            if preferred_key >= best_key:
                best_sid = preferred_sid
        self._preferred_active_speaker_id = int(best_sid)
        for sid, item in list(next_tracks.items()):
            if sid == best_sid:
                continue
            next_tracks[sid] = SpeakerGainDirection(
                speaker_id=int(item.speaker_id),
                direction_degrees=float(item.direction_degrees),
                gain_weight=float(item.gain_weight),
                confidence=float(item.confidence),
                active=False,
                activity_confidence=float(min(item.activity_confidence, item.confidence * 0.5)),
                updated_at_ms=float(item.updated_at_ms),
                identity_confidence=float(item.identity_confidence),
                identity_maturity=str(item.identity_maturity),
                predicted_direction_deg=item.predicted_direction_deg,
                angular_velocity_deg_per_chunk=float(item.angular_velocity_deg_per_chunk),
                last_separator_stream_index=item.last_separator_stream_index,
                anchor_direction_deg=item.anchor_direction_deg,
                anchor_confidence=float(item.anchor_confidence),
                anchor_locked=bool(item.anchor_locked),
                anchor_last_confirmed_ms=float(item.anchor_last_confirmed_ms),
            )

    def _match_speaker_id(self, angle_deg: float, matched: set[int], now_ms: float) -> int | None:
        best_id = None
        best_err = None
        for sid, track in self.speaker_track_states.items():
            if sid in matched:
                continue
            if now_ms - float(track.updated_at_ms) > float(self.algorithm.inactive_hold_ms):
                continue
            err = _angle_error_deg(angle_deg, self._track_centroid_deg(track))
            if err > float(self.req.speaker_match_window_deg):
                continue
            if best_err is None or err < best_err:
                best_err = err
                best_id = sid
        return best_id

    def _build_speaker_items(
        self,
        peaks: list[float],
        scores: list[float] | None,
        timestamp_ms: float,
    ) -> list[SpeakerStateItem]:
        scores_eff = list(scores) if scores is not None else [1.0] * len(peaks)
        now_ms = float(timestamp_ms)
        next_states: dict[int, _TrackState] = {}
        next_tracks: dict[int, SpeakerGainDirection] = {}
        matched: set[int] = set()
        activation_min_predictions = min(max(1, int(self.req.speaker_activation_min_predictions)), max(1, int(self.req.speaker_history_size)))

        for angle_deg, score in zip(peaks, scores_eff):
            sid = self._match_speaker_id(angle_deg, matched, now_ms)
            if sid is None:
                sid = self.next_speaker_id
                self.next_speaker_id += 1
                track = self._make_track_state(sid, angle_deg, score, now_ms)
            else:
                prev = self.speaker_track_states[sid]
                track = _TrackState(
                    speaker_id=int(prev.speaker_id),
                    direction_degrees=float(prev.direction_degrees),
                    updated_at_ms=float(now_ms),
                    angle_history_deg=deque(prev.angle_history_deg, maxlen=prev.angle_history_deg.maxlen),
                    score_history=deque(prev.score_history, maxlen=prev.score_history.maxlen),
                    support_count=int(prev.support_count) + 1,
                    confirmed=bool(prev.confirmed),
                )
                track.angle_history_deg.append(float(_normalize_angle_deg(angle_deg)))
                track.score_history.append(float(_clip01(score)))
                track.direction_degrees = float(self._track_centroid_deg(track))
                track.confirmed = bool(track.confirmed or len(track.angle_history_deg) >= activation_min_predictions)
            matched.add(sid)
            next_states[sid] = track
            next_tracks[sid] = self._build_public_track(track, now_ms, observed_now=True)

        for sid, track in self.speaker_track_states.items():
            if sid in next_states:
                continue
            age_ms = now_ms - float(track.updated_at_ms)
            if age_ms > float(self.algorithm.inactive_hold_ms):
                continue
            next_states[sid] = track
            next_tracks[sid] = self._build_public_track(track, now_ms, observed_now=False)

        self._apply_single_active_constraint(next_states, next_tracks, now_ms)
        self.speaker_track_states = next_states
        self.speaker_tracks = next_tracks
        return self.current_speaker_items()

    def _pick_focus_direction(self, items: list[SpeakerStateItem]) -> float | None:
        if not items:
            return None
        selected = self.selected_speaker_id
        gains = dict(self.speaker_gain_delta_db)
        if self.req.processing_mode == "specific_speaker_enhancement" and selected is not None:
            for item in items:
                if int(item.speaker_id) == int(selected):
                    return float(item.direction_degrees)
            return None
        if self.req.processing_mode in {"localize_and_beamform", "beamform_from_ground_truth"} or self.req.algorithm_mode == METHOD_LOCALIZATION_ONLY:
            best = max(items, key=lambda item: (float(item.activity_confidence), float(item.confidence)))
            return float(best.direction_degrees)
        if selected is not None and gains.get(selected):
            for item in items:
                if int(item.speaker_id) == int(selected):
                    return float(item.direction_degrees)
        return None

    def _render_output(self, frame_mc: np.ndarray, items: list[SpeakerStateItem]) -> np.ndarray:
        frame = np.asarray(frame_mc, dtype=np.float32)
        focus_direction = self._pick_focus_direction(items)
        if focus_direction is None:
            return np.mean(frame, axis=1).astype(np.float32, copy=False)

        mono = delay_and_sum_frame(
            frame,
            doa_deg=float(focus_direction),
            mic_geometry_xyz=self.mic_geometry_xyz,
            fs=int(self.req.sample_rate_hz),
            sound_speed_m_s=343.0,
        )
        if self.req.processing_mode == "specific_speaker_enhancement":
            selected = self.selected_speaker_id
            delta_db = 0.0 if selected is None else float(self.speaker_gain_delta_db.get(selected, 0.0))
            boost_db = max(0.0, _ratio_to_db(self.req.focus_ratio) + delta_db)
            mono = mono * float(10.0 ** (boost_db / 20.0))
        return np.clip(mono, -1.0, 1.0).astype(np.float32, copy=False)


def run_offline_live_causal(
    *,
    req: SessionStartRequest,
    mic_audio: np.ndarray,
    mic_geometry_xyz: np.ndarray,
    out_dir: str | Path,
    input_recording_path: str | Path | None = None,
    capture_trace: bool = False,
) -> dict[str, Any]:
    audio = np.asarray(mic_audio, dtype=np.float32)
    if audio.ndim != 2:
        raise ValueError("mic_audio must have shape [samples, channels]")

    processor = LiveCausalProcessor(req=req, mic_geometry_xyz=mic_geometry_xyz)
    out_root = Path(out_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    hop_ms = max(10, int(req.localization_hop_ms))
    frame_samples = max(1, int(int(req.sample_rate_hz) * hop_ms / 1000))
    enhanced_parts: list[np.ndarray] = []
    speaker_map_trace: list[dict[str, Any]] = []
    tracker_debug_last: dict[str, Any] = {}
    processing_total_ms = 0.0
    processed_frames = 0

    for frame_idx, start in enumerate(range(0, audio.shape[0], frame_samples)):
        end = min(audio.shape[0], start + frame_samples)
        frame = audio[start:end, :]
        if frame.shape[0] < frame_samples:
            frame = np.pad(frame, ((0, frame_samples - frame.shape[0]), (0, 0)))
        timestamp_ms = float(frame_idx * hop_ms)
        t0 = __import__("time").perf_counter()
        output, _raw_mono, items, tracker_debug = processor.process_frame(frame, timestamp_ms)
        processing_total_ms += (__import__("time").perf_counter() - t0) * 1000.0
        processed_frames += 1
        enhanced_parts.append(np.asarray(output, dtype=np.float32).reshape(-1))
        tracker_debug_last = dict(tracker_debug)
        if capture_trace:
            speaker_map_trace.append(
                {
                    "frame_index": int(frame_idx),
                    "timestamp_ms": float(timestamp_ms),
                    "raw_peaks_deg": [float(v) for v in tracker_debug.get("raw_peaks_deg", [])],
                    "raw_peak_scores": [float(v) for v in tracker_debug.get("raw_peak_scores", [])],
                    "speakers": processor.current_speaker_rows(),
                }
            )

    enhanced = np.concatenate(enhanced_parts)[: audio.shape[0]] if enhanced_parts else np.zeros(audio.shape[0], dtype=np.float32)
    raw_mix_mean = np.mean(np.asarray(audio, dtype=np.float64), axis=1).astype(np.float32, copy=False)
    sf.write(out_root / "enhanced_fast_path.wav", enhanced, int(req.sample_rate_hz))
    sf.write(out_root / "raw_mix_mean.wav", raw_mix_mean, int(req.sample_rate_hz))

    speaker_map_rows = processor.current_speaker_rows()
    with (out_root / "speaker_map_final.json").open("w", encoding="utf-8") as f:
        import json

        json.dump({"speakers": speaker_map_rows}, f, indent=2)

    fast_avg_ms = processing_total_ms / processed_frames if processed_frames else 0.0
    summary: dict[str, Any] = {
        "input_recording_path": "" if input_recording_path is None else str(Path(input_recording_path).resolve()),
        "sample_rate_hz": int(req.sample_rate_hz),
        "duration_s": float(audio.shape[0] / max(int(req.sample_rate_hz), 1)),
        "fast_frame_ms": int(hop_ms),
        "channel_count": int(audio.shape[1]),
        "fast_frames": int(processed_frames),
        "slow_chunks": 0,
        "speaker_map_updates": int(sum(1 for row in speaker_map_trace if row.get("speakers")) if capture_trace else len(speaker_map_rows)),
        "dropped_fast_to_slow_frames": 0,
        "fast_avg_ms": float(fast_avg_ms),
        "slow_avg_ms": 0.0,
        "fast_rtf": float(fast_avg_ms / max(float(hop_ms), 1e-6)),
        "slow_rtf": 0.0,
        "fast_stage_avg_ms": {
            "capture_queue": 0.0,
            "frame_process": float(fast_avg_ms),
        },
        "slow_stage_avg_ms": {},
        "separation_mode": "live_causal",
        "beamforming_mode": str(req.beamforming_mode),
        "fast_path_reference_mode": "live_causal",
        "slow_chunk_ms": 0,
        "slow_chunk_hop_ms": 0,
        "output_normalization_enabled": bool(req.output_normalization_enabled),
        "output_allow_amplification": bool(req.output_allow_amplification),
        "speaker_map_final": speaker_map_rows,
        "robust_mode": False,
        "localization_backend": str(req.localization_backend),
        "localization_window_ms": int(req.localization_window_ms),
        "localization_hop_ms": int(req.localization_hop_ms),
        "srp_overlap": float(req.overlap),
        "srp_freq_min_hz": int(req.freq_low_hz),
        "srp_freq_max_hz": int(req.freq_high_hz),
        "speaker_history_size": int(req.speaker_history_size),
        "speaker_activation_min_predictions": int(req.speaker_activation_min_predictions),
        "speaker_match_window_deg": float(req.speaker_match_window_deg),
        "tracking_mode": str(req.tracking_mode),
        "control_mode": str(req.algorithm_mode),
        "direction_long_memory_enabled": False,
        "direction_long_memory_window_ms": 0.0,
        "tracker_debug": tracker_debug_last,
    }
    if capture_trace:
        summary["speaker_map_trace"] = speaker_map_trace

    with (out_root / "summary.json").open("w", encoding="utf-8") as f:
        import json

        json.dump(summary, f, indent=2)
    return summary

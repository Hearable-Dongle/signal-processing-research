from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from collections.abc import Callable

import numpy as np

from mic_array_forwarder.models import SessionStartRequest
from realtime_pipeline.contracts import PipelineConfig, SRPPeakSnapshot, SpeakerGainDirection
from realtime_pipeline.orchestrator import RealtimeSpeakerPipeline
from realtime_pipeline.separation_backends import DominantSpeakerPassthroughBackend, MockSeparationBackend, build_default_backend


def frame_iter_from_audio(mic_audio: np.ndarray, frame_samples: int):
    total = mic_audio.shape[0]
    for start in range(0, total, frame_samples):
        end = min(total, start + frame_samples)
        frame = mic_audio[start:end, :]
        if frame.shape[0] < frame_samples:
            frame = np.pad(frame, ((0, frame_samples - frame.shape[0]), (0, 0)))
        yield frame.astype(np.float32, copy=False)


def build_pipeline_config_from_request(
    req: SessionStartRequest,
    *,
    sample_rate_hz: int,
    max_speakers_hint: int,
) -> PipelineConfig:
    assume_single_speaker = bool(req.assume_single_speaker)
    slow_path_enabled = bool(req.slow_path_enabled)
    beamforming_mode = str(req.beamforming_mode).strip().lower()
    target_activity_mode = req.target_activity_rnn_update_mode
    if beamforming_mode == "mvdr_fd" and target_activity_mode is None:
        raise ValueError("MVDR requires fast_path.target_activity_rnn_update_mode to be explicitly set.")
    return PipelineConfig(
        sample_rate_hz=int(sample_rate_hz),
        fast_frame_ms=max(10, int(req.localization_hop_ms)),
        slow_chunk_ms=int(req.slow_chunk_ms),
        slow_path_enabled=slow_path_enabled,
        max_speakers_hint=1 if assume_single_speaker else max(1, int(max_speakers_hint)),
        assume_single_speaker=assume_single_speaker,
        convtasnet_model_name=str(req.convtasnet_model_name),
        convtasnet_model_sample_rate_hz=int(req.convtasnet_model_sample_rate_hz),
        convtasnet_input_sample_rate_hz=int(req.convtasnet_input_sample_rate_hz),
        convtasnet_resample_mode=str(req.convtasnet_resample_mode),
        convtasnet_expected_num_sources=(
            None if req.convtasnet_expected_num_sources is None else int(req.convtasnet_expected_num_sources)
        ),
        identity_backend=str(req.identity_backend),
        identity_speaker_embedding_model=str(req.identity_speaker_embedding_model),
        beamforming_mode=beamforming_mode,
        fd_analysis_window_ms=float(req.fd_analysis_window_ms),
        fd_cov_ema_alpha=float(req.fd_cov_ema_alpha),
        fd_diag_load=float(req.fd_diag_load),
        target_activity_rnn_update_mode=target_activity_mode,
        target_activity_low_threshold=float(req.target_activity_low_threshold),
        target_activity_high_threshold=float(req.target_activity_high_threshold),
        target_activity_enter_frames=int(req.target_activity_enter_frames),
        target_activity_exit_frames=int(req.target_activity_exit_frames),
        fd_cov_update_scale_target_active=float(req.fd_cov_update_scale_target_active),
        fd_cov_update_scale_target_inactive=float(req.fd_cov_update_scale_target_inactive),
        target_activity_detector_mode=str(req.target_activity_detector_mode),
        target_activity_detector_backend=str(req.target_activity_detector_backend),
        target_activity_blocker_offset_deg=float(req.target_activity_blocker_offset_deg),
        target_activity_bootstrap_only_calibration=bool(req.target_activity_bootstrap_only_calibration),
        target_activity_ratio_floor_db=float(req.target_activity_ratio_floor_db),
        target_activity_ratio_active_db=float(req.target_activity_ratio_active_db),
        target_activity_target_rms_floor_scale=float(req.target_activity_target_rms_floor_scale),
        target_activity_blocker_rms_floor_scale=float(req.target_activity_blocker_rms_floor_scale),
        target_activity_speech_weight=float(req.target_activity_speech_weight),
        target_activity_ratio_weight=float(req.target_activity_ratio_weight),
        target_activity_blocker_weight=float(req.target_activity_blocker_weight),
        target_activity_vad_mode=int(req.target_activity_vad_mode),
        target_activity_vad_hangover_frames=int(req.target_activity_vad_hangover_frames),
        target_activity_noise_floor_rise_alpha=float(req.target_activity_noise_floor_rise_alpha),
        target_activity_noise_floor_fall_alpha=float(req.target_activity_noise_floor_fall_alpha),
        target_activity_noise_floor_margin_scale=float(req.target_activity_noise_floor_margin_scale),
        target_activity_rms_scale=float(req.target_activity_rms_scale),
        target_activity_score_exponent=float(req.target_activity_score_exponent),
        localization_backend=str(req.localization_backend),
        tracking_mode=str(req.tracking_mode),
        control_mode="speaker_tracking_mode" if slow_path_enabled else "spatial_peak_mode",
        fast_path_reference_mode="speaker_map" if slow_path_enabled else "srp_peak",
        localization_window_ms=max(int(req.localization_window_ms), max(10, int(req.localization_hop_ms))),
        localization_hop_ms=max(10, int(req.localization_hop_ms)),
        direction_long_memory_enabled=bool(req.direction_long_memory_enabled),
        direction_long_memory_window_ms=float(req.direction_long_memory_window_ms),
        postfilter_enabled=bool(req.postfilter_enabled),
        output_normalization_enabled=bool(req.output_normalization_enabled),
        output_allow_amplification=bool(req.output_allow_amplification),
        srp_overlap=float(req.overlap),
        srp_freq_min_hz=int(req.freq_low_hz),
        srp_freq_max_hz=int(req.freq_high_hz),
        localization_pair_selection_mode=str(req.localization_pair_selection_mode),
        localization_vad_enabled=bool(req.localization_vad_enabled),
        capon_spectrum_ema_alpha=float(req.capon_spectrum_ema_alpha),
        capon_peak_min_sharpness=float(req.capon_peak_min_sharpness),
        capon_peak_min_margin=float(req.capon_peak_min_margin),
        capon_hold_frames=int(req.capon_hold_frames),
        own_voice_suppression_mode=str(req.own_voice_suppression_mode),
        suppressed_user_voice_doa_deg=(
            None if req.suppressed_user_voice_doa_deg is None else float(req.suppressed_user_voice_doa_deg)
        ),
        suppressed_user_match_window_deg=float(req.suppressed_user_match_window_deg),
        suppressed_user_null_on_frames=int(req.suppressed_user_null_on_frames),
        suppressed_user_null_off_frames=int(req.suppressed_user_null_off_frames),
        suppressed_user_gate_attenuation_db=float(req.suppressed_user_gate_attenuation_db),
        suppressed_user_target_conflict_deg=float(req.suppressed_user_target_conflict_deg),
        speaker_match_window_deg=float(req.speaker_match_window_deg),
        centroid_association_mode=str(req.centroid_association_mode),
        centroid_association_sigma_deg=float(req.centroid_association_sigma_deg),
        centroid_association_min_score=float(req.centroid_association_min_score),
    )


def build_separation_backend_for_request(req: SessionStartRequest, cfg: PipelineConfig):
    separation_mode = str(req.separation_mode)
    if separation_mode == "single_dominant_no_separator":
        return DominantSpeakerPassthroughBackend()
    if separation_mode == "mock":
        return MockSeparationBackend(n_streams=cfg.max_speakers_hint)
    return build_default_backend(cfg)


def _row_from_speaker(item: SpeakerGainDirection) -> dict[str, Any]:
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
        "predicted_direction_deg": None if item.predicted_direction_deg is None else float(item.predicted_direction_deg),
        "angular_velocity_deg_per_chunk": float(item.angular_velocity_deg_per_chunk),
        "last_separator_stream_index": None if item.last_separator_stream_index is None else int(item.last_separator_stream_index),
        "anchor_direction_deg": None if item.anchor_direction_deg is None else float(item.anchor_direction_deg),
        "anchor_confidence": float(item.anchor_confidence),
        "anchor_locked": bool(item.anchor_locked),
        "anchor_last_confirmed_ms": float(item.anchor_last_confirmed_ms),
    }


def public_speaker_rows(
    snapshot: dict[int, SpeakerGainDirection],
    *,
    single_active: bool,
) -> list[dict[str, Any]]:
    rows = [_row_from_speaker(item) for item in snapshot.values()]
    rows.sort(key=lambda item: (-float(item["active"]), -float(item["confidence"]), int(item["speaker_id"])))
    if not bool(single_active) or not rows:
        return rows
    active_rows = [row for row in rows if bool(row["active"])]
    if not active_rows:
        return rows
    best = max(
        active_rows,
        key=lambda row: (
            float(row["activity_confidence"]),
            float(row["confidence"]),
            float(row["gain_weight"]),
            -int(row["speaker_id"]),
        ),
    )
    best_id = int(best["speaker_id"])
    for row in rows:
        row["active"] = bool(int(row["speaker_id"]) == best_id and bool(row["active"]))
    return rows


def run_offline_session_pipeline(
    *,
    req: SessionStartRequest,
    mic_audio: np.ndarray,
    mic_geometry_xyz: np.ndarray,
    out_dir: str | Path,
    input_recording_path: str | Path | None = None,
    capture_trace: bool = False,
    srp_override_provider: Callable[[int, float], SRPPeakSnapshot | None] | None = None,
    target_activity_override_provider: Callable[[int, float], float | None] | None = None,
) -> dict[str, Any]:
    import soundfile as sf

    audio = np.asarray(mic_audio, dtype=np.float32)
    if audio.ndim != 2:
        raise ValueError("mic_audio must have shape [samples, channels]")

    cfg = build_pipeline_config_from_request(
        req,
        sample_rate_hz=int(req.sample_rate_hz),
        max_speakers_hint=max(int(req.max_speakers_hint), int(audio.shape[1]), 1),
    )
    out_root = Path(out_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    frame_samples = max(1, int(cfg.sample_rate_hz * cfg.fast_frame_ms / 1000))
    mic_geometry_xyz = np.asarray(mic_geometry_xyz, dtype=float)
    mic_geometry_xy = mic_geometry_xyz[:2, :].T if mic_geometry_xyz.shape[0] == 3 else mic_geometry_xyz[:, :2]

    enhanced_parts: list[np.ndarray] = []
    speaker_map_trace: list[dict[str, Any]] = []
    srp_trace: list[dict[str, Any]] = []
    pipe_holder: dict[str, RealtimeSpeakerPipeline] = {}

    def _sink(frame_mono: np.ndarray) -> None:
        enhanced_parts.append(np.asarray(frame_mono, dtype=np.float32).reshape(-1))
        if not capture_trace:
            return
        pipe = pipe_holder.get("pipe")
        if pipe is None:
            return
        srp = pipe.shared_state.get_srp_snapshot()
        srp_trace.append(
            {
                "frame_index": len(enhanced_parts) - 1,
                "timestamp_ms": float(srp.timestamp_ms),
                "peaks_deg": [float(v) for v in srp.peaks_deg],
                "peak_scores": None if srp.peak_scores is None else [float(v) for v in srp.peak_scores],
                "raw_peaks_deg": [float(v) for v in srp.raw_peaks_deg],
                "raw_peak_scores": None if srp.raw_peak_scores is None else [float(v) for v in srp.raw_peak_scores],
                "debug": None if srp.debug is None else dict(srp.debug),
            }
        )
        rows = public_speaker_rows(
            pipe.shared_state.get_speaker_map_snapshot(),
            single_active=bool(req.single_active),
        )
        speaker_map_trace.append(
            {
                "frame_index": len(enhanced_parts) - 1,
                "timestamp_ms": float(srp.timestamp_ms),
                "raw_peaks_deg": [float(v) for v in srp.raw_peaks_deg],
                "raw_peak_scores": [] if srp.raw_peak_scores is None else [float(v) for v in srp.raw_peak_scores],
                "suppression": dict((srp.debug or {}).get("own_voice_suppression", {})),
                "speakers": rows,
            }
        )

    pipe = RealtimeSpeakerPipeline(
        config=cfg,
        mic_geometry_xyz=mic_geometry_xyz,
        mic_geometry_xy=mic_geometry_xy,
        frame_iterator=frame_iter_from_audio(audio, frame_samples),
        frame_sink=_sink,
        separation_backend=build_separation_backend_for_request(req, cfg),
        srp_override_provider=srp_override_provider,
        target_activity_override_provider=target_activity_override_provider,
    )
    pipe_holder["pipe"] = pipe
    pipe.run_blocking()

    enhanced = (
        np.concatenate(enhanced_parts)[: audio.shape[0]]
        if enhanced_parts
        else np.zeros(audio.shape[0], dtype=np.float32)
    )
    raw_mix_mean = np.mean(np.asarray(audio, dtype=np.float64), axis=1).astype(np.float32, copy=False)
    sf.write(out_root / "enhanced_fast_path.wav", enhanced, int(cfg.sample_rate_hz))
    sf.write(out_root / "raw_mix_mean.wav", raw_mix_mean, int(cfg.sample_rate_hz))

    final_rows = public_speaker_rows(
        pipe.shared_state.get_speaker_map_snapshot(),
        single_active=bool(req.single_active),
    )
    with (out_root / "speaker_map_final.json").open("w", encoding="utf-8") as handle:
        json.dump({"speakers": final_rows}, handle, indent=2)

    stats = pipe.stats_snapshot()
    summary: dict[str, Any] = {
        "input_recording_path": "" if input_recording_path is None else str(Path(input_recording_path).resolve()),
        "sample_rate_hz": int(cfg.sample_rate_hz),
        "duration_s": float(audio.shape[0] / max(int(cfg.sample_rate_hz), 1)),
        "fast_frame_ms": int(cfg.fast_frame_ms),
        "channel_count": int(audio.shape[1]),
        "fast_frames": int(stats.fast_frames),
        "slow_chunks": int(stats.slow_chunks),
        "speaker_map_updates": int(stats.speaker_map_updates),
        "dropped_fast_to_slow_frames": int(stats.dropped_fast_to_slow_frames),
        "fast_avg_ms": float(stats.fast_avg_ms),
        "slow_avg_ms": float(stats.slow_avg_ms),
        "fast_rtf": float(stats.fast_rtf),
        "slow_rtf": float(stats.slow_rtf),
        "fast_stage_avg_ms": {
            "srp": float(stats.fast_srp_avg_ms),
            "beamform": float(stats.fast_beamform_avg_ms),
            "safety": float(stats.fast_safety_avg_ms),
            "sink": float(stats.fast_sink_avg_ms),
            "enqueue": float(stats.fast_enqueue_avg_ms),
        },
        "slow_stage_avg_ms": {
            "separation": float(stats.slow_separation_avg_ms),
            "identity": float(stats.slow_identity_avg_ms),
            "direction_assignment": float(stats.slow_direction_avg_ms),
            "publish": float(stats.slow_publish_avg_ms),
        },
        "separation_mode": str(req.separation_mode),
        "beamforming_mode": str(cfg.beamforming_mode),
        "fd_analysis_window_ms": float(cfg.fd_analysis_window_ms),
        "fd_cov_ema_alpha": float(cfg.fd_cov_ema_alpha),
        "fd_diag_load": float(cfg.fd_diag_load),
        "target_activity_rnn_update_mode": None if cfg.target_activity_rnn_update_mode is None else str(cfg.target_activity_rnn_update_mode),
        "target_activity_low_threshold": float(cfg.target_activity_low_threshold),
        "target_activity_high_threshold": float(cfg.target_activity_high_threshold),
        "target_activity_enter_frames": int(cfg.target_activity_enter_frames),
        "target_activity_exit_frames": int(cfg.target_activity_exit_frames),
        "fd_cov_update_scale_target_active": float(cfg.fd_cov_update_scale_target_active),
        "fd_cov_update_scale_target_inactive": float(cfg.fd_cov_update_scale_target_inactive),
        "target_activity_detector_mode": str(cfg.target_activity_detector_mode),
        "target_activity_detector_backend": str(cfg.target_activity_detector_backend),
        "target_activity_blocker_offset_deg": float(cfg.target_activity_blocker_offset_deg),
        "target_activity_bootstrap_only_calibration": bool(cfg.target_activity_bootstrap_only_calibration),
        "target_activity_ratio_floor_db": float(cfg.target_activity_ratio_floor_db),
        "target_activity_ratio_active_db": float(cfg.target_activity_ratio_active_db),
        "target_activity_target_rms_floor_scale": float(cfg.target_activity_target_rms_floor_scale),
        "target_activity_blocker_rms_floor_scale": float(cfg.target_activity_blocker_rms_floor_scale),
        "target_activity_speech_weight": float(cfg.target_activity_speech_weight),
        "target_activity_ratio_weight": float(cfg.target_activity_ratio_weight),
        "target_activity_blocker_weight": float(cfg.target_activity_blocker_weight),
        "target_activity_vad_mode": int(cfg.target_activity_vad_mode),
        "target_activity_vad_hangover_frames": int(cfg.target_activity_vad_hangover_frames),
        "target_activity_noise_floor_rise_alpha": float(cfg.target_activity_noise_floor_rise_alpha),
        "target_activity_noise_floor_fall_alpha": float(cfg.target_activity_noise_floor_fall_alpha),
        "target_activity_noise_floor_margin_scale": float(cfg.target_activity_noise_floor_margin_scale),
        "target_activity_rms_scale": float(cfg.target_activity_rms_scale),
        "target_activity_score_exponent": float(cfg.target_activity_score_exponent),
        "fast_path_reference_mode": str(cfg.fast_path_reference_mode),
        "slow_chunk_ms": int(cfg.slow_chunk_ms),
        "slow_chunk_hop_ms": int(cfg.slow_chunk_ms if cfg.slow_chunk_hop_ms is None else cfg.slow_chunk_hop_ms),
        "output_normalization_enabled": bool(cfg.output_normalization_enabled),
        "output_allow_amplification": bool(cfg.output_allow_amplification),
        "postfilter_enabled": bool(cfg.postfilter_enabled),
        "postfilter_noise_ema_alpha": float(cfg.postfilter_noise_ema_alpha),
        "postfilter_speech_ema_alpha": float(cfg.postfilter_speech_ema_alpha),
        "postfilter_gain_floor": float(cfg.postfilter_gain_floor),
        "postfilter_gain_ema_alpha": float(cfg.postfilter_gain_ema_alpha),
        "postfilter_dd_alpha": float(cfg.postfilter_dd_alpha),
        "postfilter_noise_update_speech_scale": float(cfg.postfilter_noise_update_speech_scale),
        "postfilter_freq_smoothing_bins": int(cfg.postfilter_freq_smoothing_bins),
        "postfilter_gain_max_step_db": float(cfg.postfilter_gain_max_step_db),
        "speaker_map_final": final_rows,
        "localization_backend": str(cfg.localization_backend),
        "localization_window_ms": int(cfg.localization_window_ms),
        "localization_hop_ms": int(cfg.localization_hop_ms),
        "srp_overlap": float(cfg.srp_overlap),
        "srp_freq_min_hz": int(cfg.srp_freq_min_hz),
        "srp_freq_max_hz": int(cfg.srp_freq_max_hz),
        "own_voice_suppression_mode": str(cfg.own_voice_suppression_mode),
        "suppressed_user_voice_doa_deg": (
            None if cfg.suppressed_user_voice_doa_deg is None else float(cfg.suppressed_user_voice_doa_deg)
        ),
        "suppressed_user_match_window_deg": float(cfg.suppressed_user_match_window_deg),
        "suppressed_user_null_on_frames": int(cfg.suppressed_user_null_on_frames),
        "suppressed_user_null_off_frames": int(cfg.suppressed_user_null_off_frames),
        "suppressed_user_gate_attenuation_db": float(cfg.suppressed_user_gate_attenuation_db),
        "suppressed_user_target_conflict_deg": float(cfg.suppressed_user_target_conflict_deg),
        "tracking_mode": str(cfg.tracking_mode),
        "control_mode": str(cfg.control_mode),
        "assume_single_speaker": bool(cfg.assume_single_speaker),
        "direction_long_memory_enabled": bool(cfg.direction_long_memory_enabled),
        "direction_long_memory_window_ms": float(cfg.direction_long_memory_window_ms),
    }
    if capture_trace:
        summary["speaker_map_trace"] = speaker_map_trace
        summary["srp_trace"] = srp_trace

    with (out_root / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary

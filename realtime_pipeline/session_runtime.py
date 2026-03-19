from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any
from collections.abc import Callable

import numpy as np
from scipy.signal import resample_poly

from mic_array_forwarder.models import SessionStartRequest
from realtime_pipeline.contracts import FastPathAudioPacket, PipelineConfig, SRPPeakSnapshot, SpeakerGainDirection
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
    single_active = bool(req.single_active)
    slow_path_enabled = bool(req.slow_path_enabled) and (not single_active)
    beamforming_mode = str(req.beamforming_mode).strip().lower()
    target_activity_mode = req.target_activity_rnn_update_mode
    if beamforming_mode in {"mvdr_fd", "lcmv_target_band"} and target_activity_mode is None:
        raise ValueError("Covariance beamforming requires fast_path.target_activity_rnn_update_mode to be explicitly set.")
    return PipelineConfig(
        sample_rate_hz=int(sample_rate_hz),
        input_sample_rate_hz=int(req.sample_rate_hz),
        input_downsample_rate_hz=(None if req.input_downsample_rate_hz is None else int(req.input_downsample_rate_hz)),
        fast_frame_ms=max(10, int(req.fast_frame_ms)),
        slow_chunk_ms=int(req.slow_chunk_ms),
        slow_path_enabled=slow_path_enabled,
        split_runtime_mode=str(req.split_runtime_mode),
        postfilter_queue_max_frames=int(req.postfilter_queue_max_frames),
        postfilter_queue_drop_oldest=bool(req.postfilter_queue_drop_oldest),
        max_speakers_hint=1 if assume_single_speaker else max(1, int(max_speakers_hint)),
        assume_single_speaker=assume_single_speaker,
        single_active=single_active,
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
        mvdr_hop_ms=(None if req.mvdr_hop_ms is None else int(req.mvdr_hop_ms)),
        beamformer_snapshot_frame_indices=tuple(int(v) for v in req.beamformer_snapshot_frame_indices),
        fd_analysis_window_ms=float(req.fd_analysis_window_ms),
        delay_sum_update_min_delta_deg=float(req.delay_sum_update_min_delta_deg),
        delay_sum_crossfade_frames=int(req.delay_sum_crossfade_frames),
        delay_sum_use_smoothed_doa=bool(req.delay_sum_use_smoothed_doa),
        delay_sum_subtractive_alpha=float(req.delay_sum_subtractive_alpha),
        delay_sum_subtractive_interferer_doa_deg=(
            None if req.delay_sum_subtractive_interferer_doa_deg is None else float(req.delay_sum_subtractive_interferer_doa_deg)
        ),
        delay_sum_subtractive_multi_offset_deg=float(req.delay_sum_subtractive_multi_offset_deg),
        delay_sum_subtractive_use_suppressed_user_doa=bool(req.delay_sum_subtractive_use_suppressed_user_doa),
        delay_sum_subtractive_output_clip_guard=bool(req.delay_sum_subtractive_output_clip_guard),
        fd_cov_ema_alpha=float(req.fd_cov_ema_alpha),
        fd_diag_load=float(req.fd_diag_load),
        fd_trace_diagonal_loading_factor=float(req.fd_trace_diagonal_loading_factor),
        fd_identity_blend_alpha=float(req.fd_identity_blend_alpha),
        beamformer_rnn_skip_refresh_when_clean=bool(req.beamformer_rnn_skip_refresh_when_clean),
        beamformer_rnn_dirty_threshold=float(req.beamformer_rnn_dirty_threshold),
        beamformer_rnn_dirty_eps=float(req.beamformer_rnn_dirty_eps),
        beamformer_rnn_dirty_stat=str(req.beamformer_rnn_dirty_stat),
        beamformer_sparse_solve_enabled=bool(req.beamformer_sparse_solve_enabled),
        beamformer_sparse_solve_stride=int(req.beamformer_sparse_solve_stride),
        beamformer_sparse_solve_min_freq_hz=float(req.beamformer_sparse_solve_min_freq_hz),
        beamformer_sparse_solve_interp=str(req.beamformer_sparse_solve_interp),
        beamformer_weight_reuse_enabled=bool(req.beamformer_weight_reuse_enabled),
        beamformer_weight_smoothing_alpha=float(req.beamformer_weight_smoothing_alpha),
        beamformer_doa_refresh_tolerance_deg=float(req.beamformer_doa_refresh_tolerance_deg),
        fd_noise_covariance_mode=str(req.fd_noise_covariance_mode),
        target_activity_rnn_update_mode=target_activity_mode,
        target_activity_low_threshold=float(req.target_activity_low_threshold),
        target_activity_high_threshold=float(req.target_activity_high_threshold),
        target_activity_enter_frames=int(req.target_activity_enter_frames),
        target_activity_exit_frames=int(req.target_activity_exit_frames),
        fd_cov_update_scale_target_active=float(req.fd_cov_update_scale_target_active),
        fd_cov_update_scale_target_inactive=float(req.fd_cov_update_scale_target_inactive),
        target_activity_detector_mode=str(req.target_activity_detector_mode),
        target_activity_detector_backend=str(req.target_activity_detector_backend),
        target_activity_update_every_n_fast_frames=int(req.target_activity_update_every_n_fast_frames),
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
        localization_grid_size=max(8, int(req.localization_grid_size)),
        single_source_grid_size=max(8, int(req.localization_grid_size)),
        direction_long_memory_enabled=bool(req.direction_long_memory_enabled),
        direction_long_memory_window_ms=float(req.direction_long_memory_window_ms),
        postfilter_enabled=bool(req.postfilter_enabled),
        postfilter_method=str(req.postfilter_method),
        postfilter_noise_source=str(req.postfilter_noise_source),
        postfilter_input_source=str(req.postfilter_input_source),
        postfilter_noise_ema_alpha=float(req.postfilter_noise_ema_alpha),
        postfilter_speech_ema_alpha=float(req.postfilter_speech_ema_alpha),
        postfilter_gain_floor=float(req.postfilter_gain_floor),
        postfilter_gain_ema_alpha=float(req.postfilter_gain_ema_alpha),
        postfilter_dd_alpha=float(req.postfilter_dd_alpha),
        postfilter_noise_update_speech_scale=float(req.postfilter_noise_update_speech_scale),
        postfilter_oversubtraction_alpha=float(req.postfilter_oversubtraction_alpha),
        postfilter_spectral_floor_beta=float(req.postfilter_spectral_floor_beta),
        postfilter_freq_smoothing_bins=int(req.postfilter_freq_smoothing_bins),
        postfilter_gain_max_step_db=float(req.postfilter_gain_max_step_db),
        rnnoise_wet_mix=float(req.rnnoise_wet_mix),
        rnnoise_input_gain_db=float(req.rnnoise_input_gain_db),
        rnnoise_input_highpass_enabled=bool(req.rnnoise_input_highpass_enabled),
        rnnoise_input_highpass_cutoff_hz=float(req.rnnoise_input_highpass_cutoff_hz),
        rnnoise_output_highpass_enabled=bool(req.rnnoise_output_highpass_enabled),
        rnnoise_output_highpass_cutoff_hz=float(req.rnnoise_output_highpass_cutoff_hz),
        rnnoise_output_lowpass_cutoff_hz=float(req.rnnoise_output_lowpass_cutoff_hz),
        rnnoise_output_notch_freq_hz=float(req.rnnoise_output_notch_freq_hz),
        rnnoise_output_notch_q=float(req.rnnoise_output_notch_q),
        rnnoise_vad_adaptive_blend_enabled=bool(req.rnnoise_vad_adaptive_blend_enabled),
        rnnoise_vad_blend_gamma=float(req.rnnoise_vad_blend_gamma),
        rnnoise_vad_min_speech_preserve=float(req.rnnoise_vad_min_speech_preserve),
        rnnoise_vad_max_speech_preserve=float(req.rnnoise_vad_max_speech_preserve),
        rnnoise_residual_highband_enabled=bool(req.rnnoise_residual_highband_enabled),
        rnnoise_residual_highband_cutoff_hz=float(req.rnnoise_residual_highband_cutoff_hz),
        rnnoise_residual_highband_gain=float(req.rnnoise_residual_highband_gain),
        rnnoise_residual_jump_limit_enabled=bool(req.rnnoise_residual_jump_limit_enabled),
        rnnoise_residual_jump_limit_band_low_hz=float(req.rnnoise_residual_jump_limit_band_low_hz),
        rnnoise_residual_jump_limit_rise_db_per_frame=float(req.rnnoise_residual_jump_limit_rise_db_per_frame),
        rnnoise_residual_ema_enabled=bool(req.rnnoise_residual_ema_enabled),
        rnnoise_residual_ema_alpha=float(req.rnnoise_residual_ema_alpha),
        coherence_wiener_gain_floor=float(req.coherence_wiener_gain_floor),
        coherence_wiener_coherence_exponent=float(req.coherence_wiener_coherence_exponent),
        coherence_wiener_temporal_alpha=float(req.coherence_wiener_temporal_alpha),
        output_normalization_enabled=bool(req.output_normalization_enabled),
        output_allow_amplification=bool(req.output_allow_amplification),
        robust_target_band_width_deg=float(req.robust_target_band_width_deg),
        robust_target_band_conditioning_enabled=bool(req.robust_target_band_conditioning_enabled),
        robust_target_band_max_freq_hz=float(req.robust_target_band_max_freq_hz),
        robust_target_band_condition_limit=float(req.robust_target_band_condition_limit),
        srp_overlap=float(req.overlap),
        srp_freq_min_hz=int(req.freq_low_hz),
        srp_freq_max_hz=int(req.freq_high_hz),
        localization_pair_selection_mode=str(req.localization_pair_selection_mode),
        localization_vad_enabled=bool(req.localization_vad_enabled),
        localization_track_hold_frames=int(req.localization_track_hold_frames),
        localization_max_assoc_distance_deg=float(req.localization_max_assoc_distance_deg),
        localization_velocity_alpha=float(req.localization_velocity_alpha),
        localization_angle_alpha=float(req.localization_angle_alpha),
        srp_peak_ema_alpha=float(req.srp_peak_ema_alpha),
        srp_peak_hold_frames=int(req.srp_peak_hold_frames),
        capon_spectrum_ema_alpha=float(req.capon_spectrum_ema_alpha),
        capon_peak_min_sharpness=float(req.capon_peak_min_sharpness),
        capon_peak_min_margin=float(req.capon_peak_min_margin),
        capon_hold_frames=int(req.capon_hold_frames),
        capon_freq_bin_subsample_stride=max(1, int(req.capon_freq_bin_subsample_stride)),
        capon_freq_bin_min_hz=(None if req.capon_freq_bin_min_hz is None else int(req.capon_freq_bin_min_hz)),
        capon_freq_bin_max_hz=(None if req.capon_freq_bin_max_hz is None else int(req.capon_freq_bin_max_hz)),
        capon_use_cholesky_solve=bool(req.capon_use_cholesky_solve),
        capon_covariance_ema_alpha=float(req.capon_covariance_ema_alpha),
        capon_full_scan_every_n_updates=max(1, int(req.capon_full_scan_every_n_updates)),
        capon_local_refine_enabled=bool(req.capon_local_refine_enabled),
        capon_local_refine_half_width_deg=float(req.capon_local_refine_half_width_deg),
        own_voice_suppression_mode=str(req.own_voice_suppression_mode),
        suppressed_user_voice_doa_deg=(
            None if req.suppressed_user_voice_doa_deg is None else float(req.suppressed_user_voice_doa_deg)
        ),
        suppressed_user_match_window_deg=float(req.suppressed_user_match_window_deg),
        suppressed_user_null_on_frames=int(req.suppressed_user_null_on_frames),
        suppressed_user_null_off_frames=int(req.suppressed_user_null_off_frames),
        suppressed_user_gate_attenuation_db=float(req.suppressed_user_gate_attenuation_db),
        suppressed_user_target_conflict_deg=float(req.suppressed_user_target_conflict_deg),
        focus_direction_match_window_deg=float(req.focus_direction_match_window_deg),
        focus_target_hold_frames=int(req.focus_target_hold_frames),
        multi_target_max_speakers=int(req.multi_target_max_speakers),
        multi_target_hold_frames=int(req.multi_target_hold_frames),
        multi_target_min_confidence=float(req.multi_target_min_confidence),
        multi_target_min_activity=float(req.multi_target_min_activity),
        speaker_match_window_deg=float(req.speaker_match_window_deg),
        single_active_min_observation_score=float(req.single_active_min_observation_score),
        centroid_association_mode=str(req.centroid_association_mode),
        centroid_association_sigma_deg=float(req.centroid_association_sigma_deg),
        centroid_association_min_score=float(req.centroid_association_min_score),
    )


def _resample_multichannel_audio(
    mic_audio: np.ndarray,
    *,
    in_sample_rate_hz: int,
    out_sample_rate_hz: int,
) -> np.ndarray:
    audio = np.asarray(mic_audio, dtype=np.float32)
    if int(in_sample_rate_hz) <= 0 or int(out_sample_rate_hz) <= 0:
        raise ValueError("sample rates must be positive for resampling")
    if int(in_sample_rate_hz) == int(out_sample_rate_hz):
        return audio
    if audio.ndim != 2:
        raise ValueError("mic_audio must have shape [samples, channels] for resampling")
    out = resample_poly(audio, up=int(out_sample_rate_hz), down=int(in_sample_rate_hz), axis=0)
    return np.asarray(out, dtype=np.float32)


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
    del single_active
    rows = [_row_from_speaker(item) for item in snapshot.values()]
    rows.sort(key=lambda item: (-float(item["active"]), -float(item["confidence"]), int(item["speaker_id"])))
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
    oracle_noise_frame_provider: Callable[[int, float], np.ndarray | None] | None = None,
    initial_focus_direction_deg: float | None = None,
    initial_focus_speaker_ids: list[int] | None = None,
) -> dict[str, Any]:
    import soundfile as sf

    audio = np.asarray(mic_audio, dtype=np.float32)
    if audio.ndim != 2:
        raise ValueError("mic_audio must have shape [samples, channels]")

    processing_sample_rate_hz = int(req.sample_rate_hz)
    if req.input_downsample_rate_hz is not None:
        processing_sample_rate_hz = int(req.input_downsample_rate_hz)
        if processing_sample_rate_hz <= 0:
            raise ValueError("fast_path.input_downsample_rate_hz must be positive when set")
        audio = _resample_multichannel_audio(
            audio,
            in_sample_rate_hz=int(req.sample_rate_hz),
            out_sample_rate_hz=processing_sample_rate_hz,
        )

    cfg = build_pipeline_config_from_request(
        req,
        sample_rate_hz=int(processing_sample_rate_hz),
        max_speakers_hint=max(int(req.max_speakers_hint), int(audio.shape[1]), 1),
    )
    out_root = Path(out_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    frame_samples = max(1, int(cfg.sample_rate_hz * cfg.fast_frame_ms / 1000))
    total_fast_frames = max(1, int(np.ceil(float(audio.shape[0]) / float(frame_samples))))
    requested_snapshot_indices = tuple(
        max(1, min(total_fast_frames, int(v)))
        for v in getattr(req, "beamformer_snapshot_frame_indices", ())
        if int(v) > 0
    )
    if requested_snapshot_indices:
        cfg.beamformer_snapshot_frame_indices = tuple(sorted(set(requested_snapshot_indices)))
    else:
        cfg.beamformer_snapshot_frame_indices = tuple(
            sorted(
                {
                    max(1, min(total_fast_frames, int(round(total_fast_frames * frac))))
                    for frac in (0.25, 0.5, 0.75)
                }
            )
        )
    mic_geometry_xyz = np.asarray(mic_geometry_xyz, dtype=float)
    mic_geometry_xy = mic_geometry_xyz[:2, :].T if mic_geometry_xyz.shape[0] == 3 else mic_geometry_xyz[:, :2]

    enhanced_parts: list[np.ndarray] = []
    captured_packets: list[FastPathAudioPacket] = []
    speaker_map_trace: list[dict[str, Any]] = []
    srp_trace: list[dict[str, Any]] = []
    noise_model_update_trace: list[dict[str, Any]] = []
    pipe_holder: dict[str, RealtimeSpeakerPipeline] = {}

    def _sink(frame_mono: np.ndarray) -> None:
        enhanced_parts.append(np.asarray(frame_mono, dtype=np.float32).reshape(-1))
        if not capture_trace:
            return
        pipe = pipe_holder.get("pipe")
        if pipe is None:
            return
        srp = pipe.shared_state.get_srp_snapshot()
        noise_update = pipe.shared_state.get_noise_model_update_snapshot()
        srp_trace.append(
            {
                "frame_index": len(enhanced_parts) - 1,
                "timestamp_ms": float(srp.timestamp_ms),
                "peaks_deg": [float(v) for v in srp.peaks_deg],
                "peak_scores": None if srp.peak_scores is None else [float(v) for v in srp.peak_scores],
                "raw_peaks_deg": [float(v) for v in srp.raw_peaks_deg],
                "raw_peak_scores": None if srp.raw_peak_scores is None else [float(v) for v in srp.raw_peak_scores],
                "debug": None if srp.debug is None else dict(srp.debug),
                "noise_model_update": {
                    "active": bool(noise_update.active),
                    "sources": [str(v) for v in noise_update.sources],
                    "reasons": [str(v) for v in noise_update.reasons],
                    "debug": None if noise_update.debug is None else dict(noise_update.debug),
                },
            }
        )
        noise_model_update_trace.append(
            {
                "frame_index": len(enhanced_parts) - 1,
                "timestamp_ms": float(noise_update.timestamp_ms),
                "active": bool(noise_update.active),
                "sources": [str(v) for v in noise_update.sources],
                "reasons": [str(v) for v in noise_update.reasons],
                "debug": None if noise_update.debug is None else dict(noise_update.debug),
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
                "noise_model_update": {
                    "active": bool(noise_update.active),
                    "sources": [str(v) for v in noise_update.sources],
                    "reasons": [str(v) for v in noise_update.reasons],
                    "debug": None if noise_update.debug is None else dict(noise_update.debug),
                },
                "speakers": rows,
            }
        )

    def _packet_sink(packet: FastPathAudioPacket) -> None:
        captured_packets.append(packet)

    split_mode = str(getattr(cfg, "split_runtime_mode", "monolithic")).strip().lower()
    packet_iterator = None
    if split_mode == "postfilter_only":
        pre_cfg = copy.deepcopy(cfg)
        pre_cfg.split_runtime_mode = "beamforming_only"
        prepipe = RealtimeSpeakerPipeline(
            config=pre_cfg,
            mic_geometry_xyz=mic_geometry_xyz,
            mic_geometry_xy=mic_geometry_xy,
            frame_iterator=frame_iter_from_audio(audio, frame_samples),
            frame_sink=lambda _x: None,
            frame_packet_sink=_packet_sink,
            separation_backend=build_separation_backend_for_request(req, pre_cfg),
            srp_override_provider=srp_override_provider,
            target_activity_override_provider=target_activity_override_provider,
            oracle_noise_frame_provider=oracle_noise_frame_provider,
        )
        prepipe.run_blocking()
        packet_iterator = iter(captured_packets)

    pipe = RealtimeSpeakerPipeline(
        config=cfg,
        mic_geometry_xyz=mic_geometry_xyz,
        mic_geometry_xy=mic_geometry_xy,
        frame_iterator=frame_iter_from_audio(audio, frame_samples),
        frame_packet_iterator=packet_iterator,
        frame_sink=_sink,
        separation_backend=build_separation_backend_for_request(req, cfg),
        srp_override_provider=srp_override_provider,
        target_activity_override_provider=target_activity_override_provider,
        oracle_noise_frame_provider=oracle_noise_frame_provider,
        frame_packet_sink=_packet_sink if (capture_trace or split_mode == "beamforming_only") else None,
    )
    pipe_holder["pipe"] = pipe
    if initial_focus_direction_deg is not None or initial_focus_speaker_ids:
        pipe.set_focus_control(
            focused_speaker_ids=None if not initial_focus_speaker_ids else [int(v) for v in initial_focus_speaker_ids],
            focused_direction_deg=None if initial_focus_direction_deg is None else float(initial_focus_direction_deg),
            user_boost_db=0.0,
        )
    pipe.run_blocking()

    enhanced = (
        np.concatenate(enhanced_parts)[: audio.shape[0]]
        if enhanced_parts
        else np.zeros(audio.shape[0], dtype=np.float32)
    )
    raw_mix_mean = np.mean(np.asarray(audio, dtype=np.float64), axis=1).astype(np.float32, copy=False)
    sf.write(out_root / "enhanced_fast_path.wav", enhanced, int(cfg.sample_rate_hz))
    sf.write(out_root / "raw_mix_mean.wav", raw_mix_mean, int(cfg.sample_rate_hz))
    if captured_packets:
        beamformed_stage = np.concatenate(
            [np.asarray(packet.beamformed_audio, dtype=np.float32).reshape(-1) for packet in captured_packets],
            axis=0,
        )[: audio.shape[0]]
        sf.write(out_root / "post_beamforming.wav", beamformed_stage, int(cfg.sample_rate_hz))
        if any(packet.postfilter_wiener_audio is not None for packet in captured_packets):
            post_wiener = np.concatenate(
                [
                    np.asarray(
                        packet.beamformed_audio if packet.postfilter_wiener_audio is None else packet.postfilter_wiener_audio,
                        dtype=np.float32,
                    ).reshape(-1)
                    for packet in captured_packets
                ],
                axis=0,
            )[: audio.shape[0]]
            sf.write(out_root / "post_wiener.wav", post_wiener, int(cfg.sample_rate_hz))
        if any(packet.postfilter_rnnoise_audio is not None for packet in captured_packets):
            post_rnnoise = np.concatenate(
                [
                    np.asarray(
                        packet.beamformed_audio if packet.postfilter_rnnoise_audio is None else packet.postfilter_rnnoise_audio,
                        dtype=np.float32,
                    ).reshape(-1)
                    for packet in captured_packets
                ],
                axis=0,
            )[: audio.shape[0]]
            sf.write(out_root / "post_rnnoise.wav", post_rnnoise, int(cfg.sample_rate_hz))
        if any(packet.postfilter_inverse_rnnoise_audio is not None for packet in captured_packets):
            inverse_rnnoise = np.concatenate(
                [
                    np.asarray(
                        np.zeros_like(packet.beamformed_audio) if packet.postfilter_inverse_rnnoise_audio is None else packet.postfilter_inverse_rnnoise_audio,
                        dtype=np.float32,
                    ).reshape(-1)
                    for packet in captured_packets
                ],
                axis=0,
            )[: audio.shape[0]]
            sf.write(out_root / "inverse_rnnoise.wav", inverse_rnnoise, int(cfg.sample_rate_hz))
        if any(packet.postfilter_bandpass_audio is not None for packet in captured_packets):
            post_bandpass = np.concatenate(
                [
                    np.asarray(
                        packet.beamformed_audio if packet.postfilter_bandpass_audio is None else packet.postfilter_bandpass_audio,
                        dtype=np.float32,
                    ).reshape(-1)
                    for packet in captured_packets
                ],
                axis=0,
            )[: audio.shape[0]]
            sf.write(out_root / "post_bandpass.wav", post_bandpass, int(cfg.sample_rate_hz))

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
        "input_sample_rate_hz": int(cfg.input_sample_rate_hz),
        "input_downsample_rate_hz": (
            None if cfg.input_downsample_rate_hz is None else int(cfg.input_downsample_rate_hz)
        ),
        "duration_s": float(audio.shape[0] / max(int(cfg.sample_rate_hz), 1)),
        "fast_frame_ms": int(cfg.fast_frame_ms),
        "channel_count": int(audio.shape[1]),
        "fast_frames": int(stats.fast_frames),
        "slow_chunks": int(stats.slow_chunks),
        "speaker_map_updates": int(stats.speaker_map_updates),
        "dropped_fast_to_slow_frames": int(stats.dropped_fast_to_slow_frames),
        "dropped_interstage_frames": int(stats.dropped_interstage_frames),
        "fast_avg_ms": float(stats.fast_avg_ms),
        "slow_avg_ms": float(stats.slow_avg_ms),
        "fast_rtf": float(stats.fast_rtf),
        "slow_rtf": float(stats.slow_rtf),
        "beamforming_avg_ms": float(stats.beamforming_avg_ms),
        "beamforming_rtf": float(stats.beamforming_rtf),
        "beamforming_p50_ms": float(stats.beamforming_p50_ms),
        "beamforming_p95_ms": float(stats.beamforming_p95_ms),
        "postfilter_avg_ms": float(stats.postfilter_avg_ms),
        "postfilter_rtf": float(stats.postfilter_rtf),
        "postfilter_p50_ms": float(stats.postfilter_p50_ms),
        "postfilter_p95_ms": float(stats.postfilter_p95_ms),
        "pipeline_avg_ms": float(stats.pipeline_avg_ms),
        "pipeline_rtf": float(stats.pipeline_rtf),
        "interstage_queue_wait_p50_ms": float(stats.interstage_queue_wait_p50_ms),
        "interstage_queue_wait_p95_ms": float(stats.interstage_queue_wait_p95_ms),
        "interstage_queue_depth_max": int(stats.interstage_queue_depth_max),
        "end_to_end_latency_p50_ms": float(stats.end_to_end_latency_p50_ms),
        "end_to_end_latency_p95_ms": float(stats.end_to_end_latency_p95_ms),
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
        "split_runtime_mode": str(cfg.split_runtime_mode),
        "postfilter_queue_max_frames": int(cfg.postfilter_queue_max_frames),
        "postfilter_queue_drop_oldest": bool(cfg.postfilter_queue_drop_oldest),
        "mvdr_hop_ms": (None if cfg.mvdr_hop_ms is None else int(cfg.mvdr_hop_ms)),
        "fd_analysis_window_ms": float(cfg.fd_analysis_window_ms),
        "fd_cov_ema_alpha": float(cfg.fd_cov_ema_alpha),
        "fd_diag_load": float(cfg.fd_diag_load),
        "beamformer_rnn_skip_refresh_when_clean": bool(cfg.beamformer_rnn_skip_refresh_when_clean),
        "beamformer_rnn_dirty_threshold": float(cfg.beamformer_rnn_dirty_threshold),
        "beamformer_rnn_dirty_eps": float(cfg.beamformer_rnn_dirty_eps),
        "beamformer_rnn_dirty_stat": str(cfg.beamformer_rnn_dirty_stat),
        "beamformer_sparse_solve_enabled": bool(cfg.beamformer_sparse_solve_enabled),
        "beamformer_sparse_solve_stride": int(cfg.beamformer_sparse_solve_stride),
        "beamformer_sparse_solve_min_freq_hz": float(cfg.beamformer_sparse_solve_min_freq_hz),
        "beamformer_sparse_solve_interp": str(cfg.beamformer_sparse_solve_interp),
        "beamformer_weight_reuse_enabled": bool(cfg.beamformer_weight_reuse_enabled),
        "beamformer_weight_smoothing_alpha": float(cfg.beamformer_weight_smoothing_alpha),
        "beamformer_doa_refresh_tolerance_deg": float(cfg.beamformer_doa_refresh_tolerance_deg),
        "fd_noise_covariance_mode": str(cfg.fd_noise_covariance_mode),
        "target_activity_rnn_update_mode": None if cfg.target_activity_rnn_update_mode is None else str(cfg.target_activity_rnn_update_mode),
        "target_activity_low_threshold": float(cfg.target_activity_low_threshold),
        "target_activity_high_threshold": float(cfg.target_activity_high_threshold),
        "target_activity_enter_frames": int(cfg.target_activity_enter_frames),
        "target_activity_exit_frames": int(cfg.target_activity_exit_frames),
        "fd_cov_update_scale_target_active": float(cfg.fd_cov_update_scale_target_active),
        "fd_cov_update_scale_target_inactive": float(cfg.fd_cov_update_scale_target_inactive),
        "target_activity_detector_mode": str(cfg.target_activity_detector_mode),
        "target_activity_detector_backend": str(cfg.target_activity_detector_backend),
        "target_activity_update_every_n_fast_frames": int(cfg.target_activity_update_every_n_fast_frames),
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
        "postfilter_method": str(cfg.postfilter_method),
        "postfilter_noise_source": str(cfg.postfilter_noise_source),
        "postfilter_input_source": str(cfg.postfilter_input_source),
        "postfilter_noise_ema_alpha": float(cfg.postfilter_noise_ema_alpha),
        "postfilter_speech_ema_alpha": float(cfg.postfilter_speech_ema_alpha),
        "postfilter_gain_floor": float(cfg.postfilter_gain_floor),
        "postfilter_gain_ema_alpha": float(cfg.postfilter_gain_ema_alpha),
        "postfilter_dd_alpha": float(cfg.postfilter_dd_alpha),
        "postfilter_noise_update_speech_scale": float(cfg.postfilter_noise_update_speech_scale),
        "postfilter_oversubtraction_alpha": float(cfg.postfilter_oversubtraction_alpha),
        "postfilter_spectral_floor_beta": float(cfg.postfilter_spectral_floor_beta),
        "postfilter_freq_smoothing_bins": int(cfg.postfilter_freq_smoothing_bins),
        "postfilter_gain_max_step_db": float(cfg.postfilter_gain_max_step_db),
        "rnnoise_wet_mix": float(cfg.rnnoise_wet_mix),
        "rnnoise_input_gain_db": float(cfg.rnnoise_input_gain_db),
        "rnnoise_input_highpass_enabled": bool(cfg.rnnoise_input_highpass_enabled),
        "rnnoise_input_highpass_cutoff_hz": float(cfg.rnnoise_input_highpass_cutoff_hz),
        "rnnoise_output_highpass_enabled": bool(cfg.rnnoise_output_highpass_enabled),
        "rnnoise_output_highpass_cutoff_hz": float(cfg.rnnoise_output_highpass_cutoff_hz),
        "rnnoise_output_lowpass_cutoff_hz": float(cfg.rnnoise_output_lowpass_cutoff_hz),
        "rnnoise_output_notch_freq_hz": float(cfg.rnnoise_output_notch_freq_hz),
        "rnnoise_output_notch_q": float(cfg.rnnoise_output_notch_q),
        "rnnoise_vad_adaptive_blend_enabled": bool(cfg.rnnoise_vad_adaptive_blend_enabled),
        "rnnoise_vad_blend_gamma": float(cfg.rnnoise_vad_blend_gamma),
        "rnnoise_vad_min_speech_preserve": float(cfg.rnnoise_vad_min_speech_preserve),
        "rnnoise_vad_max_speech_preserve": float(cfg.rnnoise_vad_max_speech_preserve),
        "rnnoise_residual_highband_enabled": bool(cfg.rnnoise_residual_highband_enabled),
        "rnnoise_residual_highband_cutoff_hz": float(cfg.rnnoise_residual_highband_cutoff_hz),
        "rnnoise_residual_highband_gain": float(cfg.rnnoise_residual_highband_gain),
        "rnnoise_residual_jump_limit_enabled": bool(cfg.rnnoise_residual_jump_limit_enabled),
        "rnnoise_residual_jump_limit_band_low_hz": float(cfg.rnnoise_residual_jump_limit_band_low_hz),
        "rnnoise_residual_jump_limit_rise_db_per_frame": float(cfg.rnnoise_residual_jump_limit_rise_db_per_frame),
        "rnnoise_residual_ema_enabled": bool(cfg.rnnoise_residual_ema_enabled),
        "rnnoise_residual_ema_alpha": float(cfg.rnnoise_residual_ema_alpha),
        "delay_sum_subtractive_alpha": float(cfg.delay_sum_subtractive_alpha),
        "delay_sum_subtractive_interferer_doa_deg": (
            None if cfg.delay_sum_subtractive_interferer_doa_deg is None else float(cfg.delay_sum_subtractive_interferer_doa_deg)
        ),
        "delay_sum_subtractive_multi_offset_deg": float(cfg.delay_sum_subtractive_multi_offset_deg),
        "delay_sum_subtractive_use_suppressed_user_doa": bool(cfg.delay_sum_subtractive_use_suppressed_user_doa),
        "delay_sum_subtractive_output_clip_guard": bool(cfg.delay_sum_subtractive_output_clip_guard),
        "coherence_wiener_gain_floor": float(cfg.coherence_wiener_gain_floor),
        "coherence_wiener_coherence_exponent": float(cfg.coherence_wiener_coherence_exponent),
        "coherence_wiener_temporal_alpha": float(cfg.coherence_wiener_temporal_alpha),
        "speaker_map_final": final_rows,
        "localization_backend": str(cfg.localization_backend),
        "localization_window_ms": int(cfg.localization_window_ms),
        "localization_hop_ms": int(cfg.localization_hop_ms),
        "localization_grid_size": int(cfg.localization_grid_size),
        "localization_track_hold_frames": int(cfg.localization_track_hold_frames),
        "localization_max_assoc_distance_deg": float(cfg.localization_max_assoc_distance_deg),
        "localization_velocity_alpha": float(cfg.localization_velocity_alpha),
        "localization_angle_alpha": float(cfg.localization_angle_alpha),
        "srp_overlap": float(cfg.srp_overlap),
        "srp_freq_min_hz": int(cfg.srp_freq_min_hz),
        "srp_freq_max_hz": int(cfg.srp_freq_max_hz),
        "srp_peak_ema_alpha": float(cfg.srp_peak_ema_alpha),
        "srp_peak_hold_frames": int(cfg.srp_peak_hold_frames),
        "capon_spectrum_ema_alpha": float(cfg.capon_spectrum_ema_alpha),
        "capon_peak_min_sharpness": float(cfg.capon_peak_min_sharpness),
        "capon_peak_min_margin": float(cfg.capon_peak_min_margin),
        "capon_hold_frames": int(cfg.capon_hold_frames),
        "capon_freq_bin_subsample_stride": int(cfg.capon_freq_bin_subsample_stride),
        "capon_freq_bin_min_hz": (None if cfg.capon_freq_bin_min_hz is None else int(cfg.capon_freq_bin_min_hz)),
        "capon_freq_bin_max_hz": (None if cfg.capon_freq_bin_max_hz is None else int(cfg.capon_freq_bin_max_hz)),
        "capon_use_cholesky_solve": bool(cfg.capon_use_cholesky_solve),
        "capon_covariance_ema_alpha": float(cfg.capon_covariance_ema_alpha),
        "capon_full_scan_every_n_updates": int(cfg.capon_full_scan_every_n_updates),
        "capon_local_refine_enabled": bool(cfg.capon_local_refine_enabled),
        "capon_local_refine_half_width_deg": float(cfg.capon_local_refine_half_width_deg),
        "own_voice_suppression_mode": str(cfg.own_voice_suppression_mode),
        "suppressed_user_voice_doa_deg": (
            None if cfg.suppressed_user_voice_doa_deg is None else float(cfg.suppressed_user_voice_doa_deg)
        ),
        "suppressed_user_match_window_deg": float(cfg.suppressed_user_match_window_deg),
        "suppressed_user_null_on_frames": int(cfg.suppressed_user_null_on_frames),
        "suppressed_user_null_off_frames": int(cfg.suppressed_user_null_off_frames),
        "suppressed_user_gate_attenuation_db": float(cfg.suppressed_user_gate_attenuation_db),
        "suppressed_user_target_conflict_deg": float(cfg.suppressed_user_target_conflict_deg),
        "focus_direction_match_window_deg": float(cfg.focus_direction_match_window_deg),
        "focus_target_hold_frames": int(cfg.focus_target_hold_frames),
        "multi_target_max_speakers": int(cfg.multi_target_max_speakers),
        "multi_target_hold_frames": int(cfg.multi_target_hold_frames),
        "multi_target_min_confidence": float(cfg.multi_target_min_confidence),
        "multi_target_min_activity": float(cfg.multi_target_min_activity),
        "tracking_mode": str(cfg.tracking_mode),
        "control_mode": str(cfg.control_mode),
        "assume_single_speaker": bool(cfg.assume_single_speaker),
        "direction_long_memory_enabled": bool(cfg.direction_long_memory_enabled),
        "direction_long_memory_window_ms": float(cfg.direction_long_memory_window_ms),
    }
    if capture_trace:
        summary["speaker_map_trace"] = speaker_map_trace
        summary["srp_trace"] = srp_trace
        summary["noise_model_update_trace"] = noise_model_update_trace
        if pipe._fast is not None:
            summary["beamformer_snapshot_trace"] = pipe._fast.get_beamformer_snapshot_trace()
            summary.update(pipe._fast.get_beamformer_runtime_stats())

    with (out_root / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary

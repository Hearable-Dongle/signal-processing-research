from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from .contracts import PipelineConfig
from .orchestrator import RealtimeSpeakerPipeline
from .separation_backends import DominantSpeakerPassthroughBackend, MockSeparationBackend, build_default_backend


def _frame_iter(mic_audio: np.ndarray, frame_samples: int):
    total = mic_audio.shape[0]
    for start in range(0, total, frame_samples):
        end = min(total, start + frame_samples)
        frame = mic_audio[start:end, :]
        if frame.shape[0] < frame_samples:
            frame = np.pad(frame, ((0, frame_samples - frame.shape[0]), (0, 0)))
        yield frame.astype(np.float32, copy=False)


def run_recording_pipeline(
    *,
    mic_audio: np.ndarray,
    sample_rate_hz: int,
    mic_geometry_xyz: np.ndarray,
    out_dir: str | Path,
    input_recording_path: str | Path | None = None,
    separation_mode: str = "single_dominant_no_separator",
    fast_frame_ms: int = 50,
    slow_chunk_ms: int = 2000,
    slow_chunk_hop_ms: int | None = 1000,
    beamforming_mode: str = "mvdr_fd",
    fast_path_reference_mode: str = "speaker_map",
    output_normalization_enabled: bool = True,
    output_allow_amplification: bool = False,
    robust_mode: bool = True,
    capture_trace: bool = False,
    localization_backend: str = "srp_phat_localization",
    localization_window_ms: int = 200,
    localization_hop_ms: int | None = None,
    srp_overlap: float = 0.5,
    srp_freq_min_hz: int = 200,
    srp_freq_max_hz: int = 3000,
    tracking_mode: str = "doa_centroid_v1",
    control_mode: str = "spatial_peak_mode",
    direction_long_memory_enabled: bool = True,
    direction_long_memory_window_ms: float = 60000.0,
    direction_history_window_chunks: int = 4,
    direction_speaker_stale_timeout_ms: float = 2000.0,
    direction_speaker_forget_timeout_ms: float = 8000.0,
    convtasnet_model_name: str = "JorisCos/ConvTasNet_Libri2Mix_sepnoisy_16k",
    convtasnet_model_sample_rate_hz: int = 16000,
    convtasnet_input_sample_rate_hz: int = 16000,
    convtasnet_expected_num_sources: int | None = None,
    identity_backend: str = "mfcc_legacy",
    identity_speaker_embedding_model: str = "wavlm_base_plus_sv",
    max_speakers_hint: int = 4,
) -> dict:
    mic_audio = np.asarray(mic_audio, dtype=np.float32)
    if mic_audio.ndim != 2:
        raise ValueError("mic_audio must have shape [samples, channels]")

    mic_geometry_xyz = np.asarray(mic_geometry_xyz, dtype=float)
    if mic_geometry_xyz.shape[0] == 3 and mic_geometry_xyz.shape[1] == mic_audio.shape[1]:
        mic_geometry_xy = mic_geometry_xyz[:2, :].T
    elif mic_geometry_xyz.shape[1] == 3 and mic_geometry_xyz.shape[0] == mic_audio.shape[1]:
        mic_geometry_xy = mic_geometry_xyz[:, :2]
        mic_geometry_xyz = mic_geometry_xyz.T
    else:
        raise ValueError("mic geometry shape does not match channel count")

    default_cfg = PipelineConfig()
    out_root = Path(out_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    cfg = PipelineConfig(
        sample_rate_hz=int(sample_rate_hz),
        fast_frame_ms=int(fast_frame_ms),
        slow_chunk_ms=int(slow_chunk_ms),
        slow_chunk_hop_ms=None if slow_chunk_hop_ms is None else int(slow_chunk_hop_ms),
        fast_path_reference_mode=str(fast_path_reference_mode),
        convtasnet_model_name=str(convtasnet_model_name),
        convtasnet_model_sample_rate_hz=int(convtasnet_model_sample_rate_hz),
        convtasnet_input_sample_rate_hz=int(convtasnet_input_sample_rate_hz),
        convtasnet_expected_num_sources=None if convtasnet_expected_num_sources is None else int(convtasnet_expected_num_sources),
        identity_backend=str(identity_backend),
        identity_speaker_embedding_model=str(identity_speaker_embedding_model),
        localization_window_ms=int(localization_window_ms),
        localization_hop_ms=int(fast_frame_ms if localization_hop_ms is None else localization_hop_ms),
        localization_backend=str(localization_backend),
        srp_overlap=float(srp_overlap),
        srp_freq_min_hz=int(srp_freq_min_hz),
        srp_freq_max_hz=int(srp_freq_max_hz),
        tracking_mode=str(tracking_mode),
        control_mode=str(control_mode),
        direction_long_memory_enabled=bool(direction_long_memory_enabled),
        direction_long_memory_window_ms=float(direction_long_memory_window_ms),
        direction_history_window_chunks=int(direction_history_window_chunks),
        direction_speaker_stale_timeout_ms=float(direction_speaker_stale_timeout_ms),
        direction_speaker_forget_timeout_ms=float(direction_speaker_forget_timeout_ms),
        max_speakers_hint=max(1, int(max_speakers_hint)),
        beamforming_mode=str(beamforming_mode),
        target_activity_rnn_update_mode="estimated_target_activity" if str(beamforming_mode).strip().lower() == "mvdr_fd" else None,
        output_normalization_enabled=bool(output_normalization_enabled),
        output_allow_amplification=bool(output_allow_amplification),
        srp_prior_enabled=bool(robust_mode),
        srp_peak_match_tolerance_deg=12.0 if robust_mode else 20.0,
        srp_peak_hold_frames=4 if robust_mode else 0,
        identity_continuity_bonus=0.04 if robust_mode else 0.0,
        identity_switch_penalty=0.06 if robust_mode else 0.0,
        identity_hold_similarity_threshold=0.6 if robust_mode else 1.1,
        identity_carry_forward_chunks=1 if robust_mode else 0,
        identity_confidence_decay=0.92 if robust_mode else 0.0,
        identity_retire_after_chunks=25 if robust_mode else 1,
        direction_transition_penalty_deg=22.0 if robust_mode else 180.0,
        direction_min_confidence_for_switch=0.35 if robust_mode else 0.0,
        direction_hold_confidence_decay=0.9 if robust_mode else 1.0,
        direction_stale_confidence_decay=0.96 if robust_mode else 1.0,
        direction_min_persist_confidence=0.05 if robust_mode else 0.0,
        speaker_map_min_confidence_for_refresh=0.2 if robust_mode else 0.0,
        speaker_map_hold_ms=300.0 if robust_mode else 0.0,
        speaker_map_confidence_decay=0.9 if robust_mode else 1.0,
        speaker_map_activity_decay=0.92 if robust_mode else 1.0,
        identity_new_speaker_max_existing_score=float(default_cfg.identity_new_speaker_max_existing_score),
        identity_direction_mismatch_block_deg=float(default_cfg.identity_direction_mismatch_block_deg),
        direction_focus_gain_db=float(default_cfg.direction_focus_gain_db),
        direction_non_focus_attenuation_db=float(default_cfg.direction_non_focus_attenuation_db),
    )
    frame_samples = int(cfg.sample_rate_hz * cfg.fast_frame_ms / 1000)

    enhanced_parts: list[np.ndarray] = []
    speaker_map_trace: list[dict] = []
    srp_trace: list[dict] = []
    pipe_holder: dict[str, RealtimeSpeakerPipeline] = {}

    def _sink(x: np.ndarray) -> None:
        enhanced_parts.append(np.asarray(x, dtype=np.float32).reshape(-1))
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
        snapshot = pipe.shared_state.get_speaker_map_snapshot()
        speaker_map_trace.append(
            {
                "frame_index": len(enhanced_parts) - 1,
                "speakers": [
                    {
                        "speaker_id": int(v.speaker_id),
                        "direction_degrees": float(v.direction_degrees),
                        "gain_weight": float(v.gain_weight),
                        "confidence": float(v.confidence),
                        "active": bool(v.active),
                        "activity_confidence": float(v.activity_confidence),
                        "updated_at_ms": float(v.updated_at_ms),
                        "identity_confidence": float(v.identity_confidence),
                        "identity_maturity": str(v.identity_maturity),
                    }
                    for v in snapshot.values()
                ],
            }
        )

    mode = str(separation_mode)
    if mode == "single_dominant_no_separator":
        sep = DominantSpeakerPassthroughBackend()
    elif mode == "mock":
        sep = MockSeparationBackend(n_streams=cfg.max_speakers_hint)
    elif mode == "auto":
        sep = build_default_backend(cfg)
    else:
        raise ValueError(f"Unsupported separation_mode: {separation_mode}")

    pipe = RealtimeSpeakerPipeline(
        config=cfg,
        mic_geometry_xyz=mic_geometry_xyz,
        mic_geometry_xy=mic_geometry_xy,
        frame_iterator=_frame_iter(mic_audio, frame_samples),
        frame_sink=_sink,
        separation_backend=sep,
    )
    pipe_holder["pipe"] = pipe
    try:
        pipe.start()
        pipe.join()
    finally:
        pipe.stop()
        pipe.join(timeout=5.0)
        fast_alive = bool(getattr(pipe, "_fast", None) and pipe._fast.is_alive())
        slow_alive = bool(getattr(pipe, "_slow", None) and pipe._slow.is_alive())
        if fast_alive or slow_alive:
            raise RuntimeError(
                f"Pipeline threads did not terminate cleanly: fast_alive={fast_alive}, slow_alive={slow_alive}"
            )

    enhanced = np.concatenate(enhanced_parts)[: mic_audio.shape[0]] if enhanced_parts else np.zeros(mic_audio.shape[0], dtype=np.float32)
    raw_mix_mean = np.mean(np.asarray(mic_audio, dtype=np.float64), axis=1).astype(np.float32, copy=False)
    sf.write(out_root / "enhanced_fast_path.wav", enhanced, cfg.sample_rate_hz)
    sf.write(out_root / "raw_mix_mean.wav", raw_mix_mean, cfg.sample_rate_hz)

    final_speaker_map = pipe.shared_state.get_speaker_map_snapshot()
    speaker_map_rows = [
        {
            "speaker_id": int(v.speaker_id),
            "direction_degrees": float(v.direction_degrees),
            "gain_weight": float(v.gain_weight),
            "confidence": float(v.confidence),
            "active": bool(v.active),
            "activity_confidence": float(v.activity_confidence),
            "updated_at_ms": float(v.updated_at_ms),
            "identity_confidence": float(v.identity_confidence),
            "identity_maturity": str(v.identity_maturity),
        }
        for v in final_speaker_map.values()
    ]
    with (out_root / "speaker_map_final.json").open("w", encoding="utf-8") as f:
        json.dump({"speakers": speaker_map_rows}, f, indent=2)

    stats = pipe.stats_snapshot()
    summary = {
        "input_recording_path": "" if input_recording_path is None else str(Path(input_recording_path).resolve()),
        "sample_rate_hz": int(cfg.sample_rate_hz),
        "duration_s": float(mic_audio.shape[0] / max(cfg.sample_rate_hz, 1)),
        "fast_frame_ms": int(cfg.fast_frame_ms),
        "channel_count": int(mic_audio.shape[1]),
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
        "separation_mode": mode,
        "beamforming_mode": str(cfg.beamforming_mode),
        "fast_path_reference_mode": str(cfg.fast_path_reference_mode),
        "slow_chunk_ms": int(cfg.slow_chunk_ms),
        "slow_chunk_hop_ms": int(cfg.slow_chunk_ms if cfg.slow_chunk_hop_ms is None else cfg.slow_chunk_hop_ms),
        "output_normalization_enabled": bool(cfg.output_normalization_enabled),
        "output_allow_amplification": bool(cfg.output_allow_amplification),
        "speaker_map_final": speaker_map_rows,
        "robust_mode": bool(robust_mode),
        "localization_backend": str(cfg.localization_backend),
        "localization_window_ms": int(cfg.localization_window_ms),
        "localization_hop_ms": int(cfg.localization_hop_ms),
        "srp_overlap": float(cfg.srp_overlap),
        "srp_freq_min_hz": int(cfg.srp_freq_min_hz),
        "srp_freq_max_hz": int(cfg.srp_freq_max_hz),
        "tracking_mode": str(cfg.tracking_mode),
        "control_mode": str(cfg.control_mode),
        "direction_long_memory_enabled": bool(cfg.direction_long_memory_enabled),
        "direction_long_memory_window_ms": float(cfg.direction_long_memory_window_ms),
    }
    if capture_trace:
        summary["speaker_map_trace"] = speaker_map_trace
        summary["srp_trace"] = srp_trace

    with (out_root / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from simulation.simulation_config import SimulationConfig
from simulation.simulator import run_simulation
from simulation.target_policy import iter_target_source_indices

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


def run_simulation_pipeline(
    *,
    scene_config_path: str | Path,
    out_dir: str | Path,
    use_mock_separation: bool = True,
    fast_frame_ms: int = 10,
    slow_chunk_ms: int = 200,
    slow_chunk_hop_ms: int | None = None,
    beamforming_mode: str = "mvdr_fd",
    fast_path_reference_mode: str = "speaker_map",
    output_normalization_enabled: bool = True,
    output_allow_amplification: bool = False,
    write_raw_mix_output: bool = True,
    robust_mode: bool = True,
    capture_trace: bool = False,
    localization_backend: str = "srp_phat_localization",
    tracking_mode: str = "multi_peak_v2",
    control_mode: str = "spatial_peak_mode",
    localization_window_ms: int = 160,
    localization_hop_ms: int = 50,
    srp_overlap: float = 0.2,
    srp_freq_min_hz: int = 1200,
    srp_freq_max_hz: int = 5400,
    localization_vad_enabled: bool = True,
    localization_vad_rms_floor: float = 5e-4,
    localization_vad_speech_ratio_threshold: float = 0.62,
    localization_vad_rms_ratio_threshold: float = 1.2,
    localization_vad_flux_threshold: float = 0.12,
    localization_snr_gating_enabled: bool = True,
    localization_snr_threshold_db: float = 3.0,
    localization_snr_soft_range_db: float = 12.0,
    localization_snr_weight_exponent: float = 1.0,
    localization_noise_floor_alpha_fast: float = 0.35,
    localization_noise_floor_alpha_slow: float = 0.97,
    localization_msc_variance_enabled: bool = True,
    localization_msc_history_frames: int = 6,
    localization_msc_variance_floor: float = 0.002,
    localization_msc_weight_exponent: float = 1.0,
    localization_hsda_enabled: bool = True,
    localization_hsda_window_frames: int = 5,
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
    identity_retire_after_chunks: int | None = None,
    identity_new_speaker_max_existing_score: float | None = None,
    identity_direction_mismatch_block_deg: float | None = None,
    direction_focus_gain_db: float | None = None,
    direction_non_focus_attenuation_db: float | None = None,
    manual_target_speaker_id: int | None = None,
    auto_lock_first_identified_speaker: bool = False,
    target_user_boost_db: float = 0.0,
    auto_focus_active_speaker: bool = False,
    single_dominant_no_separator: bool = False,
) -> dict:
    sim_cfg = SimulationConfig.from_file(scene_config_path)
    mic_audio, mic_pos, _source_signals = run_simulation(sim_cfg)
    default_cfg = PipelineConfig()

    out_root = Path(out_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    cfg = PipelineConfig(
        sample_rate_hz=sim_cfg.audio.fs,
        fast_frame_ms=int(fast_frame_ms),
        slow_chunk_ms=int(slow_chunk_ms),
        slow_chunk_hop_ms=None if slow_chunk_hop_ms is None else int(slow_chunk_hop_ms),
        fast_path_reference_mode=str(fast_path_reference_mode),
        localization_window_ms=int(localization_window_ms),
        localization_hop_ms=int(localization_hop_ms),
        srp_overlap=float(srp_overlap),
        srp_freq_min_hz=int(srp_freq_min_hz),
        srp_freq_max_hz=int(srp_freq_max_hz),
        localization_vad_enabled=bool(localization_vad_enabled),
        localization_vad_rms_floor=float(localization_vad_rms_floor),
        localization_vad_speech_ratio_threshold=float(localization_vad_speech_ratio_threshold),
        localization_vad_rms_ratio_threshold=float(localization_vad_rms_ratio_threshold),
        localization_vad_flux_threshold=float(localization_vad_flux_threshold),
        localization_snr_gating_enabled=bool(localization_snr_gating_enabled),
        localization_snr_threshold_db=float(localization_snr_threshold_db),
        localization_snr_soft_range_db=float(localization_snr_soft_range_db),
        localization_snr_weight_exponent=float(localization_snr_weight_exponent),
        localization_noise_floor_alpha_fast=float(localization_noise_floor_alpha_fast),
        localization_noise_floor_alpha_slow=float(localization_noise_floor_alpha_slow),
        localization_msc_variance_enabled=bool(localization_msc_variance_enabled),
        localization_msc_history_frames=int(localization_msc_history_frames),
        localization_msc_variance_floor=float(localization_msc_variance_floor),
        localization_msc_weight_exponent=float(localization_msc_weight_exponent),
        localization_hsda_enabled=bool(localization_hsda_enabled),
        localization_hsda_window_frames=int(localization_hsda_window_frames),
        convtasnet_model_name=str(convtasnet_model_name),
        convtasnet_model_sample_rate_hz=int(convtasnet_model_sample_rate_hz),
        convtasnet_input_sample_rate_hz=int(convtasnet_input_sample_rate_hz),
        convtasnet_expected_num_sources=None if convtasnet_expected_num_sources is None else int(convtasnet_expected_num_sources),
        identity_backend=str(identity_backend),
        identity_speaker_embedding_model=str(identity_speaker_embedding_model),
        identity_new_speaker_max_existing_score=(
            float(identity_new_speaker_max_existing_score)
            if identity_new_speaker_max_existing_score is not None
            else float(default_cfg.identity_new_speaker_max_existing_score)
        ),
        identity_direction_mismatch_block_deg=(
            float(identity_direction_mismatch_block_deg)
            if identity_direction_mismatch_block_deg is not None
            else float(default_cfg.identity_direction_mismatch_block_deg)
        ),
        localization_backend=str(localization_backend),
        tracking_mode=str(tracking_mode),
        control_mode=str(control_mode),
        direction_long_memory_enabled=bool(direction_long_memory_enabled),
        direction_long_memory_window_ms=float(direction_long_memory_window_ms),
        direction_history_window_chunks=int(direction_history_window_chunks),
        direction_speaker_stale_timeout_ms=float(direction_speaker_stale_timeout_ms),
        direction_speaker_forget_timeout_ms=float(direction_speaker_forget_timeout_ms),
        direction_focus_gain_db=(
            float(direction_focus_gain_db)
            if direction_focus_gain_db is not None
            else float(default_cfg.direction_focus_gain_db)
        ),
        direction_non_focus_attenuation_db=(
            float(direction_non_focus_attenuation_db)
            if direction_non_focus_attenuation_db is not None
            else float(default_cfg.direction_non_focus_attenuation_db)
        ),
        max_speakers_hint=max(1, len(list(iter_target_source_indices(sim_cfg)))),
        beamforming_mode=str(beamforming_mode),
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
        identity_retire_after_chunks=(
            int(identity_retire_after_chunks)
            if identity_retire_after_chunks is not None
            else (25 if robust_mode else 1)
        ),
        direction_transition_penalty_deg=22.0 if robust_mode else 180.0,
        direction_min_confidence_for_switch=0.35 if robust_mode else 0.0,
        direction_hold_confidence_decay=0.9 if robust_mode else 1.0,
        direction_stale_confidence_decay=0.96 if robust_mode else 1.0,
        direction_min_persist_confidence=0.05 if robust_mode else 0.0,
        speaker_map_min_confidence_for_refresh=0.2 if robust_mode else 0.0,
        speaker_map_hold_ms=300.0 if robust_mode else 0.0,
        speaker_map_confidence_decay=0.9 if robust_mode else 1.0,
        speaker_map_activity_decay=0.92 if robust_mode else 1.0,
    )
    frame_samples = int(cfg.sample_rate_hz * cfg.fast_frame_ms / 1000)

    enhanced_parts: list[np.ndarray] = []
    speaker_map_trace: list[dict] = []
    srp_trace: list[dict] = []
    focus_trace: list[dict] = []
    pipe_holder: dict[str, RealtimeSpeakerPipeline] = {}
    auto_focus_state: dict[str, int | None] = {"speaker_id": None}
    target_lock_state: dict[str, int | None] = {
        "speaker_id": None if manual_target_speaker_id is None else int(manual_target_speaker_id)
    }

    def _sink(x: np.ndarray) -> None:
        enhanced_parts.append(np.asarray(x, dtype=np.float32).reshape(-1))
        pipe = pipe_holder.get("pipe")
        if pipe is None:
            return
        if target_lock_state["speaker_id"] is not None:
            pipe.set_focus_control(
                focused_speaker_ids=[int(target_lock_state["speaker_id"])],
                user_boost_db=float(target_user_boost_db),
            )
        elif auto_lock_first_identified_speaker:
            snapshot = pipe.shared_state.get_speaker_map_snapshot()
            best_sid: int | None = None
            best_key: tuple[int, float, float, float] | None = None
            for sid, item in snapshot.items():
                maturity = 1 if str(getattr(item, "identity_maturity", "unknown")) == "stable" else 0
                active_score = float(getattr(item, "activity_confidence", 0.0))
                conf = float(getattr(item, "confidence", 0.0))
                ident = float(getattr(item, "identity_confidence", 0.0))
                if maturity == 0 and conf < 0.45 and ident < 0.45:
                    continue
                key = (maturity, ident, conf, active_score)
                if best_key is None or key > best_key:
                    best_sid = int(sid)
                    best_key = key
            if best_sid is not None:
                target_lock_state["speaker_id"] = int(best_sid)
                pipe.set_focus_control(
                    focused_speaker_ids=[int(best_sid)],
                    user_boost_db=float(target_user_boost_db),
                )
        if auto_focus_active_speaker:
            snapshot = pipe.shared_state.get_speaker_map_snapshot()
            best_sid: int | None = None
            best_key: tuple[float, float, float] | None = None
            for sid, item in snapshot.items():
                active_score = float(getattr(item, "activity_confidence", 0.0))
                conf = float(getattr(item, "confidence", 0.0))
                ident = float(getattr(item, "identity_confidence", 0.0))
                if active_score < 0.2 and conf < 0.2 and ident < 0.2:
                    continue
                key = (active_score, conf, ident)
                if best_key is None or key > best_key:
                    best_sid = int(sid)
                    best_key = key
            if best_sid is not None and auto_focus_state["speaker_id"] != best_sid:
                pipe.set_focus_control(focused_speaker_ids=[best_sid], user_boost_db=0.0)
                auto_focus_state["speaker_id"] = best_sid

        if not capture_trace:
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
                        "predicted_direction_deg": None if v.predicted_direction_deg is None else float(v.predicted_direction_deg),
                        "angular_velocity_deg_per_chunk": float(v.angular_velocity_deg_per_chunk),
                        "last_separator_stream_index": None if v.last_separator_stream_index is None else int(v.last_separator_stream_index),
                        "anchor_direction_deg": None if v.anchor_direction_deg is None else float(v.anchor_direction_deg),
                        "anchor_confidence": float(v.anchor_confidence),
                        "anchor_locked": bool(v.anchor_locked),
                        "anchor_last_confirmed_ms": float(v.anchor_last_confirmed_ms),
                    }
                    for v in snapshot.values()
                ],
            }
        )
        focus = pipe.shared_state.get_focus_control_snapshot()
        focus_trace.append(
            {
                "frame_index": len(enhanced_parts) - 1,
                "focused_speaker_ids": None if focus.focused_speaker_ids is None else [int(v) for v in focus.focused_speaker_ids],
                "focused_direction_deg": None if focus.focused_direction_deg is None else float(focus.focused_direction_deg),
                "user_boost_db": float(focus.user_boost_db),
                "locked_target_speaker_id": None if target_lock_state["speaker_id"] is None else int(target_lock_state["speaker_id"]),
            }
        )

    separation_mode = "real_separator"
    if single_dominant_no_separator:
        sep = DominantSpeakerPassthroughBackend()
        separation_mode = "single_dominant_no_separator"
    elif use_mock_separation:
        sep = MockSeparationBackend(n_streams=cfg.max_speakers_hint)
        separation_mode = "mock_separator"
    else:
        sep = build_default_backend(cfg)

    mic_geometry_xyz = np.asarray(mic_pos, dtype=float)
    mic_geometry_xy = mic_geometry_xyz[:2, :].T

    pipe = RealtimeSpeakerPipeline(
        config=cfg,
        mic_geometry_xyz=mic_geometry_xyz,
        mic_geometry_xy=mic_geometry_xy,
        frame_iterator=_frame_iter(mic_audio, frame_samples),
        frame_sink=_sink,
        separation_backend=sep,
    )
    pipe_holder["pipe"] = pipe
    pipe.run_blocking()

    enhanced = np.concatenate(enhanced_parts)[: mic_audio.shape[0]] if enhanced_parts else np.zeros(mic_audio.shape[0], dtype=np.float32)
    sf.write(out_root / "enhanced_fast_path.wav", enhanced, cfg.sample_rate_hz)
    raw_mix_mean = np.mean(np.asarray(mic_audio, dtype=np.float64), axis=1).astype(np.float32, copy=False)
    if write_raw_mix_output:
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
            "predicted_direction_deg": None if v.predicted_direction_deg is None else float(v.predicted_direction_deg),
            "angular_velocity_deg_per_chunk": float(v.angular_velocity_deg_per_chunk),
            "last_separator_stream_index": None if v.last_separator_stream_index is None else int(v.last_separator_stream_index),
            "anchor_direction_deg": None if v.anchor_direction_deg is None else float(v.anchor_direction_deg),
            "anchor_confidence": float(v.anchor_confidence),
            "anchor_locked": bool(v.anchor_locked),
            "anchor_last_confirmed_ms": float(v.anchor_last_confirmed_ms),
        }
        for v in final_speaker_map.values()
    ]
    with (out_root / "speaker_map_final.json").open("w", encoding="utf-8") as f:
        json.dump({"speakers": speaker_map_rows}, f, indent=2)

    stats = pipe.stats_snapshot()
    summary = {
        "scene_config": str(Path(scene_config_path).resolve()),
        "sample_rate_hz": cfg.sample_rate_hz,
        "duration_s": float(mic_audio.shape[0] / cfg.sample_rate_hz),
        "fast_frame_ms": int(cfg.fast_frame_ms),
        "fast_frames": stats.fast_frames,
        "slow_chunks": stats.slow_chunks,
        "speaker_map_updates": stats.speaker_map_updates,
        "dropped_fast_to_slow_frames": stats.dropped_fast_to_slow_frames,
        "fast_avg_ms": stats.fast_avg_ms,
        "slow_avg_ms": stats.slow_avg_ms,
        "fast_rtf": stats.fast_rtf,
        "slow_rtf": stats.slow_rtf,
        "fast_stage_avg_ms": {
            "srp": stats.fast_srp_avg_ms,
            "beamform": stats.fast_beamform_avg_ms,
            "safety": stats.fast_safety_avg_ms,
            "sink": stats.fast_sink_avg_ms,
            "enqueue": stats.fast_enqueue_avg_ms,
        },
        "slow_stage_avg_ms": {
            "separation": stats.slow_separation_avg_ms,
            "identity": stats.slow_identity_avg_ms,
            "direction_assignment": stats.slow_direction_avg_ms,
            "publish": stats.slow_publish_avg_ms,
        },
        "use_mock_separation": bool(use_mock_separation),
        "single_dominant_no_separator": bool(single_dominant_no_separator),
        "separation_mode": str(separation_mode),
        "beamforming_mode": str(cfg.beamforming_mode),
        "fast_path_reference_mode": str(cfg.fast_path_reference_mode),
        "slow_chunk_ms": int(cfg.slow_chunk_ms),
        "slow_chunk_hop_ms": int(cfg.slow_chunk_ms if cfg.slow_chunk_hop_ms is None else cfg.slow_chunk_hop_ms),
        "output_normalization_enabled": bool(cfg.output_normalization_enabled),
        "output_allow_amplification": bool(cfg.output_allow_amplification),
        "speaker_map_final": speaker_map_rows,
        "robust_mode": bool(robust_mode),
        "localization_backend": str(cfg.localization_backend),
        "tracking_mode": str(cfg.tracking_mode),
        "control_mode": str(cfg.control_mode),
        "localization_window_ms": int(cfg.localization_window_ms),
        "localization_hop_ms": int(cfg.localization_hop_ms),
        "direction_long_memory_enabled": bool(cfg.direction_long_memory_enabled),
        "direction_long_memory_window_ms": float(cfg.direction_long_memory_window_ms),
        "direction_history_window_chunks": int(cfg.direction_history_window_chunks),
        "direction_speaker_stale_timeout_ms": float(cfg.direction_speaker_stale_timeout_ms),
        "direction_speaker_forget_timeout_ms": float(cfg.direction_speaker_forget_timeout_ms),
        "direction_focus_gain_db": float(cfg.direction_focus_gain_db),
        "direction_non_focus_attenuation_db": float(cfg.direction_non_focus_attenuation_db),
        "manual_target_speaker_id": None if manual_target_speaker_id is None else int(manual_target_speaker_id),
        "auto_lock_first_identified_speaker": bool(auto_lock_first_identified_speaker),
        "target_user_boost_db": float(target_user_boost_db),
        "final_locked_target_speaker_id": None if target_lock_state["speaker_id"] is None else int(target_lock_state["speaker_id"]),
    }
    if capture_trace:
        summary["speaker_map_trace"] = speaker_map_trace
        summary["srp_trace"] = srp_trace
        summary["focus_trace"] = focus_trace

    with (out_root / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if str(cfg.tracking_mode).strip().lower() == "dominant_lock_v1":
        (out_root / "dominant_lock_tradeoffs.txt").write_text(
            "\n".join(
                [
                    "dominant_lock_v1 tradeoffs",
                    "single dominant only",
                    "hold last lock during short uncertainty",
                    "very conservative switching",
                    "intended for low-motion scenes",
                    "fallback knobs for tomorrow morning:",
                    "- reduce dominant_lock_switch_confirm_frames",
                    "- reduce dominant_lock_challenger_margin",
                    "- reduce dominant_lock_hold_missing_frames",
                    "- increase dominant_lock_update_alpha",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

    return summary


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run integrated realtime pipeline on simulation scene JSON")
    p.add_argument("--scene-config", required=True)
    p.add_argument("--out-dir", default="realtime_pipeline/output/sim_run")
    p.add_argument("--real-separation", action="store_true", help="Use real Asteroid ConvTasNet instead of mock backend")
    p.add_argument("--validate-only", action="store_true", help="Run sanity checks and emit validation_report.json")
    p.add_argument("--beamforming-mode", choices=["mvdr_fd", "gsc_fd", "delay_sum"], default="mvdr_fd")
    p.add_argument("--fast-path-reference-mode", choices=["speaker_map", "srp_peak"], default="speaker_map")
    p.add_argument("--disable-output-normalization", action="store_true")
    p.add_argument("--allow-output-amplification", action="store_true")
    p.add_argument("--disable-robust-mode", action="store_true")
    p.add_argument(
        "--localization-backend",
        choices=[
            "srp_phat_legacy",
            "srp_phat_localization",
            "music_1src",
        ],
        default="srp_phat_localization",
    )
    p.add_argument("--tracking-mode", choices=["legacy", "multi_peak_v2", "dominant_lock_v1"], default="multi_peak_v2")
    p.add_argument("--control-mode", choices=["spatial_peak_mode", "speaker_tracking_mode"], default="spatial_peak_mode")
    p.add_argument("--disable-direction-long-memory", action="store_true")
    p.add_argument("--direction-long-memory-window-ms", type=float, default=60000.0)
    p.add_argument("--convtasnet-model-name", default="JorisCos/ConvTasNet_Libri2Mix_sepnoisy_16k")
    p.add_argument("--convtasnet-model-sample-rate-hz", type=int, default=16000)
    p.add_argument("--convtasnet-input-sample-rate-hz", type=int, default=16000)
    p.add_argument("--convtasnet-expected-num-sources", type=int, default=None)
    p.add_argument("--identity-backend", choices=["mfcc_legacy", "speaker_embed_session"], default="mfcc_legacy")
    p.add_argument(
        "--identity-speaker-embedding-model",
        choices=["ecapa_voxceleb", "wavlm_base_sv", "wavlm_base_plus_sv"],
        default="wavlm_base_plus_sv",
    )
    return p


def main() -> None:
    args = _build_arg_parser().parse_args()
    if args.validate_only:
        from .sanity_checks import run_sanity_checks

        report = run_sanity_checks(out_dir=args.out_dir, scene_config_path=args.scene_config)
        print(json.dumps(report, indent=2))
        return

    summary = run_simulation_pipeline(
        scene_config_path=args.scene_config,
        out_dir=args.out_dir,
        use_mock_separation=not args.real_separation,
        beamforming_mode=args.beamforming_mode,
        fast_path_reference_mode=args.fast_path_reference_mode,
        output_normalization_enabled=not args.disable_output_normalization,
        output_allow_amplification=bool(args.allow_output_amplification),
        robust_mode=not bool(args.disable_robust_mode),
        localization_backend=str(args.localization_backend),
        tracking_mode=str(args.tracking_mode),
        control_mode=str(args.control_mode),
        direction_long_memory_enabled=not bool(args.disable_direction_long_memory),
        direction_long_memory_window_ms=float(args.direction_long_memory_window_ms),
        convtasnet_model_name=str(args.convtasnet_model_name),
        convtasnet_model_sample_rate_hz=int(args.convtasnet_model_sample_rate_hz),
        convtasnet_input_sample_rate_hz=int(args.convtasnet_input_sample_rate_hz),
        convtasnet_expected_num_sources=(
            None if args.convtasnet_expected_num_sources is None else int(args.convtasnet_expected_num_sources)
        ),
        identity_backend=str(args.identity_backend),
        identity_speaker_embedding_model=str(args.identity_speaker_embedding_model),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

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
from .separation_backends import MockSeparationBackend, build_default_backend


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
    beamforming_mode: str = "mvdr_fd",
    output_normalization_enabled: bool = True,
    output_allow_amplification: bool = False,
    write_raw_mix_output: bool = True,
    robust_mode: bool = True,
    capture_trace: bool = False,
) -> dict:
    sim_cfg = SimulationConfig.from_file(scene_config_path)
    mic_audio, mic_pos, _source_signals = run_simulation(sim_cfg)

    out_root = Path(out_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    frame_ms = 10
    cfg = PipelineConfig(
        sample_rate_hz=sim_cfg.audio.fs,
        fast_frame_ms=frame_ms,
        slow_chunk_ms=200,
        max_speakers_hint=max(1, len(list(iter_target_source_indices(sim_cfg)))),
        beamforming_mode=str(beamforming_mode),
        output_normalization_enabled=bool(output_normalization_enabled),
        output_allow_amplification=bool(output_allow_amplification),
        srp_prior_enabled=bool(robust_mode),
        identity_continuity_bonus=0.12 if robust_mode else 0.0,
        identity_switch_penalty=0.2 if robust_mode else 0.0,
        identity_hold_similarity_threshold=0.45 if robust_mode else 1.1,
        identity_carry_forward_chunks=3 if robust_mode else 0,
        identity_confidence_decay=0.85 if robust_mode else 0.0,
        direction_transition_penalty_deg=35.0 if robust_mode else 180.0,
        direction_min_confidence_for_switch=0.55 if robust_mode else 0.0,
        direction_hold_confidence_decay=0.9 if robust_mode else 1.0,
        direction_stale_confidence_decay=0.96 if robust_mode else 1.0,
        direction_min_persist_confidence=0.05 if robust_mode else 0.0,
        speaker_map_min_confidence_for_refresh=0.2 if robust_mode else 0.0,
        speaker_map_hold_ms=800.0 if robust_mode else 0.0,
        speaker_map_confidence_decay=0.9 if robust_mode else 1.0,
        speaker_map_activity_decay=0.92 if robust_mode else 1.0,
    )
    frame_samples = int(cfg.sample_rate_hz * cfg.fast_frame_ms / 1000)

    enhanced_parts: list[np.ndarray] = []
    speaker_map_trace: list[dict] = []
    pipe_holder: dict[str, RealtimeSpeakerPipeline] = {}

    def _sink(x: np.ndarray) -> None:
        enhanced_parts.append(np.asarray(x, dtype=np.float32).reshape(-1))
        if not capture_trace:
            return
        pipe = pipe_holder.get("pipe")
        if pipe is None:
            return
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
                    }
                    for v in snapshot.values()
                ],
            }
        )

    if use_mock_separation:
        sep = MockSeparationBackend(n_streams=cfg.max_speakers_hint)
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
        "beamforming_mode": str(cfg.beamforming_mode),
        "output_normalization_enabled": bool(cfg.output_normalization_enabled),
        "output_allow_amplification": bool(cfg.output_allow_amplification),
        "speaker_map_final": speaker_map_rows,
        "robust_mode": bool(robust_mode),
    }
    if capture_trace:
        summary["speaker_map_trace"] = speaker_map_trace

    with (out_root / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run integrated realtime pipeline on simulation scene JSON")
    p.add_argument("--scene-config", required=True)
    p.add_argument("--out-dir", default="realtime_pipeline/output/sim_run")
    p.add_argument("--real-separation", action="store_true", help="Use real Asteroid ConvTasNet instead of mock backend")
    p.add_argument("--validate-only", action="store_true", help="Run sanity checks and emit validation_report.json")
    p.add_argument("--beamforming-mode", choices=["mvdr_fd", "gsc_fd", "delay_sum"], default="mvdr_fd")
    p.add_argument("--disable-output-normalization", action="store_true")
    p.add_argument("--allow-output-amplification", action="store_true")
    p.add_argument("--disable-robust-mode", action="store_true")
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
        output_normalization_enabled=not args.disable_output_normalization,
        output_allow_amplification=bool(args.allow_output_amplification),
        robust_mode=not bool(args.disable_robust_mode),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

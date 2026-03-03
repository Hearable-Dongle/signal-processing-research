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
    )
    frame_samples = int(cfg.sample_rate_hz * cfg.fast_frame_ms / 1000)

    enhanced_parts: list[np.ndarray] = []

    def _sink(x: np.ndarray) -> None:
        enhanced_parts.append(np.asarray(x, dtype=np.float32).reshape(-1))

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
    pipe.run_blocking()

    enhanced = np.concatenate(enhanced_parts)[: mic_audio.shape[0]] if enhanced_parts else np.zeros(mic_audio.shape[0], dtype=np.float32)
    sf.write(out_root / "enhanced_fast_path.wav", enhanced, cfg.sample_rate_hz)

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
    }

    with (out_root / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run integrated realtime pipeline on simulation scene JSON")
    p.add_argument("--scene-config", required=True)
    p.add_argument("--out-dir", default="realtime_pipeline/output/sim_run")
    p.add_argument("--real-separation", action="store_true", help="Use real Asteroid ConvTasNet instead of mock backend")
    p.add_argument("--validate-only", action="store_true", help="Run sanity checks and emit validation_report.json")
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
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from realtime_pipeline.streaming_adapter import RealtimeIntelligibilityAdapter
from simulation.simulation_config import SimulationConfig
from simulation.simulator import run_simulation


def _scene_paths_from_args(scene_dir: Path, scene_config: Path | None, max_scenes: int) -> list[Path]:
    if scene_config is not None:
        return [scene_config]
    paths = sorted(scene_dir.glob("*.json"))
    if not paths:
        raise FileNotFoundError(f"no scene configs found in {scene_dir}")
    return paths[: max(1, int(max_scenes))]


def _chunk_plan(total_samples: int, pattern: tuple[int, ...]) -> list[tuple[int, int]]:
    plan: list[tuple[int, int]] = []
    start = 0
    idx = 0
    while start < total_samples:
        width = int(pattern[idx % len(pattern)])
        end = min(total_samples, start + max(1, width))
        plan.append((start, end))
        start = end
        idx += 1
    return plan


def _float_mc_to_int16_channels(frame_mc: np.ndarray) -> list[np.ndarray]:
    clipped = np.clip(np.asarray(frame_mc, dtype=np.float32), -1.0, 1.0)
    pcm = np.clip(np.round(clipped * 32767.0), -32768.0, 32767.0).astype(np.int16)
    return [pcm[:, idx].copy() for idx in range(pcm.shape[1])]


def run_streaming_adapter_smoke(
    *,
    scene_paths: list[Path],
    out_dir: Path,
    processing_sample_rate_hz: int = 16000,
    chunk_pattern: tuple[int, ...] = (73, 211, 160, 97, 401),
) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    for scene_path in scene_paths:
        sim_cfg = SimulationConfig.from_file(scene_path)
        mic_audio, mic_pos, _ = run_simulation(sim_cfg)
        scene_dir = out_dir / scene_path.stem
        scene_dir.mkdir(parents=True, exist_ok=True)

        adapter = RealtimeIntelligibilityAdapter(
            mic_geometry_xyz=mic_pos,
            input_sample_rate_hz=int(sim_cfg.audio.fs),
            processing_sample_rate_hz=int(processing_sample_rate_hz),
            enable_resample=int(sim_cfg.audio.fs) != int(processing_sample_rate_hz),
            postfilter_method="off",
            postfilter_enabled=False,
        )
        try:
            parts: list[np.ndarray] = []
            for start, end in _chunk_plan(int(mic_audio.shape[0]), chunk_pattern):
                callback_channels = _float_mc_to_int16_channels(mic_audio[start:end, :])
                out = adapter.process_chunk(callback_channels)
                if out.size > 0:
                    parts.append(np.asarray(out, dtype=np.float32))
            flushed = adapter.flush()
            if flushed.size > 0:
                parts.append(np.asarray(flushed, dtype=np.float32))
        finally:
            adapter.close()

        enhanced = np.concatenate(parts, axis=0) if parts else np.zeros(0, dtype=np.float32)
        raw_mix = np.mean(np.asarray(mic_audio, dtype=np.float32), axis=1)
        if int(sim_cfg.audio.fs) != int(processing_sample_rate_hz):
            expected_samples = int(round((float(mic_audio.shape[0]) * float(processing_sample_rate_hz)) / float(sim_cfg.audio.fs)))
        else:
            expected_samples = int(mic_audio.shape[0])
        enhanced = enhanced[:expected_samples]

        sf.write(scene_dir / "enhanced_callback.wav", enhanced, int(processing_sample_rate_hz))
        sf.write(scene_dir / "raw_mix_reference.wav", raw_mix, int(sim_cfg.audio.fs))

        row = {
            "scene": scene_path.name,
            "input_sample_rate_hz": int(sim_cfg.audio.fs),
            "processing_sample_rate_hz": int(processing_sample_rate_hz),
            "input_samples": int(mic_audio.shape[0]),
            "output_samples": int(enhanced.shape[0]),
            "expected_output_samples": int(expected_samples),
            "channel_count": int(mic_audio.shape[1]),
            "chunk_calls": int(len(_chunk_plan(int(mic_audio.shape[0]), chunk_pattern))),
            "output_rms": float(np.sqrt(np.mean(enhanced.astype(np.float64) ** 2) + 1e-12)) if enhanced.size else 0.0,
            "output_peak": float(np.max(np.abs(enhanced))) if enhanced.size else 0.0,
        }
        rows.append(row)
        with (scene_dir / "summary.json").open("w", encoding="utf-8") as handle:
            json.dump(row, handle, indent=2)

    overall = {
        "scene_count": int(len(rows)),
        "processing_sample_rate_hz": int(processing_sample_rate_hz),
        "chunk_pattern": [int(v) for v in chunk_pattern],
        "scenes": rows,
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(overall, handle, indent=2)
    return overall


def main() -> None:
    parser = argparse.ArgumentParser(description="Feed simulation audio through RealtimeIntelligibilityAdapter like a callback.")
    parser.add_argument(
        "--scene-dir",
        type=Path,
        default=Path("simulation/simulations/configs/testing_specific_angles_silence_gaps"),
        help="Directory of simulation scene JSON configs.",
    )
    parser.add_argument("--scene-config", type=Path, default=None, help="Optional specific scene config to run.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("realtime_pipeline/output/streaming_adapter_sim_smoke"),
        help="Directory for smoke outputs.",
    )
    parser.add_argument("--max-scenes", type=int, default=1)
    parser.add_argument("--processing-sample-rate-hz", type=int, default=16000)
    args = parser.parse_args()

    scene_paths = _scene_paths_from_args(args.scene_dir, args.scene_config, args.max_scenes)
    summary = run_streaming_adapter_smoke(
        scene_paths=scene_paths,
        out_dir=args.out_dir,
        processing_sample_rate_hz=int(args.processing_sample_rate_hz),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

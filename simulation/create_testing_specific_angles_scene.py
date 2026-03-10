from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf

from simulation.simulation_config import MicrophoneArray, Room, SimulationAudio, SimulationConfig, SimulationSource
from sim.realistic_conversations.assets import AssetLibrary, load_asset_library


DEFAULT_CONFIG_ROOT = Path("simulation/simulations/configs/testing_specific_angles")
DEFAULT_ASSET_ROOT = Path("simulation/simulations/assets/testing_specific_angles")
DEFAULT_SEED = 42
DEFAULT_DURATION_SEC = 10.0
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_ROOM_DIMS = [8.0, 6.4, 3.1]
DEFAULT_ROOM_ABSORPTION = 0.5
DEFAULT_MIC_CENTER = [4.0, 3.2, 1.45]
DEFAULT_MIC_RADIUS = 0.1
DEFAULT_MIC_COUNT = 4
DEFAULT_SOURCE_RADIUS_SCALE = 0.28
SPEECH_WINDOW_SEC = 5.0
MAIN_ANGLES_DEG = list(range(0, 360, 45))
NOISE_SINGLE_ANGLES_DEG = list(range(0, 360, 45))
NOISE_OPPOSITE_PAIRS_DEG = [(0, 180), (45, 225), (90, 270), (135, 315)]


def _fade(samples: np.ndarray, sr: int, fade_ms: float = 20.0) -> np.ndarray:
    fade_len = min(samples.shape[0] // 2, max(1, int(sr * fade_ms / 1000.0)))
    if fade_len <= 0:
        return samples.astype(np.float32, copy=False)
    ramp = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
    out = samples.astype(np.float32, copy=True)
    out[:fade_len] *= ramp
    out[-fade_len:] *= ramp[::-1]
    return out


def _write_wav(path: Path, samples: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, samples.astype(np.float32, copy=False), sr)


def _normalize_peak(x: np.ndarray, peak: float = 0.98) -> np.ndarray:
    scale = float(np.max(np.abs(x))) if x.size else 0.0
    if scale <= 1e-8:
        return x.astype(np.float32, copy=False)
    return (x.astype(np.float32, copy=False) * float(peak / scale)).astype(np.float32, copy=False)


def _load_excerpt(path: Path, sr: int, length_samples: int, rng: np.random.Generator) -> np.ndarray:
    audio, _ = librosa.load(path, sr=sr, mono=True)
    if audio.size == 0:
        return np.zeros(length_samples, dtype=np.float32)
    if audio.shape[0] >= length_samples:
        start = int(rng.integers(0, audio.shape[0] - length_samples + 1))
        clip = audio[start : start + length_samples]
    else:
        repeats = int(np.ceil(length_samples / audio.shape[0]))
        clip = np.tile(audio, repeats)[:length_samples]
    clip = clip.astype(np.float32, copy=False)
    clip = clip - np.mean(clip)
    clip = _fade(clip, sr=sr)
    peak = float(np.max(np.abs(clip)))
    if peak > 1e-8:
        clip = 0.9 * clip / peak
    return clip.astype(np.float32, copy=False)


def _room_position(center: list[float], room_dims: list[float], angle_deg: float, radius_scale: float, z: float) -> list[float]:
    max_radius = min(float(room_dims[0]), float(room_dims[1])) * radius_scale
    theta = np.deg2rad(angle_deg)
    x = float(np.clip(center[0] + max_radius * np.cos(theta), 0.6, float(room_dims[0]) - 0.6))
    y = float(np.clip(center[1] + max_radius * np.sin(theta), 0.6, float(room_dims[1]) - 0.6))
    return [round(x, 4), round(y, 4), float(z)]


def _speech_track(
    *,
    path: Path,
    sr: int,
    total_samples: int,
    start_sec: float,
    duration_sec: float,
    rng: np.random.Generator,
) -> np.ndarray:
    start = int(round(start_sec * sr))
    length = int(round(duration_sec * sr))
    clip = _load_excerpt(path, sr=sr, length_samples=length, rng=rng)
    track = np.zeros(total_samples, dtype=np.float32)
    end = min(total_samples, start + clip.shape[0])
    track[start:end] = clip[: end - start]
    return track


def _noise_track(path: Path, sr: int, total_samples: int, rng: np.random.Generator, gain: float) -> np.ndarray:
    clip = _load_excerpt(path, sr=sr, length_samples=total_samples, rng=rng)
    return (gain * clip).astype(np.float32, copy=False)


def _render_sparse_asset(track: np.ndarray, asset_path: Path, sr: int) -> None:
    _write_wav(asset_path, _normalize_peak(track), sr)


def _scene_id(scene_idx: int) -> str:
    return f"testing_specific_angles_k2_scene{scene_idx:02d}"


def _noise_layouts() -> list[dict[str, Any]]:
    layouts: list[dict[str, Any]] = []
    for angle in NOISE_SINGLE_ANGLES_DEG:
        layouts.append(
            {
                "noise_layout_type": "single",
                "noise_angles_deg": [int(angle)],
            }
        )
    for a0, a1 in NOISE_OPPOSITE_PAIRS_DEG:
        layouts.append(
            {
                "noise_layout_type": "opposite_pair",
                "noise_angles_deg": [int(a0), int(a1)],
            }
        )
    return layouts


def _write_frame_truth(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "frame_index",
                "start_time_s",
                "end_time_s",
                "active_speaker_ids",
                "primary_speaker_id",
                "overlap",
                "overlap_count",
                "speaker_positions",
                "frame_snr_db",
            ],
        )
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out["active_speaker_ids"] = "|".join(str(v) for v in row["active_speaker_ids"])
            out["speaker_positions"] = json.dumps(row["speaker_positions"], separators=(",", ":"))
            writer.writerow(out)


def generate_testing_specific_angles_dataset(
    *,
    config_root: str | Path = DEFAULT_CONFIG_ROOT,
    asset_root: str | Path = DEFAULT_ASSET_ROOT,
    seed: int = DEFAULT_SEED,
    duration_sec: float = DEFAULT_DURATION_SEC,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    manifest_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    config_root = Path(config_root)
    asset_root = Path(asset_root)
    config_root.mkdir(parents=True, exist_ok=True)
    asset_root.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    library: AssetLibrary = load_asset_library(manifest_path)
    total_samples = int(round(duration_sec * sample_rate))
    room_dims = list(DEFAULT_ROOM_DIMS)
    center = list(DEFAULT_MIC_CENTER)
    speech_window_samples = int(round(SPEECH_WINDOW_SEC * sample_rate))
    summary_rows: list[dict[str, Any]] = []
    scene_idx = 0

    for main_angle_deg in MAIN_ANGLES_DEG:
        secondary_angle_deg = int((main_angle_deg + 45) % 360)
        for layout in _noise_layouts():
            scene_name = _scene_id(scene_idx)
            scene_dir = asset_root / scene_name
            scene_dir.mkdir(parents=True, exist_ok=True)

            speaker_labels = library.choose_speakers(rng, 2)
            speech_paths = [
                library.choose_speech_path(rng, speaker_labels[0]),
                library.choose_speech_path(rng, speaker_labels[1]),
            ]
            noise_paths = [library.choose_noise_path(rng) for _ in layout["noise_angles_deg"]]

            speech_tracks = [
                _speech_track(
                    path=speech_paths[0],
                    sr=sample_rate,
                    total_samples=total_samples,
                    start_sec=0.0,
                    duration_sec=SPEECH_WINDOW_SEC,
                    rng=rng,
                ),
                _speech_track(
                    path=speech_paths[1],
                    sr=sample_rate,
                    total_samples=total_samples,
                    start_sec=SPEECH_WINDOW_SEC,
                    duration_sec=SPEECH_WINDOW_SEC,
                    rng=rng,
                ),
            ]
            noise_tracks = [
                _noise_track(path=noise_paths[idx], sr=sample_rate, total_samples=total_samples, rng=rng, gain=0.35)
                for idx in range(len(noise_paths))
            ]

            render_assets = scene_dir / "render_assets"
            speaker_asset_paths = [
                render_assets / "speaker_0_seg000.wav",
                render_assets / "speaker_1_seg000.wav",
            ]
            for track, asset_path in zip(speech_tracks, speaker_asset_paths):
                _render_sparse_asset(track, asset_path, sample_rate)

            noise_asset_paths: list[Path] = []
            for idx, track in enumerate(noise_tracks):
                asset_path = render_assets / f"noise_{idx}_seg000.wav"
                _render_sparse_asset(track, asset_path, sample_rate)
                noise_asset_paths.append(asset_path)

            source_rows: list[SimulationSource] = []
            speech_angles = [int(main_angle_deg), int(secondary_angle_deg)]
            speech_positions = [
                _room_position(center, room_dims, speech_angles[0], DEFAULT_SOURCE_RADIUS_SCALE, 1.45),
                _room_position(center, room_dims, speech_angles[1], DEFAULT_SOURCE_RADIUS_SCALE, 1.45),
            ]
            for idx, asset_path in enumerate(speaker_asset_paths):
                source_rows.append(
                    SimulationSource(
                        loc=speech_positions[idx],
                        audio_path=str(asset_path.resolve()),
                        gain=1.0,
                        classification="speech",
                    )
                )

            noise_positions: list[list[float]] = []
            for idx, angle_deg in enumerate(layout["noise_angles_deg"]):
                noise_pos = _room_position(center, room_dims, float(angle_deg), DEFAULT_SOURCE_RADIUS_SCALE * 1.18, 1.2)
                noise_positions.append(noise_pos)
                source_rows.append(
                    SimulationSource(
                        loc=noise_pos,
                        audio_path=str(noise_asset_paths[idx].resolve()),
                        gain=1.0,
                        classification="noise",
                    )
                )

            sim_cfg = SimulationConfig(
                room=Room(dimensions=room_dims, absorption=DEFAULT_ROOM_ABSORPTION),
                microphone_array=MicrophoneArray(
                    mic_center=center,
                    mic_radius=DEFAULT_MIC_RADIUS,
                    mic_count=DEFAULT_MIC_COUNT,
                ),
                audio=SimulationAudio(
                    sources=source_rows,
                    duration=float(duration_sec),
                    fs=int(sample_rate),
                ),
            )
            scene_config_path = scene_dir / "scene_config.json"
            sim_cfg.to_file(scene_config_path)
            config_out = config_root / f"{scene_name}.json"
            sim_cfg.to_file(config_out)

            frame_rows = []
            speaker_positions_map = {"0": speech_positions[0], "1": speech_positions[1]}
            for frame_idx in range(int(duration_sec / 0.02)):
                start_sec = frame_idx * 0.02
                end_sec = min(duration_sec, start_sec + 0.02)
                active_ids = [0] if start_sec < SPEECH_WINDOW_SEC else [1]
                primary_id = active_ids[0]
                frame_rows.append(
                    {
                        "frame_index": frame_idx,
                        "start_time_s": round(start_sec, 4),
                        "end_time_s": round(end_sec, 4),
                        "active_speaker_ids": active_ids,
                        "primary_speaker_id": primary_id,
                        "overlap": False,
                        "overlap_count": len(active_ids),
                        "speaker_positions": {str(primary_id): speaker_positions_map[str(primary_id)]},
                        "frame_snr_db": 0.0,
                    }
                )
            frame_json_path = scene_dir / "frame_ground_truth.json"
            frame_json_path.write_text(json.dumps(frame_rows, indent=2), encoding="utf-8")
            _write_frame_truth(scene_dir / "frame_ground_truth.csv", frame_rows)

            metadata = {
                "scene_name": scene_name,
                "scene_type": "testing_specific_angles",
                "seed": int(seed),
                "speaker_count": 2,
                "main_angle_deg": int(main_angle_deg),
                "secondary_angle_deg": int(secondary_angle_deg),
                "noise_layout_type": str(layout["noise_layout_type"]),
                "noise_angles_deg": [int(v) for v in layout["noise_angles_deg"]],
                "room": {
                    "dimensions_m": room_dims,
                    "absorption": DEFAULT_ROOM_ABSORPTION,
                },
                "mic_array": {
                    "mic_center_m": center,
                    "mic_radius_m": DEFAULT_MIC_RADIUS,
                    "mic_count": DEFAULT_MIC_COUNT,
                },
                "assets": {
                    "speech": [
                        {
                            "speaker_id": idx,
                            "speaker_label": speaker_labels[idx],
                            "source_path": str(speech_paths[idx].resolve()),
                            "render_asset_path": str(speaker_asset_paths[idx].resolve()),
                            "active_window_sec": [0.0, 5.0] if idx == 0 else [5.0, 10.0],
                            "position_m": speech_positions[idx],
                            "angle_deg": speech_angles[idx],
                        }
                        for idx in range(2)
                    ],
                    "noise": [
                        {
                            "noise_id": idx,
                            "source_path": str(noise_paths[idx].resolve()),
                            "render_asset_path": str(noise_asset_paths[idx].resolve()),
                            "active_window_sec": [0.0, duration_sec],
                            "position_m": noise_positions[idx],
                            "angle_deg": int(layout["noise_angles_deg"][idx]),
                        }
                        for idx in range(len(noise_paths))
                    ],
                },
                "summary_metrics": {
                    "overlap_ratio": 0.0,
                    "num_turn_events": 2,
                    "num_backchannels": 0,
                    "speaker_activity_frames": {
                        "0": speech_window_samples // int(round(0.02 * sample_rate)),
                        "1": speech_window_samples // int(round(0.02 * sample_rate)),
                    },
                },
            }
            metadata_path = scene_dir / "scenario_metadata.json"
            metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            (scene_dir / "event_schedule.json").write_text(
                json.dumps(
                    {
                        "speech_events": [
                            {"speaker_id": 0, "start_sec": 0.0, "end_sec": 5.0, "angle_deg": int(main_angle_deg)},
                            {"speaker_id": 1, "start_sec": 5.0, "end_sec": 10.0, "angle_deg": int(secondary_angle_deg)},
                        ],
                        "noise_events": [
                            {
                                "noise_id": idx,
                                "start_sec": 0.0,
                                "end_sec": duration_sec,
                                "angle_deg": int(angle),
                            }
                            for idx, angle in enumerate(layout["noise_angles_deg"])
                        ],
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

            summary_rows.append(
                {
                    "scene_name": scene_name,
                    "scene_config_path": str(config_out.resolve()),
                    "scene_dir": str(scene_dir.resolve()),
                    "main_angle_deg": int(main_angle_deg),
                    "secondary_angle_deg": int(secondary_angle_deg),
                    "noise_layout_type": str(layout["noise_layout_type"]),
                    "noise_angles_deg": [int(v) for v in layout["noise_angles_deg"]],
                }
            )
            scene_idx += 1

    summary_path = asset_root / "generation_summary.json"
    summary_path.write_text(json.dumps({"scenes": summary_rows}, indent=2), encoding="utf-8")
    return summary_rows


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate fixed-angle testing scenes for ReSpeaker localization evaluation")
    parser.add_argument("--config-root", default=str(DEFAULT_CONFIG_ROOT))
    parser.add_argument("--asset-root", default=str(DEFAULT_ASSET_ROOT))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--duration-sec", type=float, default=DEFAULT_DURATION_SEC)
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE)
    parser.add_argument("--asset-manifest", default=None)
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    generated = generate_testing_specific_angles_dataset(
        config_root=args.config_root,
        asset_root=args.asset_root,
        seed=args.seed,
        duration_sec=float(args.duration_sec),
        sample_rate=int(args.sample_rate),
        manifest_path=args.asset_manifest,
    )
    print(json.dumps({"num_scenes": len(generated), "scenes": generated}, indent=2))


if __name__ == "__main__":
    main()

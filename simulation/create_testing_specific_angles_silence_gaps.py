from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from sim.realistic_conversations.assets import AssetLibrary, load_asset_library
from simulation.create_testing_specific_angles_scene import (
    DEFAULT_MIC_CENTER,
    DEFAULT_MIC_COUNT,
    DEFAULT_MIC_RADIUS,
    DEFAULT_ROOM_ABSORPTION,
    DEFAULT_ROOM_DIMS,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_SEED,
    _load_excerpt,
    _normalize_peak,
    _render_sparse_asset,
    _room_position,
)
from simulation.simulation_config import MicrophoneArray, Room, SimulationAudio, SimulationConfig, SimulationSource


DEFAULT_CONFIG_ROOT = Path("simulation/simulations/configs/testing_specific_angles_silence_gaps")
DEFAULT_ASSET_ROOT = Path("simulation/simulations/assets/testing_specific_angles_silence_gaps")
DEFAULT_DURATION_SEC = 10.0
DEFAULT_SOURCE_RADIUS_SCALE = 0.28
FRAME_MS = 20
SILENCE_WINDOWS_SEC: tuple[tuple[float, float], ...] = (
    (1.9, 2.6),
    (4.5, 5.2),
    (7.1, 7.8),
    (9.7, 10.0),
)
SPEECH_WINDOWS_BY_CANONICAL_SPEAKER: dict[int, tuple[tuple[float, float], ...]] = {
    0: ((0.0, 1.9), (5.2, 7.1)),
    1: ((2.6, 4.5), (7.8, 9.7)),
}


@dataclass(frozen=True)
class NoiseProfile:
    angle_deg: int
    gain_points: tuple[float, ...]
    keyframe_sec: tuple[float, ...] = (0.0, 2.5, 5.0, 7.5, 10.0)


@dataclass(frozen=True)
class SilenceGapSceneSpec:
    scene_idx: int
    speaker_angles_deg: tuple[int, int]
    noise_profiles: tuple[NoiseProfile, ...]


SCENE_SPECS: tuple[SilenceGapSceneSpec, ...] = (
    SilenceGapSceneSpec(0, (0, 45), (NoiseProfile(135, (0.22, 0.28, 0.36, 0.30, 0.42)),)),
    SilenceGapSceneSpec(1, (0, 60), (NoiseProfile(180, (0.18, 0.24, 0.32, 0.38, 0.46)),)),
    SilenceGapSceneSpec(2, (45, 135), (NoiseProfile(270, (0.30, 0.26, 0.22, 0.28, 0.34)),)),
    SilenceGapSceneSpec(3, (90, 180), (NoiseProfile(315, (0.20, 0.28, 0.40, 0.34, 0.30)),)),
    SilenceGapSceneSpec(
        4,
        (135, 225),
        (
            NoiseProfile(0, (0.12, 0.18, 0.24, 0.30, 0.34)),
            NoiseProfile(300, (0.20, 0.16, 0.14, 0.22, 0.28)),
        ),
    ),
    SilenceGapSceneSpec(
        5,
        (270, 330),
        (
            NoiseProfile(120, (0.22, 0.30, 0.38, 0.32, 0.26)),
            NoiseProfile(210, (0.10, 0.14, 0.20, 0.24, 0.30)),
        ),
    ),
)


def _scene_id(scene_idx: int) -> str:
    return f"testing_specific_angles_silence_gaps_k2_scene{scene_idx:02d}"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


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
                "primary_doa_deg",
                "target_active",
                "silence_gap",
                "overlap",
                "overlap_count",
                "speaker_positions",
                "frame_snr_db",
                "noise_scale_hint",
            ],
        )
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out["active_speaker_ids"] = "|".join(str(v) for v in row["active_speaker_ids"])
            out["speaker_positions"] = json.dumps(row["speaker_positions"], separators=(",", ":"))
            writer.writerow(out)


def _build_sparse_speech_track(
    *,
    path: Path,
    sr: int,
    total_samples: int,
    windows_sec: tuple[tuple[float, float], ...],
    rng: np.random.Generator,
) -> np.ndarray:
    track = np.zeros(total_samples, dtype=np.float32)
    for start_sec, end_sec in windows_sec:
        start = int(round(float(start_sec) * sr))
        end = int(round(float(end_sec) * sr))
        length = max(1, end - start)
        clip = _load_excerpt(path, sr=sr, length_samples=length, rng=rng)
        stop = min(total_samples, start + clip.shape[0])
        track[start:stop] += clip[: stop - start]
    peak = float(np.max(np.abs(track))) if track.size else 0.0
    if peak > 1e-8:
        track = 0.9 * track / peak
    return track.astype(np.float32, copy=False)


def _build_ramped_noise_track(
    *,
    path: Path,
    sr: int,
    total_samples: int,
    keyframe_sec: tuple[float, ...],
    gain_points: tuple[float, ...],
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    base = _load_excerpt(path, sr=sr, length_samples=total_samples, rng=rng).astype(np.float32, copy=False)
    key_samples = np.asarray([int(round(float(sec) * sr)) for sec in keyframe_sec], dtype=np.int64)
    key_samples = np.clip(key_samples, 0, max(0, total_samples - 1))
    envelope = np.interp(
        np.arange(total_samples, dtype=np.float64),
        key_samples.astype(np.float64),
        np.asarray(gain_points, dtype=np.float64),
    ).astype(np.float32, copy=False)
    track = base * envelope
    schedule = [
        {
            "start_sec": float(keyframe_sec[idx]),
            "end_sec": float(keyframe_sec[idx + 1]),
            "gain_start": float(gain_points[idx]),
            "gain_end": float(gain_points[idx + 1]),
        }
        for idx in range(len(keyframe_sec) - 1)
    ]
    return track.astype(np.float32, copy=False), schedule


def _noise_hint_at_time(noise_profiles: tuple[NoiseProfile, ...], time_s: float) -> float:
    total = 0.0
    for profile in noise_profiles:
        total += float(
            np.interp(
                float(time_s),
                np.asarray(profile.keyframe_sec, dtype=np.float64),
                np.asarray(profile.gain_points, dtype=np.float64),
            )
        )
    return total


def _build_frame_rows(
    *,
    duration_sec: float,
    speaker_positions: dict[int, list[float]],
    speaker_angles_deg: dict[int, int],
    noise_profiles: tuple[NoiseProfile, ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    frame_count = int(round(float(duration_sec) * 1000.0 / FRAME_MS))
    for frame_idx in range(frame_count):
        start_sec = frame_idx * (FRAME_MS / 1000.0)
        end_sec = min(float(duration_sec), start_sec + (FRAME_MS / 1000.0))
        active_ids = [
            canonical_speaker_id
            for canonical_speaker_id, windows in SPEECH_WINDOWS_BY_CANONICAL_SPEAKER.items()
            if any(float(start) <= start_sec < float(end) for start, end in windows)
        ]
        primary_id = int(active_ids[0]) if active_ids else -1
        primary_doa = float(speaker_angles_deg[primary_id]) if primary_id >= 0 else ""
        rows.append(
            {
                "frame_index": int(frame_idx),
                "start_time_s": round(start_sec, 4),
                "end_time_s": round(end_sec, 4),
                "active_speaker_ids": list(active_ids),
                "primary_speaker_id": int(primary_id),
                "primary_doa_deg": primary_doa,
                "target_active": bool(primary_id >= 0),
                "silence_gap": bool(primary_id < 0),
                "overlap": False,
                "overlap_count": int(len(active_ids)),
                "speaker_positions": {} if primary_id < 0 else {str(primary_id): speaker_positions[primary_id]},
                "frame_snr_db": 0.0,
                "noise_scale_hint": round(_noise_hint_at_time(noise_profiles, 0.5 * (start_sec + end_sec)), 4),
            }
        )
    return rows


def generate_testing_specific_angles_silence_gaps_dataset(
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
    total_samples = int(round(float(duration_sec) * sample_rate))
    room_dims = list(DEFAULT_ROOM_DIMS)
    center = list(DEFAULT_MIC_CENTER)
    summary_rows: list[dict[str, Any]] = []

    for spec in SCENE_SPECS:
        scene_name = _scene_id(spec.scene_idx)
        scene_dir = asset_root / scene_name
        render_assets = scene_dir / "render_assets"
        render_assets.mkdir(parents=True, exist_ok=True)

        speaker_labels = library.choose_speakers(rng, 2)
        speech_paths = [
            library.choose_speech_path(rng, speaker_labels[0]),
            library.choose_speech_path(rng, speaker_labels[1]),
        ]
        noise_paths = [library.choose_noise_path(rng) for _ in spec.noise_profiles]

        speaker_positions = {
            0: _room_position(center, room_dims, float(spec.speaker_angles_deg[0]), DEFAULT_SOURCE_RADIUS_SCALE, 1.45),
            1: _room_position(center, room_dims, float(spec.speaker_angles_deg[1]), DEFAULT_SOURCE_RADIUS_SCALE, 1.45),
        }
        speaker_tracks = [
            _build_sparse_speech_track(
                path=speech_paths[idx],
                sr=sample_rate,
                total_samples=total_samples,
                windows_sec=SPEECH_WINDOWS_BY_CANONICAL_SPEAKER[idx],
                rng=rng,
            )
            for idx in range(2)
        ]
        speaker_asset_paths = [render_assets / "speaker_0_seg000.wav", render_assets / "speaker_1_seg000.wav"]
        for track, path in zip(speaker_tracks, speaker_asset_paths):
            _render_sparse_asset(track, path, sample_rate)

        noise_asset_paths: list[Path] = []
        noise_schedule_rows: list[dict[str, Any]] = []
        for noise_idx, (noise_path, profile) in enumerate(zip(noise_paths, spec.noise_profiles)):
            noise_track, schedule = _build_ramped_noise_track(
                path=noise_path,
                sr=sample_rate,
                total_samples=total_samples,
                keyframe_sec=profile.keyframe_sec,
                gain_points=profile.gain_points,
                rng=rng,
            )
            asset_path = render_assets / f"noise_{noise_idx}_seg000.wav"
            _render_sparse_asset(_normalize_peak(noise_track), asset_path, sample_rate)
            noise_asset_paths.append(asset_path)
            noise_schedule_rows.append(
                {
                    "noise_id": int(noise_idx),
                    "angle_deg": int(profile.angle_deg),
                    "keyframe_sec": [float(v) for v in profile.keyframe_sec],
                    "gain_points": [float(v) for v in profile.gain_points],
                    "segments": schedule,
                }
            )

        source_rows: list[SimulationSource] = [
            SimulationSource(
                loc=list(speaker_positions[idx]),
                audio_path=str(speaker_asset_paths[idx].resolve()),
                gain=1.0,
                classification="speech",
            )
            for idx in range(2)
        ]
        noise_positions: list[list[float]] = []
        for noise_idx, profile in enumerate(spec.noise_profiles):
            noise_pos = _room_position(center, room_dims, float(profile.angle_deg), DEFAULT_SOURCE_RADIUS_SCALE * 1.18, 1.2)
            noise_positions.append(noise_pos)
            source_rows.append(
                SimulationSource(
                    loc=noise_pos,
                    audio_path=str(noise_asset_paths[noise_idx].resolve()),
                    gain=1.0,
                    classification="noise",
                )
            )

        sim_cfg = SimulationConfig(
            room=Room(dimensions=room_dims, absorption=DEFAULT_ROOM_ABSORPTION),
            microphone_array=MicrophoneArray(
                mic_center=center,
                mic_radius=float(DEFAULT_MIC_RADIUS),
                mic_count=int(DEFAULT_MIC_COUNT),
            ),
            audio=SimulationAudio(
                sources=source_rows,
                duration=float(duration_sec),
                fs=int(sample_rate),
            ),
        )
        sim_cfg.to_file(scene_dir / "scene_config.json")
        sim_cfg.to_file(config_root / f"{scene_name}.json")

        frame_rows = _build_frame_rows(
            duration_sec=duration_sec,
            speaker_positions=speaker_positions,
            speaker_angles_deg={0: int(spec.speaker_angles_deg[0]), 1: int(spec.speaker_angles_deg[1])},
            noise_profiles=spec.noise_profiles,
        )
        _write_json(scene_dir / "frame_ground_truth.json", frame_rows)
        _write_frame_truth(scene_dir / "frame_ground_truth.csv", frame_rows)

        speech_event_rows = []
        for canonical_speaker_id, windows in SPEECH_WINDOWS_BY_CANONICAL_SPEAKER.items():
            for turn_index, (start_sec, end_sec) in enumerate(windows):
                speech_event_rows.append(
                    {
                        "speaker_id": int(canonical_speaker_id),
                        "speaker_label": str(speaker_labels[canonical_speaker_id]),
                        "source_path": str(speech_paths[canonical_speaker_id].resolve()),
                        "render_asset_path": str(speaker_asset_paths[canonical_speaker_id].resolve()),
                        "active_window_sec": [float(start_sec), float(end_sec)],
                        "position_m": speaker_positions[canonical_speaker_id],
                        "angle_deg": int(spec.speaker_angles_deg[canonical_speaker_id]),
                        "turn_index": int(len(speech_event_rows)),
                    }
                )

        metadata = {
            "scene_name": scene_name,
            "scene_type": "testing_specific_angles_silence_gaps",
            "scene_prefix": "testing_specific_angles_silence_gaps",
            "seed": int(seed),
            "speaker_count": 2,
            "turn_pattern": "ABAB",
            "main_angle_deg": int(spec.speaker_angles_deg[0]),
            "secondary_angle_deg": int(spec.speaker_angles_deg[1]),
            "silence_windows_sec": [[float(start), float(end)] for start, end in SILENCE_WINDOWS_SEC],
            "speaker_turn_order": [0, 1, 0, 1],
            "target_activity_schedule": {
                "speaker_0_windows_sec": [[float(start), float(end)] for start, end in SPEECH_WINDOWS_BY_CANONICAL_SPEAKER[0]],
                "speaker_1_windows_sec": [[float(start), float(end)] for start, end in SPEECH_WINDOWS_BY_CANONICAL_SPEAKER[1]],
            },
            "noise_schedule": noise_schedule_rows,
            "noise_layout_type": "single" if len(spec.noise_profiles) == 1 else "custom_2",
            "noise_angles_deg": [int(profile.angle_deg) for profile in spec.noise_profiles],
            "room": {
                "dimensions_m": room_dims,
                "absorption": float(DEFAULT_ROOM_ABSORPTION),
            },
            "mic_array": {
                "mic_center_m": center,
                "mic_radius_m": float(DEFAULT_MIC_RADIUS),
                "mic_count": int(DEFAULT_MIC_COUNT),
            },
            "assets": {
                "speech": speech_event_rows,
                "noise": [
                    {
                        "noise_id": int(noise_idx),
                        "source_path": str(noise_paths[noise_idx].resolve()),
                        "render_asset_path": str(noise_asset_paths[noise_idx].resolve()),
                        "active_window_sec": [0.0, float(duration_sec)],
                        "position_m": noise_positions[noise_idx],
                        "angle_deg": int(spec.noise_profiles[noise_idx].angle_deg),
                        "keyframe_sec": [float(v) for v in spec.noise_profiles[noise_idx].keyframe_sec],
                        "gain_points": [float(v) for v in spec.noise_profiles[noise_idx].gain_points],
                    }
                    for noise_idx in range(len(spec.noise_profiles))
                ],
            },
            "summary_metrics": {
                "overlap_ratio": 0.0,
                "num_turn_events": 4,
                "num_silence_gaps": len(SILENCE_WINDOWS_SEC),
                "silence_ratio": float(sum(end - start for start, end in SILENCE_WINDOWS_SEC) / duration_sec),
                "mean_gap_sec": float(np.mean([end - start for start, end in SILENCE_WINDOWS_SEC])),
                "speaker_activity_frames": {
                    "0": int(sum((end - start) * 1000.0 / FRAME_MS for start, end in SPEECH_WINDOWS_BY_CANONICAL_SPEAKER[0])),
                    "1": int(sum((end - start) * 1000.0 / FRAME_MS for start, end in SPEECH_WINDOWS_BY_CANONICAL_SPEAKER[1])),
                },
            },
        }
        _write_json(scene_dir / "scenario_metadata.json", metadata)
        _write_json(
            scene_dir / "event_schedule.json",
            {
                "speech_events": [
                    {
                        "speaker_id": int(row["speaker_id"]),
                        "start_sec": float(row["active_window_sec"][0]),
                        "end_sec": float(row["active_window_sec"][1]),
                        "angle_deg": int(row["angle_deg"]),
                    }
                    for row in speech_event_rows
                ],
                "silence_events": [
                    {"start_sec": float(start), "end_sec": float(end)}
                    for start, end in SILENCE_WINDOWS_SEC
                ],
                "noise_events": [
                    {
                        "noise_id": int(row["noise_id"]),
                        "start_sec": 0.0,
                        "end_sec": float(duration_sec),
                        "angle_deg": int(row["angle_deg"]),
                        "keyframe_sec": row["keyframe_sec"],
                        "gain_points": row["gain_points"],
                    }
                    for row in noise_schedule_rows
                ],
            },
        )

        summary_rows.append(
            {
                "scene_name": scene_name,
                "scene_config_path": str((config_root / f"{scene_name}.json").resolve()),
                "scene_dir": str(scene_dir.resolve()),
                "main_angle_deg": int(spec.speaker_angles_deg[0]),
                "secondary_angle_deg": int(spec.speaker_angles_deg[1]),
                "noise_angles_deg": [int(profile.angle_deg) for profile in spec.noise_profiles],
                "silence_windows_sec": [[float(start), float(end)] for start, end in SILENCE_WINDOWS_SEC],
            }
        )

    _write_json(asset_root / "generation_summary.json", {"scenes": summary_rows})
    return summary_rows


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create deterministic two-speaker angle scenes with explicit silence gaps and drifting noise.")
    parser.add_argument("--config-root", default=str(DEFAULT_CONFIG_ROOT))
    parser.add_argument("--asset-root", default=str(DEFAULT_ASSET_ROOT))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--duration-sec", type=float, default=DEFAULT_DURATION_SEC)
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE)
    parser.add_argument("--manifest-path", default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    rows = generate_testing_specific_angles_silence_gaps_dataset(
        config_root=args.config_root,
        asset_root=args.asset_root,
        seed=int(args.seed),
        duration_sec=float(args.duration_sec),
        sample_rate=int(args.sample_rate),
        manifest_path=args.manifest_path,
    )
    print(json.dumps({"n_scenes": len(rows), "config_root": str(Path(args.config_root).resolve())}, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sim.realistic_conversations.assets import AssetLibrary, load_asset_library
from simulation.create_testing_specific_angles_near_target_far_diffuse import (
    DEFAULT_BACKGROUND_RADIUS_SCALE,
    DEFAULT_BABBLE_COUNT,
    DEFAULT_BABBLE_GAIN_MAX,
    DEFAULT_BABBLE_GAIN_MIN,
    DEFAULT_BOOTSTRAP_SEC,
    DEFAULT_DURATION_SEC,
    DEFAULT_TARGET_RADIUS_SCALE,
    DEFAULT_WHAM_GAIN,
    FRAME_MS,
    _background_centroid_deg,
    _continuous_track,
    _diffuse_babble_angles,
    _smooth_envelope,
    _target_track,
    _write_json,
)
from simulation.create_testing_specific_angles_scene import (
    DEFAULT_MIC_CENTER,
    DEFAULT_MIC_COUNT,
    DEFAULT_MIC_RADIUS,
    DEFAULT_ROOM_ABSORPTION,
    DEFAULT_ROOM_DIMS,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_SEED,
    _render_sparse_asset,
    _room_position,
)
from simulation.simulation_config import MicrophoneArray, Room, SimulationAudio, SimulationConfig, SimulationSource


DEFAULT_CONFIG_ROOT = Path("simulation/simulations/configs/testing_specific_angles_zone_overlap")
DEFAULT_ASSET_ROOT = Path("simulation/simulations/assets/testing_specific_angles_zone_overlap")
DEFAULT_ZONE_WIDTH_DEG = 30.0


@dataclass(frozen=True)
class ZoneOverlapSceneSpec:
    scene_idx: int
    target_angle_deg: int
    interferer_angle_deg: int
    wham_angle_deg: int
    babble_rotation_deg: int = 0


SCENE_SPECS: tuple[ZoneOverlapSceneSpec, ...] = (
    ZoneOverlapSceneSpec(0, 0, 65, 180, 0),
    ZoneOverlapSceneSpec(1, 35, 95, 225, 18),
    ZoneOverlapSceneSpec(2, 70, 145, 270, 36),
    ZoneOverlapSceneSpec(3, 110, 190, 315, 54),
    ZoneOverlapSceneSpec(4, 180, 255, 0, 72),
    ZoneOverlapSceneSpec(5, 250, 320, 90, 90),
)


def _scene_id(scene_idx: int) -> str:
    return f"testing_specific_angles_zone_overlap_k2_scene{scene_idx:02d}"


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
                "foreground_target_active",
                "bootstrap_noise_only",
                "background_only",
                "overlap",
                "overlap_count",
                "speaker_positions",
                "frame_snr_db",
                "noise_scale_hint",
                "target_zone_center_deg",
                "target_zone_width_deg",
                "target_zone_active",
            ],
        )
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out["active_speaker_ids"] = "|".join(str(v) for v in row["active_speaker_ids"])
            out["speaker_positions"] = json.dumps(row["speaker_positions"], separators=(",", ":"))
            writer.writerow(out)


def _build_frame_rows(
    *,
    duration_sec: float,
    bootstrap_sec: float,
    target_windows: dict[int, tuple[float, float]],
    target_positions: dict[int, list[float]],
    target_angles_deg: dict[int, int],
    noise_scale_hint_by_frame: np.ndarray,
    target_zone_center_deg: float,
    target_zone_width_deg: float,
    zone_target_speaker_id: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    frame_count = int(round(float(duration_sec) * 1000.0 / FRAME_MS))
    for frame_idx in range(frame_count):
        start_sec = frame_idx * (FRAME_MS / 1000.0)
        end_sec = min(float(duration_sec), start_sec + (FRAME_MS / 1000.0))
        active_ids = [
            speaker_id
            for speaker_id, (start, end) in target_windows.items()
            if float(start) <= start_sec < float(end)
        ]
        target_active = int(zone_target_speaker_id) in active_ids
        if target_active:
            primary_id = int(zone_target_speaker_id)
        elif active_ids:
            primary_id = int(active_ids[0])
        else:
            primary_id = -1
        rows.append(
            {
                "frame_index": int(frame_idx),
                "start_time_s": round(start_sec, 4),
                "end_time_s": round(end_sec, 4),
                "active_speaker_ids": list(active_ids),
                "primary_speaker_id": int(primary_id),
                "primary_doa_deg": "" if primary_id < 0 else float(target_angles_deg[primary_id]),
                "foreground_target_active": bool(target_active),
                "bootstrap_noise_only": bool(start_sec < float(bootstrap_sec)),
                "background_only": bool(not active_ids),
                "overlap": bool(len(active_ids) > 1),
                "overlap_count": int(len(active_ids)),
                "speaker_positions": {str(speaker_id): target_positions[speaker_id] for speaker_id in active_ids},
                "frame_snr_db": 0.0,
                "noise_scale_hint": round(float(noise_scale_hint_by_frame[min(frame_idx, noise_scale_hint_by_frame.shape[0] - 1)]), 4),
                "target_zone_center_deg": float(target_zone_center_deg),
                "target_zone_width_deg": float(target_zone_width_deg),
                "target_zone_active": bool(target_active),
            }
        )
    return rows


def generate_testing_specific_angles_zone_overlap_dataset(
    *,
    config_root: str | Path = DEFAULT_CONFIG_ROOT,
    asset_root: str | Path = DEFAULT_ASSET_ROOT,
    seed: int = DEFAULT_SEED,
    duration_sec: float = DEFAULT_DURATION_SEC,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    bootstrap_noise_only_sec: float = DEFAULT_BOOTSTRAP_SEC,
    background_babble_count: int = DEFAULT_BABBLE_COUNT,
    background_babble_gain_min: float = DEFAULT_BABBLE_GAIN_MIN,
    background_babble_gain_max: float = DEFAULT_BABBLE_GAIN_MAX,
    background_wham_gain: float = DEFAULT_WHAM_GAIN,
    manifest_path: str | Path | None = None,
    target_zone_width_deg: float = DEFAULT_ZONE_WIDTH_DEG,
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
    bootstrap_sec = float(np.clip(bootstrap_noise_only_sec, 0.5, max(0.5, duration_sec - 3.0)))
    babble_count = max(1, int(background_babble_count))
    babble_gain_min = float(max(0.0, min(background_babble_gain_min, background_babble_gain_max)))
    babble_gain_max = float(max(babble_gain_min, background_babble_gain_max))
    wham_gain = float(max(0.0, background_wham_gain))
    summary_rows: list[dict[str, Any]] = []

    target_0_start = bootstrap_sec
    target_0_end = min(float(duration_sec), bootstrap_sec + 6.0)
    target_1_start = min(float(duration_sec), bootstrap_sec + 2.0)
    target_1_end = float(duration_sec)

    for spec in SCENE_SPECS:
        scene_name = _scene_id(spec.scene_idx)
        scene_dir = asset_root / scene_name
        render_assets = scene_dir / "render_assets"
        render_assets.mkdir(parents=True, exist_ok=True)

        chosen_speakers = library.choose_speakers(rng, babble_count + 2)
        target_labels = chosen_speakers[:2]
        babble_labels = chosen_speakers[2:]
        target_paths = [library.choose_speech_path(rng, label) for label in target_labels]
        babble_paths = [library.choose_speech_path(rng, label) for label in babble_labels]
        wham_path = library.choose_noise_path(rng)

        target_positions = {
            0: _room_position(center, room_dims, float(spec.target_angle_deg), DEFAULT_TARGET_RADIUS_SCALE, 1.45),
            1: _room_position(center, room_dims, float(spec.interferer_angle_deg), DEFAULT_TARGET_RADIUS_SCALE * 1.05, 1.45),
        }
        target_angles_deg = {0: int(spec.target_angle_deg), 1: int(spec.interferer_angle_deg)}
        target_windows = {0: (target_0_start, target_0_end), 1: (target_1_start, target_1_end)}

        target_asset_paths = [render_assets / "target_0_seg000.wav", render_assets / "target_1_seg000.wav"]
        for idx, asset_path in enumerate(target_asset_paths):
            track = _target_track(
                path=target_paths[idx],
                sr=sample_rate,
                total_samples=total_samples,
                start_sec=float(target_windows[idx][0]),
                end_sec=float(target_windows[idx][1]),
                rng=rng,
            )
            _render_sparse_asset(track, asset_path, sample_rate)

        chatter_angles = _diffuse_babble_angles(babble_count, spec.babble_rotation_deg)
        chatter_gains = [float(rng.uniform(babble_gain_min, babble_gain_max)) for _ in range(babble_count)]
        chatter_positions: list[list[float]] = []
        chatter_asset_paths: list[Path] = []
        chatter_envelope_segments: list[list[dict[str, float]]] = []
        noise_mix_track = np.zeros(total_samples, dtype=np.float32)
        for idx, (speech_path, angle_deg) in enumerate(zip(babble_paths, chatter_angles, strict=True)):
            asset_path = render_assets / f"background_chatter_{idx:02d}.wav"
            track = _continuous_track(speech_path, sample_rate, total_samples, rng)
            envelope, segments = _smooth_envelope(
                total_samples=total_samples,
                sr=sample_rate,
                base_gain=float(chatter_gains[idx]),
                rng=rng,
                min_scale=0.85,
                max_scale=1.20,
                segment_sec_range=(2.0, 4.5),
            )
            track = (track * envelope).astype(np.float32, copy=False)
            noise_mix_track += track
            _render_sparse_asset(track, asset_path, sample_rate)
            chatter_asset_paths.append(asset_path)
            chatter_positions.append(_room_position(center, room_dims, float(angle_deg), DEFAULT_BACKGROUND_RADIUS_SCALE, 1.3))
            chatter_envelope_segments.append(segments)

        wham_asset_path = render_assets / "background_wham_0.wav"
        wham_track = _continuous_track(wham_path, sample_rate, total_samples, rng)
        wham_envelope, wham_segments = _smooth_envelope(
            total_samples=total_samples,
            sr=sample_rate,
            base_gain=float(wham_gain),
            rng=rng,
            min_scale=0.9,
            max_scale=1.15,
            segment_sec_range=(2.5, 5.0),
        )
        wham_track = (wham_track * wham_envelope).astype(np.float32, copy=False)
        noise_mix_track += wham_track
        _render_sparse_asset(wham_track, wham_asset_path, sample_rate)
        wham_position = _room_position(center, room_dims, float(spec.wham_angle_deg), DEFAULT_BACKGROUND_RADIUS_SCALE * 1.05, 1.2)

        source_rows: list[SimulationSource] = [
            SimulationSource(loc=list(target_positions[idx]), audio_path=str(target_asset_paths[idx].resolve()), gain=1.0, classification="speech")
            for idx in range(2)
        ]
        for idx in range(babble_count):
            source_rows.append(
                SimulationSource(
                    loc=list(chatter_positions[idx]),
                    audio_path=str(chatter_asset_paths[idx].resolve()),
                    gain=1.0,
                    classification="noise",
                )
            )
        source_rows.append(
            SimulationSource(
                loc=list(wham_position),
                audio_path=str(wham_asset_path.resolve()),
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
            audio=SimulationAudio(sources=source_rows, duration=float(duration_sec), fs=int(sample_rate)),
        )
        sim_cfg.to_file(scene_dir / "scene_config.json")
        sim_cfg.to_file(config_root / f"{scene_name}.json")

        frame_count = int(round(float(duration_sec) * 1000.0 / FRAME_MS))
        frame_samples = max(1, int(round(sample_rate * FRAME_MS / 1000.0)))
        noise_scale_hint_by_frame = np.zeros(frame_count, dtype=np.float32)
        for frame_idx in range(frame_count):
            start = frame_idx * frame_samples
            end = min(total_samples, start + frame_samples)
            if end > start:
                noise_scale_hint_by_frame[frame_idx] = float(np.sqrt(np.mean(noise_mix_track[start:end] ** 2) + 1e-12))
        frame_rows = _build_frame_rows(
            duration_sec=duration_sec,
            bootstrap_sec=bootstrap_sec,
            target_windows=target_windows,
            target_positions=target_positions,
            target_angles_deg=target_angles_deg,
            noise_scale_hint_by_frame=noise_scale_hint_by_frame,
            target_zone_center_deg=float(spec.target_angle_deg),
            target_zone_width_deg=float(target_zone_width_deg),
            zone_target_speaker_id=0,
        )
        _write_json(scene_dir / "frame_ground_truth.json", frame_rows)
        _write_frame_truth(scene_dir / "frame_ground_truth.csv", frame_rows)

        chatter_centroid_deg = _background_centroid_deg(chatter_angles, chatter_gains)
        all_noise_centroid_deg = _background_centroid_deg(chatter_angles + [float(spec.wham_angle_deg)], chatter_gains + [float(wham_gain)])
        metadata = {
            "scene_name": scene_name,
            "scene_type": "testing_specific_angles_zone_overlap",
            "scene_prefix": "testing_specific_angles_zone_overlap",
            "speaker_count": 2,
            "zone_target_speaker_id": 0,
            "target_zone_center_deg": int(spec.target_angle_deg),
            "target_zone_width_deg": float(target_zone_width_deg),
            "main_angle_deg": int(spec.target_angle_deg),
            "secondary_angle_deg": int(spec.interferer_angle_deg),
            "bootstrap_noise_only_window_sec": [0.0, float(bootstrap_sec)],
            "bootstrap_reference_doa_deg": int(spec.target_angle_deg),
            "target_radius_scale": float(DEFAULT_TARGET_RADIUS_SCALE),
            "background_radius_scale": float(DEFAULT_BACKGROUND_RADIUS_SCALE),
            "background_noise_centroid_deg": chatter_centroid_deg,
            "background_noise_all_centroid_deg": all_noise_centroid_deg,
            "background_babble_count": int(babble_count),
            "background_babble_gain_range": [float(babble_gain_min), float(babble_gain_max)],
            "background_wham_gain": float(wham_gain),
            "target_activity_schedule": {
                "speaker_0_windows_sec": [[float(target_windows[0][0]), float(target_windows[0][1])]],
                "speaker_1_windows_sec": [[float(target_windows[1][0]), float(target_windows[1][1])]],
            },
            "assets": {
                "speech": [
                    {
                        "speaker_id": int(idx),
                        "speaker_label": str(target_labels[idx]),
                        "source_path": str(target_paths[idx].resolve()),
                        "render_asset_path": str(target_asset_paths[idx].resolve()),
                        "active_window_sec": [float(target_windows[idx][0]), float(target_windows[idx][1])],
                        "position_m": target_positions[idx],
                        "angle_deg": target_angles_deg[idx],
                    }
                    for idx in range(2)
                ],
                "background_chatter": [
                    {
                        "background_speaker_id": int(idx),
                        "speaker_label": str(babble_labels[idx]),
                        "source_path": str(babble_paths[idx].resolve()),
                        "render_asset_path": str(chatter_asset_paths[idx].resolve()),
                        "active_window_sec": [0.0, float(duration_sec)],
                        "position_m": chatter_positions[idx],
                        "angle_deg": int(chatter_angles[idx]),
                        "gain": float(chatter_gains[idx]),
                        "gain_segments": chatter_envelope_segments[idx],
                    }
                    for idx in range(babble_count)
                ],
                "noise": [
                    {
                        "noise_id": 0,
                        "source_path": str(wham_path.resolve()),
                        "render_asset_path": str(wham_asset_path.resolve()),
                        "active_window_sec": [0.0, float(duration_sec)],
                        "position_m": wham_position,
                        "angle_deg": int(spec.wham_angle_deg),
                        "gain": float(wham_gain),
                        "gain_segments": wham_segments,
                    }
                ],
            },
            "summary_metrics": {
                "bootstrap_noise_only_sec": float(bootstrap_sec),
                "target_overlap_sec": float(max(0.0, min(target_0_end, target_1_end) - max(target_0_start, target_1_start))),
            },
        }
        _write_json(scene_dir / "scenario_metadata.json", metadata)
        _write_json(
            scene_dir / "event_schedule.json",
            {
                "bootstrap_noise_only_window_sec": [0.0, float(bootstrap_sec)],
                "target_zone_center_deg": int(spec.target_angle_deg),
                "target_zone_width_deg": float(target_zone_width_deg),
                "speech_events": [
                    {
                        "speaker_id": int(idx),
                        "start_sec": float(target_windows[idx][0]),
                        "end_sec": float(target_windows[idx][1]),
                        "angle_deg": target_angles_deg[idx],
                    }
                    for idx in range(2)
                ],
            },
        )
        summary_rows.append(
            {
                "scene_name": scene_name,
                "scene_config_path": str((config_root / f"{scene_name}.json").resolve()),
                "scene_dir": str(scene_dir.resolve()),
                "target_zone_center_deg": int(spec.target_angle_deg),
                "target_zone_width_deg": float(target_zone_width_deg),
            }
        )

    _write_json(asset_root / "generation_summary.json", {"scenes": summary_rows})
    return summary_rows


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create overlap scenes for zone-target MVDR benchmarking.")
    parser.add_argument("--config-root", default=str(DEFAULT_CONFIG_ROOT))
    parser.add_argument("--asset-root", default=str(DEFAULT_ASSET_ROOT))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--duration-sec", type=float, default=DEFAULT_DURATION_SEC)
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE)
    parser.add_argument("--bootstrap-noise-only-sec", type=float, default=DEFAULT_BOOTSTRAP_SEC)
    parser.add_argument("--background-babble-count", type=int, default=DEFAULT_BABBLE_COUNT)
    parser.add_argument("--background-babble-gain-min", type=float, default=DEFAULT_BABBLE_GAIN_MIN)
    parser.add_argument("--background-babble-gain-max", type=float, default=DEFAULT_BABBLE_GAIN_MAX)
    parser.add_argument("--background-wham-gain", type=float, default=DEFAULT_WHAM_GAIN)
    parser.add_argument("--target-zone-width-deg", type=float, default=DEFAULT_ZONE_WIDTH_DEG)
    parser.add_argument("--manifest-path", default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    rows = generate_testing_specific_angles_zone_overlap_dataset(
        config_root=args.config_root,
        asset_root=args.asset_root,
        seed=int(args.seed),
        duration_sec=float(args.duration_sec),
        sample_rate=int(args.sample_rate),
        bootstrap_noise_only_sec=float(args.bootstrap_noise_only_sec),
        background_babble_count=int(args.background_babble_count),
        background_babble_gain_min=float(args.background_babble_gain_min),
        background_babble_gain_max=float(args.background_babble_gain_max),
        background_wham_gain=float(args.background_wham_gain),
        manifest_path=args.manifest_path,
        target_zone_width_deg=float(args.target_zone_width_deg),
    )
    print(json.dumps({"n_scenes": len(rows), "config_root": str(Path(args.config_root).resolve())}, indent=2))


if __name__ == "__main__":
    main()

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


DEFAULT_CONFIG_ROOT = Path("simulation/simulations/configs/testing_specific_angles_babble_bootstrap")
DEFAULT_ASSET_ROOT = Path("simulation/simulations/assets/testing_specific_angles_babble_bootstrap")
DEFAULT_DURATION_SEC = 12.0
DEFAULT_BOOTSTRAP_SEC = 4.0
DEFAULT_SOURCE_RADIUS_SCALE = 0.28
DEFAULT_BABBLE_COUNT = 8
DEFAULT_BABBLE_GAIN_MIN = 0.045
DEFAULT_BABBLE_GAIN_MAX = 0.10
DEFAULT_WHAM_GAIN = 0.16
FRAME_MS = 20


@dataclass(frozen=True)
class BabbleBootstrapSceneSpec:
    scene_idx: int
    target_angles_deg: tuple[int, int]
    wham_angle_deg: int
    babble_rotation_deg: int = 0


SCENE_SPECS: tuple[BabbleBootstrapSceneSpec, ...] = (
    BabbleBootstrapSceneSpec(0, (0, 45), 180, 0),
    BabbleBootstrapSceneSpec(1, (45, 90), 225, 15),
    BabbleBootstrapSceneSpec(2, (90, 135), 270, 30),
    BabbleBootstrapSceneSpec(3, (135, 180), 315, 45),
    BabbleBootstrapSceneSpec(4, (180, 225), 0, 60),
    BabbleBootstrapSceneSpec(5, (270, 315), 90, 75),
)


def _scene_id(scene_idx: int) -> str:
    return f"testing_specific_angles_babble_bootstrap_k2_scene{scene_idx:02d}"


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
                "foreground_target_active",
                "bootstrap_noise_only",
                "background_only",
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


def _continuous_track(path: Path, sr: int, total_samples: int, rng: np.random.Generator) -> np.ndarray:
    clip = _load_excerpt(path, sr=sr, length_samples=total_samples, rng=rng)
    return _normalize_peak(clip).astype(np.float32, copy=False)


def _smooth_envelope(
    *,
    total_samples: int,
    sr: int,
    base_gain: float,
    rng: np.random.Generator,
    min_scale: float,
    max_scale: float,
    segment_sec_range: tuple[float, float],
) -> tuple[np.ndarray, list[dict[str, float]]]:
    if total_samples <= 0:
        return np.zeros(0, dtype=np.float32), []
    min_segment = max(0.35, float(segment_sec_range[0]))
    max_segment = max(min_segment, float(segment_sec_range[1]))
    anchors_t = [0.0]
    anchors_v = [float(base_gain * rng.uniform(min_scale, max_scale))]
    total_sec = float(total_samples) / max(float(sr), 1.0)
    t = 0.0
    while t < total_sec:
        t = min(total_sec, t + float(rng.uniform(min_segment, max_segment)))
        anchors_t.append(float(t))
        anchors_v.append(float(base_gain * rng.uniform(min_scale, max_scale)))
    sample_pos = np.linspace(0.0, total_sec, total_samples, endpoint=False, dtype=np.float64)
    env = np.interp(sample_pos, np.asarray(anchors_t, dtype=np.float64), np.asarray(anchors_v, dtype=np.float64))
    segments: list[dict[str, float]] = []
    for idx in range(len(anchors_t) - 1):
        segments.append(
            {
                "start_sec": float(anchors_t[idx]),
                "end_sec": float(anchors_t[idx + 1]),
                "gain_start": float(anchors_v[idx]),
                "gain_end": float(anchors_v[idx + 1]),
            }
        )
    return env.astype(np.float32, copy=False), segments


def _target_track(
    *,
    path: Path,
    sr: int,
    total_samples: int,
    start_sec: float,
    end_sec: float,
    rng: np.random.Generator,
) -> np.ndarray:
    track = np.zeros(total_samples, dtype=np.float32)
    start = int(round(float(start_sec) * sr))
    end = int(round(float(end_sec) * sr))
    length = max(1, end - start)
    clip = _load_excerpt(path, sr=sr, length_samples=length, rng=rng)
    stop = min(total_samples, start + clip.shape[0])
    track[start:stop] = clip[: stop - start]
    peak = float(np.max(np.abs(track))) if track.size else 0.0
    if peak > 1e-8:
        track = 0.9 * track / peak
    return track.astype(np.float32, copy=False)


def _background_centroid_deg(angles_deg: list[float], weights: list[float]) -> float:
    if not angles_deg or not weights or len(angles_deg) != len(weights):
        return float("nan")
    vec = np.zeros(2, dtype=np.float64)
    for angle_deg, weight in zip(angles_deg, weights):
        theta = np.deg2rad(float(angle_deg))
        vec += float(weight) * np.array([np.cos(theta), np.sin(theta)], dtype=np.float64)
    if float(np.linalg.norm(vec)) <= 1e-8:
        return float("nan")
    return float(np.mod(np.degrees(np.arctan2(vec[1], vec[0])), 360.0))


def _babble_angles(count: int, rotation_deg: int) -> list[int]:
    base_angles = np.linspace(0.0, 360.0, max(count, 1), endpoint=False)
    return [int(round((float(angle) + float(rotation_deg)) % 360.0)) for angle in base_angles[:count]]


def _build_frame_rows(
    *,
    duration_sec: float,
    bootstrap_sec: float,
    target_windows: dict[int, tuple[float, float]],
    target_positions: dict[int, list[float]],
    target_angles_deg: dict[int, int],
    noise_scale_hint_by_frame: np.ndarray,
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
        primary_id = int(active_ids[0]) if active_ids else -1
        rows.append(
            {
                "frame_index": int(frame_idx),
                "start_time_s": round(start_sec, 4),
                "end_time_s": round(end_sec, 4),
                "active_speaker_ids": list(active_ids),
                "primary_speaker_id": int(primary_id),
                "primary_doa_deg": "" if primary_id < 0 else float(target_angles_deg[primary_id]),
                "foreground_target_active": bool(primary_id >= 0),
                "bootstrap_noise_only": bool(start_sec < float(bootstrap_sec)),
                "background_only": bool(primary_id < 0),
                "overlap": False,
                "overlap_count": int(len(active_ids)),
                "speaker_positions": {} if primary_id < 0 else {str(primary_id): target_positions[primary_id]},
                "frame_snr_db": 0.0,
                "noise_scale_hint": round(float(noise_scale_hint_by_frame[min(frame_idx, noise_scale_hint_by_frame.shape[0] - 1)]), 4),
            }
        )
    return rows


def generate_testing_specific_angles_babble_bootstrap_dataset(
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
    bootstrap_sec = float(np.clip(bootstrap_noise_only_sec, 0.5, max(0.5, duration_sec - 2.0)))
    post_bootstrap_sec = max(float(duration_sec - bootstrap_sec), 1.0)
    target_turn_sec = post_bootstrap_sec / 2.0
    babble_count = max(1, int(background_babble_count))
    babble_gain_min = float(max(0.0, min(background_babble_gain_min, background_babble_gain_max)))
    babble_gain_max = float(max(babble_gain_min, background_babble_gain_max))
    wham_gain = float(max(0.0, background_wham_gain))
    summary_rows: list[dict[str, Any]] = []

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
            0: _room_position(center, room_dims, float(spec.target_angles_deg[0]), DEFAULT_SOURCE_RADIUS_SCALE, 1.45),
            1: _room_position(center, room_dims, float(spec.target_angles_deg[1]), DEFAULT_SOURCE_RADIUS_SCALE, 1.45),
        }
        target_windows = {
            0: (bootstrap_sec, bootstrap_sec + target_turn_sec),
            1: (bootstrap_sec + target_turn_sec, float(duration_sec)),
        }
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

        chatter_angles = _babble_angles(babble_count, spec.babble_rotation_deg)
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
                min_scale=0.7,
                max_scale=1.35,
                segment_sec_range=(1.5, 3.5),
            )
            track = (track * envelope).astype(np.float32, copy=False)
            noise_mix_track += track
            _render_sparse_asset(track, asset_path, sample_rate)
            chatter_asset_paths.append(asset_path)
            chatter_positions.append(_room_position(center, room_dims, float(angle_deg), DEFAULT_SOURCE_RADIUS_SCALE * 1.25, 1.3))
            chatter_envelope_segments.append(segments)

        wham_asset_path = render_assets / "background_wham_0.wav"
        wham_track = _continuous_track(wham_path, sample_rate, total_samples, rng)
        wham_envelope, wham_segments = _smooth_envelope(
            total_samples=total_samples,
            sr=sample_rate,
            base_gain=float(wham_gain),
            rng=rng,
            min_scale=0.8,
            max_scale=1.25,
            segment_sec_range=(2.0, 4.0),
        )
        wham_track = (wham_track * wham_envelope).astype(np.float32, copy=False)
        noise_mix_track += wham_track
        _render_sparse_asset(wham_track, wham_asset_path, sample_rate)
        wham_position = _room_position(center, room_dims, float(spec.wham_angle_deg), DEFAULT_SOURCE_RADIUS_SCALE * 1.35, 1.2)

        source_rows: list[SimulationSource] = [
            SimulationSource(
                loc=list(target_positions[idx]),
                audio_path=str(target_asset_paths[idx].resolve()),
                gain=1.0,
                classification="speech",
            )
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
            audio=SimulationAudio(
                sources=source_rows,
                duration=float(duration_sec),
                fs=int(sample_rate),
            ),
        )
        sim_cfg.to_file(scene_dir / "scene_config.json")
        sim_cfg.to_file(config_root / f"{scene_name}.json")

        frame_count = int(round(float(duration_sec) * 1000.0 / FRAME_MS))
        frame_samples = max(1, int(round(sample_rate * FRAME_MS / 1000.0)))
        noise_scale_hint_by_frame = np.zeros(frame_count, dtype=np.float32)
        for frame_idx in range(frame_count):
            start = frame_idx * frame_samples
            end = min(total_samples, start + frame_samples)
            if end <= start:
                continue
            noise_scale_hint_by_frame[frame_idx] = float(np.sqrt(np.mean(noise_mix_track[start:end] ** 2) + 1e-12))
        frame_rows = _build_frame_rows(
            duration_sec=duration_sec,
            bootstrap_sec=bootstrap_sec,
            target_windows=target_windows,
            target_positions=target_positions,
            target_angles_deg={0: int(spec.target_angles_deg[0]), 1: int(spec.target_angles_deg[1])},
            noise_scale_hint_by_frame=noise_scale_hint_by_frame,
        )
        _write_json(scene_dir / "frame_ground_truth.json", frame_rows)
        _write_frame_truth(scene_dir / "frame_ground_truth.csv", frame_rows)

        chatter_centroid_deg = _background_centroid_deg(chatter_angles, chatter_gains)
        all_noise_centroid_deg = _background_centroid_deg(
            chatter_angles + [float(spec.wham_angle_deg)],
            chatter_gains + [float(wham_gain)],
        )
        metadata = {
            "scene_name": scene_name,
            "scene_type": "testing_specific_angles_babble_bootstrap",
            "scene_prefix": "testing_specific_angles_babble_bootstrap",
            "seed": int(seed),
            "speaker_count": 2,
            "main_angle_deg": int(spec.target_angles_deg[0]),
            "secondary_angle_deg": int(spec.target_angles_deg[1]),
            "bootstrap_noise_only_window_sec": [0.0, float(bootstrap_sec)],
            "bootstrap_reference_doa_deg": int(spec.target_angles_deg[0]),
            "background_noise_centroid_deg": chatter_centroid_deg,
            "background_noise_all_centroid_deg": all_noise_centroid_deg,
            "background_babble_count": int(babble_count),
            "background_babble_gain_range": [float(babble_gain_min), float(babble_gain_max)],
            "background_wham_gain": float(wham_gain),
            "noise_schedule": {
                "background_chatter_gain_segments": [
                    {
                        "background_speaker_id": int(idx),
                        "segments": segments,
                    }
                    for idx, segments in enumerate(chatter_envelope_segments)
                ],
                "background_wham_gain_segments": wham_segments,
            },
            "target_activity_schedule": {
                "speaker_0_windows_sec": [[float(target_windows[0][0]), float(target_windows[0][1])]],
                "speaker_1_windows_sec": [[float(target_windows[1][0]), float(target_windows[1][1])]],
            },
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
                "speech": [
                    {
                        "speaker_id": int(idx),
                        "speaker_label": str(target_labels[idx]),
                        "source_path": str(target_paths[idx].resolve()),
                        "render_asset_path": str(target_asset_paths[idx].resolve()),
                        "active_window_sec": [float(target_windows[idx][0]), float(target_windows[idx][1])],
                        "position_m": target_positions[idx],
                        "angle_deg": int(spec.target_angles_deg[idx]),
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
                "background_only_ratio": float(bootstrap_sec / duration_sec),
                "speaker_activity_frames": {
                    "0": int((target_windows[0][1] - target_windows[0][0]) * 1000.0 / FRAME_MS),
                    "1": int((target_windows[1][1] - target_windows[1][0]) * 1000.0 / FRAME_MS),
                },
            },
        }
        _write_json(scene_dir / "scenario_metadata.json", metadata)
        _write_json(
            scene_dir / "event_schedule.json",
            {
                "bootstrap_noise_only_window_sec": [0.0, float(bootstrap_sec)],
                "speech_events": [
                    {
                        "speaker_id": int(idx),
                        "start_sec": float(target_windows[idx][0]),
                        "end_sec": float(target_windows[idx][1]),
                        "angle_deg": int(spec.target_angles_deg[idx]),
                    }
                    for idx in range(2)
                ],
                "background_chatter_events": [
                    {
                        "background_speaker_id": int(idx),
                        "start_sec": 0.0,
                        "end_sec": float(duration_sec),
                        "angle_deg": int(chatter_angles[idx]),
                        "gain": float(chatter_gains[idx]),
                        "gain_segments": chatter_envelope_segments[idx],
                    }
                    for idx in range(babble_count)
                ],
                "noise_events": [
                    {
                        "noise_id": 0,
                        "start_sec": 0.0,
                        "end_sec": float(duration_sec),
                        "angle_deg": int(spec.wham_angle_deg),
                        "gain": float(wham_gain),
                        "gain_segments": wham_segments,
                    }
                ],
            },
        )

        summary_rows.append(
            {
                "scene_name": scene_name,
                "scene_config_path": str((config_root / f"{scene_name}.json").resolve()),
                "scene_dir": str(scene_dir.resolve()),
                "main_angle_deg": int(spec.target_angles_deg[0]),
                "secondary_angle_deg": int(spec.target_angles_deg[1]),
                "bootstrap_noise_only_window_sec": [0.0, float(bootstrap_sec)],
                "background_babble_count": int(babble_count),
                "background_wham_gain": float(wham_gain),
            }
        )

    _write_json(asset_root / "generation_summary.json", {"scenes": summary_rows})
    return summary_rows


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create deterministic babble-bootstrap scenes for aggressive MVDR covariance tuning.")
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
    parser.add_argument("--manifest-path", default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    rows = generate_testing_specific_angles_babble_bootstrap_dataset(
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
    )
    print(json.dumps({"n_scenes": len(rows), "config_root": str(Path(args.config_root).resolve())}, indent=2))


if __name__ == "__main__":
    main()

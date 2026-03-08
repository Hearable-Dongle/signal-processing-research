from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf
from scipy import signal

from simulation.simulation_config import MicrophoneArray, Room, SimulationAudio, SimulationConfig, SimulationSource
from simulation.simulator import run_simulation

from .assets import AssetLibrary, load_asset_library
from .config import ConversationScenarioConfig, GenerationResult, build_preset
from .scheduler import ConversationPlan, ScheduledUtterance, build_conversation_plan


TRANSIENT_TYPES = ("door_click", "cough", "dish_clink")


def _fade(samples: np.ndarray, sr: int, fade_ms: float = 20.0) -> np.ndarray:
    fade_len = min(samples.shape[0] // 2, max(1, int(sr * fade_ms / 1000.0)))
    if fade_len <= 0:
        return samples
    ramp = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
    out = samples.astype(np.float32, copy=True)
    out[:fade_len] *= ramp
    out[-fade_len:] *= ramp[::-1]
    return out


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(np.asarray(x, dtype=np.float64))) + 1e-12))


def _normalize_peak(x: np.ndarray, peak: float = 0.98) -> np.ndarray:
    scale = np.max(np.abs(x)) if x.size else 0.0
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
    peak = np.max(np.abs(clip))
    if peak > 1e-8:
        clip = 0.9 * clip / peak
    return clip.astype(np.float32, copy=False)


def _shape_noise(base: np.ndarray, sr: int, category: str, rng: np.random.Generator) -> np.ndarray:
    x = base.astype(np.float32, copy=True)
    if category == "hvac":
        b, a = signal.butter(2, 260.0 / (sr / 2.0), btype="lowpass")
        x = signal.lfilter(b, a, x).astype(np.float32, copy=False)
        env = 0.55 + 0.25 * np.sin(2 * np.pi * rng.uniform(0.04, 0.11) * np.arange(x.size) / sr)
        x *= env.astype(np.float32, copy=False)
    elif category == "keyboard":
        b, a = signal.butter(2, [900.0 / (sr / 2.0), 4000.0 / (sr / 2.0)], btype="bandpass")
        x = signal.lfilter(b, a, x).astype(np.float32, copy=False)
        env = np.zeros_like(x)
        click_spacing = max(1, int(sr * rng.uniform(0.08, 0.22)))
        width = max(8, int(sr * 0.012))
        for idx in range(0, x.size, click_spacing):
            if rng.random() < 0.6:
                end = min(x.size, idx + width)
                env[idx:end] += np.hanning(max(2, end - idx)).astype(np.float32)
        x *= 0.15 + env
    elif category == "distant_chatter":
        b, a = signal.butter(2, [250.0 / (sr / 2.0), 2400.0 / (sr / 2.0)], btype="bandpass")
        x = signal.lfilter(b, a, x).astype(np.float32, copy=False)
        env = 0.35 + 0.3 * np.maximum(0.0, np.sin(2 * np.pi * rng.uniform(0.25, 0.6) * np.arange(x.size) / sr))
        x *= env.astype(np.float32, copy=False)
    elif category in {"street", "dishwasher"}:
        b, a = signal.butter(2, [80.0 / (sr / 2.0), 1800.0 / (sr / 2.0)], btype="bandpass")
        x = signal.lfilter(b, a, x).astype(np.float32, copy=False)
        env = 0.45 + 0.2 * np.sin(2 * np.pi * rng.uniform(0.08, 0.2) * np.arange(x.size) / sr + rng.uniform(0, np.pi))
        x *= env.astype(np.float32, copy=False)
    else:
        x *= 0.8
    peak = np.max(np.abs(x))
    if peak > 1e-8:
        x = 0.9 * x / peak
    return x.astype(np.float32, copy=False)


def _synthesize_transient(kind: str, sr: int, duration_sec: float, rng: np.random.Generator) -> np.ndarray:
    n = max(1, int(sr * duration_sec))
    t = np.arange(n, dtype=np.float32) / sr
    noise = rng.standard_normal(n).astype(np.float32)
    if kind == "door_click":
        env = np.exp(-55.0 * t)
        y = noise * env
    elif kind == "dish_clink":
        env = np.exp(-18.0 * t)
        y = 0.55 * np.sin(2 * np.pi * 2400.0 * t) * env + 0.35 * noise * env
    else:
        env = np.exp(-5.0 * t)
        y = (0.4 * noise + 0.6 * np.sin(2 * np.pi * 180.0 * t)) * env
    return _normalize_peak(y.astype(np.float32, copy=False), peak=0.8)


def _speaker_position(cfg: ConversationScenarioConfig, speaker_id: int, t_sec: float, n_speakers: int) -> list[float]:
    center = np.asarray(cfg.mic_array.mic_center_m, dtype=float)
    base_radius = min(cfg.room.dimensions_m[0], cfg.room.dimensions_m[1]) * 0.28
    angle = (2.0 * np.pi * speaker_id) / max(1, n_speakers)
    radius = base_radius
    if cfg.moving_speaker and speaker_id == 0:
        sweep = np.clip(t_sec / max(cfg.render.duration_sec, 1e-6), 0.0, 1.0)
        angle += np.deg2rad(-55.0 + 110.0 * sweep)
        radius *= 0.7 + 0.5 * sweep
    x = float(np.clip(center[0] + radius * np.cos(angle), 0.6, cfg.room.dimensions_m[0] - 0.6))
    y = float(np.clip(center[1] + radius * np.sin(angle), 0.6, cfg.room.dimensions_m[1] - 0.6))
    z = 1.45
    return [round(x, 4), round(y, 4), z]


def _mix_event(track: np.ndarray, event_audio: np.ndarray, start_sample: int, gain: float) -> None:
    if event_audio.size == 0:
        return
    end_sample = min(track.shape[0], start_sample + event_audio.shape[0])
    if end_sample <= start_sample:
        return
    track[start_sample:end_sample] += gain * event_audio[: end_sample - start_sample]


def _materialize_speech_tracks(
    cfg: ConversationScenarioConfig,
    plan: ConversationPlan,
    library: AssetLibrary,
    rng: np.random.Generator,
) -> tuple[list[np.ndarray], list[dict[str, Any]], dict[int, str]]:
    sr = cfg.render.sample_rate
    total_samples = int(cfg.render.duration_sec * sr)
    speaker_labels = library.choose_speakers(rng, len(plan.speaker_ids))
    label_map = {speaker_id: speaker_labels[idx] for idx, speaker_id in enumerate(plan.speaker_ids)}
    tracks = [np.zeros(total_samples, dtype=np.float32) for _ in plan.speaker_ids]
    event_rows: list[dict[str, Any]] = []

    for idx, utterance in enumerate(plan.utterances):
        start = int(round(utterance.start_sec * sr))
        end = int(round(utterance.end_sec * sr))
        length = max(1, end - start)
        clip_path = library.choose_speech_path(rng, label_map[utterance.speaker_id])
        excerpt = _load_excerpt(clip_path, sr=sr, length_samples=length, rng=rng)
        gain = 0.42 if utterance.kind == "backchannel" else 0.82
        if utterance.kind == "interruption":
            gain = 0.72
        _mix_event(tracks[utterance.speaker_id], excerpt, start, gain)
        event_rows.append(
            {
                "event_id": f"speech_{idx:04d}",
                "type": utterance.kind,
                "speaker_id": utterance.speaker_id,
                "speaker_label": label_map[utterance.speaker_id],
                "start_sec": round(utterance.start_sec, 4),
                "end_sec": round(utterance.end_sec, 4),
                "asset_path": str(clip_path),
                "interrupted": bool(utterance.interrupted),
            }
        )
    return tracks, event_rows, label_map


def _build_noise_layers(
    cfg: ConversationScenarioConfig,
    library: AssetLibrary,
    rng: np.random.Generator,
    speech_mix: np.ndarray,
) -> tuple[list[np.ndarray], list[dict[str, Any]], list[dict[str, Any]]]:
    sr = cfg.render.sample_rate
    total_samples = int(cfg.render.duration_sec * sr)
    layers: list[np.ndarray] = []
    layer_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []

    target_snr = float(rng.uniform(cfg.noise.base_snr_db_range[0], cfg.noise.base_snr_db_range[1]))
    speech_rms = max(_rms(speech_mix), 1e-4)
    target_noise_rms = speech_rms / (10.0 ** (target_snr / 20.0))

    for layer_idx, category in enumerate(cfg.noise.ambience_layers):
        excerpt = _load_excerpt(library.choose_noise_path(rng), sr=sr, length_samples=total_samples, rng=rng)
        shaped = _shape_noise(excerpt, sr=sr, category=category, rng=rng)
        n_segments = max(3, int(np.ceil(cfg.render.duration_sec / 2.5)))
        env = np.zeros(total_samples, dtype=np.float32)
        segment_len = max(1, total_samples // n_segments)
        for seg in range(n_segments):
            start = seg * segment_len
            end = total_samples if seg == n_segments - 1 else min(total_samples, start + segment_len)
            snr_offset = float(rng.uniform(-4.0, 4.0))
            local_snr = np.clip(target_snr + snr_offset, cfg.noise.base_snr_db_range[0] - 4.0, cfg.noise.base_snr_db_range[1] + 4.0)
            local_noise_rms = speech_rms / (10.0 ** (local_snr / 20.0))
            scale = local_noise_rms / max(_rms(shaped[start:end]), 1e-4)
            env[start:end] = scale
        layer = shaped * env
        layers.append(layer.astype(np.float32, copy=False))
        layer_rows.append(
            {
                "noise_id": f"noise_{layer_idx:03d}",
                "category": category,
                "start_sec": 0.0,
                "end_sec": round(cfg.render.duration_sec, 4),
                "target_snr_db": round(target_snr, 3),
            }
        )

    transient_min, transient_max = cfg.noise.transient_count_range
    num_transients = int(rng.integers(transient_min, transient_max + 1))
    for idx in range(num_transients):
        kind = cfg.noise.transient_types[int(rng.integers(0, len(cfg.noise.transient_types)))]
        duration = float(rng.uniform(0.04, 0.22 if kind != "cough" else 0.5))
        start_sec = float(rng.uniform(0.3, max(0.35, cfg.render.duration_sec - duration - 0.1)))
        start = int(round(start_sec * sr))
        transient = _synthesize_transient(kind=kind, sr=sr, duration_sec=duration, rng=rng)
        target_rms = target_noise_rms * float(rng.uniform(1.1, 2.5))
        transient *= target_rms / max(_rms(transient), 1e-4)
        layer = np.zeros(total_samples, dtype=np.float32)
        _mix_event(layer, transient, start, 1.0)
        layers.append(layer)
        event_rows.append(
            {
                "event_id": f"transient_{idx:03d}",
                "type": kind,
                "start_sec": round(start_sec, 4),
                "end_sec": round(start_sec + duration, 4),
            }
        )

    return layers, layer_rows, event_rows


def _write_wav(path: Path, samples: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, samples.astype(np.float32, copy=False), sr)


def _sources_for_track(
    cfg: ConversationScenarioConfig,
    samples: np.ndarray,
    base_name: str,
    classification: str,
    scene_dir: Path,
    position_fn,
    motion_update_sec: float,
) -> tuple[list[SimulationSource], list[dict[str, Any]]]:
    sr = cfg.render.sample_rate
    total_samples = samples.shape[0]
    segment_samples = max(1, int(motion_update_sec * sr))
    sources: list[SimulationSource] = []
    segment_rows: list[dict[str, Any]] = []
    for segment_idx, start in enumerate(range(0, total_samples, segment_samples)):
        end = min(total_samples, start + segment_samples)
        chunk = samples[start:end].copy()
        if np.max(np.abs(chunk)) < 1e-5:
            continue
        sparse = np.zeros(total_samples, dtype=np.float32)
        sparse[start:end] = chunk
        asset_path = scene_dir / "render_assets" / f"{base_name}_seg{segment_idx:03d}.wav"
        _write_wav(asset_path, sparse, sr=sr)
        t_mid = 0.5 * ((start / sr) + (end / sr))
        position = position_fn(t_mid)
        sources.append(
            SimulationSource(
                loc=position,
                audio_path=str(asset_path.resolve()),
                gain=1.0,
                classification=classification,
            )
        )
        segment_rows.append(
            {
                "asset_path": str(asset_path.resolve()),
                "start_sec": round(start / sr, 4),
                "end_sec": round(end / sr, 4),
                "position_m": position,
                "classification": classification,
            }
        )
    return sources, segment_rows


def _build_scene_config(
    cfg: ConversationScenarioConfig,
    scene_dir: Path,
    speech_tracks: list[np.ndarray],
    noise_tracks: list[np.ndarray],
) -> tuple[SimulationConfig, list[dict[str, Any]]]:
    sim_sources: list[SimulationSource] = []
    segment_rows: list[dict[str, Any]] = []
    n_speakers = len(speech_tracks)

    for speaker_id, track in enumerate(speech_tracks):
        sources, rows = _sources_for_track(
            cfg=cfg,
            samples=track,
            base_name=f"speaker_{speaker_id}",
            classification="speech",
            scene_dir=scene_dir,
            position_fn=lambda t, sid=speaker_id: _speaker_position(cfg, sid, t, n_speakers),
            motion_update_sec=cfg.noise.motion_update_sec if cfg.moving_speaker and speaker_id == 0 else cfg.render.duration_sec + 1.0,
        )
        sim_sources.extend(sources)
        segment_rows.extend(
            [{"source_id": f"speaker_{speaker_id}", "speaker_id": speaker_id, **row} for row in rows]
        )

    for noise_id, track in enumerate(noise_tracks):
        angle = 2.0 * np.pi * (noise_id + 0.5) / max(1, len(noise_tracks))
        center = np.asarray(cfg.mic_array.mic_center_m, dtype=float)
        radius = min(cfg.room.dimensions_m[0], cfg.room.dimensions_m[1]) * 0.33
        pos = [
            float(np.clip(center[0] + radius * np.cos(angle), 0.5, cfg.room.dimensions_m[0] - 0.5)),
            float(np.clip(center[1] + radius * np.sin(angle), 0.5, cfg.room.dimensions_m[1] - 0.5)),
            1.2,
        ]
        sources, rows = _sources_for_track(
            cfg=cfg,
            samples=track,
            base_name=f"noise_{noise_id}",
            classification="noise",
            scene_dir=scene_dir,
            position_fn=lambda _t, p=pos: p,
            motion_update_sec=cfg.render.duration_sec + 1.0,
        )
        sim_sources.extend(sources)
        segment_rows.extend([{"source_id": f"noise_{noise_id}", **row} for row in rows])

    sim_cfg = SimulationConfig(
        room=Room(dimensions=cfg.room.dimensions_m, absorption=cfg.room.absorption),
        microphone_array=MicrophoneArray(
            mic_center=cfg.mic_array.mic_center_m,
            mic_radius=cfg.mic_array.mic_radius_m,
            mic_count=cfg.mic_array.mic_count,
        ),
        audio=SimulationAudio(
            sources=sim_sources,
            duration=cfg.render.duration_sec,
            fs=cfg.render.sample_rate,
        ),
    )
    return sim_cfg, segment_rows


def _compute_frame_truth(
    cfg: ConversationScenarioConfig,
    plan: ConversationPlan,
    speech_tracks: list[np.ndarray],
    noise_tracks: list[np.ndarray],
) -> tuple[list[dict[str, Any]], list[dict[str, float]]]:
    sr = cfg.render.sample_rate
    frame_samples = max(1, int(sr * cfg.render.frame_ms / 1000.0))
    total_samples = int(cfg.render.duration_sec * sr)
    n_speakers = len(speech_tracks)
    overlap_rows: list[dict[str, float]] = []
    frame_rows: list[dict[str, Any]] = []

    activity = np.stack([np.abs(track) > 5e-3 for track in speech_tracks], axis=0) if speech_tracks else np.zeros((0, total_samples), dtype=bool)
    active_count = np.sum(activity, axis=0) if activity.size else np.zeros(total_samples, dtype=int)

    in_overlap = False
    overlap_start = 0
    for sample_idx, count in enumerate(active_count):
        if count >= 2 and not in_overlap:
            in_overlap = True
            overlap_start = sample_idx
        elif count < 2 and in_overlap:
            overlap_rows.append(
                {
                    "start_sec": round(overlap_start / sr, 4),
                    "end_sec": round(sample_idx / sr, 4),
                }
            )
            in_overlap = False
    if in_overlap:
        overlap_rows.append({"start_sec": round(overlap_start / sr, 4), "end_sec": round(total_samples / sr, 4)})

    speech_mix = np.sum(np.stack(speech_tracks, axis=0), axis=0) if speech_tracks else np.zeros(total_samples, dtype=np.float32)
    noise_mix = np.sum(np.stack(noise_tracks, axis=0), axis=0) if noise_tracks else np.zeros(total_samples, dtype=np.float32)

    for frame_idx, start in enumerate(range(0, total_samples, frame_samples)):
        end = min(total_samples, start + frame_samples)
        active_ids = [int(sid) for sid in range(n_speakers) if np.any(activity[sid, start:end])]
        primary_id = active_ids[0] if active_ids else None
        if len(active_ids) > 1:
            energies = [float(np.sum(np.square(speech_tracks[sid][start:end]))) for sid in active_ids]
            primary_id = active_ids[int(np.argmax(energies))]
        frame_speech = speech_mix[start:end]
        frame_noise = noise_mix[start:end]
        snr_db = 20.0 * np.log10(max(_rms(frame_speech), 1e-4) / max(_rms(frame_noise), 1e-4))
        frame_rows.append(
            {
                "frame_index": frame_idx,
                "start_time_s": round(start / sr, 4),
                "end_time_s": round(end / sr, 4),
                "active_speaker_ids": active_ids,
                "primary_speaker_id": primary_id,
                "overlap": bool(len(active_ids) > 1),
                "overlap_count": len(active_ids),
                "speaker_positions": {
                    str(sid): _speaker_position(cfg, sid, 0.5 * ((start / sr) + (end / sr)), n_speakers)
                    for sid in active_ids
                },
                "frame_snr_db": round(float(snr_db), 4),
            }
        )
    return frame_rows, overlap_rows


def _write_frame_csv(path: Path, frame_rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "frame_index",
        "start_time_s",
        "end_time_s",
        "active_speaker_ids",
        "primary_speaker_id",
        "overlap",
        "overlap_count",
        "speaker_positions",
        "frame_snr_db",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in frame_rows:
            out = dict(row)
            out["active_speaker_ids"] = "|".join(str(v) for v in row["active_speaker_ids"])
            out["speaker_positions"] = json.dumps(row["speaker_positions"], separators=(",", ":"))
            writer.writerow(out)


def _compute_summary_metrics(frame_rows: list[dict[str, Any]], plan: ConversationPlan) -> dict[str, Any]:
    total_frames = max(1, len(frame_rows))
    overlap_frames = sum(1 for row in frame_rows if row["overlap"])
    speaker_activity: dict[int, int] = {}
    for row in frame_rows:
        for speaker_id in row["active_speaker_ids"]:
            speaker_activity[int(speaker_id)] = speaker_activity.get(int(speaker_id), 0) + 1
    snrs = np.asarray([float(row["frame_snr_db"]) for row in frame_rows], dtype=np.float64)
    return {
        "overlap_ratio": float(overlap_frames / total_frames),
        "num_turn_events": len([event for event in plan.utterances if event.kind != "backchannel"]),
        "num_backchannels": len([event for event in plan.utterances if event.kind == "backchannel"]),
        "speaker_activity_frames": {str(k): int(v) for k, v in sorted(speaker_activity.items())},
        "snr_distribution_db": {
            "min": float(np.min(snrs)) if snrs.size else 0.0,
            "mean": float(np.mean(snrs)) if snrs.size else 0.0,
            "max": float(np.max(snrs)) if snrs.size else 0.0,
        },
    }


def _render_scene(scene_config_path: Path) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    sim_cfg = SimulationConfig.from_file(scene_config_path)
    return run_simulation(sim_cfg)


def _dump_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def generate_scenario(
    *,
    preset: str,
    out_dir: str | Path,
    seed: int,
    duration_sec: float | None = None,
    sample_rate: int | None = None,
    frame_ms: int | None = None,
    manifest_path: str | Path | None = None,
    export_audio: bool = True,
    scene_name: str | None = None,
) -> GenerationResult:
    rng = np.random.default_rng(seed)
    cfg = build_preset(preset)
    if duration_sec is not None:
        cfg.render.duration_sec = float(duration_sec)
    if sample_rate is not None:
        cfg.render.sample_rate = int(sample_rate)
    if frame_ms is not None:
        cfg.render.frame_ms = int(frame_ms)

    scene_name = scene_name or f"{cfg.preset}_seed{seed}"
    scene_dir = Path(out_dir) / scene_name
    scene_dir.mkdir(parents=True, exist_ok=True)

    library = load_asset_library(manifest_path)
    plan = build_conversation_plan(cfg, rng)
    speech_tracks, speech_events, label_map = _materialize_speech_tracks(cfg, plan, library, rng)
    speech_mix = np.sum(np.stack(speech_tracks, axis=0), axis=0) if speech_tracks else np.zeros(int(cfg.render.duration_sec * cfg.render.sample_rate), dtype=np.float32)
    noise_tracks, noise_layers, transient_rows = _build_noise_layers(cfg, library, rng, speech_mix=speech_mix)

    sim_cfg, render_segments = _build_scene_config(cfg, scene_dir, speech_tracks, noise_tracks)
    scene_config_path = scene_dir / "scene_config.json"
    sim_cfg.to_file(scene_config_path)

    frame_rows, overlap_rows = _compute_frame_truth(cfg, plan, speech_tracks, noise_tracks)
    summary_metrics = _compute_summary_metrics(frame_rows, plan)

    if export_audio:
        mic_audio, mic_pos, _ = _render_scene(scene_config_path)
        _write_wav(scene_dir / "audio" / "mic_array.wav", _normalize_peak(mic_audio), cfg.render.sample_rate)
        _write_wav(scene_dir / "audio" / "mix_mono.wav", _normalize_peak(np.mean(mic_audio, axis=1)), cfg.render.sample_rate)
    else:
        mic_pos = np.zeros((3, cfg.mic_array.mic_count), dtype=np.float32)

    for speaker_id, track in enumerate(speech_tracks):
        _write_wav(scene_dir / "audio" / f"speaker_{speaker_id}_dry.wav", _normalize_peak(track), cfg.render.sample_rate)
    if noise_tracks:
        noise_mix = np.sum(np.stack(noise_tracks, axis=0), axis=0)
        _write_wav(scene_dir / "audio" / "background_noise_dry.wav", _normalize_peak(noise_mix), cfg.render.sample_rate)

    metadata_path = scene_dir / "scenario_metadata.json"
    frame_truth_path = scene_dir / "frame_ground_truth.csv"
    frame_truth_json_path = scene_dir / "frame_ground_truth.json"
    metrics_path = scene_dir / "metrics_summary.json"
    events_path = scene_dir / "event_schedule.json"

    metadata = {
        "scene_name": scene_name,
        "preset": cfg.preset,
        "seed": int(seed),
        "config": cfg.to_dict(),
        "speaker_count": len(plan.speaker_ids),
        "speakers": [
            {
                "speaker_id": int(sid),
                "speaker_label": label_map[sid],
                "moving": bool(cfg.moving_speaker and sid == 0),
            }
            for sid in plan.speaker_ids
        ],
        "mic_geometry": {
            "center_m": cfg.mic_array.mic_center_m,
            "radius_m": cfg.mic_array.mic_radius_m,
            "count": cfg.mic_array.mic_count,
            "rendered_positions_xyz": np.asarray(mic_pos).tolist(),
        },
        "overlap_intervals": overlap_rows,
        "summary_metrics": summary_metrics,
        "assets": {
            "speech_events": speech_events,
            "noise_layers": noise_layers,
            "transients": transient_rows,
            "render_segments": render_segments,
            "manifest_path": None if manifest_path is None else str(Path(manifest_path).resolve()),
        },
    }
    _dump_json(metadata_path, metadata)
    _dump_json(frame_truth_json_path, {"frames": frame_rows})
    _dump_json(metrics_path, summary_metrics)
    _dump_json(events_path, {"speech_events": speech_events, "noise_events": transient_rows, "noise_layers": noise_layers})
    _write_frame_csv(frame_truth_path, frame_rows)

    return GenerationResult(
        scene_name=scene_name,
        scene_dir=scene_dir,
        scene_config_path=scene_config_path,
        metadata_path=metadata_path,
        frame_truth_path=frame_truth_path,
        metrics_path=metrics_path,
    )


def generate_dataset(
    *,
    preset: str,
    out_dir: str | Path,
    seed: int,
    num_scenes: int = 1,
    duration_sec: float | None = None,
    sample_rate: int | None = None,
    frame_ms: int | None = None,
    manifest_path: str | Path | None = None,
    export_audio: bool = True,
) -> list[GenerationResult]:
    results = []
    for scene_idx in range(num_scenes):
        results.append(
            generate_scenario(
                preset=preset,
                out_dir=out_dir,
                seed=seed + scene_idx,
                duration_sec=duration_sec,
                sample_rate=sample_rate,
                frame_ms=frame_ms,
                manifest_path=manifest_path,
                export_audio=export_audio,
                scene_name=f"{preset}_scene{scene_idx:02d}",
            )
        )
    return results


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate realistic conversational mic-array simulations")
    parser.add_argument("--preset", choices=["quiet_room", "office", "cafe", "moving_speaker", "noisy_home"], default="quiet_room")
    parser.add_argument("--out-dir", default="sim/output/realistic_conversations")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num-scenes", type=int, default=1)
    parser.add_argument("--duration-sec", type=float, default=None)
    parser.add_argument("--sample-rate", type=int, default=None)
    parser.add_argument("--frame-ms", type=int, default=None)
    parser.add_argument("--asset-manifest", type=str, default=None)
    parser.add_argument("--no-audio", action="store_true", help="Skip multichannel rendering and only emit dry references/metadata")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    results = generate_dataset(
        preset=args.preset,
        out_dir=args.out_dir,
        seed=args.seed,
        num_scenes=args.num_scenes,
        duration_sec=args.duration_sec,
        sample_rate=args.sample_rate,
        frame_ms=args.frame_ms,
        manifest_path=args.asset_manifest,
        export_audio=not bool(args.no_audio),
    )
    print(
        json.dumps(
            {
                "preset": args.preset,
                "num_scenes": len(results),
                "scenes": [str(result.scene_dir.resolve()) for result in results],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

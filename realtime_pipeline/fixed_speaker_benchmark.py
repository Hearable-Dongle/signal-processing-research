from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

from sim.realistic_conversations.assets import load_asset_library
from sim.realistic_conversations.config import build_preset
from sim.realistic_conversations.generator import _load_excerpt, _normalize_peak, _rms
from simulation.simulation_config import SimulationConfig, SimulationSource

from .plot_scene_setup import render_scene_setup
from .simulation_runner import run_simulation_pipeline


TRUE_SPEAKER_ANGLES_DEG = (0.0, 75.0, 150.0)


def _true_segments(duration_sec: float) -> tuple[tuple[float, float], ...]:
    segment_sec = float(duration_sec) / float(len(TRUE_SPEAKER_ANGLES_DEG))
    return tuple((idx * segment_sec, (idx + 1) * segment_sec) for idx in range(len(TRUE_SPEAKER_ANGLES_DEG)))


@dataclass(frozen=True)
class BenchmarkArtifacts:
    scene_config_path: Path
    scenario_metadata_path: Path
    metrics_summary_path: Path
    frame_truth_path: Path


def _normalize_deg(v: float) -> float:
    return float(v % 360.0)


def _circular_distance_deg(a: float, b: float) -> float:
    return abs((float(a) - float(b) + 180.0) % 360.0 - 180.0)


def _position_from_angle(center_xyz: np.ndarray, radius_m: float, angle_deg: float) -> list[float]:
    angle_rad = np.deg2rad(float(angle_deg))
    return [
        float(center_xyz[0] + radius_m * np.cos(angle_rad)),
        float(center_xyz[1] + radius_m * np.sin(angle_rad)),
        float(center_xyz[2]),
    ]


def _load_noise_clip(path: Path, sr: int, length_samples: int, rng: np.random.Generator) -> np.ndarray:
    clip = _load_excerpt(path, sr=sr, length_samples=length_samples, rng=rng)
    clip = clip - np.mean(clip)
    return _normalize_peak(clip, peak=0.9)


def generate_fixed_speaker_scene(
    *,
    out_root: str | Path,
    noise_scale: float,
    seed: int = 20260318,
    duration_sec: float = 30.0,
    sample_rate_hz: int = 16000,
) -> BenchmarkArtifacts:
    out_root = Path(out_root).resolve()
    scene_name = f"fixed_speaker_seq_noise{str(noise_scale).replace('.', 'p')}x_{int(duration_sec)}s"
    scene_dir = out_root / scene_name
    audio_dir = scene_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(int(seed))
    preset = build_preset("restaurant_meeting")
    preset.render.duration_sec = float(duration_sec)
    preset.render.sample_rate = int(sample_rate_hz)

    sr = int(sample_rate_hz)
    total_samples = int(round(float(duration_sec) * sr))
    segments_sec = _true_segments(float(duration_sec))
    segment_sec = float(duration_sec) / float(len(TRUE_SPEAKER_ANGLES_DEG))
    segment_samples = int(round(float(segment_sec) * sr))
    center = np.asarray(preset.mic_array.mic_center_m, dtype=float)
    speaker_radius_m = 1.8
    noise_radius_m = 2.3

    library = load_asset_library(None)
    speaker_labels = library.choose_speakers(rng, len(TRUE_SPEAKER_ANGLES_DEG))

    speaker_tracks: list[np.ndarray] = []
    speech_events: list[dict[str, object]] = []
    render_segments: list[dict[str, object]] = []
    sources: list[SimulationSource] = []

    for speaker_id, (speaker_label, angle_deg, (start_sec, end_sec)) in enumerate(
        zip(speaker_labels, TRUE_SPEAKER_ANGLES_DEG, segments_sec, strict=True)
    ):
        track = np.zeros(total_samples, dtype=np.float32)
        clip_path = library.choose_speech_path(rng, speaker_label)
        excerpt = _load_excerpt(clip_path, sr=sr, length_samples=segment_samples, rng=rng)
        start_idx = int(round(float(start_sec) * sr))
        end_idx = start_idx + excerpt.shape[0]
        track[start_idx:end_idx] += excerpt
        track = _normalize_peak(track, peak=0.95)
        speaker_tracks.append(track)
        wav_path = audio_dir / f"speaker_{speaker_id}.wav"
        sf.write(wav_path, track, sr)
        loc = _position_from_angle(center, speaker_radius_m, angle_deg)
        sources.append(SimulationSource(loc=loc, audio_path=str(wav_path.resolve()), gain=1.0, classification="speech"))
        speech_events.append(
            {
                "speaker_id": int(speaker_id),
                "speaker_label": str(speaker_label),
                "start_sec": float(start_sec),
                "end_sec": float(end_sec),
                "direction_deg": float(angle_deg),
            }
        )
        render_segments.append(
            {
                "source_id": f"speaker_{speaker_id}",
                "speaker_id": int(speaker_id),
                "classification": "speech",
                "position_m": [float(v) for v in loc],
                "start_sec": float(start_sec),
                "end_sec": float(end_sec),
            }
        )

    speech_rms = float(np.mean([_rms(track[np.abs(track) > 1e-8]) for track in speaker_tracks]))
    noise_angles = (40.0, 140.0, 220.0, 320.0)
    noise_paths = [library.choose_noise_path(rng) for _ in noise_angles]
    for idx, (angle_deg, noise_path) in enumerate(zip(noise_angles, noise_paths, strict=True)):
        noise_track = _load_noise_clip(noise_path, sr=sr, length_samples=total_samples, rng=rng)
        if float(noise_scale) > 0.0:
            target_rms = speech_rms * float(noise_scale) / max(np.sqrt(len(noise_angles)), 1.0)
            current_rms = _rms(noise_track)
            if current_rms > 1e-8:
                noise_track = (noise_track * float(target_rms / current_rms)).astype(np.float32, copy=False)
        else:
            noise_track = np.zeros(total_samples, dtype=np.float32)
        wav_path = audio_dir / f"noise_{idx}.wav"
        sf.write(wav_path, noise_track, sr)
        loc = _position_from_angle(center, noise_radius_m, angle_deg)
        sources.append(SimulationSource(loc=loc, audio_path=str(wav_path.resolve()), gain=1.0, classification="noise"))
        render_segments.append(
            {
                "source_id": f"noise_{idx}",
                "classification": "noise",
                "position_m": [float(v) for v in loc],
                "start_sec": 0.0,
                "end_sec": float(duration_sec),
            }
        )

    sim_cfg = SimulationConfig.from_dict(
        {
            "room": {
                "dimensions": list(preset.room.dimensions_m),
                "absorption": float(preset.room.absorption),
            },
            "microphone_array": {
                "mic_center": [float(v) for v in center.tolist()],
                "mic_radius": float(preset.mic_array.mic_radius_m),
                "mic_count": int(preset.mic_array.mic_count),
            },
            "audio": {
                "sources": [
                    {
                        "loc": [float(v) for v in source.loc],
                        "audio": str(source.audio_path),
                        "gain": float(source.gain),
                        "classification": str(source.classification),
                    }
                    for source in sources
                ],
                "duration": float(duration_sec),
                "fs": int(sr),
            },
        }
    )
    scene_config_path = scene_dir / "scene_config.json"
    sim_cfg.to_file(scene_config_path)

    noise_rms = 0.0
    if float(noise_scale) > 0.0:
        noise_rms = float(
            np.sqrt(
                sum((_rms(sf.read(audio_dir / f"noise_{idx}.wav")[0]) ** 2 for idx in range(len(noise_angles))))
            )
        )
    mean_snr_db = 120.0 if noise_rms <= 1e-8 else float(20.0 * np.log10(max(speech_rms, 1e-8) / noise_rms))
    metrics_summary = {
        "overlap_ratio": 0.0,
        "num_turn_events": 3,
        "speaker_activity_frames": {str(idx): int(segment_sec * 1000.0 / 20.0) for idx in range(3)},
        "snr_distribution_db": {
            "min": float(mean_snr_db),
            "mean": float(mean_snr_db),
            "max": float(mean_snr_db),
        },
    }
    metrics_summary_path = scene_dir / "metrics_summary.json"
    metrics_summary_path.write_text(json.dumps(metrics_summary, indent=2), encoding="utf-8")

    scenario_metadata = {
        "speaker_count": 3,
        "config": {"render": {"duration_sec": float(duration_sec), "sample_rate": int(sr), "frame_ms": 20}},
        "assets": {
            "speech_events": speech_events,
            "render_segments": render_segments,
        },
    }
    scenario_metadata_path = scene_dir / "scenario_metadata.json"
    scenario_metadata_path.write_text(json.dumps(scenario_metadata, indent=2), encoding="utf-8")

    frame_truth_path = scene_dir / "frame_ground_truth.csv"
    with frame_truth_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["time_sec", "active_speaker_id", "active_direction_deg"])
        writer.writeheader()
        frame_times = np.arange(0.0, float(duration_sec), 0.02)
        for t_sec in frame_times:
            active_speaker_id = -1
            active_direction_deg = np.nan
            for speaker_id, (start_sec, end_sec) in enumerate(segments_sec):
                if float(start_sec) <= float(t_sec) < float(end_sec):
                    active_speaker_id = int(speaker_id)
                    active_direction_deg = float(TRUE_SPEAKER_ANGLES_DEG[speaker_id])
                    break
            writer.writerow(
                {
                    "time_sec": f"{float(t_sec):.3f}",
                    "active_speaker_id": int(active_speaker_id),
                    "active_direction_deg": "" if not np.isfinite(active_direction_deg) else f"{float(active_direction_deg):.3f}",
                }
            )

    return BenchmarkArtifacts(
        scene_config_path=scene_config_path,
        scenario_metadata_path=scenario_metadata_path,
        metrics_summary_path=metrics_summary_path,
        frame_truth_path=frame_truth_path,
    )


def _ground_truth_timeline(*, duration_s: float, frame_step_ms: float = 10.0) -> dict[str, object]:
    num_frames = max(1, int(round(float(duration_s) * 1000.0 / float(frame_step_ms))))
    time_s = np.arange(num_frames, dtype=np.float64) * (float(frame_step_ms) / 1000.0)
    active_speaker_ids = np.full(num_frames, -1, dtype=np.int32)
    active_directions_deg = np.full(num_frames, np.nan, dtype=np.float64)
    for speaker_id, (start_sec, end_sec) in enumerate(_true_segments(float(duration_s))):
        mask = (time_s >= float(start_sec)) & (time_s < float(end_sec))
        active_speaker_ids[mask] = int(speaker_id)
        active_directions_deg[mask] = float(TRUE_SPEAKER_ANGLES_DEG[speaker_id])
    return {
        "time_s": time_s.tolist(),
        "active_speaker_ids": active_speaker_ids.tolist(),
        "active_directions_deg": active_directions_deg.tolist(),
        "doa_by_speaker": {str(idx): float(v) for idx, v in enumerate(TRUE_SPEAKER_ANGLES_DEG)},
    }


def _match_predicted_ids_to_truth(summary: dict) -> dict[int, int]:
    speaker_dirs: dict[int, list[float]] = {}
    for row in summary.get("speaker_map_trace", []):
        for item in row.get("speakers", []):
            sid = int(item["speaker_id"])
            conf = float(item.get("confidence", 0.0))
            ident = float(item.get("identity_confidence", 0.0))
            if conf < 0.2 and ident < 0.2:
                continue
            speaker_dirs.setdefault(sid, []).append(float(item["direction_degrees"]))
    unmatched_truth = set(range(len(TRUE_SPEAKER_ANGLES_DEG)))
    out: dict[int, int] = {}
    for sid, median_deg in sorted(
        ((sid, float(np.median(vals))) for sid, vals in speaker_dirs.items() if vals),
        key=lambda item: item[1],
    ):
        if not unmatched_truth:
            break
        best_truth = min(unmatched_truth, key=lambda idx: _circular_distance_deg(median_deg, TRUE_SPEAKER_ANGLES_DEG[idx]))
        out[int(sid)] = int(best_truth)
        unmatched_truth.remove(best_truth)
    return out


def _speaker_count_timeline(summary: dict, *, frame_count: int) -> tuple[list[int], list[dict[str, object]]]:
    cumulative: list[int] = []
    discovery_events: list[dict[str, object]] = []
    seen: set[int] = set()
    trace_by_frame = {int(row["frame_index"]): row for row in summary.get("speaker_map_trace", [])}
    for frame_idx in range(frame_count):
        row = trace_by_frame.get(frame_idx, {})
        for item in row.get("speakers", []):
            sid = int(item["speaker_id"])
            conf = float(item.get("confidence", 0.0))
            ident = float(item.get("identity_confidence", 0.0))
            maturity = str(item.get("identity_maturity", "unknown"))
            if sid in seen:
                continue
            if maturity == "stable" or (conf >= 0.25 and ident >= 0.25):
                seen.add(sid)
                discovery_events.append(
                    {
                        "frame_index": int(frame_idx),
                        "time_s": float(frame_idx * 0.01),
                        "speaker_id": int(sid),
                        "confidence": float(conf),
                        "identity_confidence": float(ident),
                    }
                )
        cumulative.append(len(seen))
    return cumulative, discovery_events


def _plot_direction_timeline(*, out_path: Path, summary: dict, ground_truth: dict[str, object]) -> dict[int, int]:
    time_s = np.asarray(ground_truth["time_s"], dtype=float)
    gt_speakers = np.asarray(ground_truth["active_speaker_ids"], dtype=int)
    gt_deg = np.asarray(ground_truth["active_directions_deg"], dtype=float)
    speaker_match = _match_predicted_ids_to_truth(summary)
    cmap = plt.get_cmap("tab10")
    gt_colors = {sid: cmap(sid % 10) for sid in range(len(TRUE_SPEAKER_ANGLES_DEG))}

    predicted_series: dict[int, np.ndarray] = {}
    for row in summary.get("speaker_map_trace", []):
        frame_idx = int(row["frame_index"])
        for item in row.get("speakers", []):
            conf = float(item.get("confidence", 0.0))
            ident = float(item.get("identity_confidence", 0.0))
            activity = float(item.get("activity_confidence", 0.0))
            if activity < 0.2 and conf < 0.3 and ident < 0.3:
                continue
            sid = int(item["speaker_id"])
            series = predicted_series.setdefault(sid, np.full(time_s.shape[0], np.nan, dtype=np.float64))
            if frame_idx < series.shape[0]:
                series[frame_idx] = float(item["direction_degrees"])

    plt.figure(figsize=(14, 5))
    for true_sid in range(len(TRUE_SPEAKER_ANGLES_DEG)):
        mask = gt_speakers == true_sid
        gt_series = np.where(mask, gt_deg, np.nan)
        plt.plot(time_s, gt_series, color=gt_colors[true_sid], linewidth=2.0, alpha=0.3, label=f"gt spk {true_sid}")

    for pred_sid, series in sorted(predicted_series.items()):
        color_sid = speaker_match.get(pred_sid, pred_sid)
        color = gt_colors.get(color_sid, cmap(pred_sid % 10))
        label = f"pred spk {pred_sid}"
        if pred_sid in speaker_match:
            label += f" -> gt {speaker_match[pred_sid]}"
        plt.plot(time_s, series, color=color, linewidth=1.6, linestyle=":", label=label)

    plt.xlabel("time (s)")
    plt.ylabel("direction (deg)")
    plt.ylim(-5.0, 365.0)
    plt.grid(True, alpha=0.2)
    plt.legend(loc="upper right", ncol=2)
    plt.title("Ground truth vs predicted speaker directions over time")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    return speaker_match


def _plot_unique_speakers(*, out_path: Path, time_s: np.ndarray, cumulative_count: list[int]) -> None:
    plt.figure(figsize=(14, 3.2))
    plt.step(time_s, cumulative_count, where="post", linewidth=2.0)
    plt.xlabel("time (s)")
    plt.ylabel("unique speakers")
    plt.ylim(-0.1, max(cumulative_count + [0]) + 0.5)
    plt.grid(True, alpha=0.2)
    plt.title("Unique assigned speakers found over time")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _wrap_diff_deg(a: float, b: float) -> float:
    return float((float(a) - float(b) + 180.0) % 360.0 - 180.0)


def _smooth_direction_series(
    *,
    raw_deg: np.ndarray,
    conf: np.ndarray,
    ident: np.ndarray,
    activity: np.ndarray,
    small_change_deg: float = 12.0,
    medium_change_deg: float = 30.0,
    relock_persist_frames: int = 2,
    max_hold_frames: int = 150,
) -> dict[str, object]:
    n = raw_deg.shape[0]
    smoothed = np.full(n, np.nan, dtype=np.float64)
    confidence_gate = np.clip((0.55 * conf) + (0.30 * ident) + (0.15 * activity), 0.0, 1.0)
    decision = np.full(n, "", dtype=object)
    pending_counts = np.zeros(n, dtype=np.int32)
    pending_candidate = np.nan
    pending_count = 0
    hold_frames = 0

    for idx in range(n):
        raw = float(raw_deg[idx])
        if not np.isfinite(raw):
            if idx > 0 and np.isfinite(smoothed[idx - 1]) and hold_frames < max_hold_frames:
                smoothed[idx] = smoothed[idx - 1]
                hold_frames += 1
                decision[idx] = "hold_missing"
            else:
                hold_frames = max_hold_frames
                decision[idx] = "missing"
            pending_counts[idx] = pending_count
            continue

        hold_frames = 0
        if idx == 0 or not np.isfinite(smoothed[idx - 1]):
            smoothed[idx] = raw
            decision[idx] = "init"
            pending_candidate = np.nan
            pending_count = 0
            pending_counts[idx] = pending_count
            continue

        prev = float(smoothed[idx - 1])
        delta = abs(_wrap_diff_deg(raw, prev))
        gate = float(confidence_gate[idx])
        if delta <= float(small_change_deg):
            smoothed[idx] = raw
            pending_candidate = np.nan
            pending_count = 0
            decision[idx] = "fast"
        elif delta <= float(medium_change_deg):
            alpha = float(np.clip(0.20 + (0.50 * gate), 0.20, 0.70))
            smoothed[idx] = float((prev + alpha * _wrap_diff_deg(raw, prev)) % 360.0)
            pending_candidate = np.nan
            pending_count = 0
            decision[idx] = "smooth"
        else:
            if np.isfinite(pending_candidate) and abs(_wrap_diff_deg(raw, float(pending_candidate))) <= float(small_change_deg):
                pending_count += 1
            else:
                pending_candidate = raw
                pending_count = 1
            if pending_count >= int(relock_persist_frames) and gate >= 0.45:
                smoothed[idx] = raw
                pending_candidate = np.nan
                pending_count = 0
                decision[idx] = "relock"
            else:
                smoothed[idx] = prev
                decision[idx] = "hold_large"
        pending_counts[idx] = pending_count

    lower_small = np.mod(smoothed - float(small_change_deg), 360.0)
    upper_small = np.mod(smoothed + float(small_change_deg), 360.0)
    lower_medium = np.mod(smoothed - float(medium_change_deg), 360.0)
    upper_medium = np.mod(smoothed + float(medium_change_deg), 360.0)
    return {
        "smoothed_deg": smoothed.tolist(),
        "confidence_gate": confidence_gate.tolist(),
        "small_change_deg": [float(small_change_deg)] * n,
        "medium_change_deg": [float(medium_change_deg)] * n,
        "small_band_lower_deg": lower_small.tolist(),
        "small_band_upper_deg": upper_small.tolist(),
        "medium_band_lower_deg": lower_medium.tolist(),
        "medium_band_upper_deg": upper_medium.tolist(),
        "pending_count": pending_counts.tolist(),
        "decision": [str(v) for v in decision.tolist()],
    }


def _build_smoothing_payload(summary: dict, ground_truth: dict[str, object]) -> dict[str, object]:
    time_s = np.asarray(ground_truth["time_s"], dtype=float)
    traces: dict[int, dict[str, np.ndarray]] = {}
    for row in summary.get("speaker_map_trace", []):
        frame_idx = int(row["frame_index"])
        for item in row.get("speakers", []):
            sid = int(item["speaker_id"])
            speaker = traces.setdefault(
                sid,
                {
                    "raw_deg": np.full(time_s.shape[0], np.nan, dtype=np.float64),
                    "conf": np.zeros(time_s.shape[0], dtype=np.float64),
                    "ident": np.zeros(time_s.shape[0], dtype=np.float64),
                    "activity": np.zeros(time_s.shape[0], dtype=np.float64),
                },
            )
            if frame_idx >= time_s.shape[0]:
                continue
            speaker["raw_deg"][frame_idx] = float(item.get("direction_degrees", np.nan))
            speaker["conf"][frame_idx] = float(item.get("confidence", 0.0))
            speaker["ident"][frame_idx] = float(item.get("identity_confidence", 0.0))
            speaker["activity"][frame_idx] = float(item.get("activity_confidence", 0.0))

    payload: dict[str, object] = {"time_s": ground_truth["time_s"], "speakers": {}}
    for sid, vals in sorted(traces.items()):
        smooth = _smooth_direction_series(
            raw_deg=vals["raw_deg"],
            conf=vals["conf"],
            ident=vals["ident"],
            activity=vals["activity"],
        )
        payload["speakers"][str(sid)] = {
            "raw_deg": vals["raw_deg"].tolist(),
            "confidence": vals["conf"].tolist(),
            "identity_confidence": vals["ident"].tolist(),
            "activity_confidence": vals["activity"].tolist(),
            **smooth,
        }
    return payload


def _plot_smoothed_directions(
    *,
    out_path: Path,
    summary: dict,
    ground_truth: dict[str, object],
    smoothing_payload: dict[str, object],
) -> None:
    time_s = np.asarray(ground_truth["time_s"], dtype=float)
    gt_speakers = np.asarray(ground_truth["active_speaker_ids"], dtype=int)
    gt_deg = np.asarray(ground_truth["active_directions_deg"], dtype=float)
    speaker_match = _match_predicted_ids_to_truth(summary)
    cmap = plt.get_cmap("tab10")
    gt_colors = {sid: cmap(sid % 10) for sid in range(len(TRUE_SPEAKER_ANGLES_DEG))}

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    ax_dir, ax_gate = axes

    for true_sid in range(len(TRUE_SPEAKER_ANGLES_DEG)):
        mask = gt_speakers == true_sid
        gt_series = np.where(mask, gt_deg, np.nan)
        ax_dir.plot(time_s, gt_series, color=gt_colors[true_sid], linewidth=2.0, alpha=0.3, label=f"gt spk {true_sid}")

    for sid_str, speaker in sorted(smoothing_payload["speakers"].items(), key=lambda item: int(item[0])):
        sid = int(sid_str)
        color_sid = speaker_match.get(sid, sid)
        color = gt_colors.get(color_sid, cmap(sid % 10))
        raw = np.asarray(speaker["raw_deg"], dtype=float)
        smooth = np.asarray(speaker["smoothed_deg"], dtype=float)
        band_lo = np.asarray(speaker["small_band_lower_deg"], dtype=float)
        band_hi = np.asarray(speaker["small_band_upper_deg"], dtype=float)
        gate = np.asarray(speaker["confidence_gate"], dtype=float)

        ax_dir.plot(time_s, raw, color=color, linestyle=":", linewidth=1.0, alpha=0.7, label=f"raw spk {sid}")
        ax_dir.plot(time_s, smooth, color=color, linewidth=2.0, label=f"smooth spk {sid}")
        if np.nanmax(np.abs(_wrap_diff_deg(band_hi[np.isfinite(smooth)][0], band_lo[np.isfinite(smooth)][0])) if np.any(np.isfinite(smooth)) else [0]) >= 0:
            lower = np.where(np.isfinite(smooth), smooth - 12.0, np.nan)
            upper = np.where(np.isfinite(smooth), smooth + 12.0, np.nan)
            ax_dir.fill_between(time_s, lower, upper, color=color, alpha=0.08)
        ax_gate.plot(time_s, gate, color=color, linewidth=1.5, label=f"gate spk {sid}")

    ax_dir.set_ylabel("direction (deg)")
    ax_dir.set_ylim(-5.0, 365.0)
    ax_dir.grid(True, alpha=0.2)
    ax_dir.legend(loc="upper right", ncol=2)
    ax_dir.set_title("Raw vs smoothed accepted speaker directions")

    ax_gate.axhline(0.45, color="black", linestyle="--", linewidth=1.0, alpha=0.6, label="relock gate")
    ax_gate.set_ylabel("confidence gate")
    ax_gate.set_xlabel("time (s)")
    ax_gate.set_ylim(-0.02, 1.02)
    ax_gate.grid(True, alpha=0.2)
    ax_gate.legend(loc="upper right", ncol=2)
    ax_gate.set_title("Smoothing gate / threshold over time")

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_target_lock(
    *,
    out_path: Path,
    summary: dict,
    ground_truth: dict[str, object],
) -> None:
    time_s = np.asarray(ground_truth["time_s"], dtype=float)
    gt_ids = np.asarray(ground_truth["active_speaker_ids"], dtype=int)
    focus_trace = summary.get("focus_trace", [])
    locked = np.full(time_s.shape[0], np.nan, dtype=np.float64)
    boost = np.zeros(time_s.shape[0], dtype=np.float64)
    for row in focus_trace:
        idx = int(row["frame_index"])
        if idx >= time_s.shape[0]:
            continue
        if row.get("locked_target_speaker_id") is not None:
            locked[idx] = float(row["locked_target_speaker_id"])
        boost[idx] = float(row.get("user_boost_db", 0.0))

    fig, axes = plt.subplots(2, 1, figsize=(14, 5), sharex=True)
    axes[0].plot(time_s, gt_ids, linewidth=1.8, alpha=0.35, label="ground truth active speaker")
    axes[0].plot(time_s, locked, linewidth=1.8, linestyle="--", label="locked target speaker")
    axes[0].set_ylabel("speaker id")
    axes[0].grid(True, alpha=0.2)
    axes[0].legend(loc="upper right")
    axes[0].set_title("Target speaker lock over time")

    axes[1].plot(time_s, boost, linewidth=1.8)
    axes[1].set_ylabel("boost dB")
    axes[1].set_xlabel("time (s)")
    axes[1].grid(True, alpha=0.2)
    axes[1].set_title("Applied target boost over time")

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_output_comparison(*, out_path: Path, base_wav: Path, target_wav: Path) -> None:
    y_base, sr = sf.read(base_wav, always_2d=False)
    y_tgt, _ = sf.read(target_wav, always_2d=False)
    y_base = np.asarray(y_base, dtype=np.float32).reshape(-1)
    y_tgt = np.asarray(y_tgt, dtype=np.float32).reshape(-1)
    t = np.arange(min(y_base.shape[0], y_tgt.shape[0]), dtype=np.float64) / max(int(sr), 1)
    y_base = y_base[: t.shape[0]]
    y_tgt = y_tgt[: t.shape[0]]

    fig, axes = plt.subplots(2, 1, figsize=(14, 5), sharex=True)
    axes[0].plot(t, y_base, linewidth=0.8, label="baseline")
    axes[0].plot(t, y_tgt, linewidth=0.8, alpha=0.8, label="target amplified")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.2)
    axes[0].set_title("Baseline vs target-amplified output")

    win = max(1, int(sr * 0.25))
    def _window_rms(x: np.ndarray) -> np.ndarray:
        vals = []
        for start in range(0, x.shape[0], win):
            seg = x[start : start + win]
            vals.append(float(np.sqrt(np.mean(seg.astype(np.float64) ** 2) + 1e-12)))
        return np.asarray(vals, dtype=np.float64)

    rms_t = np.arange(_window_rms(y_base).shape[0], dtype=np.float64) * (win / max(int(sr), 1))
    axes[1].plot(rms_t, _window_rms(y_base), linewidth=1.5, label="baseline rms")
    axes[1].plot(rms_t, _window_rms(y_tgt), linewidth=1.5, label="target rms")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.2)
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("rms")

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def run_fixed_speaker_benchmark(
    *,
    out_dir: str | Path,
    noise_scale: float,
    use_mock_separation: bool = False,
    single_dominant_no_separator: bool = False,
    identity_backend: str = "mfcc_legacy",
    duration_sec: float = 30.0,
    target_first_identified_speaker: bool = False,
) -> dict[str, object]:
    output_root = Path(out_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    artifacts = generate_fixed_speaker_scene(
        out_root=output_root.parent / f"{output_root.name}_assets",
        noise_scale=float(noise_scale),
        duration_sec=float(duration_sec),
    )
    visualizations_root = output_root / "visualizations"
    visualizations_root.mkdir(parents=True, exist_ok=True)

    run_summary = run_simulation_pipeline(
        scene_config_path=artifacts.scene_config_path,
        out_dir=output_root / "baseline" if target_first_identified_speaker else output_root,
        use_mock_separation=bool(use_mock_separation),
        single_dominant_no_separator=bool(single_dominant_no_separator),
        capture_trace=True,
        slow_chunk_ms=2000,
        slow_chunk_hop_ms=1000,
        control_mode="speaker_tracking_mode",
        direction_long_memory_enabled=False,
        direction_long_memory_window_ms=2000.0,
        direction_history_window_chunks=2,
        direction_speaker_stale_timeout_ms=2000.0,
        direction_speaker_forget_timeout_ms=2000.0,
        identity_backend=str(identity_backend),
        identity_retire_after_chunks=2,
        identity_new_speaker_max_existing_score=0.22,
        identity_direction_mismatch_block_deg=45.0,
        fast_path_reference_mode="speaker_map",
        auto_focus_active_speaker=True,
        robust_mode=True,
    )
    target_summary = None
    if target_first_identified_speaker:
        target_summary = run_simulation_pipeline(
            scene_config_path=artifacts.scene_config_path,
            out_dir=output_root / "target_first_identified",
            use_mock_separation=bool(use_mock_separation),
            single_dominant_no_separator=bool(single_dominant_no_separator),
            capture_trace=True,
            slow_chunk_ms=2000,
            slow_chunk_hop_ms=1000,
            control_mode="speaker_tracking_mode",
            direction_long_memory_enabled=False,
            direction_long_memory_window_ms=2000.0,
            direction_history_window_chunks=2,
            direction_speaker_stale_timeout_ms=2000.0,
            direction_speaker_forget_timeout_ms=2000.0,
            identity_backend=str(identity_backend),
            identity_retire_after_chunks=2,
            identity_new_speaker_max_existing_score=0.22,
            identity_direction_mismatch_block_deg=45.0,
            fast_path_reference_mode="speaker_map",
            auto_lock_first_identified_speaker=True,
            target_user_boost_db=3.0,
            direction_focus_gain_db=3.0,
            direction_non_focus_attenuation_db=-18.0,
            robust_mode=True,
        )

    primary_summary = target_summary or run_summary
    frame_step_ms = float(primary_summary.get("fast_frame_ms", 10.0))
    ground_truth = _ground_truth_timeline(duration_s=float(primary_summary["duration_s"]), frame_step_ms=frame_step_ms)
    speaker_match = _plot_direction_timeline(
        out_path=visualizations_root / "speaker_directions_over_time.png",
        summary=primary_summary,
        ground_truth=ground_truth,
    )
    time_s = np.asarray(ground_truth["time_s"], dtype=float)
    cumulative_count, discovery_events = _speaker_count_timeline(primary_summary, frame_count=time_s.shape[0])
    _plot_unique_speakers(
        out_path=visualizations_root / "unique_speakers_over_time.png",
        time_s=time_s,
        cumulative_count=cumulative_count,
    )
    render_scene_setup(
        scene_config_path=artifacts.scene_config_path,
        scenario_metadata_path=artifacts.scenario_metadata_path,
        comparison_summary_path=None,
        out_path=visualizations_root / "scene_layout_user_view.png",
    )
    smoothing_payload = _build_smoothing_payload(primary_summary, ground_truth)
    (output_root / "speaker_smoothing_trace.json").write_text(json.dumps(smoothing_payload, indent=2), encoding="utf-8")
    _plot_smoothed_directions(
        out_path=visualizations_root / "speaker_directions_smoothed.png",
        summary=primary_summary,
        ground_truth=ground_truth,
        smoothing_payload=smoothing_payload,
    )
    if target_summary is not None:
        _plot_target_lock(
            out_path=visualizations_root / "target_lock_over_time.png",
            summary=target_summary,
            ground_truth=ground_truth,
        )
        _plot_output_comparison(
            out_path=visualizations_root / "baseline_vs_target_output.png",
            base_wav=(output_root / "baseline" / "enhanced_fast_path.wav"),
            target_wav=(output_root / "target_first_identified" / "enhanced_fast_path.wav"),
        )

    speaker_timeline = {
        "time_s": ground_truth["time_s"],
        "ground_truth_active_speaker_ids": ground_truth["active_speaker_ids"],
        "ground_truth_active_directions_deg": ground_truth["active_directions_deg"],
        "speaker_map_trace": primary_summary.get("speaker_map_trace", []),
        "focus_trace": primary_summary.get("focus_trace", []),
        "predicted_to_ground_truth_mapping": {str(k): int(v) for k, v in sorted(speaker_match.items())},
    }
    (output_root / "speaker_timeline.json").write_text(json.dumps(speaker_timeline, indent=2), encoding="utf-8")

    unique_payload = {
        "time_s": ground_truth["time_s"],
        "cumulative_unique_speakers": cumulative_count,
        "discovery_events": discovery_events,
    }
    (output_root / "unique_speakers_over_time.json").write_text(json.dumps(unique_payload, indent=2), encoding="utf-8")

    final_summary = {
        "scene_config": str(artifacts.scene_config_path.resolve()),
        "scenario_metadata": str(artifacts.scenario_metadata_path.resolve()),
        "metrics_summary": str(artifacts.metrics_summary_path.resolve()),
        "noise_scale": float(noise_scale),
        "identity_backend": str(identity_backend),
        "use_mock_separation": bool(use_mock_separation),
        "single_dominant_no_separator": bool(single_dominant_no_separator),
        "separation_mode": str(primary_summary.get("separation_mode", "unknown")),
        "target_first_identified_speaker": bool(target_first_identified_speaker),
        "slow_chunk_ms": int(primary_summary["slow_chunk_ms"]),
        "slow_chunk_hop_ms": int(primary_summary["slow_chunk_hop_ms"]),
        "fast_rtf": float(primary_summary["fast_rtf"]),
        "slow_rtf": float(primary_summary["slow_rtf"]),
        "speaker_map_updates": int(primary_summary["speaker_map_updates"]),
        "final_unique_speakers_found": int(max(cumulative_count) if cumulative_count else 0),
        "artifacts": {
            "enhanced_wav": str(((output_root / "target_first_identified" / "enhanced_fast_path.wav") if target_summary is not None else (output_root / "enhanced_fast_path.wav")).resolve()),
            "summary_json": str(((output_root / "target_first_identified" / "summary.json") if target_summary is not None else (output_root / "summary.json")).resolve()),
            "speaker_timeline": str((output_root / "speaker_timeline.json").resolve()),
            "unique_speakers": str((output_root / "unique_speakers_over_time.json").resolve()),
            "direction_plot": str((visualizations_root / "speaker_directions_over_time.png").resolve()),
            "unique_count_plot": str((visualizations_root / "unique_speakers_over_time.png").resolve()),
            "scene_layout": str((visualizations_root / "scene_layout_user_view.png").resolve()),
            "smoothed_direction_plot": str((visualizations_root / "speaker_directions_smoothed.png").resolve()),
        },
    }
    if target_summary is not None:
        final_summary["artifacts"]["baseline_enhanced_wav"] = str((output_root / "baseline" / "enhanced_fast_path.wav").resolve())
        final_summary["artifacts"]["target_lock_plot"] = str((visualizations_root / "target_lock_over_time.png").resolve())
        final_summary["artifacts"]["baseline_vs_target_output"] = str((visualizations_root / "baseline_vs_target_output.png").resolve())
        final_summary["artifacts"]["target_summary_json"] = str((output_root / "target_first_identified" / "summary.json").resolve())
    (output_root / "benchmark_summary.json").write_text(json.dumps(final_summary, indent=2), encoding="utf-8")
    return final_summary


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run fixed 30s speaker-direction benchmark.")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--noise-scale", type=float, required=True)
    p.add_argument("--mock-separation", action="store_true")
    p.add_argument("--single-dominant-no-separator", action="store_true")
    p.add_argument("--identity-backend", default="mfcc_legacy")
    p.add_argument("--target-first-identified-speaker", action="store_true")
    return p


def main() -> None:
    args = _build_arg_parser().parse_args()
    summary = run_fixed_speaker_benchmark(
        out_dir=args.out_dir,
        noise_scale=float(args.noise_scale),
        use_mock_separation=bool(args.mock_separation),
        single_dominant_no_separator=bool(args.single_dominant_no_separator),
        identity_backend=str(args.identity_backend),
        target_first_identified_speaker=bool(args.target_first_identified_speaker),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

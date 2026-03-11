from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import math
import os
import re
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path

try:
    import numpy as np
except ModuleNotFoundError as exc:  # pragma: no cover - environment-specific dependency guard
    missing = exc.name or "numpy"
    raise SystemExit(
        f"Missing dependency '{missing}'. Install the beamforming benchmark dependencies first, "
        f"for example: `python3 -m pip install -r beamforming/requirements.txt`."
    ) from exc

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError as exc:  # pragma: no cover - environment-specific dependency guard
    missing = exc.name or "matplotlib"
    raise SystemExit(
        f"Missing dependency '{missing}'. Install the beamforming benchmark dependencies first, "
        f"for example: `python3 -m pip install -r beamforming/requirements.txt`."
    ) from exc

from realtime_pipeline.recording_runner import run_recording_pipeline
from simulation.mic_array_profiles import mic_positions_xyz

try:
    import soundfile as sf
except ModuleNotFoundError as exc:  # pragma: no cover - environment-specific dependency guard
    missing = exc.name or "soundfile"
    raise SystemExit(
        f"Missing dependency '{missing}'. Install the beamforming benchmark dependencies first, "
        f"for example: `python3 -m pip install -r beamforming/requirements.txt`."
    ) from exc


CHANNEL_RE = re.compile(r"(\d+)")


def _slug(v: str) -> str:
    return re.sub(r"[^a-z0-9._-]+", "_", str(v).strip().lower())


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _clip_rate(x: np.ndarray, threshold: float = 0.99) -> float:
    y = np.asarray(x, dtype=np.float64).reshape(-1)
    if y.size == 0:
        return 0.0
    return float(np.mean(np.abs(y) >= threshold))


def _high_band_noise_ratio(x: np.ndarray, fs: int, high_hz: int = 4000) -> float:
    y = np.asarray(x, dtype=np.float64).reshape(-1)
    if y.size == 0:
        return 0.0
    spec = np.fft.rfft(y)
    power = np.abs(spec) ** 2
    freq = np.fft.rfftfreq(y.size, d=1.0 / float(fs))
    total = float(np.sum(power) + 1e-12)
    high = float(np.sum(power[freq >= float(high_hz)]))
    return high / total


def _frame_gain_delta_p95(raw: np.ndarray, proc: np.ndarray, frame_len: int = 160) -> float:
    x = np.asarray(raw, dtype=np.float64).reshape(-1)
    y = np.asarray(proc, dtype=np.float64).reshape(-1)
    n = min(x.size, y.size)
    if n <= frame_len:
        return 0.0
    x = x[:n]
    y = y[:n]
    ratios: list[float] = []
    for start in range(0, n - frame_len + 1, frame_len):
        xr = float(np.sqrt(np.mean(x[start : start + frame_len] ** 2) + 1e-12))
        yr = float(np.sqrt(np.mean(y[start : start + frame_len] ** 2) + 1e-12))
        ratios.append(yr / max(xr, 1e-6))
    if len(ratios) < 3:
        return 0.0
    delta = np.abs(np.diff(np.asarray(ratios, dtype=np.float64)))
    return float(np.percentile(delta, 95))


def _rms_db_ratio(raw: np.ndarray, proc: np.ndarray) -> float:
    raw_rms = float(np.sqrt(np.mean(np.asarray(raw, dtype=np.float64) ** 2) + 1e-12))
    proc_rms = float(np.sqrt(np.mean(np.asarray(proc, dtype=np.float64) ** 2) + 1e-12))
    return float(20.0 * math.log10(max(proc_rms, 1e-6) / max(raw_rms, 1e-6)))


def _parse_channel_key(path: Path) -> tuple[int, str]:
    match = CHANNEL_RE.search(path.stem)
    if match:
        return int(match.group(1)), path.name
    return 10**9, path.name


def _load_multichannel_wavs(raw_dir: Path) -> tuple[np.ndarray, int, list[str]]:
    wav_paths = sorted(raw_dir.glob("*.wav"), key=_parse_channel_key)
    if not wav_paths:
        raise FileNotFoundError(f"no WAV files found in {raw_dir}")

    chans: list[np.ndarray] = []
    sample_rate_hz: int | None = None
    min_len: int | None = None
    for wav_path in wav_paths:
        audio, sr = sf.read(wav_path, dtype="float32", always_2d=False)
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        if sample_rate_hz is None:
            sample_rate_hz = int(sr)
        elif int(sr) != sample_rate_hz:
            raise ValueError(f"sample rate mismatch in {raw_dir}: {wav_path} has {sr}, expected {sample_rate_hz}")
        min_len = int(audio.shape[0]) if min_len is None else min(min_len, int(audio.shape[0]))
        chans.append(audio)

    assert sample_rate_hz is not None
    assert min_len is not None
    stacked = np.stack([ch[:min_len] for ch in chans], axis=1)
    return stacked, sample_rate_hz, [p.name for p in wav_paths]


def _recording_dir_from_path(path: Path) -> Path | None:
    if path.is_dir() and (path / "raw").is_dir():
        return path
    if path.is_dir() and any(path.glob("*.wav")):
        return path.parent if path.name == "raw" else path
    return None


def _discover_recordings(root: Path) -> tuple[Path, list[tuple[str, Path]]]:
    if root.is_file() and root.suffix.lower() == ".zip":
        raise ValueError("zip inputs must be extracted before discovery")

    if (root / "collection.json").exists():
        recordings_root = root / "recordings"
        items = []
        for recording_dir in sorted(p for p in recordings_root.iterdir() if p.is_dir()):
            if (recording_dir / "raw").is_dir():
                items.append((recording_dir.name, recording_dir))
        if not items:
            raise FileNotFoundError(f"no recordings found in {recordings_root}")
        return root, items

    recording_dir = _recording_dir_from_path(root)
    if recording_dir is not None:
        return recording_dir, [(recording_dir.name, recording_dir)]

    raise FileNotFoundError(
        "input path must be a Data Collection export root, a specific recording directory, a raw WAV directory, or a zip export"
    )


def _extract_if_needed(input_path: Path) -> tuple[Path, tempfile.TemporaryDirectory[str] | None]:
    if input_path.is_file() and input_path.suffix.lower() == ".zip":
        temp_dir = tempfile.TemporaryDirectory(prefix="data-collection-benchmark-")
        with zipfile.ZipFile(input_path) as zf:
            zf.extractall(temp_dir.name)
        return Path(temp_dir.name), temp_dir
    return input_path, None


def _plot_waveform_compare(raw: np.ndarray, proc: np.ndarray, fs: int, out_path: Path, title: str) -> None:
    n = min(raw.size, proc.size)
    t = np.arange(n, dtype=np.float64) / max(int(fs), 1)
    fig, axes = plt.subplots(2, 1, figsize=(14, 5), sharex=True)
    axes[0].plot(t, raw[:n], linewidth=0.6, color="#666666")
    axes[0].set_ylabel("raw")
    axes[0].grid(True, alpha=0.2)
    axes[1].plot(t, proc[:n], linewidth=0.6, color="#005f73")
    axes[1].set_ylabel("processed")
    axes[1].set_xlabel("time (s)")
    axes[1].grid(True, alpha=0.2)
    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_spectrogram_compare(raw: np.ndarray, proc: np.ndarray, fs: int, out_path: Path, title: str) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    axes[0].specgram(raw, NFFT=512, Fs=int(fs), noverlap=256, cmap="magma")
    axes[0].set_ylabel("raw Hz")
    axes[1].specgram(proc, NFFT=512, Fs=int(fs), noverlap=256, cmap="magma")
    axes[1].set_ylabel("proc Hz")
    axes[1].set_xlabel("time (s)")
    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
def _load_ground_truth_tracks(recording_dir: Path) -> list[dict]:
    metadata_path = recording_dir / "metadata.json"
    if not metadata_path.exists():
        return []
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    scene_config_path = metadata.get("sceneConfigPath")
    if not scene_config_path:
        return []
    scene_path = Path(str(scene_config_path))
    scenario_metadata = (
        scene_path.parent.parent.parent
        / "assets"
        / scene_path.parent.name
        / scene_path.stem
        / "scenario_metadata.json"
    )
    if not scenario_metadata.exists():
        return []
    try:
        payload = json.loads(scenario_metadata.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    tracks: list[dict] = []
    for item in payload.get("assets", {}).get("speech", []):
        window = item.get("active_window_sec", [])
        if len(window) != 2:
            continue
        tracks.append(
            {
                "speaker_id": int(item.get("speaker_id", len(tracks))),
                "start_sec": float(window[0]),
                "end_sec": float(window[1]),
                "angle_deg": float(item.get("angle_deg", 0.0)),
            }
        )
    return tracks


def _plot_speaker_timeline(summary: dict, out_path: Path, title: str, ground_truth_tracks: list[dict] | None = None) -> None:
    trace = list(summary.get("speaker_map_trace", []))
    ground_truth_tracks = ground_truth_tracks or []
    if not trace and not ground_truth_tracks:
        return
    frame_step_s = float(summary.get("fast_frame_ms", 10.0)) / 1000.0
    fig, ax = plt.subplots(figsize=(14, 4))
    xs: list[float] = []
    ys: list[float] = []
    ss: list[float] = []
    cs: list[float] = []
    for row in trace:
        x = float(row.get("frame_index", 0)) * frame_step_s
        for speaker in row.get("speakers", []):
            xs.append(x)
            ys.append(float(speaker.get("direction_degrees", 0.0)))
            ss.append(20.0 + 80.0 * float(max(0.0, speaker.get("gain_weight", 0.0))))
            cs.append(float(max(0.0, speaker.get("confidence", 0.0))))
    gt_handles = []
    gt_labels = []
    gt_colors = ["#c1121f", "#003049", "#669bbc", "#fb8500", "#2a9d8f", "#6a4c93"]
    for idx, item in enumerate(ground_truth_tracks):
        color = gt_colors[idx % len(gt_colors)]
        line = ax.hlines(
            y=float(item["angle_deg"]),
            xmin=float(item["start_sec"]),
            xmax=float(item["end_sec"]),
            colors=color,
            linewidth=3.0,
            linestyles="-",
            zorder=4,
        )
        gt_handles.append(line)
        gt_labels.append(f"GT speaker {int(item['speaker_id'])}: {float(item['angle_deg']):.1f} deg")

    if not xs and not gt_handles:
        plt.close(fig)
        return
    scatter = None
    if xs:
        scatter = ax.scatter(xs, ys, s=ss, c=cs, cmap="viridis", alpha=0.65, edgecolors="none", zorder=2)
    ax.set_title(title)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("direction (deg)")
    ax.set_ylim(-5.0, 365.0)
    ax.grid(True, alpha=0.2)
    if scatter is not None:
        fig.colorbar(scatter, ax=ax, label="confidence")
    if gt_handles:
        pred_handle = None
        if scatter is not None:
            pred_handle = ax.scatter([], [], s=40, c="#6c757d", alpha=0.8, label="Predicted speakers")
        legend_handles = list(gt_handles)
        legend_labels = list(gt_labels)
        if pred_handle is not None:
            legend_handles.append(pred_handle)
            legend_labels.append("Predicted speakers")
        ax.legend(legend_handles, legend_labels, loc="upper right")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _trace_metrics(summary: dict) -> dict[str, float]:
    trace = list(summary.get("speaker_map_trace", []))
    if not trace:
        return {
            "tracked_speakers_avg": 0.0,
            "active_speakers_avg": 0.0,
            "dominant_confidence_avg": 0.0,
            "dominant_direction_step_p95_deg": 0.0,
        }
    tracked_counts: list[float] = []
    active_counts: list[float] = []
    dominant_conf: list[float] = []
    dominant_dirs: list[float] = []
    for row in trace:
        speakers = list(row.get("speakers", []))
        tracked_counts.append(float(len(speakers)))
        active_counts.append(float(sum(1 for speaker in speakers if bool(speaker.get("active", False)))))
        if speakers:
            best = max(speakers, key=lambda speaker: (float(speaker.get("gain_weight", 0.0)), float(speaker.get("confidence", 0.0))))
            dominant_conf.append(float(best.get("confidence", 0.0)))
            dominant_dirs.append(float(best.get("direction_degrees", 0.0)))
    if len(dominant_dirs) >= 2:
        steps = [
            abs((dominant_dirs[idx] - dominant_dirs[idx - 1] + 180.0) % 360.0 - 180.0)
            for idx in range(1, len(dominant_dirs))
        ]
        direction_step = float(np.percentile(np.asarray(steps, dtype=np.float64), 95))
    else:
        direction_step = 0.0
    return {
        "tracked_speakers_avg": float(np.mean(tracked_counts)) if tracked_counts else 0.0,
        "active_speakers_avg": float(np.mean(active_counts)) if active_counts else 0.0,
        "dominant_confidence_avg": float(np.mean(dominant_conf)) if dominant_conf else 0.0,
        "dominant_direction_step_p95_deg": direction_step,
    }


def _plot_summary_bars(summary_rows: list[dict], out_dir: Path) -> None:
    if not summary_rows:
        return
    methods = [str(row["method"]) for row in summary_rows]
    metric_specs = [
        ("fast_rtf_mean", "Fast RTF mean"),
        ("clip_rate_processed_mean", "Processed clip rate mean"),
        ("dominant_confidence_avg_mean", "Dominant confidence mean"),
        ("speaker_count_final_mean", "Final speaker count mean"),
    ]
    fig, axes = plt.subplots(len(metric_specs), 1, figsize=(11, 10), sharex=True)
    for ax, (key, label) in zip(np.atleast_1d(axes), metric_specs):
        values = [float(row.get(key, 0.0)) for row in summary_rows]
        ax.bar(methods, values, color="#0a9396")
        ax.set_ylabel(label)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_title("Data collection benchmark summary")
    axes[-1].tick_params(axis="x", rotation=20)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "summary_bars.png", dpi=160)
    plt.close(fig)


def _run_recording_method_job(
    *,
    recording_id: str,
    recording_dir: Path,
    method: str,
    out_dir: Path,
    mic_geometry_xyz: np.ndarray,
    separation_mode: str,
    fast_frame_ms: int,
    localization_window_ms: int,
    slow_chunk_ms: int,
    slow_chunk_hop_ms: int,
    fast_path_reference_mode: str,
    output_normalization_enabled: bool,
    output_allow_amplification: bool,
    localization_backend: str,
    tracking_mode: str,
    control_mode: str,
    direction_long_memory_enabled: bool,
    identity_backend: str,
    identity_speaker_embedding_model: str,
    max_speakers_hint: int,
) -> dict:
    raw_dir = recording_dir / "raw" if (recording_dir / "raw").is_dir() else recording_dir
    mic_audio, sample_rate_hz, channel_filenames = _load_multichannel_wavs(raw_dir)
    raw_mix = np.mean(np.asarray(mic_audio, dtype=np.float64), axis=1)

    method_slug = _slug(method)
    run_dir = out_dir / "runs" / _slug(recording_id) / method_slug
    summary = run_recording_pipeline(
        mic_audio=mic_audio,
        sample_rate_hz=sample_rate_hz,
        mic_geometry_xyz=mic_geometry_xyz,
        out_dir=run_dir,
        input_recording_path=recording_dir,
        separation_mode=str(separation_mode),
        fast_frame_ms=int(fast_frame_ms),
        slow_chunk_ms=int(slow_chunk_ms),
        slow_chunk_hop_ms=int(slow_chunk_hop_ms),
        beamforming_mode=str(method),
        fast_path_reference_mode=str(fast_path_reference_mode),
        output_normalization_enabled=bool(output_normalization_enabled),
        output_allow_amplification=bool(output_allow_amplification),
        capture_trace=True,
        localization_backend=str(localization_backend),
        localization_window_ms=int(localization_window_ms),
        tracking_mode=str(tracking_mode),
        control_mode=str(control_mode),
        direction_long_memory_enabled=bool(direction_long_memory_enabled),
        identity_backend=str(identity_backend),
        identity_speaker_embedding_model=str(identity_speaker_embedding_model),
        max_speakers_hint=int(max_speakers_hint),
    )
    proc, proc_sr = sf.read(run_dir / "enhanced_fast_path.wav", dtype="float32", always_2d=False)
    proc = np.asarray(proc, dtype=np.float32).reshape(-1)
    fs = int(proc_sr if proc_sr > 0 else sample_rate_hz)
    n = min(raw_mix.size, proc.size)
    trace_metrics = _trace_metrics(summary)
    ground_truth_tracks = _load_ground_truth_tracks(recording_dir)
    row = {
        "recording": recording_id,
        "method": method,
        "sample_rate_hz": fs,
        "duration_s": float(n / max(fs, 1)),
        "channel_count": int(mic_audio.shape[1]),
        "raw_channel_filenames": ",".join(channel_filenames),
        "separation_mode": str(separation_mode),
        "fast_rtf": float(summary["fast_rtf"]),
        "slow_rtf": float(summary["slow_rtf"]),
        "fast_avg_ms": float(summary["fast_avg_ms"]),
        "slow_avg_ms": float(summary["slow_avg_ms"]),
        "clip_rate_raw": _clip_rate(raw_mix[:n]),
        "clip_rate_processed": _clip_rate(proc[:n]),
        "peak_abs_raw": float(np.max(np.abs(raw_mix[:n]))) if n else 0.0,
        "peak_abs_processed": float(np.max(np.abs(proc[:n]))) if n else 0.0,
        "rms_gain_db": _rms_db_ratio(raw_mix[:n], proc[:n]),
        "high_band_noise_ratio_raw": _high_band_noise_ratio(raw_mix[:n], fs),
        "high_band_noise_ratio_processed": _high_band_noise_ratio(proc[:n], fs),
        "frame_gain_delta_p95": _frame_gain_delta_p95(raw_mix[:n], proc[:n]),
        "speaker_count_final": int(len(summary.get("speaker_map_final", []))),
        "active_speaker_count_final": int(sum(1 for speaker in summary.get("speaker_map_final", []) if bool(speaker.get("active", False)))),
        "max_confidence_final": float(max((float(speaker.get("confidence", 0.0)) for speaker in summary.get("speaker_map_final", [])), default=0.0)),
        "mean_gain_weight_final": float(np.mean([float(speaker.get("gain_weight", 0.0)) for speaker in summary.get("speaker_map_final", [])])) if summary.get("speaker_map_final") else 0.0,
        **trace_metrics,
    }

    label = f"{recording_id} | {method} | {separation_mode}"
    viz_dir = run_dir / "visualizations"
    _plot_waveform_compare(raw_mix[:n], proc[:n], fs, viz_dir / "waveforms.png", label)
    _plot_spectrogram_compare(raw_mix[:n], proc[:n], fs, viz_dir / "spectrograms.png", label)
    _plot_speaker_timeline(summary, viz_dir / "speaker_directions.png", label, ground_truth_tracks=ground_truth_tracks)
    return row


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run realtime beamforming over Data Collection raw-channel recordings.")
    parser.add_argument("--input-path", required=True, help="Data Collection export root/zip, recording dir, or raw WAV dir")
    parser.add_argument("--out-dir", required=True, help="Directory for benchmark outputs")
    parser.add_argument("--methods", nargs="+", choices=["mvdr_fd", "gsc_fd", "delay_sum"], default=["mvdr_fd"])
    parser.add_argument(
        "--separation-mode",
        choices=["single_dominant_no_separator", "mock", "auto"],
        default="single_dominant_no_separator",
        help="Realtime path backend to use before beamforming",
    )
    parser.add_argument("--mic-array-profile", choices=["respeaker_v3_0457", "respeaker_cross_0640", "respeaker_xvf3800_0650"], default="respeaker_v3_0457")
    parser.add_argument("--fast-frame-ms", type=int, default=50)
    parser.add_argument("--localization-window-ms", type=int, default=200)
    parser.add_argument("--slow-chunk-ms", type=int, default=2000)
    parser.add_argument("--slow-chunk-hop-ms", type=int, default=1000)
    parser.add_argument("--max-speakers-hint", type=int, default=4)
    parser.add_argument("--localization-backend", choices=["tiny_dp_ipd", "weighted_srp_dp", "srp_phat_legacy", "music_1src", "gcc_tdoa_1src", "snr_weighted_srp_phat", "peak_confidence_srp_phat", "particle_filter_tracker", "neural_mask_gcc_phat", "ipd_regressor"], default="weighted_srp_dp")
    parser.add_argument("--tracking-mode", choices=["legacy", "multi_peak_v2"], default="multi_peak_v2")
    parser.add_argument("--control-mode", choices=["spatial_peak_mode", "speaker_tracking_mode"], default="spatial_peak_mode")
    parser.add_argument("--fast-path-reference-mode", choices=["speaker_map", "srp_peak"], default="speaker_map")
    parser.add_argument("--disable-output-normalization", action="store_true")
    parser.add_argument("--allow-output-amplification", action="store_true")
    parser.add_argument("--disable-direction-long-memory", action="store_true")
    parser.add_argument("--identity-backend", choices=["mfcc_legacy", "speaker_embed_session"], default="mfcc_legacy")
    parser.add_argument("--identity-speaker-embedding-model", choices=["ecapa_voxceleb", "wavlm_base_sv", "wavlm_base_plus_sv"], default="wavlm_base_plus_sv")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 1) - 1))
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    input_path = Path(args.input_path).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    extracted_root, temp_dir = _extract_if_needed(input_path)
    try:
        resolved_root, recordings = _discover_recordings(extracted_root.resolve())
        mic_geometry_xyz = mic_positions_xyz(str(args.mic_array_profile)).T

        print(
            json.dumps(
                {
                    "status": "starting",
                    "input_path": str(input_path),
                    "resolved_input_root": str(resolved_root),
                    "out_dir": str(out_dir),
                    "n_recordings": len(recordings),
                    "methods": list(args.methods),
                    "separation_mode": str(args.separation_mode),
                    "workers": int(args.workers),
                },
                indent=2,
            ),
            flush=True,
        )

        scene_rows: list[dict] = []
        summary_by_method: dict[str, list[dict]] = defaultdict(list)
        jobs: list[tuple[int, int, str, Path, str]] = []
        for recording_idx, (recording_id, recording_dir) in enumerate(recordings, start=1):
            for method_idx, method in enumerate(args.methods, start=1):
                jobs.append((recording_idx, method_idx, recording_id, recording_dir, method))

        def _submit_job(job: tuple[int, int, str, Path, str]) -> dict:
            recording_idx, method_idx, recording_id, recording_dir, method = job
            print(
                f"[{recording_idx}/{len(recordings)}] recording={recording_id} "
                f"[{method_idx}/{len(args.methods)}] method={method}",
                flush=True,
            )
            return _run_recording_method_job(
                recording_id=recording_id,
                recording_dir=recording_dir,
                method=method,
                out_dir=out_dir,
                mic_geometry_xyz=mic_geometry_xyz,
                separation_mode=str(args.separation_mode),
                fast_frame_ms=int(args.fast_frame_ms),
                localization_window_ms=int(args.localization_window_ms),
                slow_chunk_ms=int(args.slow_chunk_ms),
                slow_chunk_hop_ms=int(args.slow_chunk_hop_ms),
                fast_path_reference_mode=str(args.fast_path_reference_mode),
                output_normalization_enabled=not bool(args.disable_output_normalization),
                output_allow_amplification=bool(args.allow_output_amplification),
                localization_backend=str(args.localization_backend),
                tracking_mode=str(args.tracking_mode),
                control_mode=str(args.control_mode),
                direction_long_memory_enabled=not bool(args.disable_direction_long_memory),
                identity_backend=str(args.identity_backend),
                identity_speaker_embedding_model=str(args.identity_speaker_embedding_model),
                max_speakers_hint=int(args.max_speakers_hint),
            )

        if int(args.workers) <= 1:
            for job in jobs:
                row = _submit_job(job)
                scene_rows.append(row)
                summary_by_method[str(row["method"])].append(row)
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
                futures = [pool.submit(_submit_job, job) for job in jobs]
                for fut in concurrent.futures.as_completed(futures):
                    row = fut.result()
                    scene_rows.append(row)
                    summary_by_method[str(row["method"])].append(row)

        scene_rows.sort(key=lambda row: (str(row["recording"]), str(row["method"])))

        _write_csv(out_dir / "scene_metrics.csv", scene_rows)

        summary_rows: list[dict] = []
        for method, rows in summary_by_method.items():
            summary_rows.append(
                {
                    "method": method,
                    "n_recordings": len(rows),
                    "fast_rtf_mean": float(np.mean([float(row["fast_rtf"]) for row in rows])),
                    "slow_rtf_mean": float(np.mean([float(row["slow_rtf"]) for row in rows])),
                    "clip_rate_processed_mean": float(np.mean([float(row["clip_rate_processed"]) for row in rows])),
                    "rms_gain_db_mean": float(np.mean([float(row["rms_gain_db"]) for row in rows])),
                    "high_band_noise_ratio_processed_mean": float(np.mean([float(row["high_band_noise_ratio_processed"]) for row in rows])),
                    "speaker_count_final_mean": float(np.mean([float(row["speaker_count_final"]) for row in rows])),
                    "active_speaker_count_final_mean": float(np.mean([float(row["active_speaker_count_final"]) for row in rows])),
                    "dominant_confidence_avg_mean": float(np.mean([float(row["dominant_confidence_avg"]) for row in rows])),
                    "dominant_direction_step_p95_deg_mean": float(np.mean([float(row["dominant_direction_step_p95_deg"]) for row in rows])),
                }
            )
        summary_rows.sort(key=lambda row: row["method"])
        _write_csv(out_dir / "summary_by_method.csv", summary_rows)
        _plot_summary_bars(summary_rows, out_dir / "visualizations")

        report = {
            "input_path": str(input_path),
            "resolved_input_root": str(resolved_root),
            "recordings": [recording_id for recording_id, _recording_dir in recordings],
            "methods": list(args.methods),
            "separation_mode": str(args.separation_mode),
            "fast_frame_ms": int(args.fast_frame_ms),
            "localization_window_ms": int(args.localization_window_ms),
            "slow_chunk_ms": int(args.slow_chunk_ms),
            "slow_chunk_hop_ms": int(args.slow_chunk_hop_ms),
            "mic_array_profile": str(args.mic_array_profile),
            "note": "Real Data Collection recordings do not have clean-reference speech, so this benchmark reports runtime and blind/relative diagnostics instead of SI-SDR/STOI/SII.",
        }
        with (out_dir / "report.json").open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        print(json.dumps({"out_dir": str(out_dir), "n_recordings": len(recordings), "methods": list(args.methods)}, indent=2))
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()


if __name__ == "__main__":
    main()

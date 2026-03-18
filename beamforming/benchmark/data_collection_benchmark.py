from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import math
import os
import re
import sys
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

try:
    import numpy as np
except ModuleNotFoundError as exc:  # pragma: no cover - environment-specific dependency guard
    missing = exc.name or "numpy"
    raise SystemExit(
        f"Missing dependency '{missing}'. Install the beamforming benchmark dependencies first, "
        f"for example: `python3 -m pip install -r beamforming/requirements.txt`."
    ) from exc

try:
    from scipy.optimize import linear_sum_assignment
    from scipy.signal import welch
except ModuleNotFoundError as exc:  # pragma: no cover - environment-specific dependency guard
    missing = exc.name or "scipy"
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

from mic_array_forwarder.models import SessionStartRequest
from realtime_pipeline.contracts import SRPPeakSnapshot
from realtime_pipeline.fast_path import _steering_vector_f_domain
from realtime_pipeline.session_runtime import run_offline_session_pipeline
from realtime_pipeline.tracking_modes import TRACKING_MODE_CHOICES, validate_tracking_mode
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


def _normalize_angle_deg(v: float) -> float:
    return float(v % 360.0)


def _raw_math_angle_to_display_angle_deg(angle_deg: float, *, mic_array_profile: str) -> float:
    angle = _normalize_angle_deg(float(angle_deg))
    if str(mic_array_profile) == "respeaker_xvf3800_0650":
        # Backend/localization math uses +x as 0 deg; the UI uses the cable edge
        # between mics 1 and 2 as 0 deg, which is +y for the XVF3800 geometry.
        return _normalize_angle_deg(90.0 - angle)
    return angle


def _backend_prediction_to_display_angle_deg(angle_deg: float, *, mic_array_profile: str) -> float:
    angle = _normalize_angle_deg(float(angle_deg))
    if str(mic_array_profile) == "respeaker_xvf3800_0650":
        # Backend localization emits incoming-wave direction. Manual annotations
        # from Data Collection are source bearing, so flip by 180 deg before
        # rotating into the UI convention.
        return _normalize_angle_deg(270.0 - angle)
    return angle


def _display_angle_to_backend_prediction_deg(angle_deg: float, *, mic_array_profile: str) -> float:
    angle = _normalize_angle_deg(float(angle_deg))
    if str(mic_array_profile) == "respeaker_xvf3800_0650":
        return _normalize_angle_deg(270.0 - angle)
    return angle


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


def _plot_psd_compare(raw: np.ndarray, proc: np.ndarray, fs: int, out_path: Path, title: str) -> None:
    n = min(raw.size, proc.size)
    if n <= 1:
        return
    seg = min(4096, max(256, n))
    freqs_raw, psd_raw = welch(np.asarray(raw[:n], dtype=np.float64), fs=int(fs), nperseg=seg)
    freqs_proc, psd_proc = welch(np.asarray(proc[:n], dtype=np.float64), fs=int(fs), nperseg=seg)
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.semilogx(freqs_raw[1:], 10.0 * np.log10(np.maximum(psd_raw[1:], 1e-12)), color="#666666", linewidth=1.3, label="raw")
    ax.semilogx(freqs_proc[1:], 10.0 * np.log10(np.maximum(psd_proc[1:], 1e-12)), color="#005f73", linewidth=1.3, label="processed")
    ax.set_title(f"{title} PSD")
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("PSD (dB/Hz)")
    ax.grid(True, alpha=0.25, which="both")
    ax.legend(loc="best")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_psd_stages(stage_signals: list[tuple[str, np.ndarray]], fs: int, out_path: Path, title: str) -> None:
    valid: list[tuple[str, np.ndarray]] = []
    for name, signal in stage_signals:
        arr = np.asarray(signal, dtype=np.float64).reshape(-1)
        if arr.size > 1:
            valid.append((str(name), arr))
    if not valid:
        return
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ["#666666", "#005f73", "#0a9396", "#ee9b00", "#bb3e03"]
    for idx, (name, signal) in enumerate(valid):
        seg = min(4096, max(256, signal.size))
        freqs, psd = welch(signal, fs=int(fs), nperseg=seg)
        ax.semilogx(
            freqs[1:],
            10.0 * np.log10(np.maximum(psd[1:], 1e-12)),
            color=colors[idx % len(colors)],
            linewidth=1.3,
            label=name,
        )
    ax.set_title(f"{title} Stage PSD")
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("PSD (dB/Hz)")
    ax.grid(True, alpha=0.25, which="both")
    ax.legend(loc="best")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_single_psd(signal: np.ndarray, fs: int, out_path: Path, title: str, *, color: str = "#005f73") -> None:
    arr = np.asarray(signal, dtype=np.float64).reshape(-1)
    if arr.size <= 1:
        return
    seg = min(4096, max(256, arr.size))
    freqs, psd = welch(arr, fs=int(fs), nperseg=seg)
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.semilogx(freqs[1:], 10.0 * np.log10(np.maximum(psd[1:], 1e-12)), color=color, linewidth=1.3)
    ax.set_title(title)
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("PSD (dB/Hz)")
    ax.grid(True, alpha=0.25, which="both")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _snapshot_clip_bounds(center_sample: int, total_samples: int, fs: int, clip_duration_s: float = 2.0) -> tuple[int, int]:
    half = max(1, int(round(0.5 * clip_duration_s * max(fs, 1))))
    start = max(0, int(center_sample) - half)
    end = min(int(total_samples), int(center_sample) + half)
    return start, max(start + 1, end)


def _plot_beam_pattern_from_snapshot(
    *,
    snapshot: dict,
    mic_geometry_xyz: np.ndarray,
    sample_rate_hz: int,
    out_path: Path,
    title: str,
) -> None:
    weights_real = np.asarray(snapshot.get("weights_real", []), dtype=np.float64)
    weights_imag = np.asarray(snapshot.get("weights_imag", []), dtype=np.float64)
    if weights_real.size == 0 or weights_imag.size == 0:
        return
    weights = weights_real + 1j * weights_imag
    if weights.ndim != 2:
        return
    angles = np.linspace(0.0, 360.0, 361, endpoint=True)
    freqs = np.fft.rfftfreq((weights.shape[0] - 1) * 2, d=1.0 / float(sample_rate_hz))
    freq_mask = np.asarray((freqs >= 300.0) & (freqs <= 3000.0), dtype=bool)
    if not np.any(freq_mask):
        freq_mask = np.ones_like(freqs, dtype=bool)
    response = np.zeros(angles.shape[0], dtype=np.float64)
    for idx, angle_deg in enumerate(angles):
        steering = _steering_vector_f_domain(
            doa_deg=float(angle_deg),
            n_fft=int((weights.shape[0] - 1) * 2),
            fs=int(sample_rate_hz),
            mic_geometry_xyz=np.asarray(mic_geometry_xyz, dtype=np.float64),
            sound_speed_m_s=343.0,
        )
        gain = np.abs(np.einsum("fm,fm->f", np.conj(weights), steering))
        response[idx] = float(np.mean(gain[freq_mask]))
    max_resp = float(np.max(response))
    if max_resp > 0.0:
        response = response / max_resp
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="polar")
    ax.plot(np.deg2rad(angles), response, color="#0a9396", linewidth=1.5)
    target = snapshot.get("target_doa_deg")
    if target is not None:
        ax.scatter([np.deg2rad(float(target))], [1.0], color="#ae2012", s=40)
    ax.set_title(title)
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_rlim(0.0, 1.05)
    ax.grid(alpha=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _emit_beamformer_snapshot_artifacts(
    *,
    summary: dict,
    proc: np.ndarray,
    fs: int,
    run_dir: Path,
    mic_geometry_xyz: np.ndarray,
) -> None:
    snapshots = list(summary.get("beamformer_snapshot_trace", []))
    if not snapshots:
        return
    artifact_dir = run_dir / "beamformer_snapshots"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    total_samples = int(proc.size)
    fast_frame_ms = float(summary.get("fast_frame_ms", 10.0))
    frame_samples = max(1, int(round((fast_frame_ms / 1000.0) * max(fs, 1))))
    for idx, snapshot in enumerate(snapshots[:3], start=1):
        center_sample = int(max(0, int(snapshot.get("frame_index", 1)) - 1) * frame_samples)
        start, end = _snapshot_clip_bounds(center_sample, total_samples, fs)
        clip_path = artifact_dir / f"snapshot_{idx:02d}.wav"
        sf.write(clip_path, np.asarray(proc[start:end], dtype=np.float32), fs)
        target = snapshot.get("target_doa_deg")
        title = (
            f"{summary.get('beamforming_mode', 'beamformer')} snapshot {idx} | "
            f"frame {int(snapshot.get('frame_index', -1))} | "
            f"target {float(target):.1f} deg"
            if target is not None
            else f"{summary.get('beamforming_mode', 'beamformer')} snapshot {idx}"
        )
        _plot_beam_pattern_from_snapshot(
            snapshot=snapshot,
            mic_geometry_xyz=np.asarray(mic_geometry_xyz, dtype=np.float64),
            sample_rate_hz=int(fs),
            out_path=artifact_dir / f"snapshot_{idx:02d}_polar.png",
            title=title,
        )
def _load_ground_truth_tracks(recording_dir: Path, *, duration_s: float, mic_array_profile: str) -> list[dict]:
    metadata_path = recording_dir / "metadata.json"
    if not metadata_path.exists():
        return []
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    manual_speakers = list(metadata.get("speakers", []))
    if manual_speakers:
        tracks: list[dict] = []
        for idx, item in enumerate(manual_speakers):
            speaker_name = str(item.get("speakerName", f"speaker-{idx + 1}"))
            speaking_periods = list(item.get("speakingPeriods", []))
            if speaking_periods:
                for period_idx, period in enumerate(speaking_periods):
                    try:
                        angle_deg = float(period.get("directionDeg", 0.0))
                        start_sec = float(period.get("startSec", 0.0))
                        end_sec = float(period.get("endSec", duration_s))
                    except (TypeError, ValueError):
                        continue
                    start_sec = max(0.0, start_sec)
                    end_sec = max(start_sec, end_sec)
                    if start_sec == 0.0 and end_sec == 0.0:
                        end_sec = float(max(duration_s, 0.0))
                    tracks.append(
                        {
                            "speaker_id": int(idx + 1),
                            "speaker_name": speaker_name,
                            "start_sec": float(start_sec),
                            "end_sec": float(end_sec),
                            "angle_deg": _normalize_angle_deg(angle_deg),
                            "source": "manual_metadata",
                            "period_index": int(period_idx),
                        }
                    )
                continue
            try:
                angle_deg = float(item.get("directionDeg", 0.0))
            except (TypeError, ValueError):
                continue
            tracks.append(
                {
                    "speaker_id": int(idx + 1),
                    "speaker_name": speaker_name,
                    "start_sec": 0.0,
                    "end_sec": float(max(duration_s, 0.0)),
                    "angle_deg": _normalize_angle_deg(angle_deg),
                    "source": "manual_metadata",
                }
            )
        return tracks
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
                "speaker_name": str(item.get("speaker_name", f"speaker-{len(tracks) + 1}")),
                "start_sec": float(window[0]),
                "end_sec": float(window[1]),
                "angle_deg": _raw_math_angle_to_display_angle_deg(float(item.get("angle_deg", 0.0)), mic_array_profile=mic_array_profile),
                "source": "simulation",
            }
        )
    return tracks


def _angular_error_deg(a_deg: float, b_deg: float) -> float:
    return abs((float(a_deg) - float(b_deg) + 180.0) % 360.0 - 180.0)


def _active_ground_truth_tracks_at_time(ground_truth_tracks: list[dict], timestamp_s: float) -> list[dict]:
    active: list[dict] = []
    t = float(timestamp_s)
    for item in ground_truth_tracks:
        start_sec = float(item.get("start_sec", 0.0))
        end_sec = float(item.get("end_sec", 0.0))
        if start_sec <= t < end_sec:
            active.append(item)
    return active


def _ground_truth_transition_times(ground_truth_tracks: list[dict], *, duration_s: float) -> list[float]:
    times: set[float] = set()
    max_duration = max(0.0, float(duration_s))
    for item in ground_truth_tracks:
        start_sec = max(0.0, float(item.get("start_sec", 0.0)))
        end_sec = min(max_duration, float(item.get("end_sec", max_duration)))
        if 0.0 < start_sec < max_duration:
            times.add(float(start_sec))
        if 0.0 < end_sec < max_duration:
            times.add(float(end_sec))
    return sorted(times)


def _is_transition_ignored(
    *,
    timestamp_s: float,
    transition_times: list[float],
    transition_ignore_sec: float,
) -> bool:
    ignore_sec = max(0.0, float(transition_ignore_sec))
    if ignore_sec <= 0.0 or not transition_times:
        return False
    return any(abs(float(timestamp_s) - float(change_s)) <= ignore_sec for change_s in transition_times)


def _filter_tracks_by_speaker_name(ground_truth_tracks: list[dict], speaker_name: str | None) -> list[dict]:
    if not speaker_name:
        return []
    expected = str(speaker_name).strip().lower()
    if not expected:
        return []
    return [item for item in ground_truth_tracks if str(item.get("speaker_name", "")).strip().lower() == expected]


def _ground_truth_srp_override_provider(
    ground_truth_tracks: list[dict],
    *,
    mic_array_profile: str,
):
    def _provider(frame_index: int, timestamp_ms: float) -> SRPPeakSnapshot | None:
        timestamp_s = float(timestamp_ms) / 1000.0
        active_tracks = _active_ground_truth_tracks_at_time(ground_truth_tracks, timestamp_s)
        if not active_tracks:
            return None
        backend_angles = tuple(
            _display_angle_to_backend_prediction_deg(float(item["angle_deg"]), mic_array_profile=mic_array_profile)
            for item in active_tracks
        )
        scores = tuple(1.0 for _ in backend_angles)
        return SRPPeakSnapshot(
            timestamp_ms=float(timestamp_ms),
            peaks_deg=backend_angles,
            peak_scores=scores,
            raw_peaks_deg=backend_angles,
            raw_peak_scores=scores,
            debug={
                "source": "ground_truth_override",
                "frame_index": int(frame_index),
                "active_speaker_names": [str(item.get("speaker_name", "")) for item in active_tracks],
            },
        )

    return _provider


def _match_angle_sets(gt_angles: list[float], pred_angles: list[float]) -> tuple[list[float], int, int, int]:
    if not gt_angles or not pred_angles:
        return [], 0, len(gt_angles), len(pred_angles)
    cost = np.asarray(
        [[_angular_error_deg(gt_angle, pred_angle) for pred_angle in pred_angles] for gt_angle in gt_angles],
        dtype=np.float64,
    )
    rows, cols = linear_sum_assignment(cost)
    errors = [float(cost[int(r), int(c)]) for r, c in zip(rows, cols)]
    matched = len(errors)
    return errors, matched, max(0, len(gt_angles) - matched), max(0, len(pred_angles) - matched)


def _ground_truth_metrics(
    summary: dict,
    ground_truth_tracks: list[dict],
    *,
    mic_array_profile: str,
    transition_ignore_sec: float = 0.0,
) -> dict[str, float | str]:
    if not ground_truth_tracks:
        return {
            "ground_truth_source": "",
            "ground_truth_speaker_count": 0,
            "gt_final_mae_deg": float("nan"),
            "gt_final_acc_10": float("nan"),
            "gt_final_acc_20": float("nan"),
            "gt_final_match_rate": float("nan"),
            "gt_trace_active_mae_deg": float("nan"),
            "gt_trace_active_acc_10": float("nan"),
            "gt_trace_active_acc_20": float("nan"),
            "gt_trace_active_match_rate": float("nan"),
        }

    gt_speaker_keys = {
        (str(item.get("speaker_name", "")), int(item.get("speaker_id", 0)))
        for item in ground_truth_tracks
    }
    gt_source = str(ground_truth_tracks[0].get("source", ""))
    final_speakers = list(summary.get("speaker_map_final", []))
    final_active_angles = [
        _backend_prediction_to_display_angle_deg(float(item.get("direction_degrees", 0.0)), mic_array_profile=mic_array_profile)
        for item in final_speakers
        if bool(item.get("active", False))
    ]
    duration_s = float(summary.get("duration_s", 0.0))
    final_gt_tracks = _active_ground_truth_tracks_at_time(ground_truth_tracks, max(0.0, duration_s - 1e-6))
    final_gt_angles = [float(item["angle_deg"]) for item in final_gt_tracks]
    final_errors, final_matches, final_missed_gt, _final_extra_pred = _match_angle_sets(final_gt_angles, final_active_angles)

    trace_rows = list(summary.get("speaker_map_trace", []))
    transition_times = _ground_truth_transition_times(ground_truth_tracks, duration_s=duration_s)
    active_errors: list[float] = []
    total_active_gt = 0
    total_matched_gt = 0
    for row in trace_rows:
        timestamp_s = float(row.get("timestamp_ms", 0.0)) / 1000.0
        if _is_transition_ignored(
            timestamp_s=timestamp_s,
            transition_times=transition_times,
            transition_ignore_sec=transition_ignore_sec,
        ):
            continue
        active_gt_tracks = _active_ground_truth_tracks_at_time(ground_truth_tracks, timestamp_s)
        active_gt_angles = [float(item["angle_deg"]) for item in active_gt_tracks]
        if not active_gt_angles:
            continue
        active_angles = [
            _backend_prediction_to_display_angle_deg(float(item.get("direction_degrees", 0.0)), mic_array_profile=mic_array_profile)
            for item in row.get("speakers", [])
            if bool(item.get("active", False))
        ]
        frame_errors, matched, _missed_gt, _extra_pred = _match_angle_sets(active_gt_angles, active_angles)
        active_errors.extend(frame_errors)
        total_active_gt += int(len(active_gt_angles))
        total_matched_gt += int(matched)

    def _acc(errors: list[float], threshold: float) -> float:
        if not errors:
            return float("nan")
        return float(np.mean(np.asarray(errors, dtype=np.float64) <= float(threshold)))

    return {
        "ground_truth_source": gt_source,
        "ground_truth_speaker_count": int(len(gt_speaker_keys)),
        "gt_final_mae_deg": float(np.mean(final_errors)) if final_errors else float("nan"),
        "gt_final_acc_10": _acc(final_errors, 10.0),
        "gt_final_acc_20": _acc(final_errors, 20.0),
        "gt_final_match_rate": (float(final_matches) / float(len(final_gt_angles))) if final_gt_angles else float("nan"),
        "gt_trace_active_mae_deg": float(np.mean(active_errors)) if active_errors else float("nan"),
        "gt_trace_active_acc_10": _acc(active_errors, 10.0),
        "gt_trace_active_acc_20": _acc(active_errors, 20.0),
        "gt_trace_active_match_rate": (float(total_matched_gt) / float(total_active_gt)) if total_active_gt > 0 else float("nan"),
    }


def _suppression_metrics(summary: dict, suppressed_user_tracks: list[dict], *, transition_ignore_sec: float = 0.0) -> dict[str, float | str]:
    if not suppressed_user_tracks:
        return {
            "suppressed_user_name": "",
            "suppression_user_recall": float("nan"),
            "suppression_false_positive_rate": float("nan"),
            "suppression_precision": float("nan"),
            "suppression_activation_rate": float("nan"),
        }
    trace = list(summary.get("speaker_map_trace", []))
    duration_s = float(summary.get("duration_s", 0.0))
    transition_times = _ground_truth_transition_times(suppressed_user_tracks, duration_s=duration_s)
    if not trace:
        return {
            "suppressed_user_name": str(suppressed_user_tracks[0].get("speaker_name", "")),
            "suppression_user_recall": float("nan"),
            "suppression_false_positive_rate": float("nan"),
            "suppression_precision": float("nan"),
            "suppression_activation_rate": float("nan"),
        }
    total_user_frames = 0
    total_non_user_frames = 0
    suppressed_on_user = 0
    suppressed_on_non_user = 0
    total_suppressed_frames = 0
    for row in trace:
        timestamp_s = float(row.get("timestamp_ms", 0.0)) / 1000.0
        if _is_transition_ignored(
            timestamp_s=timestamp_s,
            transition_times=transition_times,
            transition_ignore_sec=transition_ignore_sec,
        ):
            continue
        user_active = bool(_active_ground_truth_tracks_at_time(suppressed_user_tracks, timestamp_s))
        suppression_active = bool(row.get("suppression", {}).get("suppression_active", False))
        if suppression_active:
            total_suppressed_frames += 1
        if user_active:
            total_user_frames += 1
            if suppression_active:
                suppressed_on_user += 1
        else:
            total_non_user_frames += 1
            if suppression_active:
                suppressed_on_non_user += 1
    return {
        "suppressed_user_name": str(suppressed_user_tracks[0].get("speaker_name", "")),
        "suppression_user_recall": (float(suppressed_on_user) / float(total_user_frames)) if total_user_frames else float("nan"),
        "suppression_false_positive_rate": (float(suppressed_on_non_user) / float(total_non_user_frames)) if total_non_user_frames else float("nan"),
        "suppression_precision": (float(suppressed_on_user) / float(total_suppressed_frames)) if total_suppressed_frames else float("nan"),
        "suppression_activation_rate": (float(total_suppressed_frames) / float(len(trace))) if trace else float("nan"),
    }


def _plot_speaker_timeline(
    summary: dict,
    out_path: Path,
    title: str,
    ground_truth_tracks: list[dict] | None = None,
    suppressed_user_tracks: list[dict] | None = None,
    *,
    mic_array_profile: str,
) -> None:
    trace = list(summary.get("speaker_map_trace", []))
    ground_truth_tracks = ground_truth_tracks or []
    suppressed_user_tracks = suppressed_user_tracks or []
    frame_step_s = float(summary.get("fast_frame_ms", 10.0)) / 1000.0
    duration_s = float(summary.get("duration_s", 0.0))
    if duration_s <= 0.0 and trace:
        duration_s = max(0.0, float(trace[-1].get("frame_index", 0)) * frame_step_s)
    fig, ax = plt.subplots(figsize=(14, 4))
    raw_xs: list[float] = []
    raw_ys: list[float] = []
    raw_cs: list[float] = []
    centroid_tracks: dict[int, dict[str, list[float]]] = {}
    suppression_xs: list[float] = []
    for row in trace:
        x = float(row.get("frame_index", 0)) * frame_step_s
        if bool(row.get("suppression", {}).get("suppression_active", False)):
            suppression_xs.append(x)
        for angle_deg, score in zip(row.get("raw_peaks_deg", []), row.get("raw_peak_scores", [])):
            raw_xs.append(x)
            raw_ys.append(_backend_prediction_to_display_angle_deg(float(angle_deg), mic_array_profile=mic_array_profile))
            raw_cs.append(float(max(0.0, score)))
        for speaker in row.get("speakers", []):
            if not bool(speaker.get("active", False)):
                continue
            speaker_id = int(speaker.get("speaker_id", 0))
            track = centroid_tracks.setdefault(
                speaker_id,
                {"xs": [], "ys": [], "conf": []},
            )
            track["xs"].append(x)
            track["ys"].append(
                _backend_prediction_to_display_angle_deg(float(speaker.get("direction_degrees", 0.0)), mic_array_profile=mic_array_profile)
            )
            track["conf"].append(float(max(0.0, speaker.get("confidence", 0.0))))
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
        gt_name = str(item.get("speaker_name", f"speaker {int(item['speaker_id'])}"))
        gt_labels.append(f"GT {gt_name}: {float(item['angle_deg']):.1f} deg")
    user_handles = []
    user_labels = []
    for item in suppressed_user_tracks:
        user_line = ax.hlines(
            y=float(item["angle_deg"]),
            xmin=float(item["start_sec"]),
            xmax=float(item["end_sec"]),
            colors="#d00000",
            linewidth=2.2,
            linestyles="--",
            zorder=5,
        )
        user_handles.append(user_line)
        user_labels.append(f"Suppressed user GT: {str(item.get('speaker_name', 'user'))}")

    raw_scatter = None
    if raw_xs:
        raw_scatter = ax.scatter(
            raw_xs,
            raw_ys,
            s=12,
            c=raw_cs,
            cmap="viridis",
            alpha=0.2,
            edgecolors="none",
            zorder=1,
        )

    centroid_handles = []
    centroid_labels = []
    centroid_palette = plt.get_cmap("tab10", max(1, len(centroid_tracks)))
    for color_idx, speaker_id in enumerate(sorted(centroid_tracks)):
        track = centroid_tracks[speaker_id]
        color = centroid_palette(color_idx)
        ax.plot(track["xs"], track["ys"], color=color, linewidth=2.0, alpha=0.9, zorder=3)
        ax.scatter(track["xs"], track["ys"], s=28, color=[color], alpha=0.95, edgecolors="white", linewidths=0.3, zorder=4)
        centroid_handles.append(ax.plot([], [], color=color, linewidth=2.0)[0])
        centroid_labels.append(f"Active speaker centroid {speaker_id}")
    ax.set_title(title)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("direction (deg)")
    ax.set_ylim(-5.0, 365.0)
    ax.set_xlim(0.0, max(duration_s, 1.0))
    ax.grid(True, alpha=0.2)
    if not raw_xs and not centroid_tracks:
        note = "No speaker predictions were found in the trace."
        if trace:
            note = "No speaker was found in these frames."
        ax.text(
            0.5,
            0.5,
            note,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=11,
            color="#6c757d",
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#ced4da", "alpha": 0.9},
            zorder=5,
        )
    if raw_scatter is not None:
        fig.colorbar(raw_scatter, ax=ax, label="raw detection confidence")
    if suppression_xs:
        ymin, ymax = ax.get_ylim()
        ax.vlines(suppression_xs, ymin=ymin, ymax=ymax, colors="#d00000", linewidth=0.8, alpha=0.12, zorder=0)
    legend_handles = list(gt_handles)
    legend_labels = list(gt_labels)
    if user_handles:
        legend_handles.extend(user_handles[:1])
        legend_labels.extend(user_labels[:1])
    if raw_scatter is not None:
        legend_handles.append(ax.scatter([], [], s=18, c="#6c757d", alpha=0.25))
        legend_labels.append("Raw detections")
    if centroid_handles:
        legend_handles.extend(centroid_handles)
        legend_labels.extend(centroid_labels)
    if legend_handles:
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


def _mean_numeric(rows: list[dict], key: str) -> float:
    vals = [float(row[key]) for row in rows if key in row]
    if not vals:
        return float("nan")
    arr = np.asarray(vals, dtype=np.float64)
    if np.all(np.isnan(arr)):
        return float("nan")
    return float(np.nanmean(arr))


def _run_recording_method_job(
    *,
    recording_id: str,
    recording_dir: Path,
    method: str,
    out_dir: Path,
    mic_array_profile: str,
    mic_geometry_xyz: np.ndarray,
    algorithm_mode: str,
    separation_mode: str,
    localization_window_ms: int,
    localization_hop_ms: int,
    localization_overlap: float,
    localization_freq_low_hz: int,
    localization_freq_high_hz: int,
    localization_pair_selection_mode: str,
    localization_vad_enabled: bool,
    capon_peak_min_sharpness: float,
    capon_peak_min_margin: float,
    speaker_history_size: int,
    speaker_activation_min_predictions: int,
    speaker_match_window_deg: float,
    centroid_association_mode: str,
    centroid_association_sigma_deg: float,
    centroid_association_min_score: float,
    own_voice_suppression_mode: str,
    suppressed_user_voice_doa_deg: float | None,
    suppressed_user_match_window_deg: float,
    suppressed_user_null_on_frames: int,
    suppressed_user_null_off_frames: int,
    suppressed_user_gate_attenuation_db: float,
    suppressed_user_target_conflict_deg: float,
    suppressed_user_speaker_name: str | None,
    ground_truth_transition_ignore_sec: float,
    enhancement_tier: str,
    output_enhancer_mode: str,
    postfilter_enabled: bool,
    postfilter_method: str,
    postfilter_noise_ema_alpha: float,
    postfilter_speech_ema_alpha: float,
    postfilter_gain_floor: float,
    postfilter_gain_ema_alpha: float,
    postfilter_dd_alpha: float,
    postfilter_noise_update_speech_scale: float,
    postfilter_gain_max_step_db: float,
    rnnoise_wet_mix: float,
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
    assume_single_speaker: bool,
    use_ground_truth_doa_override: bool,
    mvdr_hop_ms: int | None,
    fd_analysis_window_ms: float,
    robust_target_band_width_deg: float,
    robust_target_band_max_freq_hz: float,
    robust_target_band_condition_limit: float,
    fd_noise_covariance_mode: str,
    target_activity_rnn_update_mode: str,
    target_activity_detector_mode: str,
    target_activity_detector_backend: str,
    target_activity_update_every_n_fast_frames: int,
    fd_cov_ema_alpha: float,
    fd_diag_load: float,
    fd_trace_diagonal_loading_factor: float,
    fd_identity_blend_alpha: float,
    target_activity_low_threshold: float,
    target_activity_high_threshold: float,
    target_activity_enter_frames: int,
    target_activity_exit_frames: int,
    fd_cov_update_scale_target_active: float,
    fd_cov_update_scale_target_inactive: float,
) -> dict:
    raw_dir = recording_dir / "raw" if (recording_dir / "raw").is_dir() else recording_dir
    mic_audio, sample_rate_hz, channel_filenames = _load_multichannel_wavs(raw_dir)
    raw_mix = np.mean(np.asarray(mic_audio, dtype=np.float64), axis=1)

    method_slug = _slug(method)
    run_dir = out_dir / "runs" / _slug(recording_id) / method_slug
    is_mvdr = str(method).strip().lower() in {"mvdr_fd", "lcmv_target_band"}
    if str(fd_noise_covariance_mode) == "oracle_non_target_residual":
        raise ValueError(
            "fd_noise_covariance_mode=oracle_non_target_residual is not supported for real-data benchmarks "
            "because there is no oracle non-target noise provider."
        )
    req = SessionStartRequest(
        input_source="respeaker_live",
        channel_count=int(mic_audio.shape[1]),
        sample_rate_hz=int(sample_rate_hz),
        monitor_source="processed",
        mic_array_profile=str(mic_array_profile),
        fast_path={
            "localization_hop_ms": int(localization_hop_ms),
            "localization_window_ms": int(localization_window_ms),
            "overlap": float(localization_overlap),
            "freq_low_hz": int(localization_freq_low_hz),
            "freq_high_hz": int(localization_freq_high_hz),
            "localization_pair_selection_mode": str(localization_pair_selection_mode),
            "localization_vad_enabled": bool(localization_vad_enabled),
            "capon_peak_min_sharpness": float(capon_peak_min_sharpness),
            "capon_peak_min_margin": float(capon_peak_min_margin),
            "enhancement_tier": str(enhancement_tier),
            "output_enhancer_mode": str(output_enhancer_mode),
            "postfilter_enabled": bool(postfilter_enabled),
            "postfilter_method": str(postfilter_method),
            "postfilter_noise_ema_alpha": float(postfilter_noise_ema_alpha),
            "postfilter_speech_ema_alpha": float(postfilter_speech_ema_alpha),
            "postfilter_gain_floor": float(postfilter_gain_floor),
            "postfilter_gain_ema_alpha": float(postfilter_gain_ema_alpha),
            "postfilter_dd_alpha": float(postfilter_dd_alpha),
            "postfilter_noise_update_speech_scale": float(postfilter_noise_update_speech_scale),
            "postfilter_gain_max_step_db": float(postfilter_gain_max_step_db),
            "rnnoise_wet_mix": float(rnnoise_wet_mix),
            "own_voice_suppression_mode": str(own_voice_suppression_mode),
            "suppressed_user_voice_doa_deg": (
                None if suppressed_user_voice_doa_deg is None else float(suppressed_user_voice_doa_deg)
            ),
            "suppressed_user_match_window_deg": float(suppressed_user_match_window_deg),
            "suppressed_user_null_on_frames": int(suppressed_user_null_on_frames),
            "suppressed_user_null_off_frames": int(suppressed_user_null_off_frames),
            "suppressed_user_gate_attenuation_db": float(suppressed_user_gate_attenuation_db),
            "suppressed_user_target_conflict_deg": float(suppressed_user_target_conflict_deg),
            "assume_single_speaker": bool(assume_single_speaker),
            "localization_backend": str(localization_backend),
            "beamforming_mode": str(method),
            "robust_target_band_width_deg": float(robust_target_band_width_deg),
            "robust_target_band_max_freq_hz": float(robust_target_band_max_freq_hz),
            "robust_target_band_condition_limit": float(robust_target_band_condition_limit),
            "mvdr_hop_ms": (None if mvdr_hop_ms is None else int(mvdr_hop_ms)),
            "fd_analysis_window_ms": float(fd_analysis_window_ms),
            "fd_noise_covariance_mode": str(fd_noise_covariance_mode),
            "target_activity_rnn_update_mode": (str(target_activity_rnn_update_mode) if is_mvdr else None),
            "target_activity_detector_mode": str(target_activity_detector_mode),
            "target_activity_detector_backend": str(target_activity_detector_backend),
            "target_activity_update_every_n_fast_frames": int(target_activity_update_every_n_fast_frames),
            "fd_cov_ema_alpha": float(fd_cov_ema_alpha),
            "fd_diag_load": float(fd_diag_load),
            "fd_trace_diagonal_loading_factor": float(fd_trace_diagonal_loading_factor),
            "fd_identity_blend_alpha": float(fd_identity_blend_alpha),
            "target_activity_low_threshold": float(target_activity_low_threshold),
            "target_activity_high_threshold": float(target_activity_high_threshold),
            "target_activity_enter_frames": int(target_activity_enter_frames),
            "target_activity_exit_frames": int(target_activity_exit_frames),
            "fd_cov_update_scale_target_active": float(fd_cov_update_scale_target_active),
            "fd_cov_update_scale_target_inactive": float(fd_cov_update_scale_target_inactive),
            "output_normalization_enabled": bool(output_normalization_enabled),
            "output_allow_amplification": bool(output_allow_amplification),
        },
        slow_path={
            "enabled": str(algorithm_mode) != "localization_only",
            "tracking_mode": str(tracking_mode),
            "single_active": str(algorithm_mode) == "speaker_tracking_single_active",
            "speaker_history_size": int(speaker_history_size),
            "speaker_activation_min_predictions": int(speaker_activation_min_predictions),
            "speaker_match_window_deg": float(speaker_match_window_deg),
            "centroid_association_mode": str(centroid_association_mode),
            "centroid_association_sigma_deg": float(centroid_association_sigma_deg),
            "centroid_association_min_score": float(centroid_association_min_score),
            "slow_chunk_ms": int(slow_chunk_ms),
            "long_memory_enabled": bool(direction_long_memory_enabled),
            "long_memory_window_ms": 60000.0 if bool(direction_long_memory_enabled) else 2000.0,
        },
        focus_ratio=2.0,
        separation_mode=str(separation_mode),
        processing_mode="specific_speaker_enhancement",
    )
    duration_s_guess = float(raw_mix.size / max(int(sample_rate_hz), 1))
    ground_truth_tracks = _load_ground_truth_tracks(
        recording_dir,
        duration_s=duration_s_guess,
        mic_array_profile=str(mic_array_profile),
    )
    srp_override_provider = None
    initial_focus_direction_deg = None
    if bool(use_ground_truth_doa_override) and ground_truth_tracks:
        srp_override_provider = _ground_truth_srp_override_provider(
            ground_truth_tracks,
            mic_array_profile=str(mic_array_profile),
        )
        initial_focus_direction_deg = _display_angle_to_backend_prediction_deg(
            float(ground_truth_tracks[0]["angle_deg"]),
            mic_array_profile=str(mic_array_profile),
        )
    summary = run_offline_session_pipeline(
        req=req,
        mic_audio=mic_audio,
        mic_geometry_xyz=mic_geometry_xyz,
        out_dir=run_dir,
        input_recording_path=recording_dir,
        capture_trace=True,
        srp_override_provider=srp_override_provider,
        initial_focus_direction_deg=initial_focus_direction_deg,
    )
    proc, proc_sr = sf.read(run_dir / "enhanced_fast_path.wav", dtype="float32", always_2d=False)
    proc = np.asarray(proc, dtype=np.float32).reshape(-1)
    fs = int(proc_sr if proc_sr > 0 else sample_rate_hz)
    stage_paths = {
        "post_beamforming": run_dir / "post_beamforming.wav",
        "post_wiener": run_dir / "post_wiener.wav",
        "post_rnnoise": run_dir / "post_rnnoise.wav",
        "post_bandpass": run_dir / "post_bandpass.wav",
    }
    stage_audio: dict[str, np.ndarray] = {}
    for stage_name, stage_path in stage_paths.items():
        if not stage_path.exists():
            continue
        stage_sig, _stage_sr = sf.read(stage_path, dtype="float32", always_2d=False)
        stage_audio[stage_name] = np.asarray(stage_sig, dtype=np.float32).reshape(-1)
    if str(method).strip().lower() in {"mvdr_fd", "lcmv_target_band"}:
        _emit_beamformer_snapshot_artifacts(
            summary=summary,
            proc=proc,
            fs=fs,
            run_dir=run_dir,
            mic_geometry_xyz=np.asarray(mic_geometry_xyz, dtype=np.float64),
        )
    n = min(raw_mix.size, proc.size)
    trace_metrics = _trace_metrics(summary)
    duration_s = float(n / max(fs, 1))
    suppressed_user_tracks = _filter_tracks_by_speaker_name(ground_truth_tracks, suppressed_user_speaker_name)
    gt_metrics = _ground_truth_metrics(
        summary,
        ground_truth_tracks,
        mic_array_profile=str(mic_array_profile),
        transition_ignore_sec=float(ground_truth_transition_ignore_sec),
    )
    suppression_metrics = _suppression_metrics(
        summary,
        suppressed_user_tracks,
        transition_ignore_sec=float(ground_truth_transition_ignore_sec),
    )
    row = {
        "recording": recording_id,
        "requested_method": method,
        "method": str(summary.get("beamforming_mode", method)),
        "method_variant": (
            f"{method}__cov_{fd_noise_covariance_mode}__det_{target_activity_detector_backend}"
            if is_mvdr
            else str(summary.get("beamforming_mode", method))
        ),
        "sample_rate_hz": fs,
        "duration_s": duration_s,
        "channel_count": int(mic_audio.shape[1]),
        "raw_channel_filenames": ",".join(channel_filenames),
        "separation_mode": str(summary.get("separation_mode", "realtime_pipeline")),
        "use_ground_truth_doa_override": bool(use_ground_truth_doa_override),
        "enhancement_tier": str(summary.get("enhancement_tier", enhancement_tier)),
        "output_enhancer_mode": str(summary.get("output_enhancer_mode", output_enhancer_mode)),
        "postfilter_method": str(postfilter_method),
        "rnnoise_wet_mix": float(rnnoise_wet_mix),
        "mvdr_hop_ms": (float("nan") if mvdr_hop_ms is None else int(mvdr_hop_ms)),
        "fd_analysis_window_ms": float(fd_analysis_window_ms),
        "robust_target_band_width_deg": float(robust_target_band_width_deg),
        "robust_target_band_max_freq_hz": float(robust_target_band_max_freq_hz),
        "robust_target_band_condition_limit": float(robust_target_band_condition_limit),
        "fd_noise_covariance_mode": str(fd_noise_covariance_mode),
        "target_activity_rnn_update_mode": str(target_activity_rnn_update_mode) if is_mvdr else "",
        "target_activity_detector_mode": str(target_activity_detector_mode) if is_mvdr else "",
        "target_activity_detector_backend": str(target_activity_detector_backend) if is_mvdr else "",
        "target_activity_update_every_n_fast_frames": int(target_activity_update_every_n_fast_frames) if is_mvdr else 0,
        "fd_cov_ema_alpha": float(fd_cov_ema_alpha) if is_mvdr else float("nan"),
        "fd_diag_load": float(fd_diag_load) if is_mvdr else float("nan"),
        "fd_trace_diagonal_loading_factor": float(fd_trace_diagonal_loading_factor) if is_mvdr else float("nan"),
        "fd_identity_blend_alpha": float(fd_identity_blend_alpha) if is_mvdr else float("nan"),
        "fd_cov_update_scale_target_active": float(fd_cov_update_scale_target_active) if is_mvdr else float("nan"),
        "fd_cov_update_scale_target_inactive": float(fd_cov_update_scale_target_inactive) if is_mvdr else float("nan"),
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
        **gt_metrics,
        **suppression_metrics,
    }

    label = f"{recording_id} | {method} | realtime_pipeline"
    viz_dir = run_dir / "visualizations"
    _plot_waveform_compare(raw_mix[:n], proc[:n], fs, viz_dir / "waveforms.png", label)
    _plot_spectrogram_compare(raw_mix[:n], proc[:n], fs, viz_dir / "spectrograms.png", label)
    _plot_psd_compare(raw_mix[:n], proc[:n], fs, viz_dir / "psd.png", label)
    _plot_psd_stages(
        [
            ("raw", raw_mix[:n]),
            ("post_beamforming", stage_audio.get("post_beamforming", proc)[:n]),
            *([("post_wiener", stage_audio["post_wiener"][:n])] if "post_wiener" in stage_audio else []),
            *([("post_rnnoise", stage_audio["post_rnnoise"][:n])] if "post_rnnoise" in stage_audio else []),
            *([("post_bandpass", stage_audio["post_bandpass"][:n])] if "post_bandpass" in stage_audio else []),
            ("final", proc[:n]),
        ],
        fs,
        viz_dir / "psd_stages.png",
        label,
    )
    _plot_single_psd(raw_mix[:n], fs, viz_dir / "psd_raw.png", f"{label} raw PSD", color="#666666")
    _plot_single_psd(
        stage_audio.get("post_beamforming", proc)[:n],
        fs,
        viz_dir / "psd_post_beamforming.png",
        f"{label} post beamforming PSD",
        color="#005f73",
    )
    if "post_wiener" in stage_audio:
        _plot_single_psd(
            stage_audio["post_wiener"][:n],
            fs,
            viz_dir / "psd_post_wiener.png",
            f"{label} post wiener PSD",
            color="#0a9396",
        )
    if "post_rnnoise" in stage_audio:
        _plot_single_psd(
            stage_audio["post_rnnoise"][:n],
            fs,
            viz_dir / "psd_post_rnnoise.png",
            f"{label} post rnnoise PSD",
            color="#ee9b00",
        )
    if "post_bandpass" in stage_audio:
        _plot_single_psd(
            stage_audio["post_bandpass"][:n],
            fs,
            viz_dir / "psd_post_bandpass.png",
            f"{label} post bandpass PSD",
            color="#94d2bd",
        )
    _plot_single_psd(proc[:n], fs, viz_dir / "psd_final.png", f"{label} final PSD", color="#bb3e03")
    _plot_speaker_timeline(
        summary,
        viz_dir / "speaker_directions.png",
        label,
        ground_truth_tracks=ground_truth_tracks,
        suppressed_user_tracks=suppressed_user_tracks,
        mic_array_profile=str(mic_array_profile),
    )
    return row


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run realtime beamforming over Data Collection raw-channel recordings.")
    parser.add_argument("--input-path", required=True, help="Data Collection export root/zip, recording dir, or raw WAV dir")
    parser.add_argument("--out-dir", required=True, help="Directory for benchmark outputs")
    parser.add_argument("--methods", nargs="+", choices=["mvdr_fd", "lcmv_target_band", "sd_mvdr_fd", "gsc_fd", "delay_sum"], default=["mvdr_fd"])
    parser.add_argument(
        "--separation-mode",
        choices=["single_dominant_no_separator", "mock", "auto"],
        default="single_dominant_no_separator",
        help="Deprecated for this benchmark. Offline runs now use the same causal path as live sessions.",
    )
    parser.add_argument(
        "--algorithm-mode",
        choices=[
            "localization_only",
            "spatial_baseline",
            "speaker_tracking",
            "speaker_tracking_long_memory",
            "speaker_tracking_single_active",
            "single_dominant_no_separator",
        ],
        default="single_dominant_no_separator",
    )
    parser.add_argument("--mic-array-profile", choices=["respeaker_v3_0457", "respeaker_xvf3800_0650"], default="respeaker_xvf3800_0650")
    parser.add_argument("--fast-frame-ms", type=int, default=None, help="Fast-path frame cadence in ms. Defaults to localization hop when omitted.")
    parser.add_argument("--localization-window-ms", type=int, default=200)
    parser.add_argument("--localization-hop-ms", type=int, default=50)
    parser.add_argument("--localization-overlap", type=float, default=0.2)
    parser.add_argument("--localization-freq-low-hz", type=int, default=200)
    parser.add_argument("--localization-freq-high-hz", type=int, default=3000)
    parser.add_argument("--localization-pair-selection-mode", choices=["all", "adjacent_only"], default="all")
    parser.add_argument("--localization-vad-enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--capon-peak-min-sharpness", type=float, default=0.12)
    parser.add_argument("--capon-peak-min-margin", type=float, default=0.04)
    parser.add_argument("--own-voice-suppression-mode", choices=["off", "lcmv_null_hysteresis", "soft_output_gate"], default="lcmv_null_hysteresis")
    parser.add_argument("--suppressed-user-voice-doa-deg", type=float, default=None)
    parser.add_argument("--suppressed-user-match-window-deg", type=float, default=33.0)
    parser.add_argument("--suppressed-user-null-on-frames", type=int, default=3)
    parser.add_argument("--suppressed-user-null-off-frames", type=int, default=10)
    parser.add_argument("--suppressed-user-gate-attenuation-db", type=float, default=18.0)
    parser.add_argument("--suppressed-user-target-conflict-deg", type=float, default=30.0)
    parser.add_argument("--suppressed-user-speaker-name", type=str, default=None)
    parser.add_argument("--ground-truth-transition-ignore-sec", type=float, default=0.0)
    parser.add_argument("--enhancement-tier", choices=["custom", "baseline_pi", "classical_plus", "quality_cpu", "quality_heavy"], default="custom")
    parser.add_argument("--output-enhancer-mode", choices=["off", "wiener"], default="off")
    parser.add_argument("--postfilter-enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--postfilter-method", choices=["off", "wiener_dd", "rnnoise", "coherence_wiener", "wiener_then_rnnoise", "voice_bandpass", "rnnoise_then_voice_bandpass", "wiener_then_voice_bandpass"], default="off")
    parser.add_argument("--postfilter-noise-ema-alpha", type=float, default=0.08)
    parser.add_argument("--postfilter-speech-ema-alpha", type=float, default=0.12)
    parser.add_argument("--postfilter-gain-floor", type=float, default=0.22)
    parser.add_argument("--postfilter-gain-ema-alpha", type=float, default=0.3)
    parser.add_argument("--postfilter-dd-alpha", type=float, default=0.92)
    parser.add_argument("--postfilter-noise-update-speech-scale", type=float, default=0.2)
    parser.add_argument("--postfilter-gain-max-step-db", type=float, default=2.5)
    parser.add_argument("--rnnoise-wet-mix", type=float, default=1.0)
    parser.add_argument("--mvdr-hop-ms", type=int, default=60)
    parser.add_argument("--fd-analysis-window-ms", type=float, default=120.0)
    parser.add_argument("--robust-target-band-width-deg", type=float, default=10.0)
    parser.add_argument("--robust-target-band-max-freq-hz", type=float, default=0.0)
    parser.add_argument("--robust-target-band-condition-limit", type=float, default=1e3)
    parser.add_argument(
        "--fd-noise-covariance-mode",
        choices=["estimated_target_subtractive", "estimated_target_subtractive_frozen", "oracle_non_target_residual"],
        default="estimated_target_subtractive",
    )
    parser.add_argument(
        "--target-activity-rnn-update-mode",
        choices=["off", "estimated_target_activity", "oracle_target_activity"],
        default="estimated_target_activity",
    )
    parser.add_argument(
        "--target-activity-detector-mode",
        choices=["target_blocker_calibrated", "localization_peak_confidence"],
        default="target_blocker_calibrated",
    )
    parser.add_argument(
        "--target-activity-detector-backend",
        choices=["webrtc_fused", "silero_fused"],
        default="silero_fused",
    )
    parser.add_argument("--target-activity-update-every-n-fast-frames", type=int, default=2)
    parser.add_argument("--fd-cov-ema-alpha", type=float, default=0.6)
    parser.add_argument("--fd-diag-load", type=float, default=0.01)
    parser.add_argument("--fd-trace-diagonal-loading-factor", type=float, default=0.0)
    parser.add_argument("--fd-identity-blend-alpha", type=float, default=0.0)
    parser.add_argument("--target-activity-low-threshold", type=float, default=0.35)
    parser.add_argument("--target-activity-high-threshold", type=float, default=0.55)
    parser.add_argument("--target-activity-enter-frames", type=int, default=2)
    parser.add_argument("--target-activity-exit-frames", type=int, default=3)
    parser.add_argument("--fd-cov-update-scale-target-active", type=float, default=0.15)
    parser.add_argument("--fd-cov-update-scale-target-inactive", type=float, default=1.0)
    parser.add_argument(
        "--speaker-centroid-history-size",
        "--speaker-history-size",
        dest="speaker_history_size",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--speaker-activation-observations",
        "--speaker-activation-min-predictions",
        dest="speaker_activation_min_predictions",
        type=int,
        default=3,
    )
    parser.add_argument("--speaker-match-window-deg", type=float, default=25.0)
    parser.add_argument("--centroid-association-mode", choices=["hard_window", "gaussian"], default="hard_window")
    parser.add_argument("--centroid-association-sigma-deg", type=float, default=10.0)
    parser.add_argument("--centroid-association-min-score", type=float, default=0.15)
    parser.add_argument("--slow-chunk-ms", type=int, default=2000)
    parser.add_argument("--slow-chunk-hop-ms", type=int, default=1000)
    parser.add_argument("--max-speakers-hint", type=int, default=4)
    parser.add_argument("--assume-single-speaker", action="store_true")
    parser.add_argument("--localization-backend", choices=["srp_phat_localization", "srp_phat_legacy", "capon_1src", "capon_multisrc", "capon_mvdr_refine_1src", "music_1src"], default="srp_phat_localization")
    parser.add_argument("--use-ground-truth-doa-override", action="store_true", help="Use annotated speaker DOAs as localization peaks.")
    parser.add_argument("--tracking-mode", choices=TRACKING_MODE_CHOICES, default="doa_centroid_v1")
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
    args.tracking_mode = validate_tracking_mode(str(args.tracking_mode))
    input_path = Path(args.input_path).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    extracted_root, temp_dir = _extract_if_needed(input_path)
    try:
        resolved_root, recordings = _discover_recordings(extracted_root.resolve())
        mic_geometry_xyz = mic_positions_xyz(str(args.mic_array_profile))

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
                mic_array_profile=str(args.mic_array_profile),
                mic_geometry_xyz=mic_geometry_xyz,
                algorithm_mode=str(args.algorithm_mode),
                separation_mode=str(args.separation_mode),
                localization_window_ms=int(args.localization_window_ms),
                localization_hop_ms=int(args.localization_hop_ms),
                localization_overlap=float(args.localization_overlap),
                localization_freq_low_hz=int(args.localization_freq_low_hz),
                localization_freq_high_hz=int(args.localization_freq_high_hz),
                localization_pair_selection_mode=str(args.localization_pair_selection_mode),
                localization_vad_enabled=bool(args.localization_vad_enabled),
                capon_peak_min_sharpness=float(args.capon_peak_min_sharpness),
                capon_peak_min_margin=float(args.capon_peak_min_margin),
                speaker_history_size=int(args.speaker_history_size),
                speaker_activation_min_predictions=int(args.speaker_activation_min_predictions),
                speaker_match_window_deg=float(args.speaker_match_window_deg),
                centroid_association_mode=str(args.centroid_association_mode),
                centroid_association_sigma_deg=float(args.centroid_association_sigma_deg),
                centroid_association_min_score=float(args.centroid_association_min_score),
                own_voice_suppression_mode=str(args.own_voice_suppression_mode),
                suppressed_user_voice_doa_deg=(
                    None if args.suppressed_user_voice_doa_deg is None else float(args.suppressed_user_voice_doa_deg)
                ),
                suppressed_user_match_window_deg=float(args.suppressed_user_match_window_deg),
                suppressed_user_null_on_frames=int(args.suppressed_user_null_on_frames),
                suppressed_user_null_off_frames=int(args.suppressed_user_null_off_frames),
                suppressed_user_gate_attenuation_db=float(args.suppressed_user_gate_attenuation_db),
                suppressed_user_target_conflict_deg=float(args.suppressed_user_target_conflict_deg),
                suppressed_user_speaker_name=None if args.suppressed_user_speaker_name is None else str(args.suppressed_user_speaker_name),
                ground_truth_transition_ignore_sec=float(args.ground_truth_transition_ignore_sec),
                enhancement_tier=str(args.enhancement_tier),
                output_enhancer_mode=str(args.output_enhancer_mode),
                postfilter_enabled=bool(args.postfilter_enabled),
                postfilter_method=str(args.postfilter_method),
                postfilter_noise_ema_alpha=float(args.postfilter_noise_ema_alpha),
                postfilter_speech_ema_alpha=float(args.postfilter_speech_ema_alpha),
                postfilter_gain_floor=float(args.postfilter_gain_floor),
                postfilter_gain_ema_alpha=float(args.postfilter_gain_ema_alpha),
                postfilter_dd_alpha=float(args.postfilter_dd_alpha),
                postfilter_noise_update_speech_scale=float(args.postfilter_noise_update_speech_scale),
                postfilter_gain_max_step_db=float(args.postfilter_gain_max_step_db),
                rnnoise_wet_mix=float(args.rnnoise_wet_mix),
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
                assume_single_speaker=bool(args.assume_single_speaker),
                use_ground_truth_doa_override=bool(args.use_ground_truth_doa_override),
                mvdr_hop_ms=(None if args.mvdr_hop_ms is None else int(args.mvdr_hop_ms)),
                fd_analysis_window_ms=float(args.fd_analysis_window_ms),
                robust_target_band_width_deg=float(args.robust_target_band_width_deg),
                robust_target_band_max_freq_hz=float(args.robust_target_band_max_freq_hz),
                robust_target_band_condition_limit=float(args.robust_target_band_condition_limit),
                fd_noise_covariance_mode=str(args.fd_noise_covariance_mode),
                target_activity_rnn_update_mode=str(args.target_activity_rnn_update_mode),
                target_activity_detector_mode=str(args.target_activity_detector_mode),
                target_activity_detector_backend=str(args.target_activity_detector_backend),
                target_activity_update_every_n_fast_frames=int(args.target_activity_update_every_n_fast_frames),
                fd_cov_ema_alpha=float(args.fd_cov_ema_alpha),
                fd_diag_load=float(args.fd_diag_load),
                fd_trace_diagonal_loading_factor=float(args.fd_trace_diagonal_loading_factor),
                fd_identity_blend_alpha=float(args.fd_identity_blend_alpha),
                target_activity_low_threshold=float(args.target_activity_low_threshold),
                target_activity_high_threshold=float(args.target_activity_high_threshold),
                target_activity_enter_frames=int(args.target_activity_enter_frames),
                target_activity_exit_frames=int(args.target_activity_exit_frames),
                fd_cov_update_scale_target_active=float(args.fd_cov_update_scale_target_active),
                fd_cov_update_scale_target_inactive=float(args.fd_cov_update_scale_target_inactive),
            )

        if int(args.workers) <= 1:
            for job in jobs:
                row = _submit_job(job)
                scene_rows.append(row)
                summary_by_method[str(row.get("method_variant", row["method"]))].append(row)
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
                futures = [pool.submit(_submit_job, job) for job in jobs]
                for fut in concurrent.futures.as_completed(futures):
                    row = fut.result()
                    scene_rows.append(row)
                    summary_by_method[str(row.get("method_variant", row["method"]))].append(row)

        scene_rows.sort(key=lambda row: (str(row["recording"]), str(row["method"])))

        _write_csv(out_dir / "scene_metrics.csv", scene_rows)

        summary_rows: list[dict] = []
        for method, rows in summary_by_method.items():
            summary_rows.append(
                {
                    "method": method,
                    "n_recordings": len(rows),
                    "fast_rtf_mean": _mean_numeric(rows, "fast_rtf"),
                    "slow_rtf_mean": _mean_numeric(rows, "slow_rtf"),
                    "clip_rate_processed_mean": _mean_numeric(rows, "clip_rate_processed"),
                    "rms_gain_db_mean": _mean_numeric(rows, "rms_gain_db"),
                    "high_band_noise_ratio_processed_mean": _mean_numeric(rows, "high_band_noise_ratio_processed"),
                    "speaker_count_final_mean": _mean_numeric(rows, "speaker_count_final"),
                    "active_speaker_count_final_mean": _mean_numeric(rows, "active_speaker_count_final"),
                    "dominant_confidence_avg_mean": _mean_numeric(rows, "dominant_confidence_avg"),
                    "dominant_direction_step_p95_deg_mean": _mean_numeric(rows, "dominant_direction_step_p95_deg"),
                    "enhancement_tier": str(rows[0].get("enhancement_tier", "")) if rows else "",
                    "output_enhancer_mode": str(rows[0].get("output_enhancer_mode", "")) if rows else "",
                    "postfilter_method": str(rows[0].get("postfilter_method", "")) if rows else "",
                    "fd_noise_covariance_mode": str(rows[0].get("fd_noise_covariance_mode", "")) if rows else "",
                    "target_activity_detector_mode": str(rows[0].get("target_activity_detector_mode", "")) if rows else "",
                    "target_activity_detector_backend": str(rows[0].get("target_activity_detector_backend", "")) if rows else "",
                    "mvdr_hop_ms": _mean_numeric(rows, "mvdr_hop_ms"),
                    "fd_analysis_window_ms": _mean_numeric(rows, "fd_analysis_window_ms"),
                    "suppression_user_recall_mean": _mean_numeric(rows, "suppression_user_recall"),
                    "suppression_false_positive_rate_mean": _mean_numeric(rows, "suppression_false_positive_rate"),
                    "suppression_precision_mean": _mean_numeric(rows, "suppression_precision"),
                    "suppression_activation_rate_mean": _mean_numeric(rows, "suppression_activation_rate"),
                    "gt_final_mae_mean": _mean_numeric(rows, "gt_final_mae_deg"),
                    "gt_final_acc_10_mean": _mean_numeric(rows, "gt_final_acc_10"),
                    "gt_final_acc_20_mean": _mean_numeric(rows, "gt_final_acc_20"),
                    "gt_final_match_rate_mean": _mean_numeric(rows, "gt_final_match_rate"),
                    "gt_trace_active_mae_mean": _mean_numeric(rows, "gt_trace_active_mae_deg"),
                    "gt_trace_active_acc_10_mean": _mean_numeric(rows, "gt_trace_active_acc_10"),
                    "gt_trace_active_acc_20_mean": _mean_numeric(rows, "gt_trace_active_acc_20"),
                    "gt_trace_active_match_rate_mean": _mean_numeric(rows, "gt_trace_active_match_rate"),
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
            "separation_mode": "realtime_pipeline",
            "algorithm_mode": str(args.algorithm_mode),
            "assume_single_speaker": bool(args.assume_single_speaker),
            "fast_frame_ms": int(args.localization_hop_ms),
            "localization_window_ms": int(args.localization_window_ms),
            "localization_hop_ms": int(args.localization_hop_ms),
            "localization_overlap": float(args.localization_overlap),
            "localization_freq_low_hz": int(args.localization_freq_low_hz),
            "localization_freq_high_hz": int(args.localization_freq_high_hz),
            "localization_pair_selection_mode": str(args.localization_pair_selection_mode),
            "localization_vad_enabled": bool(args.localization_vad_enabled),
            "capon_peak_min_sharpness": float(args.capon_peak_min_sharpness),
            "capon_peak_min_margin": float(args.capon_peak_min_margin),
            "own_voice_suppression_mode": str(args.own_voice_suppression_mode),
            "suppressed_user_voice_doa_deg": (
                None if args.suppressed_user_voice_doa_deg is None else float(args.suppressed_user_voice_doa_deg)
            ),
            "suppressed_user_match_window_deg": float(args.suppressed_user_match_window_deg),
            "suppressed_user_null_on_frames": int(args.suppressed_user_null_on_frames),
            "suppressed_user_null_off_frames": int(args.suppressed_user_null_off_frames),
            "suppressed_user_gate_attenuation_db": float(args.suppressed_user_gate_attenuation_db),
            "suppressed_user_target_conflict_deg": float(args.suppressed_user_target_conflict_deg),
            "suppressed_user_speaker_name": None if args.suppressed_user_speaker_name is None else str(args.suppressed_user_speaker_name),
            "ground_truth_transition_ignore_sec": float(args.ground_truth_transition_ignore_sec),
            "enhancement_tier": str(args.enhancement_tier),
            "output_enhancer_mode": str(args.output_enhancer_mode),
            "postfilter_enabled": bool(args.postfilter_enabled),
            "postfilter_method": str(args.postfilter_method),
            "rnnoise_wet_mix": float(args.rnnoise_wet_mix),
            "mvdr_hop_ms": int(args.mvdr_hop_ms),
            "fd_analysis_window_ms": float(args.fd_analysis_window_ms),
            "fd_noise_covariance_mode": str(args.fd_noise_covariance_mode),
            "target_activity_rnn_update_mode": str(args.target_activity_rnn_update_mode),
            "target_activity_detector_mode": str(args.target_activity_detector_mode),
            "target_activity_detector_backend": str(args.target_activity_detector_backend),
            "target_activity_update_every_n_fast_frames": int(args.target_activity_update_every_n_fast_frames),
            "fd_cov_ema_alpha": float(args.fd_cov_ema_alpha),
            "fd_diag_load": float(args.fd_diag_load),
            "fd_trace_diagonal_loading_factor": float(args.fd_trace_diagonal_loading_factor),
            "fd_identity_blend_alpha": float(args.fd_identity_blend_alpha),
            "target_activity_low_threshold": float(args.target_activity_low_threshold),
            "target_activity_high_threshold": float(args.target_activity_high_threshold),
            "target_activity_enter_frames": int(args.target_activity_enter_frames),
            "target_activity_exit_frames": int(args.target_activity_exit_frames),
            "fd_cov_update_scale_target_active": float(args.fd_cov_update_scale_target_active),
            "fd_cov_update_scale_target_inactive": float(args.fd_cov_update_scale_target_inactive),
            "speaker_history_size": int(args.speaker_history_size),
            "speaker_activation_min_predictions": int(args.speaker_activation_min_predictions),
            "speaker_match_window_deg": float(args.speaker_match_window_deg),
            "localization_backend": str(args.localization_backend),
            "use_ground_truth_doa_override": bool(args.use_ground_truth_doa_override),
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

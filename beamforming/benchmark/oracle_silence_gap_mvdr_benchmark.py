from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import soundfile as sf

from beamforming.benchmark.metrics import compute_metric_bundle
from beamforming.benchmark.oracle_xvf3800_enhancement_sweep import (
    DEFAULT_PROFILE,
    _build_active_target_schedule,
    _build_clean_reference,
    _build_oracle_frame_states,
    _load_scene_metadata,
    _oracle_srp_override_provider,
    _oracle_target_activity_override_provider,
    _speaker_source_index_map,
    _stage_scene,
)
from mic_array_forwarder.models import SessionStartRequest
from realtime_pipeline.session_runtime import run_offline_session_pipeline
from simulation.mic_array_profiles import SUPPORTED_MIC_ARRAY_PROFILES
from simulation.simulation_config import SimulationConfig


DEFAULT_SCENES_ROOT = Path("simulation/simulations/configs/testing_specific_angles_silence_gaps")
DEFAULT_ASSETS_ROOT = Path("simulation/simulations/assets/testing_specific_angles_silence_gaps")
DEFAULT_OUT_ROOT = Path("beamforming/benchmark/oracle_silence_gap_mvdr")
DEFAULT_METHODS = ["delay_sum", "mvdr_fd_oracle_activity", "mvdr_fd_estimated_activity"]
FAST_FRAME_MS = 10
FD_ANALYSIS_WINDOW_MS = 40.0
ACTIVE_UPDATE_SCALE = 0.15
INACTIVE_UPDATE_SCALE = 1.0


@dataclass(frozen=True)
class MethodSpec:
    method_key: str
    beamforming_mode: str
    target_activity_mode: str | None


@dataclass(frozen=True)
class JobResult:
    row: dict[str, object]
    run_summary: dict[str, object]


METHOD_SPECS: dict[str, MethodSpec] = {
    "delay_sum": MethodSpec("delay_sum", "delay_sum", None),
    "mvdr_fd_oracle_activity": MethodSpec("mvdr_fd_oracle_activity", "mvdr_fd", "oracle_target_activity"),
    "mvdr_fd_estimated_activity": MethodSpec("mvdr_fd_estimated_activity", "mvdr_fd", "estimated_target_activity"),
}


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _build_session_request(*, method_spec: MethodSpec, sample_rate: int, channel_count: int, profile: str) -> SessionStartRequest:
    return SessionStartRequest(
        input_source="simulation",
        channel_count=int(channel_count),
        sample_rate_hz=int(sample_rate),
        monitor_source="processed",
        mic_array_profile=str(profile),
        fast_path={
            "localization_hop_ms": int(FAST_FRAME_MS),
            "localization_window_ms": 160,
            "overlap": 0.2,
            "freq_low_hz": 200,
            "freq_high_hz": 3000,
            "localization_pair_selection_mode": "all",
            "localization_vad_enabled": True,
            "localization_backend": "srp_phat_localization",
            "beamforming_mode": str(method_spec.beamforming_mode),
            "fd_analysis_window_ms": float(FD_ANALYSIS_WINDOW_MS),
            "target_activity_rnn_update_mode": method_spec.target_activity_mode,
            "target_activity_low_threshold": 0.25,
            "target_activity_high_threshold": 0.45,
            "target_activity_enter_frames": 2,
            "target_activity_exit_frames": 3,
            "fd_cov_update_scale_target_active": float(ACTIVE_UPDATE_SCALE),
            "fd_cov_update_scale_target_inactive": float(INACTIVE_UPDATE_SCALE),
            "postfilter_enabled": False,
            "output_normalization_enabled": False,
            "output_allow_amplification": False,
            "own_voice_suppression_mode": "off",
        },
        slow_path={
            "enabled": False,
            "tracking_mode": "doa_centroid_v1",
            "speaker_match_window_deg": 25.0,
            "centroid_association_mode": "hard_window",
            "centroid_association_sigma_deg": 10.0,
            "centroid_association_min_score": 0.15,
            "slow_chunk_ms": 200,
        },
        max_speakers_hint=max(1, int(channel_count)),
        separation_mode="mock",
        processing_mode="beamform_from_ground_truth",
    )


def _metric_dict(bundle) -> dict[str, float]:
    return {
        "snr_db_raw": float(bundle.snr_db_raw),
        "snr_db_processed": float(bundle.snr_db_processed),
        "delta_snr_db": float(bundle.delta_snr_db),
        "sii_raw": float(bundle.sii_raw),
        "sii_processed": float(bundle.sii_processed),
        "delta_sii": float(bundle.delta_sii),
        "si_sdr_db_raw": float(bundle.si_sdr_db_raw),
        "si_sdr_db_processed": float(bundle.si_sdr_db_processed),
        "delta_si_sdr_db": float(bundle.delta_si_sdr_db),
    }


def _masked_metric_dict(
    *,
    clean_ref: np.ndarray,
    raw_audio: np.ndarray,
    processed_audio: np.ndarray,
    sample_rate: int,
    mask: np.ndarray,
) -> dict[str, float]:
    mask_arr = np.asarray(mask, dtype=bool).reshape(-1)
    if mask_arr.size == 0 or int(np.sum(mask_arr)) < max(256, sample_rate // 8):
        return {key: float("nan") for key in _metric_dict(compute_metric_bundle(clean_ref=np.zeros(512), raw_audio=np.zeros(512), processed_audio=np.zeros(512), sample_rate=sample_rate)).keys()}
    bundle = compute_metric_bundle(
        clean_ref=np.asarray(clean_ref, dtype=np.float64)[mask_arr],
        raw_audio=np.asarray(raw_audio, dtype=np.float64)[mask_arr],
        processed_audio=np.asarray(processed_audio, dtype=np.float64)[mask_arr],
        sample_rate=int(sample_rate),
    )
    return _metric_dict(bundle)


def _silence_metrics(raw_audio: np.ndarray, processed_audio: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    mask_arr = np.asarray(mask, dtype=bool).reshape(-1)
    if mask_arr.size == 0 or not np.any(mask_arr):
        return {
            "silence_rms_raw_dbfs": float("nan"),
            "silence_rms_processed_dbfs": float("nan"),
            "silence_noise_reduction_db": float("nan"),
        }
    raw = np.asarray(raw_audio, dtype=np.float64)[mask_arr]
    proc = np.asarray(processed_audio, dtype=np.float64)[mask_arr]
    raw_rms = float(np.sqrt(np.mean(raw**2) + 1e-12))
    proc_rms = float(np.sqrt(np.mean(proc**2) + 1e-12))
    return {
        "silence_rms_raw_dbfs": float(20.0 * np.log10(max(raw_rms, 1e-12))),
        "silence_rms_processed_dbfs": float(20.0 * np.log10(max(proc_rms, 1e-12))),
        "silence_noise_reduction_db": float(20.0 * np.log10(max(raw_rms, 1e-12) / max(proc_rms, 1e-12))),
    }


def _summarize_trace(summary: dict[str, object]) -> dict[str, float]:
    srp_trace = list(summary.get("srp_trace", []))
    alphas_active: list[float] = []
    alphas_inactive: list[float] = []
    for row in srp_trace:
        debug = dict(row.get("debug", {}))
        target_debug = dict(debug.get("target_activity", {}))
        alpha = target_debug.get("covariance_alpha")
        active = target_debug.get("active")
        if alpha is None or active is None:
            continue
        if bool(active):
            alphas_active.append(float(alpha))
        else:
            alphas_inactive.append(float(alpha))
    return {
        "trace_target_active_frames": float(len(alphas_active)),
        "trace_target_inactive_frames": float(len(alphas_inactive)),
        "trace_cov_alpha_active_mean": float(np.mean(alphas_active)) if alphas_active else float("nan"),
        "trace_cov_alpha_inactive_mean": float(np.mean(alphas_inactive)) if alphas_inactive else float("nan"),
    }


def _mean_or_nan(values: list[float]) -> float:
    finite = [float(v) for v in values if np.isfinite(float(v))]
    return float(np.mean(finite)) if finite else float("nan")


def _aggregate(rows: list[dict[str, object]], group_fields: list[str]) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(field) for field in group_fields)].append(row)
    out_rows: list[dict[str, object]] = []
    metric_fields = [
        "delta_snr_db",
        "delta_sii",
        "speech_delta_snr_db",
        "speech_delta_sii",
        "silence_noise_reduction_db",
        "trace_cov_alpha_active_mean",
        "trace_cov_alpha_inactive_mean",
        "rtf",
    ]
    for key, items in sorted(grouped.items(), key=lambda pair: tuple(str(v) for v in pair[0])):
        out = {field: value for field, value in zip(group_fields, key)}
        out["n_runs"] = len(items)
        for metric in metric_fields:
            out[f"{metric}_mean"] = _mean_or_nan([float(item.get(metric, float("nan"))) for item in items])
        out_rows.append(out)
    return out_rows


def _run_job(*, scene_path: str, assets_root: str, out_root: str, profile: str, method: str) -> JobResult:
    method_spec = METHOD_SPECS[str(method)]
    scene_cfg_path = Path(scene_path)
    scene_id = scene_cfg_path.stem
    root = Path(out_root)
    staged_scene, staged_metadata = _stage_scene(
        scene_cfg_path,
        root / "_staged_scenes",
        profile,
        Path(assets_root),
        1.0,
        stage_key=method,
    )
    metadata_payload = json.loads(staged_metadata.read_text(encoding="utf-8")) if staged_metadata.exists() else {}
    scene_meta = _load_scene_metadata(scene_id, staged_metadata)
    sim_cfg = SimulationConfig.from_file(staged_scene)

    from simulation.simulator import run_simulation

    mic_audio, mic_pos, source_signals = run_simulation(sim_cfg)
    sample_rate = int(sim_cfg.audio.fs)
    active_speaker_ids, active_doa_deg = _build_active_target_schedule(
        metadata_payload,
        sample_rate=sample_rate,
        n_samples=int(mic_audio.shape[0]),
    )
    speech_mask = np.asarray(active_speaker_ids >= 0, dtype=bool)
    silence_mask = np.logical_not(speech_mask)
    speaker_to_source_idx = _speaker_source_index_map(sim_cfg, metadata_payload)
    clean_ref = _build_clean_reference(source_signals, speaker_to_source_idx, active_speaker_ids)
    raw_mix = np.mean(np.asarray(mic_audio, dtype=np.float64), axis=1).astype(np.float32, copy=False)

    frame_states = _build_oracle_frame_states(
        active_speaker_ids=active_speaker_ids,
        active_doa_deg=active_doa_deg,
        sample_rate=sample_rate,
        fast_frame_ms=FAST_FRAME_MS,
    )
    req = _build_session_request(
        method_spec=method_spec,
        sample_rate=sample_rate,
        channel_count=int(mic_audio.shape[1]),
        profile=profile,
    )

    run_dir = root / "runs" / scene_id / method_spec.method_key
    run_dir.mkdir(parents=True, exist_ok=True)
    sf.write(run_dir / "raw_mix_mean.wav", raw_mix, sample_rate)
    sf.write(run_dir / "clean_active_target.wav", clean_ref.astype(np.float32), sample_rate)

    t0 = perf_counter()
    runtime_summary = run_offline_session_pipeline(
        req=req,
        mic_audio=np.asarray(mic_audio, dtype=np.float32),
        mic_geometry_xyz=np.asarray(mic_pos, dtype=np.float64),
        out_dir=run_dir,
        capture_trace=True,
        srp_override_provider=_oracle_srp_override_provider(frame_states),
        target_activity_override_provider=(
            _oracle_target_activity_override_provider(frame_states)
            if method_spec.target_activity_mode == "oracle_target_activity"
            else None
        ),
    )
    elapsed_s = max(perf_counter() - t0, 1e-9)
    processed, proc_sr = sf.read(run_dir / "enhanced_fast_path.wav", dtype="float32", always_2d=False)
    processed_audio = np.asarray(processed, dtype=np.float32).reshape(-1)
    if int(proc_sr) != int(sample_rate):
        raise ValueError(f"Unexpected sample rate {proc_sr}, expected {sample_rate}")
    sf.write(run_dir / "enhanced.wav", processed_audio, sample_rate)

    overall = compute_metric_bundle(
        clean_ref=np.asarray(clean_ref, dtype=np.float64),
        raw_audio=np.asarray(raw_mix, dtype=np.float64),
        processed_audio=np.asarray(processed_audio, dtype=np.float64),
        sample_rate=sample_rate,
    )
    speech_metrics = _masked_metric_dict(
        clean_ref=clean_ref,
        raw_audio=raw_mix,
        processed_audio=processed_audio,
        sample_rate=sample_rate,
        mask=speech_mask,
    )
    silence_metrics = _silence_metrics(raw_mix, processed_audio, silence_mask)
    trace_metrics = _summarize_trace(runtime_summary)
    duration_s = float(len(processed_audio) / max(sample_rate, 1))

    row: dict[str, object] = {
        "scene": scene_id,
        "method": method_spec.method_key,
        "beamforming_mode_runtime": str(runtime_summary.get("beamforming_mode", "")),
        "target_activity_mode": "" if method_spec.target_activity_mode is None else str(method_spec.target_activity_mode),
        "main_angle_deg": scene_meta["main_angle_deg"],
        "secondary_angle_deg": scene_meta["secondary_angle_deg"],
        "noise_layout_type": scene_meta["noise_layout_type"],
        "noise_angles_deg": json.dumps(scene_meta["noise_angles_deg"]),
        "scene_layout_family": scene_meta["scene_layout_family"],
        "speech_frame_fraction": float(np.mean(speech_mask.astype(np.float64))),
        "silence_frame_fraction": float(np.mean(silence_mask.astype(np.float64))),
        "rtf": float(elapsed_s / max(duration_s, 1e-9)),
        **_metric_dict(overall),
        "speech_delta_snr_db": float(speech_metrics["delta_snr_db"]),
        "speech_delta_sii": float(speech_metrics["delta_sii"]),
        "speech_delta_si_sdr_db": float(speech_metrics["delta_si_sdr_db"]),
        **silence_metrics,
        **trace_metrics,
        "run_dir": str(run_dir.resolve()),
    }

    run_summary: dict[str, object] = {
        "scene": scene_id,
        "method": method_spec.method_key,
        "profile": profile,
        "staged_scene": str(staged_scene.resolve()),
        "scenario_metadata": str(staged_metadata.resolve()) if staged_metadata.exists() else "",
        "run_dir": str(run_dir.resolve()),
        "runtime_summary": runtime_summary,
        "metrics": {
            "overall": _metric_dict(overall),
            "speech_only": speech_metrics,
            "silence_only": silence_metrics,
            "trace": trace_metrics,
        },
    }
    _write_json(run_dir / "summary.json", run_summary)
    return JobResult(row=row, run_summary=run_summary)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark GT-DOA beamforming on silence-gap scenes with oracle and estimated silence gating.")
    parser.add_argument("--scenes-root", default=str(DEFAULT_SCENES_ROOT))
    parser.add_argument("--assets-root", default=str(DEFAULT_ASSETS_ROOT))
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--profile", default=DEFAULT_PROFILE, choices=list(SUPPORTED_MIC_ARRAY_PROFILES))
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS))
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 1) - 1))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_id = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.out_root) / run_id
    out_root.mkdir(parents=True, exist_ok=True)

    scenes = sorted(Path(args.scenes_root).glob("*.json"))
    if args.max_scenes is not None:
        scenes = scenes[: int(args.max_scenes)]
    if not scenes:
        raise FileNotFoundError(f"No scenes found under {args.scenes_root}")

    jobs = [
        {
            "scene_path": str(scene.resolve()),
            "assets_root": str(Path(args.assets_root).resolve()),
            "out_root": str(out_root.resolve()),
            "profile": str(args.profile),
            "method": str(method),
        }
        for scene in scenes
        for method in list(args.methods)
    ]

    results: list[JobResult] = []
    with ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        future_map = {pool.submit(_run_job, **job): (Path(job["scene_path"]).stem, job["method"]) for job in jobs}
        for future in as_completed(future_map):
            scene_id, method = future_map[future]
            print(f"completed {scene_id} :: {method}")
            results.append(future.result())

    scene_rows = [result.row for result in sorted(results, key=lambda item: (str(item.row["scene"]), str(item.row["method"])))]
    _write_csv(out_root / "summary_rows.csv", scene_rows)
    _write_json(out_root / "summary_rows.json", {"rows": scene_rows})

    summary_by_method = _aggregate(scene_rows, ["method"])
    summary_by_layout = _aggregate(scene_rows, ["method", "scene_layout_family"])
    _write_csv(out_root / "summary_by_method.csv", summary_by_method)
    _write_csv(out_root / "summary_by_scene_layout.csv", summary_by_layout)
    _write_json(
        out_root / "summary.json",
        {
            "run_id": run_id,
            "profile": str(args.profile),
            "scenes_root": str(Path(args.scenes_root).resolve()),
            "assets_root": str(Path(args.assets_root).resolve()),
            "methods": list(args.methods),
            "n_scenes": len(scenes),
            "summary_by_method": summary_by_method,
            "summary_by_scene_layout": summary_by_layout,
            "runs": [result.run_summary for result in results],
        },
    )
    latest = Path(args.out_root) / "latest"
    if latest.exists() or latest.is_symlink():
        latest.unlink()
    latest.symlink_to(out_root.resolve(), target_is_directory=True)
    print(json.dumps({"out_root": str(out_root.resolve()), "n_runs": len(results), "n_scenes": len(scenes)}, indent=2))


if __name__ == "__main__":
    main()

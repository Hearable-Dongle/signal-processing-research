from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import soundfile as sf

from beamforming.benchmark.metrics import compute_metric_bundle
from beamforming.benchmark.oracle_xvf3800_enhancement_sweep import (
    DEFAULT_ASSETS_ROOT,
    DEFAULT_PROFILE,
    DEFAULT_SCENES_ROOT,
    NULL_CONFLICT_DEG,
    NULL_USER_SPEAKER_ID,
    _build_active_target_schedule,
    _build_clean_reference,
    _build_oracle_frame_states,
    _load_scene_metadata,
    _oracle_srp_override_provider,
    _speaker_doa_map,
    _speaker_source_index_map,
    _stage_scene,
)
from mic_array_forwarder.models import SessionStartRequest
from realtime_pipeline.contracts import PipelineConfig
from realtime_pipeline.output_enhancer import apply_output_enhancer_audio
from realtime_pipeline.session_runtime import build_pipeline_config_from_request, run_offline_session_pipeline
from simulation.simulator import run_simulation
from simulation.simulation_config import SimulationConfig

DEFAULT_OUT_ROOT = Path("beamforming/benchmark/output_denoiser_optuna")
DEFAULT_NOISE_GAIN_SCALES = [0.0, 0.5, 1.0]
DEFAULT_METHODS = ["shared_wiener", "rnnoise", "shared_wiener_rnnoise"]
DEFAULT_FAST_FRAME_MS = 50
DEFAULT_FD_ANALYSIS_WINDOW_MS = 40.0
DEFAULT_BEAMFORMING_METHOD = "lcmv_null"


@dataclass(frozen=True)
class CachedCase:
    split: str
    scene_id: str
    noise_gain_scale: float
    scene_layout_family: str
    base_audio: np.ndarray
    clean_ref: np.ndarray
    raw_audio: np.ndarray
    speech_activity: np.ndarray
    sample_rate_hz: int
    duration_s: float
    cache_dir: Path
    base_summary: dict[str, Any]


def _get_optuna():
    try:
        import optuna
    except ModuleNotFoundError as exc:
        raise RuntimeError("optuna is required for this tuner.") from exc
    return optuna


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quick Optuna tuner for realtime output denoisers on cached synthetic beamformer output.")
    parser.add_argument("--scenes-root", default=str(DEFAULT_SCENES_ROOT))
    parser.add_argument("--assets-root", default=str(DEFAULT_ASSETS_ROOT))
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--profile", default=DEFAULT_PROFILE)
    parser.add_argument("--duration-min", type=float, default=15.0)
    parser.add_argument("--max-scenes", type=int, default=3)
    parser.add_argument("--noise-gain-scales", nargs="+", type=float, default=list(DEFAULT_NOISE_GAIN_SCALES))
    parser.add_argument("--fast-frame-ms", type=int, default=DEFAULT_FAST_FRAME_MS)
    parser.add_argument("--beamforming-method", choices=["lcmv_null", "mvdr_fd", "delay_sum"], default=DEFAULT_BEAMFORMING_METHOD)
    parser.add_argument("--fd-analysis-window-ms", type=float, default=DEFAULT_FD_ANALYSIS_WINDOW_MS)
    parser.add_argument("--rtf-target", type=float, default=0.35)
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS))
    parser.add_argument("--cache-workers", type=int, default=max(1, min(4, (os.cpu_count() or 1) - 1)))
    parser.add_argument("--study-db", default=None, help="Optional SQLite path for persistent Optuna storage.")
    return parser.parse_args()


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


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _storage_url(db_path: Path) -> str:
    return f"sqlite:///{db_path.resolve()}"


def _build_base_request(
    *,
    beamforming_method: str,
    sample_rate_hz: int,
    channel_count: int,
    profile: str,
    fast_frame_ms: int,
    fd_analysis_window_ms: float,
    null_user_doa_deg: float | None,
) -> SessionStartRequest:
    shared_beamforming_mode = "delay_sum" if beamforming_method == "delay_sum" else "mvdr_fd"
    suppression_mode = "off" if beamforming_method != "lcmv_null" else "lcmv_null_hysteresis"
    return SessionStartRequest(
        algorithm_mode="localization_only",
        input_source="simulation",
        channel_count=int(channel_count),
        sample_rate_hz=int(sample_rate_hz),
        monitor_source="processed",
        mic_array_profile=str(profile),
        localization_hop_ms=int(fast_frame_ms),
        localization_window_ms=max(160, int(fast_frame_ms)),
        overlap=0.2,
        freq_low_hz=200,
        freq_high_hz=3000,
        localization_pair_selection_mode="all",
        localization_vad_enabled=True,
        own_voice_suppression_mode=suppression_mode,
        suppressed_user_voice_doa_deg=(None if beamforming_method != "lcmv_null" else null_user_doa_deg),
        suppressed_user_match_window_deg=33.0,
        suppressed_user_null_on_frames=1,
        suppressed_user_null_off_frames=1,
        suppressed_user_gate_attenuation_db=18.0,
        suppressed_user_target_conflict_deg=float(NULL_CONFLICT_DEG),
        speaker_match_window_deg=25.0,
        centroid_association_mode="hard_window",
        centroid_association_sigma_deg=10.0,
        centroid_association_min_score=0.15,
        slow_chunk_ms=200,
        max_speakers_hint=max(1, int(channel_count)),
        assume_single_speaker=False,
        separation_mode="mock",
        localization_backend="srp_phat_localization",
        tracking_mode="doa_centroid_v1",
        beamforming_mode=shared_beamforming_mode,
        fd_analysis_window_ms=float(fd_analysis_window_ms),
        postfilter_enabled=False,
        output_enhancer_mode="off",
        output_normalization_enabled=True,
        output_allow_amplification=False,
        processing_mode="beamform_from_ground_truth",
    )


def _frame_activity_from_active_ids(active_speaker_ids: np.ndarray, frame_samples: int) -> np.ndarray:
    out = np.zeros(active_speaker_ids.shape[0], dtype=np.float32)
    for start in range(0, active_speaker_ids.shape[0], frame_samples):
        end = min(active_speaker_ids.shape[0], start + frame_samples)
        val = 1.0 if np.any(active_speaker_ids[start:end] >= 0) else 0.0
        out[start:end] = val
    return out


def _cache_case(job: dict[str, Any]) -> dict[str, Any]:
    scene_path = Path(job["scene_path"])
    out_root = Path(job["out_root"])
    assets_root = Path(job["assets_root"])
    profile = str(job["profile"])
    noise_gain_scale = float(job["noise_gain_scale"])
    fast_frame_ms = int(job["fast_frame_ms"])
    fd_analysis_window_ms = float(job["fd_analysis_window_ms"])
    beamforming_method = str(job["beamforming_method"])
    split = str(job["split"])

    staged_scene, staged_metadata = _stage_scene(
        scene_path,
        out_root / "_staged_scenes",
        profile,
        assets_root,
        noise_gain_scale,
        stage_key=f"{beamforming_method}_{split}",
    )
    sim_cfg = SimulationConfig.from_file(staged_scene)
    metadata_payload = json.loads(staged_metadata.read_text(encoding="utf-8")) if staged_metadata.exists() else {}
    scene_meta = _load_scene_metadata(scene_path.stem, staged_metadata)
    mic_audio, mic_pos, source_signals = run_simulation(sim_cfg)
    sample_rate = int(sim_cfg.audio.fs)
    active_speaker_ids, active_doa_deg = _build_active_target_schedule(
        metadata_payload,
        sample_rate=sample_rate,
        n_samples=int(mic_audio.shape[0]),
    )
    frame_samples = max(1, int(round(sample_rate * (float(fast_frame_ms) / 1000.0))))
    speech_activity = _frame_activity_from_active_ids(active_speaker_ids, frame_samples)
    speaker_to_source_idx = _speaker_source_index_map(sim_cfg, metadata_payload)
    doa_by_speaker = _speaker_doa_map(metadata_payload)
    null_user_doa_deg = doa_by_speaker.get(int(NULL_USER_SPEAKER_ID))
    clean_ref = _build_clean_reference(source_signals, speaker_to_source_idx, active_speaker_ids).astype(np.float32, copy=False)
    raw_mix = np.mean(np.asarray(mic_audio, dtype=np.float64), axis=1).astype(np.float32, copy=False)
    frame_states = _build_oracle_frame_states(
        active_speaker_ids=active_speaker_ids,
        active_doa_deg=active_doa_deg,
        sample_rate=sample_rate,
        fast_frame_ms=fast_frame_ms,
        null_user_speaker_id=(NULL_USER_SPEAKER_ID if beamforming_method == "lcmv_null" else None),
        null_user_doa_deg=(null_user_doa_deg if beamforming_method == "lcmv_null" else None),
        null_conflict_deg=float(NULL_CONFLICT_DEG),
    )
    req = _build_base_request(
        beamforming_method=beamforming_method,
        sample_rate_hz=sample_rate,
        channel_count=int(mic_audio.shape[1]),
        profile=profile,
        fast_frame_ms=fast_frame_ms,
        fd_analysis_window_ms=fd_analysis_window_ms,
        null_user_doa_deg=null_user_doa_deg,
    )
    case_dir = out_root / "cache" / split / scene_path.stem / f"noise_{str(noise_gain_scale).replace('.', 'p')}"
    case_dir.mkdir(parents=True, exist_ok=True)
    summary = run_offline_session_pipeline(
        req=req,
        mic_audio=np.asarray(mic_audio, dtype=np.float32),
        mic_geometry_xyz=np.asarray(mic_pos, dtype=np.float64),
        out_dir=case_dir,
        capture_trace=False,
        srp_override_provider=_oracle_srp_override_provider(frame_states),
    )
    base_audio, sr = sf.read(case_dir / "enhanced_fast_path.wav", dtype="float32", always_2d=False)
    base_audio = np.asarray(base_audio, dtype=np.float32).reshape(-1)
    if int(sr) != sample_rate:
        raise ValueError(f"Unexpected sample rate {sr}, expected {sample_rate}")
    sf.write(case_dir / "clean_active_target.wav", clean_ref, sample_rate)
    sf.write(case_dir / "speech_activity.wav", speech_activity, sample_rate)
    bundle = compute_metric_bundle(
        clean_ref=clean_ref,
        raw_audio=raw_mix,
        processed_audio=base_audio,
        sample_rate=sample_rate,
    )
    cache_meta = {
        "split": split,
        "scene": scene_path.stem,
        "scene_layout_family": str(scene_meta["scene_layout_family"]),
        "noise_gain_scale": noise_gain_scale,
        "sample_rate_hz": sample_rate,
        "duration_s": float(base_audio.shape[0] / max(sample_rate, 1)),
        "beamforming_method": beamforming_method,
        "base_fast_rtf": float(summary.get("fast_rtf", math.nan)),
        "base_metrics": {
            "delta_sii": float(bundle.delta_sii),
            "delta_si_sdr_db": float(bundle.delta_si_sdr_db),
            "delta_snr_db": float(bundle.delta_snr_db),
        },
    }
    _write_json(case_dir / "cache_summary.json", cache_meta)
    return {
        "split": split,
        "scene_id": scene_path.stem,
        "noise_gain_scale": noise_gain_scale,
        "scene_layout_family": str(scene_meta["scene_layout_family"]),
        "cache_dir": str(case_dir.resolve()),
        "sample_rate_hz": sample_rate,
        "duration_s": cache_meta["duration_s"],
        "base_audio": base_audio,
        "clean_ref": clean_ref,
        "raw_audio": raw_mix,
        "speech_activity": speech_activity,
        "base_summary": summary,
    }


def _score_rows(rows: list[dict[str, float]], *, rtf_target: float) -> float:
    mean_delta_sii = float(np.mean([row["delta_sii"] for row in rows]))
    mean_delta_si_sdr = float(np.mean([row["delta_si_sdr_db"] for row in rows]))
    mean_rtf = float(np.mean([row["rtf"] for row in rows]))
    score = mean_delta_si_sdr + (12.0 * mean_delta_sii)
    if mean_rtf > float(rtf_target):
        score -= 80.0 * float(mean_rtf - rtf_target)
    return float(score)


def _cfg_for_trial(base_cfg: PipelineConfig, method: str, params: dict[str, Any]) -> PipelineConfig:
    cfg = PipelineConfig(**asdict(base_cfg))
    cfg.output_enhancer_mode = str(method)
    cfg.postfilter_enabled = bool(method in {"shared_wiener", "shared_wiener_rnnoise"})
    cfg.postfilter_noise_ema_alpha = float(params.get("postfilter_noise_ema_alpha", cfg.postfilter_noise_ema_alpha))
    cfg.postfilter_speech_ema_alpha = float(params.get("postfilter_speech_ema_alpha", cfg.postfilter_speech_ema_alpha))
    cfg.postfilter_gain_floor = float(params.get("postfilter_gain_floor", cfg.postfilter_gain_floor))
    cfg.postfilter_gain_ema_alpha = float(params.get("postfilter_gain_ema_alpha", cfg.postfilter_gain_ema_alpha))
    cfg.postfilter_dd_alpha = float(params.get("postfilter_dd_alpha", cfg.postfilter_dd_alpha))
    cfg.postfilter_noise_update_speech_scale = float(params.get("postfilter_noise_update_speech_scale", cfg.postfilter_noise_update_speech_scale))
    cfg.postfilter_freq_smoothing_bins = int(params.get("postfilter_freq_smoothing_bins", cfg.postfilter_freq_smoothing_bins))
    cfg.postfilter_gain_max_step_db = float(params.get("postfilter_gain_max_step_db", cfg.postfilter_gain_max_step_db))
    cfg.rnnoise_input_gain_db = float(params.get("rnnoise_input_gain_db", cfg.rnnoise_input_gain_db))
    cfg.rnnoise_wet_mix = float(params.get("rnnoise_wet_mix", cfg.rnnoise_wet_mix))
    return cfg


def _trial_params_with_defaults(trial, base_cfg: PipelineConfig) -> dict[str, Any]:
    params = dict(trial.params)
    method = str(params["method"])
    defaults: dict[str, Any] = {}
    if method in {"shared_wiener", "shared_wiener_rnnoise"}:
        defaults.update(
            {
                "postfilter_noise_ema_alpha": float(base_cfg.postfilter_noise_ema_alpha),
                "postfilter_speech_ema_alpha": float(base_cfg.postfilter_speech_ema_alpha),
                "postfilter_gain_floor": float(base_cfg.postfilter_gain_floor),
                "postfilter_gain_ema_alpha": float(base_cfg.postfilter_gain_ema_alpha),
                "postfilter_dd_alpha": float(base_cfg.postfilter_dd_alpha),
                "postfilter_noise_update_speech_scale": float(base_cfg.postfilter_noise_update_speech_scale),
                "postfilter_freq_smoothing_bins": int(base_cfg.postfilter_freq_smoothing_bins),
                "postfilter_gain_max_step_db": float(base_cfg.postfilter_gain_max_step_db),
            }
        )
    if method in {"rnnoise", "shared_wiener_rnnoise"}:
        defaults.update(
            {
                "rnnoise_input_gain_db": float(base_cfg.rnnoise_input_gain_db),
                "rnnoise_wet_mix": float(base_cfg.rnnoise_wet_mix),
            }
        )
    defaults.update(params)
    return defaults


def _suggest_params(trial, method: str) -> dict[str, Any]:
    params: dict[str, Any] = {}
    if method in {"shared_wiener", "shared_wiener_rnnoise"}:
        params.update(
            {
                "postfilter_noise_ema_alpha": trial.suggest_float("postfilter_noise_ema_alpha", 0.03, 0.20),
                "postfilter_speech_ema_alpha": trial.suggest_float("postfilter_speech_ema_alpha", 0.05, 0.25),
                "postfilter_gain_floor": trial.suggest_float("postfilter_gain_floor", 0.08, 0.35),
                "postfilter_gain_ema_alpha": trial.suggest_float("postfilter_gain_ema_alpha", 0.10, 0.60),
                "postfilter_dd_alpha": trial.suggest_float("postfilter_dd_alpha", 0.85, 0.98),
                "postfilter_noise_update_speech_scale": trial.suggest_float("postfilter_noise_update_speech_scale", 0.02, 0.40),
                "postfilter_freq_smoothing_bins": trial.suggest_int("postfilter_freq_smoothing_bins", 0, 4),
                "postfilter_gain_max_step_db": trial.suggest_float("postfilter_gain_max_step_db", 1.0, 6.0),
            }
        )
    if method in {"rnnoise", "shared_wiener_rnnoise"}:
        params.update(
            {
                "rnnoise_input_gain_db": trial.suggest_float("rnnoise_input_gain_db", -6.0, 6.0),
                "rnnoise_wet_mix": trial.suggest_float("rnnoise_wet_mix", 0.55, 1.0),
            }
        )
    return params


def _evaluate_method(
    *,
    cases: list[CachedCase],
    base_cfg: PipelineConfig,
    method: str,
    params: dict[str, Any],
) -> tuple[list[dict[str, float]], dict[str, Any]]:
    cfg = _cfg_for_trial(base_cfg, method, params)
    frame_samples = max(1, int(round(cfg.sample_rate_hz * (float(cfg.fast_frame_ms) / 1000.0))))
    rows: list[dict[str, float]] = []
    started = perf_counter()
    rnnoise_backend = ""
    rnnoise_available = True
    rnnoise_error = ""
    for case in cases:
        enhanced, meta = apply_output_enhancer_audio(
            case.base_audio,
            cfg=cfg,
            frame_samples=frame_samples,
            speech_activity=case.speech_activity,
        )
        bundle = compute_metric_bundle(
            clean_ref=case.clean_ref,
            raw_audio=case.raw_audio,
            processed_audio=enhanced,
            sample_rate=case.sample_rate_hz,
        )
        rnnoise_backend = meta.rnnoise_backend or rnnoise_backend
        rnnoise_available = rnnoise_available and bool(meta.rnnoise_available or method not in {"rnnoise", "shared_wiener_rnnoise"})
        rnnoise_error = rnnoise_error or meta.rnnoise_error
        rows.append(
            {
                "delta_sii": float(bundle.delta_sii),
                "delta_si_sdr_db": float(bundle.delta_si_sdr_db),
                "delta_snr_db": float(bundle.delta_snr_db),
                "scene_noise_key": f"{case.scene_id}_{case.noise_gain_scale}",
                "duration_s": float(case.duration_s),
            }
        )
    elapsed = max(perf_counter() - started, 1e-9)
    total_duration = max(sum(case.duration_s for case in cases), 1e-9)
    mean_rtf = float(elapsed / total_duration)
    for row in rows:
        row["rtf"] = mean_rtf
    return rows, {
        "rnnoise_backend": rnnoise_backend,
        "rnnoise_available": rnnoise_available,
        "rnnoise_error": rnnoise_error,
        "mean_rtf": mean_rtf,
    }


def _build_case_objects(raw_cases: list[dict[str, Any]]) -> list[CachedCase]:
    return [
        CachedCase(
            split=str(item["split"]),
            scene_id=str(item["scene_id"]),
            noise_gain_scale=float(item["noise_gain_scale"]),
            scene_layout_family=str(item["scene_layout_family"]),
            base_audio=np.asarray(item["base_audio"], dtype=np.float32),
            clean_ref=np.asarray(item["clean_ref"], dtype=np.float32),
            raw_audio=np.asarray(item["raw_audio"], dtype=np.float32),
            speech_activity=np.asarray(item["speech_activity"], dtype=np.float32),
            sample_rate_hz=int(item["sample_rate_hz"]),
            duration_s=float(item["duration_s"]),
            cache_dir=Path(item["cache_dir"]),
            base_summary=dict(item["base_summary"]),
        )
        for item in raw_cases
    ]


def main() -> None:
    args = _parse_args()
    optuna = _get_optuna()
    run_id = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.out_root) / run_id
    out_root.mkdir(parents=True, exist_ok=True)
    study_db_path = Path(args.study_db).resolve() if args.study_db else (out_root / "study.db").resolve()
    study_db_path.parent.mkdir(parents=True, exist_ok=True)

    scenes = sorted(Path(args.scenes_root).glob("*.json"))[: max(1, int(args.max_scenes))]
    if not scenes:
        raise FileNotFoundError(f"No scenes found under {args.scenes_root}")
    train_scenes = scenes[:-1] if len(scenes) > 1 else scenes
    val_scenes = scenes[-1:] if len(scenes) > 2 else []

    cache_jobs = [
        {
            "scene_path": str(scene.resolve()),
            "assets_root": str(Path(args.assets_root).resolve()),
            "out_root": str(out_root.resolve()),
            "profile": str(args.profile),
            "noise_gain_scale": float(scale),
            "fast_frame_ms": int(args.fast_frame_ms),
            "fd_analysis_window_ms": float(args.fd_analysis_window_ms),
            "beamforming_method": str(args.beamforming_method),
            "split": ("val" if scene in val_scenes else "train"),
        }
        for scene in scenes
        for scale in args.noise_gain_scales
    ]

    cached_raw: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=max(1, int(args.cache_workers))) as pool:
        future_map = {pool.submit(_cache_case, job): (Path(job["scene_path"]).stem, job["noise_gain_scale"]) for job in cache_jobs}
        for future in as_completed(future_map):
            scene_id, scale = future_map[future]
            print(f"cached {scene_id} noise={scale}")
            cached_raw.append(future.result())
    cached_cases = _build_case_objects(sorted(cached_raw, key=lambda item: (item["split"], item["scene_id"], item["noise_gain_scale"])))
    train_cases = [case for case in cached_cases if case.split == "train"]
    val_cases = [case for case in cached_cases if case.split == "val"]
    if not train_cases:
        raise RuntimeError("No training cases were cached.")

    _write_json(
        out_root / "cache_manifest.json",
        {
            "beamforming_method": str(args.beamforming_method),
            "fast_frame_ms": int(args.fast_frame_ms),
            "fd_analysis_window_ms": float(args.fd_analysis_window_ms),
            "study_db": str(study_db_path),
            "train_cases": [str(case.cache_dir) for case in train_cases],
            "val_cases": [str(case.cache_dir) for case in val_cases],
        },
    )

    sample_rate_hz = int(train_cases[0].sample_rate_hz)
    base_req = SessionStartRequest(
        sample_rate_hz=sample_rate_hz,
        localization_hop_ms=int(args.fast_frame_ms),
        postfilter_enabled=True,
        output_enhancer_mode="shared_wiener",
    )
    base_cfg = build_pipeline_config_from_request(base_req, sample_rate_hz=sample_rate_hz, max_speakers_hint=4)

    methods = [str(method) for method in args.methods]
    for method in methods:
        if method not in {"shared_wiener", "rnnoise", "shared_wiener_rnnoise"}:
            raise ValueError(f"Unsupported method: {method}")

    def objective(trial):
        method = trial.suggest_categorical("method", methods)
        params = _suggest_params(trial, method)
        rows, meta = _evaluate_method(cases=train_cases, base_cfg=base_cfg, method=method, params=params)
        if method in {"rnnoise", "shared_wiener_rnnoise"} and not bool(meta["rnnoise_available"]):
            trial.set_user_attr("status", "rnnoise_unavailable")
            trial.set_user_attr("rnnoise_error", str(meta["rnnoise_error"]))
            return -1e9
        score = _score_rows(rows, rtf_target=float(args.rtf_target))
        trial.set_user_attr("method", method)
        trial.set_user_attr("params", params)
        trial.set_user_attr("mean_delta_sii", float(np.mean([row["delta_sii"] for row in rows])))
        trial.set_user_attr("mean_delta_si_sdr_db", float(np.mean([row["delta_si_sdr_db"] for row in rows])))
        trial.set_user_attr("mean_delta_snr_db", float(np.mean([row["delta_snr_db"] for row in rows])))
        trial.set_user_attr("mean_rtf", float(meta["mean_rtf"]))
        trial.set_user_attr("rnnoise_backend", str(meta["rnnoise_backend"]))
        return float(score)

    storage = optuna.storages.RDBStorage(
        url=_storage_url(study_db_path),
        engine_kwargs={"connect_args": {"timeout": 120}},
    )
    study = optuna.create_study(
        study_name=f"output_denoiser_{run_id}",
        storage=storage,
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=6),
    )
    if "shared_wiener" in methods:
        study.enqueue_trial({"method": "shared_wiener"})
    if "rnnoise" in methods:
        study.enqueue_trial({"method": "rnnoise"})
    if "shared_wiener_rnnoise" in methods:
        study.enqueue_trial({"method": "shared_wiener_rnnoise"})
    deadline = time.time() + max(10.0, float(args.duration_min) * 60.0)
    while time.time() < deadline:
        remaining = max(1.0, deadline - time.time())
        study.optimize(objective, n_trials=1, timeout=remaining, show_progress_bar=False, catch=(Exception,))

    trial_rows: list[dict[str, object]] = []
    for trial in study.trials:
        row = {
            "trial_number": int(trial.number),
            "state": str(trial.state.name),
            "value": float(trial.value) if trial.value is not None else float("nan"),
            "method": str(trial.params.get("method", "")),
            "params_json": json.dumps(trial.params, sort_keys=True),
            "mean_delta_sii": trial.user_attrs.get("mean_delta_sii", float("nan")),
            "mean_delta_si_sdr_db": trial.user_attrs.get("mean_delta_si_sdr_db", float("nan")),
            "mean_delta_snr_db": trial.user_attrs.get("mean_delta_snr_db", float("nan")),
            "mean_rtf": trial.user_attrs.get("mean_rtf", float("nan")),
            "rnnoise_backend": trial.user_attrs.get("rnnoise_backend", ""),
            "status": trial.user_attrs.get("status", "ok"),
        }
        trial_rows.append(row)
    _write_csv(out_root / "trial_results.csv", trial_rows)

    complete_trials = [trial for trial in study.trials if trial.state.name == "COMPLETE" and trial.value is not None]
    complete_trials.sort(key=lambda item: float(item.value), reverse=True)
    if not complete_trials:
        raise RuntimeError("No completed Optuna trials were produced.")
    best_trial = complete_trials[0]
    best_method = str(best_trial.params["method"])
    best_params = _trial_params_with_defaults(best_trial, base_cfg)
    best_train_rows, best_train_meta = _evaluate_method(cases=train_cases, base_cfg=base_cfg, method=best_method, params=best_params)
    validation_rows: list[dict[str, float]] = []
    validation_meta: dict[str, Any] = {}
    if val_cases:
        validation_rows, validation_meta = _evaluate_method(cases=val_cases, base_cfg=base_cfg, method=best_method, params=best_params)

    best_by_method: list[dict[str, object]] = []
    for method in methods:
        method_trials = [trial for trial in complete_trials if str(trial.params.get("method")) == method]
        if not method_trials:
            continue
        trial = method_trials[0]
        best_by_method.append(
            {
                "method": method,
                "trial_number": int(trial.number),
                "value": float(trial.value),
                "mean_delta_sii": trial.user_attrs.get("mean_delta_sii", float("nan")),
                "mean_delta_si_sdr_db": trial.user_attrs.get("mean_delta_si_sdr_db", float("nan")),
                "mean_delta_snr_db": trial.user_attrs.get("mean_delta_snr_db", float("nan")),
                "mean_rtf": trial.user_attrs.get("mean_rtf", float("nan")),
                "params_json": json.dumps(trial.params, sort_keys=True),
            }
        )
    _write_csv(out_root / "best_by_method.csv", best_by_method)

    best_payload = {
        "best_trial_number": int(best_trial.number),
        "best_method": best_method,
        "best_value": float(best_trial.value),
        "study_db": str(study_db_path),
        "best_params": best_params,
        "rtf_target": float(args.rtf_target),
        "beamforming_method": str(args.beamforming_method),
        "fast_frame_ms": int(args.fast_frame_ms),
        "fd_analysis_window_ms": float(args.fd_analysis_window_ms),
        "train_summary": {
            "mean_delta_sii": float(np.mean([row["delta_sii"] for row in best_train_rows])),
            "mean_delta_si_sdr_db": float(np.mean([row["delta_si_sdr_db"] for row in best_train_rows])),
            "mean_delta_snr_db": float(np.mean([row["delta_snr_db"] for row in best_train_rows])),
            "mean_rtf": float(best_train_meta["mean_rtf"]),
            "rnnoise_backend": str(best_train_meta.get("rnnoise_backend", "")),
        },
        "validation_summary": {
            "n_cases": len(validation_rows),
            "mean_delta_sii": float(np.mean([row["delta_sii"] for row in validation_rows])) if validation_rows else float("nan"),
            "mean_delta_si_sdr_db": float(np.mean([row["delta_si_sdr_db"] for row in validation_rows])) if validation_rows else float("nan"),
            "mean_delta_snr_db": float(np.mean([row["delta_snr_db"] for row in validation_rows])) if validation_rows else float("nan"),
            "mean_rtf": float(validation_meta.get("mean_rtf", float("nan"))),
        },
    }
    _write_json(out_root / "best_params.json", best_payload)
    _write_json(out_root / "pareto_like_summary.json", {"best_by_method": best_by_method, "best_overall": best_payload})

    validation_csv_rows = [
        {
            "split": "train",
            "method": best_method,
            "delta_sii": row["delta_sii"],
            "delta_si_sdr_db": row["delta_si_sdr_db"],
            "delta_snr_db": row["delta_snr_db"],
            "rtf": row["rtf"],
        }
        for row in best_train_rows
    ] + [
        {
            "split": "val",
            "method": best_method,
            "delta_sii": row["delta_sii"],
            "delta_si_sdr_db": row["delta_si_sdr_db"],
            "delta_snr_db": row["delta_snr_db"],
            "rtf": row["rtf"],
        }
        for row in validation_rows
    ]
    _write_csv(out_root / "validation_summary.csv", validation_csv_rows)
    print(json.dumps(best_payload, indent=2))


if __name__ == "__main__":
    main()

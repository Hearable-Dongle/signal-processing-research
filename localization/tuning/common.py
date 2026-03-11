from __future__ import annotations

import concurrent.futures
import json
import math
import os
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from localization.benchmark.algo_runner import run_method_on_scene
from localization.benchmark.matching import match_predictions
from localization.benchmark.metrics import compute_scene_metrics
from simulation.mic_array_profiles import mic_positions_xyz
from simulation.simulation_config import MicrophoneArray, SimulationConfig


DEFAULT_SCENES_ROOT = Path("simulation/simulations/configs/testing_specific_angles")
DEFAULT_ASSETS_ROOT = Path("simulation/simulations/assets/testing_specific_angles")
DEFAULT_PROFILE = "respeaker_v3_0457"
DEFAULT_BENCHMARK_CONFIG = Path("localization/benchmark/configs/default.json")
SUPPORTED_METHODS = ("SSZ", "SRP-PHAT", "GMDA", "MUSIC", "NormMUSIC", "CSSM", "WAVES")


@dataclass(frozen=True)
class SceneGeometry:
    scene_id: str
    scene_path: Path
    metadata_path: Path | None
    scene_type: str
    main_angle_deg: int | None
    secondary_angle_deg: int | None
    noise_layout_type: str | None
    noise_angles_deg: tuple[int, ...]


def load_benchmark_methods(path: Path = DEFAULT_BENCHMARK_CONFIG) -> dict[str, dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    methods = data.get("methods", {})
    return {str(name): dict(cfg) for name, cfg in methods.items()}


def load_scene_geometries(
    scenes_root: Path = DEFAULT_SCENES_ROOT,
    assets_root: Path = DEFAULT_ASSETS_ROOT,
) -> list[SceneGeometry]:
    scenes: list[SceneGeometry] = []
    for scene_path in sorted(Path(scenes_root).glob("*.json")):
        metadata_path = Path(assets_root) / scene_path.stem / "scenario_metadata.json"
        metadata: dict[str, Any] = {}
        if metadata_path.exists():
            with metadata_path.open("r", encoding="utf-8") as handle:
                metadata = json.load(handle)
        scenes.append(
            SceneGeometry(
                scene_id=scene_path.stem,
                scene_path=scene_path,
                metadata_path=metadata_path if metadata_path.exists() else None,
                scene_type=str(metadata.get("scene_type", "testing_specific_angles")),
                main_angle_deg=_maybe_int(metadata.get("main_angle_deg")),
                secondary_angle_deg=_maybe_int(metadata.get("secondary_angle_deg")),
                noise_layout_type=_maybe_str(metadata.get("noise_layout_type")),
                noise_angles_deg=tuple(_normalize_angle(v) for v in metadata.get("noise_angles_deg", []) if v is not None),
            )
        )
    if not scenes:
        raise FileNotFoundError(f"No scenes found under {scenes_root}")
    return scenes


def apply_profile(sim_cfg: SimulationConfig, profile: str = DEFAULT_PROFILE) -> SimulationConfig:
    rel_positions = mic_positions_xyz(profile)
    sim_cfg.microphone_array = MicrophoneArray(
        mic_center=list(sim_cfg.microphone_array.mic_center),
        mic_radius=float(np.max(np.linalg.norm(rel_positions[:, :2], axis=1))),
        mic_count=int(rel_positions.shape[0]),
        mic_positions=rel_positions.tolist(),
    )
    return sim_cfg


def window_ms_to_nfft(window_ms: int, fs: int) -> int:
    nfft = max(16, int(round(fs * (float(window_ms) / 1000.0))))
    return int(nfft)


def build_eval_config(
    *,
    base_cfg: dict[str, Any],
    fs: int,
    window_ms: int | None = None,
    overlap: float | None = None,
    freq_low_hz: int | None = None,
    freq_high_hz: int | None = None,
    extra_updates: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = dict(base_cfg)
    if window_ms is not None:
        cfg["window_ms"] = int(window_ms)
        cfg["nfft"] = window_ms_to_nfft(int(window_ms), fs)
    if overlap is not None:
        cfg["overlap"] = float(overlap)
    if freq_low_hz is not None or freq_high_hz is not None:
        current = list(cfg.get("freq_range", [200, 3000]))
        if freq_low_hz is not None:
            current[0] = int(freq_low_hz)
        if freq_high_hz is not None:
            current[1] = int(freq_high_hz)
        if current[0] >= current[1]:
            raise ValueError(f"Invalid frequency range: {current}")
        cfg["freq_range"] = current
    if extra_updates:
        cfg.update(extra_updates)
    return cfg


def stratified_scene_subset(
    scenes: list[SceneGeometry],
    *,
    per_bucket: int,
    seed: int,
) -> list[SceneGeometry]:
    if per_bucket <= 0:
        raise ValueError("per_bucket must be positive")
    rng = np.random.default_rng(seed)
    buckets: dict[tuple[int | None, str | None], list[SceneGeometry]] = {}
    for scene in scenes:
        key = (scene.main_angle_deg, scene.noise_layout_type)
        buckets.setdefault(key, []).append(scene)

    selected: list[SceneGeometry] = []
    for key in sorted(buckets.keys(), key=lambda item: (item[0] if item[0] is not None else -1, str(item[1]))):
        bucket = sorted(buckets[key], key=lambda item: item.scene_id)
        if len(bucket) <= per_bucket:
            selected.extend(bucket)
            continue
        indices = sorted(int(v) for v in rng.choice(len(bucket), size=per_bucket, replace=False))
        selected.extend(bucket[idx] for idx in indices)
    return sorted(selected, key=lambda item: item.scene_id)


def evaluate_method_on_scenes(
    *,
    method: str,
    method_cfg: dict[str, Any],
    scenes: list[SceneGeometry],
    profile: str = DEFAULT_PROFILE,
    workers: int | None = None,
) -> list[dict[str, Any]]:
    max_workers = max(1, workers or max(1, (os.cpu_count() or 1) - 1))
    if max_workers == 1:
        return [
            _evaluate_scene_job(method, dict(method_cfg), scene, profile)
            for scene in sorted(scenes, key=lambda item: item.scene_id)
        ]
    jobs = [(method, dict(method_cfg), scene, profile) for scene in scenes]
    rows: list[dict[str, Any]] = []

    def _consume(pool):
        fut_to_scene = {
            pool.submit(_evaluate_scene_job, job_method, cfg, scene, profile_name): scene
            for job_method, cfg, scene, profile_name in jobs
        }
        for future in concurrent.futures.as_completed(fut_to_scene):
            rows.append(future.result())

    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as pool:
            _consume(pool)
    except PermissionError:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            _consume(pool)

    return sorted(rows, key=lambda row: str(row.get("scene_id")))


def aggregate_method_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    if not ok_rows:
        raise ValueError("No successful rows to aggregate")

    mae_values = [float(row["mae_deg_matched"]) for row in ok_rows if row.get("mae_deg_matched") is not None]
    acc10_values = [float(row["acc_within_10deg"]) for row in ok_rows if row.get("acc_within_10deg") is not None]
    recall_values = [float(row["recall"]) for row in ok_rows if row.get("recall") is not None]
    miss_values = [float(row["misses"]) for row in ok_rows]
    false_alarm_values = [float(row["false_alarms"]) for row in ok_rows]
    runtime_values = [float(row["runtime_seconds"]) for row in ok_rows if row.get("runtime_seconds") is not None]

    angle_mae = _group_mean(ok_rows, "main_angle_deg", "mae_deg_matched")
    layout_mae = _group_mean(ok_rows, "noise_layout_type", "mae_deg_matched")

    mae_mean = _safe_mean(mae_values)
    acc10_mean = _safe_mean(acc10_values)
    recall_mean = _safe_mean(recall_values)
    miss_mean = _safe_mean(miss_values)
    false_alarm_mean = _safe_mean(false_alarm_values)
    runtime_mean = _safe_mean(runtime_values)
    angle_std = _safe_pstdev(list(angle_mae.values()))
    layout_std = _safe_pstdev(list(layout_mae.values()))

    score = balanced_robustness_score(
        mae_deg=mae_mean,
        acc10=acc10_mean,
        recall=recall_mean,
        misses_mean=miss_mean,
        false_alarms_mean=false_alarm_mean,
        angle_std=angle_std,
        layout_std=layout_std,
    )

    return {
        "n_scenes": len(ok_rows),
        "mae_deg_matched_mean": mae_mean,
        "acc_within_10deg_mean": acc10_mean,
        "recall_mean": recall_mean,
        "misses_mean": miss_mean,
        "false_alarms_mean": false_alarm_mean,
        "runtime_seconds_mean": runtime_mean,
        "angle_mae_std": angle_std,
        "layout_mae_std": layout_std,
        "balanced_score": score,
    }


def balanced_robustness_score(
    *,
    mae_deg: float | None,
    acc10: float | None,
    recall: float | None,
    misses_mean: float | None,
    false_alarms_mean: float | None,
    angle_std: float | None,
    layout_std: float | None,
) -> float:
    mae_term = 1.0 / (1.0 + max(0.0, float(mae_deg if mae_deg is not None else 180.0)))
    acc_term = _clamp01(acc10 if acc10 is not None else 0.0)
    recall_term = _clamp01(recall if recall is not None else 0.0)
    miss_term = 1.0 / (1.0 + max(0.0, float(misses_mean if misses_mean is not None else 10.0)))
    fa_term = 1.0 / (1.0 + max(0.0, float(false_alarms_mean if false_alarms_mean is not None else 10.0)))
    angle_term = 1.0 / (1.0 + max(0.0, float(angle_std if angle_std is not None else 180.0)))
    layout_term = 1.0 / (1.0 + max(0.0, float(layout_std if layout_std is not None else 180.0)))
    return (
        0.30 * mae_term
        + 0.22 * acc_term
        + 0.18 * recall_term
        + 0.12 * miss_term
        + 0.08 * fa_term
        + 0.06 * angle_term
        + 0.04 * layout_term
    )


def summarize_by_geometry(rows: list[dict[str, Any]], group_key: str) -> list[dict[str, Any]]:
    groups: dict[Any, list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        groups.setdefault(row.get(group_key), []).append(row)
    summary: list[dict[str, Any]] = []
    for key, entries in sorted(groups.items(), key=lambda item: str(item[0])):
        summary.append(
            {
                group_key: key,
                "n_scenes": len(entries),
                "mae_deg_matched_mean": _safe_mean(float(e["mae_deg_matched"]) for e in entries if e.get("mae_deg_matched") is not None),
                "acc_within_10deg_mean": _safe_mean(float(e["acc_within_10deg"]) for e in entries if e.get("acc_within_10deg") is not None),
                "recall_mean": _safe_mean(float(e["recall"]) for e in entries if e.get("recall") is not None),
                "misses_mean": _safe_mean(float(e["misses"]) for e in entries),
                "false_alarms_mean": _safe_mean(float(e["false_alarms"]) for e in entries),
            }
        )
    return summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_normalize_json(data), handle, indent=2)


def _evaluate_scene_job(method: str, method_cfg: dict[str, Any], scene: SceneGeometry, profile: str) -> dict[str, Any]:
    sim_cfg = SimulationConfig.from_file(scene.scene_path)
    sim_cfg = apply_profile(sim_cfg, profile=profile)
    result = run_method_on_scene(method, method_cfg, sim_cfg)
    match = match_predictions(result.true_doas_deg, result.estimated_doas_deg)
    metrics = compute_scene_metrics(match=match, n_true=len(result.true_doas_deg), n_pred=len(result.estimated_doas_deg))
    return {
        "scene_id": scene.scene_id,
        "scene_path": str(scene.scene_path),
        "scene_type": scene.scene_type,
        "main_angle_deg": scene.main_angle_deg,
        "secondary_angle_deg": scene.secondary_angle_deg,
        "noise_layout_type": scene.noise_layout_type,
        "noise_angles_deg": ",".join(str(v) for v in scene.noise_angles_deg),
        "method": method,
        "status": "ok",
        "runtime_seconds": result.runtime_seconds,
        "true_doas_deg": ",".join(_fmt_float(v) for v in result.true_doas_deg),
        "pred_doas_deg": ",".join(_fmt_float(v) for v in result.estimated_doas_deg),
        "matched_errors_deg": ",".join(_fmt_float(v) for v in match.matched_errors_deg),
        "mae_deg_matched": metrics.mae_deg_matched,
        "rmse_deg_matched": metrics.rmse_deg_matched,
        "median_ae_deg": metrics.median_ae_deg,
        "acc_within_5deg": metrics.acc_within_5deg,
        "acc_within_10deg": metrics.acc_within_10deg,
        "acc_within_15deg": metrics.acc_within_15deg,
        "recall": metrics.recall,
        "precision": metrics.precision,
        "f1": metrics.f1,
        "n_true": metrics.n_true,
        "n_pred": metrics.n_pred,
        "n_matched": metrics.n_matched,
        "misses": metrics.misses,
        "false_alarms": metrics.false_alarms,
        "nfft": method_cfg.get("nfft"),
        "window_ms": method_cfg.get("window_ms"),
        "overlap": method_cfg.get("overlap"),
        "freq_low_hz": (method_cfg.get("freq_range") or [None, None])[0],
        "freq_high_hz": (method_cfg.get("freq_range") or [None, None])[1],
    }


def _group_mean(rows: list[dict[str, Any]], group_key: str, metric_key: str) -> dict[Any, float]:
    groups: dict[Any, list[float]] = {}
    for row in rows:
        metric = row.get(metric_key)
        if metric is None:
            continue
        groups.setdefault(row.get(group_key), []).append(float(metric))
    return {key: float(sum(vals) / len(vals)) for key, vals in groups.items() if vals}


def _safe_mean(values) -> float | None:
    seq = list(values)
    if not seq:
        return None
    return float(sum(seq) / len(seq))


def _safe_pstdev(values: list[float]) -> float | None:
    clean = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    if len(clean) <= 1:
        return 0.0 if clean else None
    return float(statistics.pstdev(clean))


def _clamp01(value: float) -> float:
    return float(min(1.0, max(0.0, value)))


def _fmt_float(value: float) -> str:
    return f"{float(value):.6f}"


def _normalize_json(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _normalize_json(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_normalize_json(v) for v in value]
    if isinstance(value, list):
        return [_normalize_json(v) for v in value]
    return value


def _normalize_angle(value: Any) -> int:
    return int(round(float(value))) % 360


def _maybe_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _maybe_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)

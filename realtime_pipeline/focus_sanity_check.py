from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from tqdm.auto import tqdm

from simulation.simulation_config import SimulationConfig
from simulation.simulator import run_simulation
from simulation.target_policy import iter_target_source_indices

from .contracts import PipelineConfig, SpeakerGainDirection
from .orchestrator import RealtimeSpeakerPipeline
from .separation_backends import MockSeparationBackend, build_default_backend


def _normalize_angle_deg(v: float) -> float:
    return float(v % 360.0)


def _circular_distance_deg(a: float, b: float) -> float:
    return abs((float(a) - float(b) + 180.0) % 360.0 - 180.0)


def ratio_to_db(ratio: float) -> float:
    r = max(float(ratio), 1e-6)
    return float(20.0 * np.log10(r))


def compute_target_azimuth_from_scene(scene_cfg: SimulationConfig, target_source_index: int | None = None) -> float:
    target_ids = list(iter_target_source_indices(scene_cfg))
    if not target_ids:
        target_idx = 0
    elif target_source_index is None:
        target_idx = int(target_ids[0])
    else:
        target_idx = int(target_source_index)
    target_idx = int(np.clip(target_idx, 0, max(0, len(scene_cfg.audio.sources) - 1)))

    src = scene_cfg.audio.sources[target_idx]
    mic_center = scene_cfg.microphone_array.mic_center
    dx = float(src.loc[0] - mic_center[0])
    dy = float(src.loc[1] - mic_center[1])
    return _normalize_angle_deg(np.degrees(np.arctan2(dy, dx)))


def _parse_k_from_scene_name(scene_path: Path) -> int:
    stem = scene_path.stem
    marker = "_k"
    idx = stem.find(marker)
    if idx < 0:
        return -1
    rem = stem[idx + 2 :]
    digits = ""
    for ch in rem:
        if ch.isdigit():
            digits += ch
        else:
            break
    if not digits:
        return -1
    return int(digits)


def discover_sanity_scenes(
    *,
    beamforming_config: str | Path,
    scene_types: list[str],
    scenes_per_type: int,
    seed: int,
) -> list[Path]:
    cfg_path = Path(beamforming_config)
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    roots = cfg.get("scene_roots", {})

    rng = random.Random(seed)
    picked: list[Path] = []

    for scene_type in scene_types:
        root = roots.get(scene_type)
        if root is None:
            continue
        files = sorted(Path(root).glob("*.json"))
        if not files:
            continue

        by_k: dict[int, list[Path]] = defaultdict(list)
        for p in files:
            by_k[_parse_k_from_scene_name(p)].append(p)

        local: list[Path] = []
        for k in sorted(by_k.keys()):
            bucket = by_k[k]
            local.append(rng.choice(bucket))
            if len(local) >= scenes_per_type:
                break
        if len(local) < scenes_per_type:
            remaining = [p for p in files if p not in local]
            rng.shuffle(remaining)
            local.extend(remaining[: max(0, scenes_per_type - len(local))])

        picked.extend(sorted(local[:scenes_per_type]))

    return picked


@dataclass
class FocusControllerConfig:
    target_azimuth_deg: float
    boost_db: float
    lock_tolerance_deg: float = 20.0
    lock_consecutive_hits: int = 3
    min_confidence: float = 0.2
    lock_timeout_ms: float = 3000.0
    stale_allowance_ms: float = 600.0


class FocusLockController:
    def __init__(self, cfg: FocusControllerConfig):
        self.cfg = cfg
        self.mode = "direction"
        self.locked_speaker_id: int | None = None
        self._candidate_id: int | None = None
        self._candidate_hits = 0
        self._last_seen_ms: float | None = None
        self.gain_ratio_samples: list[float] = []
        self.events: list[dict[str, Any]] = []
        self.observations: list[dict[str, Any]] = []

    def initialize(self, pipe: RealtimeSpeakerPipeline) -> None:
        pipe.set_focus_control(
            focused_speaker_ids=None,
            focused_direction_deg=self.cfg.target_azimuth_deg,
            user_boost_db=self.cfg.boost_db,
        )
        self.events.append(
            {
                "timestamp_ms": 0.0,
                "event": "direction_bootstrap",
                "speaker_id": "",
                "mode": self.mode,
            }
        )

    def _pick_lock_candidate(self, speaker_map: dict[int, SpeakerGainDirection]) -> int | None:
        best_id = None
        best_dist = None
        for sid, item in speaker_map.items():
            if float(item.confidence) < self.cfg.min_confidence:
                continue
            dist = _circular_distance_deg(item.direction_degrees, self.cfg.target_azimuth_deg)
            if dist > self.cfg.lock_tolerance_deg:
                continue
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_id = int(sid)
        return best_id

    def _record_gain_ratio(self, speaker_map: dict[int, SpeakerGainDirection]) -> None:
        if len(speaker_map) < 2:
            return

        focus_sid = self.locked_speaker_id
        if focus_sid is None or focus_sid not in speaker_map:
            nearest_sid = None
            nearest_dist = None
            for sid, item in speaker_map.items():
                dist = _circular_distance_deg(item.direction_degrees, self.cfg.target_azimuth_deg)
                if nearest_dist is None or dist < nearest_dist:
                    nearest_dist = dist
                    nearest_sid = int(sid)
            focus_sid = nearest_sid

        if focus_sid is None or focus_sid not in speaker_map:
            return

        others = [v.gain_weight for sid, v in speaker_map.items() if int(sid) != int(focus_sid)]
        if not others:
            return
        denom = float(np.mean(np.asarray(others, dtype=float)))
        if denom <= 1e-9:
            return
        ratio = float(speaker_map[int(focus_sid)].gain_weight) / denom
        if np.isfinite(ratio):
            self.gain_ratio_samples.append(ratio)

    def _nearest_to_target(self, speaker_map: dict[int, SpeakerGainDirection]) -> tuple[int | None, float]:
        nearest_sid = None
        nearest_dist = float("nan")
        for sid, item in speaker_map.items():
            dist = _circular_distance_deg(item.direction_degrees, self.cfg.target_azimuth_deg)
            if nearest_sid is None or dist < nearest_dist:
                nearest_sid = int(sid)
                nearest_dist = float(dist)
        return nearest_sid, nearest_dist

    def on_frame(self, *, frame_idx: int, timestamp_ms: float, pipe: RealtimeSpeakerPipeline) -> None:
        speaker_map = dict(pipe.shared_state.get_speaker_map_snapshot())
        nearest_sid, nearest_dist = self._nearest_to_target(speaker_map)
        self.observations.append(
            {
                "frame_idx": int(frame_idx),
                "timestamp_ms": float(timestamp_ms),
                "mode": self.mode,
                "locked_speaker_id": "" if self.locked_speaker_id is None else int(self.locked_speaker_id),
                "nearest_speaker_id": "" if nearest_sid is None else int(nearest_sid),
                "nearest_dist_deg": float(nearest_dist) if np.isfinite(nearest_dist) else float("nan"),
            }
        )

        if self.mode == "direction":
            cand = self._pick_lock_candidate(speaker_map)
            if cand is None:
                self._candidate_id = None
                self._candidate_hits = 0
            elif self._candidate_id == cand:
                self._candidate_hits += 1
            else:
                self._candidate_id = cand
                self._candidate_hits = 1

            if self._candidate_id is not None and self._candidate_hits >= self.cfg.lock_consecutive_hits:
                self.locked_speaker_id = int(self._candidate_id)
                self.mode = "speaker_id"
                self._last_seen_ms = float(timestamp_ms)
                pipe.set_focus_control(
                    focused_speaker_ids=[self.locked_speaker_id],
                    focused_direction_deg=None,
                    user_boost_db=self.cfg.boost_db,
                )
                self.events.append(
                    {
                        "timestamp_ms": float(timestamp_ms),
                        "event": "id_locked",
                        "speaker_id": self.locked_speaker_id,
                        "mode": self.mode,
                        "frame_idx": int(frame_idx),
                    }
                )
        else:
            locked_id = self.locked_speaker_id
            if locked_id is not None:
                item = speaker_map.get(locked_id)
                if item is not None:
                    age = float(timestamp_ms) - float(item.updated_at_ms)
                    if age <= self.cfg.stale_allowance_ms:
                        self._last_seen_ms = float(timestamp_ms)
                if self._last_seen_ms is None:
                    self._last_seen_ms = float(timestamp_ms)
                if float(timestamp_ms) - float(self._last_seen_ms) > self.cfg.lock_timeout_ms:
                    self.mode = "direction"
                    self.locked_speaker_id = None
                    self._candidate_id = None
                    self._candidate_hits = 0
                    pipe.set_focus_control(
                        focused_speaker_ids=None,
                        focused_direction_deg=self.cfg.target_azimuth_deg,
                        user_boost_db=self.cfg.boost_db,
                    )
                    self.events.append(
                        {
                            "timestamp_ms": float(timestamp_ms),
                            "event": "reacquire_direction",
                            "speaker_id": "",
                            "mode": self.mode,
                            "frame_idx": int(frame_idx),
                        }
                    )

        self._record_gain_ratio(speaker_map)


def _compute_catchup_metrics(observations: list[dict[str, Any]], stable_frames: int = 3) -> dict[str, Any]:
    if not observations:
        return {
            "startup_lock_ms": 0.0,
            "reacquire_catchup_ms_median": 0.0,
            "reacquire_catchup_count": 0,
            "nearest_change_catchup_ms_median": 0.0,
            "nearest_change_events": 0,
            "nearest_change_caught": 0,
        }

    rows = sorted(observations, key=lambda r: int(r["frame_idx"]))

    def _as_int_or_none(v: Any) -> int | None:
        if v == "" or v is None:
            return None
        return int(v)

    startup_lock_ms = 0.0
    for r in rows:
        if _as_int_or_none(r.get("locked_speaker_id")) is not None:
            startup_lock_ms = float(r["timestamp_ms"])
            break

    # Track stable nearest-speaker changes and measure lock catchup.
    nearest_change_events = 0
    nearest_change_caught = 0
    nearest_catchups: list[float] = []
    stable_nearest: int | None = None
    pending_target: int | None = None
    pending_t0: float | None = None

    for idx, r in enumerate(rows):
        if idx + 1 < stable_frames:
            continue
        window = rows[idx - stable_frames + 1 : idx + 1]
        nearest_vals = [_as_int_or_none(w.get("nearest_speaker_id")) for w in window]
        if nearest_vals[0] is None:
            continue
        if not all(v == nearest_vals[0] for v in nearest_vals):
            continue
        candidate = nearest_vals[0]
        if candidate is None:
            continue

        if stable_nearest is None:
            stable_nearest = candidate
        elif candidate != stable_nearest:
            stable_nearest = candidate
            nearest_change_events += 1
            pending_target = candidate
            pending_t0 = float(r["timestamp_ms"])

        if pending_target is not None and pending_t0 is not None:
            locked = _as_int_or_none(r.get("locked_speaker_id"))
            if locked == pending_target:
                nearest_change_caught += 1
                nearest_catchups.append(max(0.0, float(r["timestamp_ms"]) - pending_t0))
                pending_target = None
                pending_t0 = None

    return {
        "startup_lock_ms": float(startup_lock_ms),
        "reacquire_catchup_ms_median": 0.0,  # populated from events separately
        "reacquire_catchup_count": 0,
        "nearest_change_catchup_ms_median": float(np.median(nearest_catchups)) if nearest_catchups else 0.0,
        "nearest_change_events": int(nearest_change_events),
        "nearest_change_caught": int(nearest_change_caught),
    }


def _frame_iter_with_controller(
    mic_audio: np.ndarray,
    frame_samples: int,
    sample_rate_hz: int,
    on_frame,
):
    total = mic_audio.shape[0]
    frame_idx = 0
    for start in range(0, total, frame_samples):
        end = min(total, start + frame_samples)
        timestamp_ms = 1000.0 * float(start) / float(sample_rate_hz)
        on_frame(frame_idx=frame_idx, timestamp_ms=timestamp_ms)
        frame = mic_audio[start:end, :]
        if frame.shape[0] < frame_samples:
            frame = np.pad(frame, ((0, frame_samples - frame.shape[0]), (0, 0)))
        frame_idx += 1
        yield frame.astype(np.float32, copy=False)


def _run_scene(
    *,
    scene_path: Path,
    scene_out: Path,
    scene_repeat_index: int,
    focus_ratio: float,
    lock_timeout_ms: float,
    use_mock_separation: bool,
) -> dict[str, Any]:
    sim_cfg = SimulationConfig.from_file(scene_path)
    mic_audio, mic_pos, _src = run_simulation(sim_cfg)
    target_az = compute_target_azimuth_from_scene(sim_cfg)
    boost_db = ratio_to_db(focus_ratio)

    cfg = PipelineConfig(
        sample_rate_hz=sim_cfg.audio.fs,
        fast_frame_ms=10,
        slow_chunk_ms=200,
        max_speakers_hint=max(1, len(list(iter_target_source_indices(sim_cfg)))),
        direction_focus_gain_db=0.0,
        direction_non_focus_attenuation_db=0.0,
    )
    frame_samples = int(cfg.sample_rate_hz * cfg.fast_frame_ms / 1000)
    enhanced_parts: list[np.ndarray] = []

    def sink(x: np.ndarray) -> None:
        enhanced_parts.append(np.asarray(x, dtype=np.float32).reshape(-1))

    sep = MockSeparationBackend(n_streams=cfg.max_speakers_hint) if use_mock_separation else build_default_backend(cfg)
    controller = FocusLockController(
        FocusControllerConfig(
            target_azimuth_deg=target_az,
            boost_db=boost_db,
            lock_timeout_ms=float(lock_timeout_ms),
        )
    )

    pipe_ref: dict[str, RealtimeSpeakerPipeline] = {}

    def on_frame(frame_idx: int, timestamp_ms: float) -> None:
        if "pipe" not in pipe_ref:
            return
        controller.on_frame(frame_idx=frame_idx, timestamp_ms=timestamp_ms, pipe=pipe_ref["pipe"])

    frame_iter = _frame_iter_with_controller(
        mic_audio=mic_audio,
        frame_samples=frame_samples,
        sample_rate_hz=cfg.sample_rate_hz,
        on_frame=on_frame,
    )

    mic_geometry_xyz = np.asarray(mic_pos, dtype=float)
    mic_geometry_xy = mic_geometry_xyz[:2, :].T
    pipe = RealtimeSpeakerPipeline(
        config=cfg,
        mic_geometry_xyz=mic_geometry_xyz,
        mic_geometry_xy=mic_geometry_xy,
        frame_iterator=frame_iter,
        frame_sink=sink,
        separation_backend=sep,
    )
    pipe_ref["pipe"] = pipe
    controller.initialize(pipe)
    pipe.start()
    pipe.join()

    enhanced = np.concatenate(enhanced_parts)[: mic_audio.shape[0]] if enhanced_parts else np.zeros(mic_audio.shape[0], dtype=np.float32)
    scene_out.mkdir(parents=True, exist_ok=True)
    sf.write(scene_out / "enhanced_fast_path.wav", enhanced, cfg.sample_rate_hz)

    stats = pipe.stats_snapshot()
    clipping_fraction = float(np.mean(np.abs(enhanced) >= 0.999)) if enhanced.size else 0.0
    ratio_med = float(np.median(controller.gain_ratio_samples)) if controller.gain_ratio_samples else 0.0
    lock_success = any(e.get("event") == "id_locked" for e in controller.events)
    reacquire_count = sum(1 for e in controller.events if e.get("event") == "reacquire_direction")
    reacquire_lat_ms: list[float] = []
    for idx, ev in enumerate(controller.events):
        if ev.get("event") != "reacquire_direction":
            continue
        t0 = float(ev.get("timestamp_ms", 0.0))
        for ev2 in controller.events[idx + 1 :]:
            if ev2.get("event") == "id_locked":
                reacquire_lat_ms.append(max(0.0, float(ev2.get("timestamp_ms", 0.0)) - t0))
                break
    catchup = _compute_catchup_metrics(controller.observations, stable_frames=3)

    scene_summary = {
        "scene": scene_path.stem,
        "scene_repeat_index": int(scene_repeat_index),
        "scene_run_id": f"{scene_path.stem}__r{int(scene_repeat_index):02d}",
        "scene_config": str(scene_path.resolve()),
        "target_azimuth_deg": float(target_az),
        "boost_db": float(boost_db),
        "focus_ratio_target": float(focus_ratio),
        "focus_to_nonfocus_gain_ratio_median": ratio_med,
        "focus_gain_ratio_sample_count": int(len(controller.gain_ratio_samples)),
        "focus_lock_success": bool(lock_success),
        "reacquire_count": int(reacquire_count),
        "startup_lock_ms": float(catchup["startup_lock_ms"]),
        "reacquire_catchup_ms_median": float(np.median(reacquire_lat_ms)) if reacquire_lat_ms else 0.0,
        "reacquire_catchup_count": int(len(reacquire_lat_ms)),
        "nearest_change_catchup_ms_median": float(catchup["nearest_change_catchup_ms_median"]),
        "nearest_change_events": int(catchup["nearest_change_events"]),
        "nearest_change_caught": int(catchup["nearest_change_caught"]),
        "clipping_fraction": clipping_fraction,
        "fast_frames": int(stats.fast_frames),
        "slow_chunks": int(stats.slow_chunks),
        "speaker_map_updates": int(stats.speaker_map_updates),
        "fast_avg_ms": float(stats.fast_avg_ms),
        "slow_avg_ms": float(stats.slow_avg_ms),
        "fast_rtf": float(stats.fast_rtf),
        "slow_rtf": float(stats.slow_rtf),
        "fast_stage_avg_ms": {
            "srp": float(stats.fast_srp_avg_ms),
            "beamform": float(stats.fast_beamform_avg_ms),
            "safety": float(stats.fast_safety_avg_ms),
            "sink": float(stats.fast_sink_avg_ms),
            "enqueue": float(stats.fast_enqueue_avg_ms),
        },
        "slow_stage_avg_ms": {
            "separation": float(stats.slow_separation_avg_ms),
            "identity": float(stats.slow_identity_avg_ms),
            "direction_assignment": float(stats.slow_direction_avg_ms),
            "publish": float(stats.slow_publish_avg_ms),
        },
        "enhanced_wav": str((scene_out / "enhanced_fast_path.wav").resolve()),
        "events_count": len(controller.events),
    }

    with (scene_out / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(scene_summary, f, indent=2)

    if controller.events:
        with (scene_out / "selection_trace.csv").open("w", encoding="utf-8", newline="") as f:
            fields = sorted({k for row in controller.events for k in row.keys()})
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(controller.events)

    return scene_summary


def run_focus_sanity_check(
    *,
    out_dir: str | Path,
    beamforming_config: str | Path = "beamforming/benchmark/configs/default.json",
    scene_types: list[str] | None = None,
    scenes_per_type: int = 3,
    scene_repeats: int = 1,
    seed: int = 7,
    focus_ratio: float = 2.0,
    lock_timeout_ms: float = 3000.0,
    use_mock_separation: bool = True,
    show_progress: bool = True,
    scene_paths: list[str | Path] | None = None,
) -> dict[str, Any]:
    scene_types = scene_types or ["library", "restaurant"]
    out_root = Path(out_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    if scene_paths:
        selected = [Path(p) for p in scene_paths]
    else:
        selected = discover_sanity_scenes(
            beamforming_config=beamforming_config,
            scene_types=scene_types,
            scenes_per_type=int(scenes_per_type),
            seed=int(seed),
        )

    run_items: list[tuple[Path, int]] = []
    repeats = max(1, int(scene_repeats))
    for s in selected:
        for rep in range(1, repeats + 1):
            run_items.append((s, rep))

    per_scene: list[dict[str, Any]] = []
    scene_iter = run_items
    if show_progress:
        scene_iter = tqdm(run_items, desc="Focus sanity scenes", unit="run")
    for scene_path, rep_idx in scene_iter:
        scene_out = out_root / f"{scene_path.stem}__r{rep_idx:02d}"
        scene_result = _run_scene(
            scene_path=scene_path,
            scene_out=scene_out,
            scene_repeat_index=rep_idx,
            focus_ratio=focus_ratio,
            lock_timeout_ms=lock_timeout_ms,
            use_mock_separation=use_mock_separation,
        )
        per_scene.append(scene_result)
        if show_progress and hasattr(scene_iter, "set_postfix"):
            scene_iter.set_postfix(
                {
                    "lock": int(bool(scene_result["focus_lock_success"])),
                    "ratio_med": (
                        f"{float(scene_result['focus_to_nonfocus_gain_ratio_median']):.2f}"
                        if np.isfinite(float(scene_result["focus_to_nonfocus_gain_ratio_median"]))
                        else "nan"
                    ),
                    "clip": f"{float(scene_result['clipping_fraction']):.4f}",
                }
            )

    liveness_ok = all(
        (int(r["fast_frames"]) > 0 and int(r["slow_chunks"]) > 0 and int(r["speaker_map_updates"]) > 0) for r in per_scene
    )
    clipping_ok = all(float(r["clipping_fraction"]) <= 0.005 for r in per_scene)
    focus_ok_count = sum(
        int(bool(r["focus_lock_success"]) and np.isfinite(float(r["focus_to_nonfocus_gain_ratio_median"])) and float(r["focus_to_nonfocus_gain_ratio_median"]) >= 1.6)
        for r in per_scene
    )
    required_focus_ok = int(math.ceil(0.8 * len(per_scene))) if per_scene else 0
    overall_pass = bool(liveness_ok and clipping_ok and focus_ok_count >= required_focus_ok and len(per_scene) > 0)

    summary = {
        "overall_pass": overall_pass,
        "num_scenes": len(selected),
        "scene_repeats": repeats,
        "num_runs": len(per_scene),
        "liveness_ok": bool(liveness_ok),
        "clipping_ok": bool(clipping_ok),
        "focus_ok_count": int(focus_ok_count),
        "required_focus_ok": int(required_focus_ok),
        "scenes": per_scene,
    }

    with (out_root / "per_scene_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(per_scene, f, indent=2)
    with (out_root / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Causal realtime focus-amplification sanity check")
    p.add_argument("--out-dir", default="realtime_pipeline/output/focus_sanity")
    p.add_argument("--beamforming-config", default="beamforming/benchmark/configs/default.json")
    p.add_argument("--scene-types", nargs="*", default=["library", "restaurant"])
    p.add_argument("--scenes-per-type", type=int, default=3)
    p.add_argument("--scene-repeats", type=int, default=3, help="Run each selected scene multiple times")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--focus-ratio", type=float, default=2.0)
    p.add_argument("--lock-timeout-ms", type=float, default=3000.0)
    p.add_argument("--real-separation", action="store_true")
    p.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bar")
    p.add_argument("--scene-paths", nargs="*", default=None)
    return p


def main() -> None:
    args = _build_arg_parser().parse_args()
    summary = run_focus_sanity_check(
        out_dir=args.out_dir,
        beamforming_config=args.beamforming_config,
        scene_types=list(args.scene_types),
        scenes_per_type=int(args.scenes_per_type),
        scene_repeats=int(args.scene_repeats),
        seed=int(args.seed),
        focus_ratio=float(args.focus_ratio),
        lock_timeout_ms=float(args.lock_timeout_ms),
        use_mock_separation=not bool(args.real_separation),
        show_progress=not bool(args.no_progress),
        scene_paths=None if args.scene_paths is None or len(args.scene_paths) == 0 else list(args.scene_paths),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

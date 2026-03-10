from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from time import perf_counter

import matplotlib
import numpy as np
import soundfile as sf
from scipy.optimize import linear_sum_assignment

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from direction_assignment import DirectionAssignmentConfig
from direction_assignment.metrics import angular_distance_deg, compute_id_switch_rate, compute_track_jump_rate
from direction_assignment.tracking import SpeakerObservation, aggregate_speaker_observations, update_speaker_states
from direction_assignment.visualize import plot_doa_timeline, plot_error_histogram
from realtime_pipeline.separation_backends import AsteroidConvTasNetBackend, MockSeparationBackend
from simulation.simulation_config import SimulationConfig
from simulation.simulator import run_simulation
from speaker_identity_grouping import IdentityChunkInput, IdentityConfig, SpeakerIdentityGrouper


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for row in rows for k in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _plot_grouping_trace(path: Path, chunk_times_s: list[float], oracle_to_sid_seq: dict[int, list[int | None]], title: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for oracle_id in sorted(oracle_to_sid_seq):
        ys = [np.nan if sid is None else float(sid) for sid in oracle_to_sid_seq[oracle_id]]
        ax.plot(chunk_times_s, ys, marker="o", linewidth=1.1, markersize=2.8, label=f"oracle {oracle_id}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Pred speaker id")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _scene_dir_from_config(scene_path: Path, sim_cfg: SimulationConfig) -> Path:
    for src in sim_cfg.audio.sources:
        p = src.get_absolute_path()
        if "render_assets" in p.parts:
            return p.parent.parent
    alt = scene_path.parent.parent / "assets" / scene_path.stem
    if alt.exists():
        return alt
    raise FileNotFoundError(f"Could not infer scene directory for {scene_path}")


def _load_oracle_tracks(scene_dir: Path) -> tuple[dict[int, np.ndarray], int]:
    tracks: dict[int, np.ndarray] = {}
    sample_rate = 16000
    for wav in sorted((scene_dir / "audio").glob("speaker_*_dry.wav")):
        speaker_id = int(wav.stem.split("_")[1])
        audio, sr = sf.read(wav, dtype="float32")
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        tracks[speaker_id] = np.asarray(audio, dtype=np.float32).reshape(-1)
        sample_rate = int(sr)
    if not tracks:
        raise FileNotFoundError(f"No oracle speaker dry tracks found under {scene_dir / 'audio'}")
    return tracks, sample_rate


def _true_doa_by_speaker(scene_dir: Path, sim_cfg: SimulationConfig) -> dict[int, float]:
    metadata = json.loads((scene_dir / "scenario_metadata.json").read_text(encoding="utf-8"))
    center = np.asarray(metadata["config"]["mic_array"]["mic_center_m"][:2], dtype=float)
    by_speaker: dict[int, float] = {}
    for row in metadata.get("assets", {}).get("render_segments", []):
        if row.get("classification") != "speech" or "speaker_id" not in row:
            continue
        sid = int(row["speaker_id"])
        if sid in by_speaker:
            continue
        pos = np.asarray(row["position_m"][:2], dtype=float)
        doa = float(np.degrees(np.arctan2(pos[1] - center[1], pos[0] - center[0])) % 360.0)
        by_speaker[sid] = doa
    if by_speaker:
        return by_speaker
    return {idx: float(v) for idx, v in enumerate([])}


def _oracle_doa_chunks(scene_dir: Path, *, chunk_ms: int) -> list[dict[int, float]]:
    frame_gt = json.loads((scene_dir / "frame_ground_truth.json").read_text(encoding="utf-8"))
    metadata = json.loads((scene_dir / "scenario_metadata.json").read_text(encoding="utf-8"))
    center = np.asarray(metadata["config"]["mic_array"]["mic_center_m"][:2], dtype=float)
    frames = list(frame_gt.get("frames", []))
    if not frames:
        return []
    chunk_sec = float(chunk_ms) / 1000.0
    total_time_s = max(float(frames[-1].get("end_time_s", 0.0)), 0.0)
    n_chunks = max(1, int(np.ceil(total_time_s / max(chunk_sec, 1e-6))))
    out: list[dict[int, float]] = []
    for chunk_id in range(n_chunks):
        start_s = chunk_id * chunk_sec
        end_s = start_s + chunk_sec
        angles_by_sid: dict[int, list[float]] = defaultdict(list)
        for frame in frames:
            frame_start = float(frame.get("start_time_s", 0.0))
            frame_end = float(frame.get("end_time_s", frame_start))
            if frame_end <= start_s or frame_start >= end_s:
                continue
            for sid_str, pos_xyz in frame.get("speaker_positions", {}).items():
                pos = np.asarray(pos_xyz[:2], dtype=float)
                doa = float(np.degrees(np.arctan2(pos[1] - center[1], pos[0] - center[0])) % 360.0)
                angles_by_sid[int(sid_str)].append(doa)
        chunk_map: dict[int, float] = {}
        for sid, vals in angles_by_sid.items():
            rad = np.deg2rad(np.asarray(vals, dtype=float))
            mean_ang = float(np.degrees(np.arctan2(np.mean(np.sin(rad)), np.mean(np.cos(rad)))) % 360.0)
            chunk_map[int(sid)] = mean_ang
        out.append(chunk_map)
    return out


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64).reshape(-1)
    bb = np.asarray(b, dtype=np.float64).reshape(-1)
    denom = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    if denom <= 1e-8:
        return 0.0
    return float(np.dot(aa, bb) / denom)


def _assign_streams_to_oracle_speakers(
    streams: list[np.ndarray],
    oracle_chunks: dict[int, np.ndarray],
    *,
    similarity_threshold: float = 0.12,
    rms_threshold: float = 0.003,
) -> tuple[dict[int, int | None], dict[int, float]]:
    active_oracles = [
        sid for sid, chunk in oracle_chunks.items()
        if float(np.sqrt(np.mean(np.square(np.asarray(chunk, dtype=np.float64))) + 1e-12)) >= rms_threshold
    ]
    mapping = {i: None for i in range(len(streams))}
    confidence = {i: 0.0 for i in range(len(streams))}
    if not streams or not active_oracles:
        return mapping, confidence

    sim = np.zeros((len(streams), len(active_oracles)), dtype=np.float64)
    for i, stream in enumerate(streams):
        for j, sid in enumerate(active_oracles):
            sim[i, j] = _cosine_similarity(stream, oracle_chunks[sid])

    rows, cols = linear_sum_assignment(1.0 - sim)
    for r, c in zip(rows.tolist(), cols.tolist()):
        score = float(sim[r, c])
        confidence[r] = score
        if score >= similarity_threshold:
            mapping[r] = int(active_oracles[c])
    return mapping, confidence


def _mean_or_zero(values: list[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64))) if values else 0.0


def _build_separator(spec: str, sample_rate_hz: int, expected_num_sources: int | None = None):
    if spec == "mock":
        return MockSeparationBackend(n_streams=max(1, int(expected_num_sources or 2)))
    if spec == "librimix2_8k":
        return AsteroidConvTasNetBackend(
            model_name="JorisCos/ConvTasNet_Libri2Mix_sepnoisy_8k",
            device="cpu",
            model_sample_rate_hz=8000,
            input_sample_rate_hz=sample_rate_hz,
            expected_num_sources=2,
        )
    if spec == "librimix3_8k":
        return AsteroidConvTasNetBackend(
            model_name="JorisCos/ConvTasNet_Libri3Mix_sepnoisy_8k",
            device="cpu",
            model_sample_rate_hz=8000,
            input_sample_rate_hz=sample_rate_hz,
            expected_num_sources=3,
        )
    raise ValueError(f"Unknown separator spec: {spec}")


def evaluate_scene(
    *,
    scene_path: Path,
    separator_spec: str,
    out_dir: Path,
    chunk_ms: int = 200,
    use_ground_truth_localization: bool = True,
    identity_backend: str = "mfcc_legacy",
    identity_speaker_embedding_model: str = "wavlm_base_plus_sv",
) -> dict:
    sim_cfg = SimulationConfig.from_file(scene_path)
    scene_dir = _scene_dir_from_config(scene_path, sim_cfg)
    oracle_tracks, sample_rate = _load_oracle_tracks(scene_dir)
    true_doa_by_speaker = _true_doa_by_speaker(scene_dir, sim_cfg)
    oracle_doa_by_chunk = _oracle_doa_chunks(scene_dir, chunk_ms=chunk_ms)
    raw_mic, mic_pos, _ = run_simulation(sim_cfg)
    raw_mic = np.asarray(raw_mic, dtype=np.float32)
    if raw_mic.ndim != 2:
        raise ValueError(f"unexpected raw_mic shape: {raw_mic.shape}")

    chunk_samples = max(1, int(sample_rate * chunk_ms / 1000))
    expected_num_sources = len(oracle_tracks)
    separator = _build_separator(separator_spec, sample_rate, expected_num_sources=expected_num_sources)
    grouper = SpeakerIdentityGrouper(
        IdentityConfig(
            sample_rate_hz=sample_rate,
            chunk_duration_ms=chunk_ms,
            max_speakers=max(1, expected_num_sources),
            backend=str(identity_backend),
            speaker_embedding_model=str(identity_speaker_embedding_model),
        )
    )
    direction_cfg = DirectionAssignmentConfig(sample_rate=sample_rate, chunk_ms=chunk_ms)
    direction_states = {}

    oracle_to_sid_seq: dict[int, list[int | None]] = {sid: [] for sid in sorted(oracle_tracks)}
    speaker_to_oracle_votes: dict[int, Counter[int]] = defaultdict(Counter)
    direction_rows: list[dict] = []
    grouping_rows: list[dict] = []
    track_history: dict[int, list[float]] = defaultdict(list)
    runtime_ms = {"separation": [], "grouping": [], "direction": [], "localization": []}
    oracle_doa_chunks: list[dict[int, float]] = []
    est_doa_chunks: list[dict[int, float]] = []
    chunk_times_s: list[float] = []
    all_errors: list[float] = []

    total_samples = raw_mic.shape[0]
    for chunk_id, start in enumerate(range(0, total_samples, chunk_samples)):
        end = min(total_samples, start + chunk_samples)
        raw_chunk_mc = raw_mic[start:end, :]
        if raw_chunk_mc.shape[0] < chunk_samples:
            raw_chunk_mc = np.pad(raw_chunk_mc, ((0, chunk_samples - raw_chunk_mc.shape[0]), (0, 0)))
        mono_chunk = np.mean(raw_chunk_mc, axis=1).astype(np.float32, copy=False)
        oracle_chunk_map = {
            sid: np.pad(track[start:end], (0, max(0, chunk_samples - (end - start)))).astype(np.float32, copy=False)
            if end - start < chunk_samples
            else np.asarray(track[start:end], dtype=np.float32)
            for sid, track in oracle_tracks.items()
        }

        t0 = perf_counter()
        separated = separator.separate(mono_chunk, expected_speakers=expected_num_sources)
        runtime_ms["separation"].append((perf_counter() - t0) * 1000.0)
        oracle_map, oracle_conf = _assign_streams_to_oracle_speakers(separated, oracle_chunk_map)

        t0 = perf_counter()
        identity_out = grouper.update(
            IdentityChunkInput(
                chunk_id=chunk_id,
                timestamp_ms=chunk_id * float(chunk_ms),
                sample_rate_hz=sample_rate,
                streams=separated,
            )
        )
        runtime_ms["grouping"].append((perf_counter() - t0) * 1000.0)

        for oracle_id in sorted(oracle_tracks):
            sid_match: int | None = None
            best_score = -1.0
            for stream_idx, mapped_oracle in oracle_map.items():
                if mapped_oracle != oracle_id:
                    continue
                sid = identity_out.stream_to_speaker.get(stream_idx)
                score = float(identity_out.per_stream_confidence.get(stream_idx, 0.0))
                if sid is not None and score > best_score:
                    best_score = score
                    sid_match = int(sid)
            oracle_to_sid_seq[oracle_id].append(sid_match)
            grouping_rows.append(
                {
                    "chunk_id": chunk_id,
                    "timestamp_s": start / sample_rate,
                    "oracle_speaker_id": oracle_id,
                    "pred_speaker_id": sid_match,
                    "assignment_confidence": best_score if best_score >= 0.0 else 0.0,
                }
            )

        for stream_idx, oracle_id in oracle_map.items():
            sid = identity_out.stream_to_speaker.get(stream_idx)
            if sid is None or oracle_id is None:
                continue
            speaker_to_oracle_votes[int(sid)][int(oracle_id)] += 1

        t0 = perf_counter()
        chunk_oracle_doa = oracle_doa_by_chunk[chunk_id] if chunk_id < len(oracle_doa_by_chunk) else {}
        observations: list[SpeakerObservation] = []
        stream_debug: dict[int, dict] = {}
        for stream_idx, speaker_id in identity_out.stream_to_speaker.items():
            if speaker_id is None:
                continue
            oracle_id = oracle_map.get(stream_idx)
            if oracle_id is None:
                stream_debug[int(stream_idx)] = {
                    "speaker_id": int(speaker_id),
                    "skipped": "no_oracle_match",
                }
                continue
            oracle_doa = chunk_oracle_doa.get(int(oracle_id))
            if oracle_doa is None:
                stream_debug[int(stream_idx)] = {
                    "speaker_id": int(speaker_id),
                    "oracle_speaker_id": int(oracle_id),
                    "skipped": "oracle_inactive",
                }
                continue
            conf = float(
                np.clip(
                    identity_out.per_stream_confidence.get(stream_idx, 0.0) * oracle_conf.get(stream_idx, 0.0),
                    0.0,
                    1.0,
                )
            )
            observations.append(
                SpeakerObservation(
                    speaker_id=int(speaker_id),
                    doa_deg=float(oracle_doa),
                    confidence=conf,
                )
            )
            stream_debug[int(stream_idx)] = {
                "speaker_id": int(speaker_id),
                "oracle_speaker_id": int(oracle_id),
                "doa_deg": float(oracle_doa),
                "confidence": conf,
            }
        aggregated, agg_debug = aggregate_speaker_observations(observations)
        direction_states, track_debug = update_speaker_states(
            states=direction_states,
            aggregated_obs=aggregated,
            timestamp_ms=chunk_id * float(chunk_ms),
            cfg=direction_cfg,
            srp_peaks_deg=[float(v) for v in chunk_oracle_doa.values()],
            srp_peak_scores=[1.0] * len(chunk_oracle_doa),
        )
        runtime_ms["direction"].append((perf_counter() - t0) * 1000.0)

        chunk_times_s.append(start / sample_rate)
        oracle_doa_chunks.append({int(sid): float(doa) for sid, doa in chunk_oracle_doa.items()})
        est_map: dict[int, float] = {}
        for sid, st in direction_states.items():
            sid_int = int(sid)
            doa = float(st.direction_deg)
            track_history[sid_int].append(doa)
            if not speaker_to_oracle_votes[sid_int]:
                continue
            oracle_id = int(speaker_to_oracle_votes[sid_int].most_common(1)[0][0])
            prev = est_map.get(oracle_id)
            if prev is None:
                est_map[oracle_id] = doa
            else:
                # Prefer the estimate closer to the oracle direction when two predicted IDs collapse.
                oracle_ref = chunk_oracle_doa.get(oracle_id, true_doa_by_speaker.get(oracle_id, 0.0))
                prev_err = angular_distance_deg(prev, oracle_ref)
                new_err = angular_distance_deg(doa, oracle_ref)
                if new_err < prev_err:
                    est_map[oracle_id] = doa
            oracle_ref = chunk_oracle_doa.get(oracle_id, true_doa_by_speaker.get(oracle_id, 0.0))
            direction_rows.append(
                {
                    "chunk_id": chunk_id,
                    "timestamp_s": start / sample_rate,
                    "pred_speaker_id": sid_int,
                    "oracle_speaker_id": oracle_id,
                    "doa_deg": doa,
                    "confidence": float(st.confidence),
                    "true_doa_deg": float(oracle_ref),
                    "abs_error_deg": angular_distance_deg(doa, oracle_ref),
                }
            )
            all_errors.append(angular_distance_deg(doa, oracle_ref))
        est_doa_chunks.append(est_map)

    switch_rate = compute_id_switch_rate(oracle_to_sid_seq)
    total_assignments = 0
    majority_hits = 0
    for seq in oracle_to_sid_seq.values():
        counts = Counter(v for v in seq if v is not None)
        total_assignments += sum(counts.values())
        majority_hits += counts.most_common(1)[0][1] if counts else 0
    grouping_summary = {
        "switch_rate": float(switch_rate),
        "stability": float(majority_hits / total_assignments) if total_assignments else 0.0,
        "mean_assignment_confidence": _mean_or_zero([float(r["assignment_confidence"]) for r in grouping_rows]),
        "mean_detected_speakers": float(
            np.mean(
                [
                    len({sid for sid in oracle_to_sid_seq.keys() if oracle_to_sid_seq[sid][idx] is not None})
                    for idx in range(len(chunk_times_s))
                ]
            )
        )
        if chunk_times_s
        else 0.0,
    }
    direction_summary = {
        "mae_deg": _mean_or_zero(all_errors),
        "mismatch_rate": float(np.mean(np.asarray(all_errors, dtype=np.float64) > 15.0)) if all_errors else 0.0,
        "jump_rate": float(compute_track_jump_rate(track_history)),
        "avg_separation_ms": _mean_or_zero(runtime_ms["separation"]),
        "avg_grouping_ms": _mean_or_zero(runtime_ms["grouping"]),
        "avg_localization_ms": 0.0 if use_ground_truth_localization else _mean_or_zero(runtime_ms["localization"]),
        "avg_direction_ms": _mean_or_zero(runtime_ms["direction"]),
        "slow_rtf": float(
            (_mean_or_zero(runtime_ms["separation"]) + _mean_or_zero(runtime_ms["grouping"]) + _mean_or_zero(runtime_ms["direction"]))
            / float(chunk_ms)
        ),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "grouping_timeline.csv", grouping_rows)
    _write_csv(out_dir / "direction_timeline.csv", direction_rows)
    _plot_grouping_trace(out_dir / "grouping_trace.png", chunk_times_s, oracle_to_sid_seq, f"{scene_path.stem} :: {separator_spec}")
    plot_doa_timeline(out_dir / "direction_timeline.png", chunk_times_s, oracle_doa_chunks, est_doa_chunks)
    plot_error_histogram(out_dir / "direction_error_hist.png", all_errors)

    summary = {
        "scene": scene_path.stem,
        "separator": separator_spec,
        "ground_truth_localization": bool(use_ground_truth_localization),
        "identity_backend": str(identity_backend),
        "identity_speaker_embedding_model": str(identity_speaker_embedding_model),
        "grouping": grouping_summary,
        "direction_assignment": direction_summary,
        "true_doa_by_speaker": {str(k): float(v) for k, v in true_doa_by_speaker.items()},
    }
    _write_json(out_dir / "summary.json", summary)
    return summary


def _default_scene_paths() -> list[Path]:
    root = Path("simulation/simulations/configs/restaurant_meeting_scene")
    if root.exists():
        picks = []
        for k in [2, 3, 4, 5]:
            cand = sorted(root.glob(f"restaurant_meeting_k{k}_scene*.json"))
            if cand:
                picks.append(cand[0])
        if picks:
            return picks
    alt = Path("simulation/output/restaurant_meeting_eval/configs")
    return sorted(alt.glob("restaurant_meeting_k*_scene00.json"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare grouping and direction assignment on restaurant meeting scenes.")
    parser.add_argument("--scenes", nargs="*", default=None)
    parser.add_argument("--out-dir", default="realtime_pipeline/output/meeting_separator_compare")
    parser.add_argument("--chunk-ms", type=int, default=200)
    parser.add_argument("--estimated-localization", action="store_true", help="Use estimated localization instead of oracle chunk DOAs.")
    parser.add_argument("--identity-backend", choices=["mfcc_legacy", "speaker_embed_session"], default="mfcc_legacy")
    parser.add_argument(
        "--identity-speaker-embedding-model",
        choices=["ecapa_voxceleb", "wavlm_base_sv", "wavlm_base_plus_sv"],
        default="wavlm_base_plus_sv",
    )
    args = parser.parse_args()

    scene_paths = [Path(p) for p in args.scenes] if args.scenes else _default_scene_paths()
    if not scene_paths:
        raise SystemExit("No restaurant meeting scenes found to evaluate.")

    out_root = Path(args.out_dir)
    rows: list[dict] = []
    for scene_path in scene_paths:
        k = int(scene_path.stem.split("_k", 1)[1].split("_", 1)[0])
        separator_specs = ["mock"]
        if k in {2, 3}:
            separator_specs.extend(["librimix2_8k", "librimix3_8k"])
        for spec in separator_specs:
            scene_out = out_root / scene_path.stem / spec
            summary = evaluate_scene(
                scene_path=scene_path,
                separator_spec=spec,
                out_dir=scene_out,
                chunk_ms=int(args.chunk_ms),
                use_ground_truth_localization=not bool(args.estimated_localization),
                identity_backend=str(args.identity_backend),
                identity_speaker_embedding_model=str(args.identity_speaker_embedding_model),
            )
            rows.append(
                {
                    "scene": scene_path.stem,
                    "k": k,
                    "separator": spec,
                    "identity_backend": str(args.identity_backend),
                    "identity_speaker_embedding_model": str(args.identity_speaker_embedding_model),
                    **{f"grouping_{k2}": v for k2, v in summary["grouping"].items()},
                    **{f"direction_{k2}": v for k2, v in summary["direction_assignment"].items()},
                    "summary_json": str((scene_out / "summary.json").resolve()),
                }
            )
    _write_csv(out_root / "summary.csv", rows)
    _write_json(out_root / "summary.json", {"rows": rows})
    print(json.dumps({"out_dir": str(out_root.resolve()), "rows": rows}, indent=2))


if __name__ == "__main__":
    main()

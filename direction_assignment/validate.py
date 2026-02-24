from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

from .audio_render import StitchAccumulator, write_audio_bundle
from .config import DirectionAssignmentConfig
from .engine import DirectionAssignmentEngine
from .identity_bridge import IdentityBridge, IdentityBridgeConfig
from .metrics import angular_distance_deg, per_speaker_error_rows, summarize_scene_metrics
from .payload_runtime import build_payload_for_chunk
from .scenario_synth import generate_scene
from .validation_contracts import SceneConfig
from .visualize import plot_doa_timeline, plot_error_histogram, plot_room_topdown, plot_weight_timeline


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = sorted({k for r in rows for k in r.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _oracle_estimate_map_for_chunk(
    stream_to_oracle: dict[int, int],
    stream_to_speaker: dict[int, int | None],
    speaker_dirs: dict[int, float],
    speaker_conf: dict[int, float],
) -> tuple[dict[int, float], dict[int, int | None]]:
    oracle_to_est: dict[int, tuple[float, float]] = {}
    oracle_to_sid: dict[int, int | None] = {}

    for sidx, oracle_label in stream_to_oracle.items():
        sid = stream_to_speaker.get(sidx)
        oracle_to_sid[oracle_label] = sid
        if sid is None:
            continue
        if sid not in speaker_dirs:
            continue
        conf = float(speaker_conf.get(sid, 0.0))
        doa = float(speaker_dirs[sid])
        prev = oracle_to_est.get(oracle_label)
        if prev is None or conf > prev[1]:
            oracle_to_est[oracle_label] = (doa, conf)

    return {k: v[0] for k, v in oracle_to_est.items()}, oracle_to_sid


def _parse_speaker_choices(s: str) -> list[int]:
    out = [int(x.strip()) for x in s.split(",") if x.strip()]
    if not out:
        raise ValueError("speaker choices must be non-empty")
    return out


def run_validation(args: argparse.Namespace) -> dict:
    rng = np.random.default_rng(args.seed)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.out_dir or f"direction_assignment/output/validation_{ts}").resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    run_scene_rows = []
    per_chunk_rows_all = []
    per_speaker_rows_all = []

    for scene_idx in range(args.num_scenes):
        n_speakers = int(rng.choice(args.speaker_choices))
        scfg = SceneConfig(
            sample_rate=args.sample_rate,
            duration_sec=args.duration_sec,
            chunk_ms=args.chunk_ms,
            n_speakers=n_speakers,
            n_mics=args.n_mics,
            mic_spacing_m=args.mic_spacing_m,
            noise_std=args.noise_std,
            bleed_ratio=args.bleed_ratio,
            moving_probability=args.moving_probability,
            srp_peak_noise_deg=args.srp_peak_noise_deg,
            srp_num_distractors=args.srp_num_distractors,
        )
        scene_id = f"scene_{scene_idx:03d}_k{n_speakers}"
        scene = generate_scene(scene_id=scene_id, cfg=scfg, rng=rng)

        scene_dir = out_root / scene_id
        scene_dir.mkdir(parents=True, exist_ok=True)

        ib = IdentityBridge(
            IdentityBridgeConfig(
                sample_rate_hz=args.sample_rate,
                chunk_duration_ms=args.chunk_ms,
                identity_vad_rms=args.identity_vad_rms,
                match_threshold=args.match_threshold,
                ema_alpha=args.ema_alpha,
                max_speakers=args.max_speakers,
                retire_after_chunks=args.retire_after_chunks,
                new_speaker_confidence=args.new_speaker_confidence,
                identity_mode=args.identity_mode,
                enroll_audio_manifest=args.enroll_audio_manifest,
                seed_embeddings_npz=args.seed_embeddings_npz,
            )
        )
        ib.maybe_seed(synthetic_oracle_sources=scene.oracle_sources)

        cfg = DirectionAssignmentConfig(
            sample_rate=args.sample_rate,
            chunk_ms=args.chunk_ms,
        )
        eng_on = DirectionAssignmentEngine(mic_geometry=scene.mic_geometry_xy, config=cfg)

        est_stitch = StitchAccumulator(total_samples=scene.raw_mic.shape[0])

        errors_on: list[float] = []
        errors_off: list[float] = []
        errors_by_oracle_on: dict[int, list[float]] = defaultdict(list)
        oracle_to_sid_seq: dict[int, list[int | None]] = defaultdict(list)
        track_history_on: dict[int, list[float]] = defaultdict(list)
        oracle_doa_chunks: list[dict[int, float]] = []
        est_doa_chunks_on: list[dict[int, float]] = []
        runtime_ms: list[float] = []

        chunk_times_s: list[float] = []
        target_speaker_ids_seq: list[list[int]] = []
        target_weights_seq: list[list[float]] = []
        track_rows: list[dict] = []
        per_chunk_rows: list[dict] = []

        n_chunks = len(scene.separated_streams_by_chunk)
        for chunk_id in range(n_chunks):
            start = chunk_id * scene.chunk_samples
            end = min(scene.raw_mic.shape[0], start + scene.chunk_samples)

            raw_chunk = scene.raw_mic[start:end, :]
            sep_streams = scene.separated_streams_by_chunk[chunk_id]
            stream_to_oracle = scene.stream_to_oracle_by_chunk[chunk_id]
            srp_peaks = scene.srp_peaks_by_chunk[chunk_id]
            srp_scores = scene.srp_scores_by_chunk[chunk_id]

            identity_out = ib.update(
                chunk_id=chunk_id,
                timestamp_ms=1000.0 * start / args.sample_rate,
                streams=sep_streams,
            )

            payload, payload_dbg = build_payload_for_chunk(
                chunk_id=chunk_id,
                timestamp_ms=1000.0 * start / args.sample_rate,
                raw_mic_chunk=raw_chunk,
                separated_streams=sep_streams,
                identity_out=identity_out,
                srp_peaks=srp_peaks,
                srp_scores=srp_scores,
            )

            t0 = time.perf_counter()
            out_on = eng_on.update(payload)
            # prior-off: fresh engine each chunk
            eng_off = DirectionAssignmentEngine(mic_geometry=scene.mic_geometry_xy, config=cfg)
            out_off = eng_off.update(payload)
            t1 = time.perf_counter()
            runtime_ms.append((t1 - t0) * 1000.0)

            oracle_map = {i: float(scene.oracle_doa_deg_by_chunk[chunk_id, i]) for i in range(scene.oracle_sources.shape[0])}
            est_map_on, oracle_to_sid = _oracle_estimate_map_for_chunk(
                stream_to_oracle=stream_to_oracle,
                stream_to_speaker=identity_out.stream_to_speaker,
                speaker_dirs=out_on.speaker_directions_deg,
                speaker_conf=out_on.speaker_confidence,
            )
            est_map_off, _ = _oracle_estimate_map_for_chunk(
                stream_to_oracle=stream_to_oracle,
                stream_to_speaker=identity_out.stream_to_speaker,
                speaker_dirs=out_off.speaker_directions_deg,
                speaker_conf=out_off.speaker_confidence,
            )

            for label in sorted(oracle_map.keys()):
                oracle_to_sid_seq[label].append(oracle_to_sid.get(label))
                if label in est_map_on:
                    e = angular_distance_deg(est_map_on[label], oracle_map[label])
                    errors_on.append(e)
                    errors_by_oracle_on[label].append(e)
                if label in est_map_off:
                    eoff = angular_distance_deg(est_map_off[label], oracle_map[label])
                    errors_off.append(eoff)

            for sid, doa in out_on.speaker_directions_deg.items():
                track_history_on[int(sid)].append(float(doa))

            # Audio stitching using identity assignment.
            for sidx, sid in identity_out.stream_to_speaker.items():
                if sid is None:
                    continue
                if sidx < 0 or sidx >= len(sep_streams):
                    continue
                est_stitch.add_chunk(int(sid), sep_streams[sidx], start, end)

            # Rows
            chunk_times_s.append(start / args.sample_rate)
            target_speaker_ids_seq.append(list(out_on.target_speaker_ids))
            target_weights_seq.append(list(out_on.target_weights))

            mae_on_chunk = float(np.mean([angular_distance_deg(est_map_on[k], oracle_map[k]) for k in est_map_on.keys()])) if est_map_on else np.nan
            mae_off_chunk = float(np.mean([angular_distance_deg(est_map_off[k], oracle_map[k]) for k in est_map_off.keys()])) if est_map_off else np.nan
            per_chunk_rows.append(
                {
                    "scene_id": scene_id,
                    "chunk_id": chunk_id,
                    "timestamp_s": start / args.sample_rate,
                    "num_active_speakers": len(identity_out.active_speakers),
                    "num_estimated_tracks": len(out_on.speaker_directions_deg),
                    "mae_on_chunk": mae_on_chunk,
                    "mae_off_chunk": mae_off_chunk,
                    "payload_trimmed_streams": len(payload_dbg.trimmed_stream_indices),
                    "payload_padded_streams": len(payload_dbg.padded_stream_indices),
                }
            )

            for sidx, oracle_label in stream_to_oracle.items():
                sid = identity_out.stream_to_speaker.get(sidx)
                track_rows.append(
                    {
                        "scene_id": scene_id,
                        "chunk_id": chunk_id,
                        "stream_idx": sidx,
                        "oracle_label": oracle_label,
                        "pred_speaker_id": "" if sid is None else int(sid),
                        "oracle_doa_deg": oracle_map[oracle_label],
                        "pred_doa_deg": "" if (sid is None or sid not in out_on.speaker_directions_deg) else float(out_on.speaker_directions_deg[sid]),
                        "identity_conf": float(identity_out.per_stream_confidence.get(sidx, 0.0)),
                    }
                )

            oracle_doa_chunks.append(oracle_map)
            est_doa_chunks_on.append(est_map_on)

        scene_metrics = summarize_scene_metrics(
            errors_on=errors_on,
            errors_off=errors_off,
            errors_by_oracle_on=errors_by_oracle_on,
            oracle_to_sid_seq=oracle_to_sid_seq,
            track_history_on=track_history_on,
            oracle_doa_chunks=oracle_doa_chunks,
            est_doa_chunks_on=est_doa_chunks_on,
            runtime_ms_per_chunk=runtime_ms,
            chunk_ms=float(args.chunk_ms),
        )

        run_scene_rows.append({"scene_id": scene_id, "n_speakers": n_speakers, **scene_metrics})
        per_chunk_rows_all.extend(per_chunk_rows)

        ps_rows = per_speaker_error_rows(errors_by_oracle_on)
        for r in ps_rows:
            r["scene_id"] = scene_id
        per_speaker_rows_all.extend(ps_rows)

        _write_csv(scene_dir / "per_chunk_metrics.csv", per_chunk_rows)
        _write_csv(scene_dir / "track_assignments.csv", track_rows)
        _write_csv(scene_dir / "per_speaker_metrics.csv", ps_rows)

        with (scene_dir / "scenario_metadata.json").open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "scene_id": scene_id,
                    "n_speakers": n_speakers,
                    "bootstrap": ib.bootstrap_debug,
                    "scene_config": vars(scfg),
                    "direction_config": vars(cfg),
                },
                f,
                indent=2,
            )

        if args.export_audio:
            est_tracks = est_stitch.render()
            mixture_mono = np.mean(scene.raw_mic, axis=1)
            write_audio_bundle(
                out_dir=scene_dir / "audio",
                sample_rate=args.sample_rate,
                mixture_mono=mixture_mono,
                oracle_sources=scene.oracle_sources,
                estimated_tracks=est_tracks,
            )

        if args.export_plots:
            plot_doa_timeline(scene_dir / "plots" / "doa_timeline.png", chunk_times_s, oracle_doa_chunks, est_doa_chunks_on)
            plot_error_histogram(scene_dir / "plots" / "error_histogram.png", errors_on)
            plot_room_topdown(scene_dir / "plots" / "room_topdown.png", scene.mic_geometry_xy, oracle_doa_chunks)
            plot_weight_timeline(
                scene_dir / "plots" / "weight_timeline.png",
                chunk_times_s,
                target_speaker_ids_seq,
                target_weights_seq,
            )

    _write_csv(out_root / "per_scene_metrics.csv", run_scene_rows)
    _write_csv(out_root / "per_chunk_metrics.csv", per_chunk_rows_all)
    _write_csv(out_root / "per_speaker_metrics.csv", per_speaker_rows_all)

    agg = {}
    if run_scene_rows:
        keys = [k for k in run_scene_rows[0].keys() if k not in {"scene_id", "n_speakers"}]
        for k in keys:
            vals = [float(r[k]) for r in run_scene_rows]
            agg[k] = float(np.mean(vals))

    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "config": vars(args),
        "num_scenes": len(run_scene_rows),
        "overall_metrics": agg,
    }
    with (out_root / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Direction assignment validation complete")
    print(f"  Output root: {out_root}")
    print(f"  Summary: {out_root / 'summary.json'}")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Validate direction assignment with synthetic multi-speaker scenes")
    p.add_argument("--num-scenes", type=int, default=20)
    p.add_argument("--duration-sec", type=float, default=12.0)
    p.add_argument("--sample-rate", type=int, default=16000)
    p.add_argument("--chunk-ms", type=int, default=200)
    p.add_argument("--speaker-choices", type=str, default="2,3,4")
    p.add_argument("--seed", type=int, default=7)

    p.add_argument("--n-mics", type=int, default=4)
    p.add_argument("--mic-spacing-m", type=float, default=0.04)
    p.add_argument("--noise-std", type=float, default=0.01)
    p.add_argument("--bleed-ratio", type=float, default=0.12)
    p.add_argument("--moving-probability", type=float, default=0.6)
    p.add_argument("--srp-peak-noise-deg", type=float, default=5.0)
    p.add_argument("--srp-num-distractors", type=int, default=1)

    p.add_argument("--identity-mode", choices=["online_only", "enroll_audio", "seed_embeddings"], default="online_only")
    p.add_argument("--enroll-audio-manifest", type=str, default=None)
    p.add_argument("--seed-embeddings-npz", type=str, default=None)
    p.add_argument("--identity-vad-rms", type=float, default=0.01)
    p.add_argument("--match-threshold", type=float, default=0.82)
    p.add_argument("--ema-alpha", type=float, default=0.1)
    p.add_argument("--max-speakers", type=int, default=16)
    p.add_argument("--retire-after-chunks", type=int, default=25)
    p.add_argument("--new-speaker-confidence", type=float, default=0.5)

    p.add_argument("--export-audio", action="store_true")
    p.add_argument("--export-plots", action="store_true")
    p.add_argument("--out-dir", type=str, default=None)
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    args.speaker_choices = _parse_speaker_choices(args.speaker_choices)
    run_validation(args)


if __name__ == "__main__":
    main()

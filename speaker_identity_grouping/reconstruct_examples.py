from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf

from general_utils.constants import LIBRIMIX_PATH
from speaker_identity_grouping import IdentityChunkInput, IdentityConfig, SpeakerIdentityGrouper


DEFAULT_MODEL_NAME = "JorisCos/ConvTasNet_Libri2Mix_sepnoisy_16k"


def _read_mono(path: Path) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    return audio.astype(np.float32, copy=False), int(sr)


def _pad_to_len(x: np.ndarray, target_len: int) -> np.ndarray:
    if x.shape[0] >= target_len:
        return x[:target_len]
    return np.pad(x, (0, target_len - x.shape[0]))


def _resolve_metadata_csv(root: Path, mix: str, sample_rate: int, mode: str, subset: str) -> Path:
    sr_tag = "wav16k" if sample_rate == 16000 else "wav8k"
    csv_path = root / mix / sr_tag / mode / "metadata" / f"mixture_{subset}_mix_clean.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Metadata csv not found: {csv_path}")
    return csv_path


def _load_rows(csv_path: Path, num_rows: int, seed: int) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if num_rows > 0 and len(rows) > num_rows:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(rows), size=num_rows, replace=False)
        idx.sort()
        rows = [rows[int(i)] for i in idx]
    return rows


def _source_cols(row: dict[str, str]) -> list[str]:
    cols = [k for k in row.keys() if k.startswith("source_") and k.endswith("_path")]
    cols.sort(key=lambda x: int(x.split("_")[1]))
    return cols


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def _serialize_mapping(mapping: dict[int, int | None]) -> str:
    norm = {int(k): (None if v is None else int(v)) for k, v in mapping.items()}
    return json.dumps(norm, sort_keys=True)


def _build_ola_window(n: int) -> np.ndarray:
    # Hann window with floor so very edge chunks still get non-zero weight normalization.
    w = np.hanning(max(2, n)).astype(np.float32)
    return np.maximum(w, 1e-4)


def _load_model(model_name: str, device: str):
    import torch
    from asteroid.models import ConvTasNet

    model = ConvTasNet.from_pretrained(model_name)
    model = model.eval().to(torch.device(device))
    return model


def run_reconstruction(args: argparse.Namespace) -> dict:
    import torch

    librimix_root = Path(args.librimix_root).expanduser().resolve()
    out_root = Path(args.out_dir).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    csv_path = _resolve_metadata_csv(
        root=librimix_root,
        mix=args.mix,
        sample_rate=args.sample_rate,
        mode=args.mode,
        subset=args.subset,
    )
    rows = _load_rows(csv_path, num_rows=args.num_mixtures, seed=args.seed)
    if not rows:
        raise RuntimeError(f"No rows loaded from {csv_path}")

    model = _load_model(args.model_name, args.device)

    identity_cfg = IdentityConfig(
        sample_rate_hz=args.sample_rate,
        chunk_duration_ms=int(args.chunk_ms),
        vad_rms_threshold=args.identity_vad_rms,
        match_threshold=args.match_threshold,
        ema_alpha=args.ema_alpha,
        max_speakers=args.max_speakers,
        retire_after_chunks=args.retire_after_chunks,
        new_speaker_confidence=args.new_speaker_confidence,
    )

    chunk_samples = max(1, int(args.chunk_ms * args.sample_rate / 1000.0))
    hop_samples = max(1, int(chunk_samples * args.hop_ratio))
    ola_window = _build_ola_window(chunk_samples)

    run_rows: list[dict] = []

    for mix_idx, row in enumerate(rows):
        mixture_id = row["mixture_ID"]
        mix_path = Path(row["mixture_path"])
        src_paths = [Path(row[c]) for c in _source_cols(row)]

        mix_audio, sr = _read_mono(mix_path)
        if sr != args.sample_rate:
            raise ValueError(f"Expected {args.sample_rate}Hz, got {sr}Hz: {mix_path}")

        gt_sources: list[np.ndarray] = []
        for p in src_paths:
            src, src_sr = _read_mono(p)
            if src_sr != sr:
                raise ValueError(f"Sample rate mismatch in {p}: {src_sr} != {sr}")
            gt_sources.append(src)

        n_samples = min([len(mix_audio), *[len(s) for s in gt_sources]])
        mix_audio = mix_audio[:n_samples]
        gt_sources = [s[:n_samples] for s in gt_sources]

        mix_out_dir = out_root / f"{mix_idx:03d}_{mixture_id}"
        mix_out_dir.mkdir(parents=True, exist_ok=True)

        # reference audio exports
        sf.write(mix_out_dir / "mixture.wav", mix_audio, sr)
        for i, s in enumerate(gt_sources, start=1):
            sf.write(mix_out_dir / f"gt_source_{i}.wav", s, sr)

        grouper = SpeakerIdentityGrouper(identity_cfg)

        speaker_buffers: dict[int, np.ndarray] = {}
        speaker_weights: dict[int, np.ndarray] = {}
        speaker_activity_chunks: dict[int, int] = defaultdict(int)

        chunk_rows: list[dict] = []
        sep_times_ms: list[float] = []
        identity_times_ms: list[float] = []
        stream_prev_speaker: dict[int, int | None] = {}
        stream_label_flip_events = 0
        mapping_pattern_changes = 0
        prev_mapping_serialized: str | None = None

        chunk_id = 0
        for start in range(0, n_samples, hop_samples):
            end = min(start + chunk_samples, n_samples)
            current_len = end - start

            mix_chunk = _pad_to_len(mix_audio[start:end], chunk_samples)

            t0 = time.perf_counter()
            with torch.no_grad():
                x = torch.tensor(mix_chunk, dtype=torch.float32, device=args.device).unsqueeze(0)
                est = model.separate(x)
                est_np = est.squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
            t1 = time.perf_counter()
            if est_np.ndim == 1:
                est_np = est_np[None, :]

            t2 = time.perf_counter()
            out = grouper.update(
                IdentityChunkInput(
                    chunk_id=chunk_id,
                    timestamp_ms=1000.0 * start / sr,
                    sample_rate_hz=sr,
                    streams=[est_np[i] for i in range(est_np.shape[0])],
                )
            )
            t3 = time.perf_counter()

            sep_ms = (t1 - t0) * 1000.0
            ident_ms = (t3 - t2) * 1000.0
            sep_times_ms.append(sep_ms)
            identity_times_ms.append(ident_ms)
            mapping_serialized = _serialize_mapping(out.stream_to_speaker)
            if prev_mapping_serialized is not None and mapping_serialized != prev_mapping_serialized:
                mapping_pattern_changes += 1
            prev_mapping_serialized = mapping_serialized

            w_chunk = ola_window.copy()
            if current_len < chunk_samples:
                w_chunk[current_len:] = 0.0

            for est_stream_idx in range(est_np.shape[0]):
                assigned = out.stream_to_speaker.get(est_stream_idx)
                prev_assigned = stream_prev_speaker.get(est_stream_idx)
                if prev_assigned is not None and assigned is not None and int(prev_assigned) != int(assigned):
                    stream_label_flip_events += 1
                stream_prev_speaker[est_stream_idx] = assigned
                stream_rms = float(np.sqrt(np.mean(est_np[est_stream_idx] ** 2) + 1e-12))
                conf = float(out.per_stream_confidence.get(est_stream_idx, 0.0))

                chunk_rows.append(
                    {
                        "chunk_id": chunk_id,
                        "start_sample": start,
                        "end_sample": end,
                        "est_stream": est_stream_idx,
                        "speaker_id": "" if assigned is None else int(assigned),
                        "identity_confidence": conf,
                        "stream_rms": stream_rms,
                        "new_speakers": json.dumps([int(s) for s in out.new_speakers]),
                        "retired_speakers": json.dumps([int(s) for s in out.retired_speakers]),
                        "stream_to_speaker": mapping_serialized,
                        "sep_ms": sep_ms,
                        "identity_ms": ident_ms,
                    }
                )

                if assigned is None:
                    continue

                sid = int(assigned)
                if sid not in speaker_buffers:
                    speaker_buffers[sid] = np.zeros(n_samples, dtype=np.float32)
                    speaker_weights[sid] = np.zeros(n_samples, dtype=np.float32)

                src_chunk = est_np[est_stream_idx][:current_len]
                win = w_chunk[:current_len]

                speaker_buffers[sid][start:end] += src_chunk * win
                speaker_weights[sid][start:end] += win
                speaker_activity_chunks[sid] += 1

            chunk_id += 1
            if end >= n_samples:
                break

        # finalize stitched tracks
        stitched_ids = sorted(speaker_buffers.keys())
        for sid in stitched_ids:
            sig = speaker_buffers[sid]
            w = speaker_weights[sid]
            valid = w > 1e-8
            out_sig = np.zeros_like(sig)
            out_sig[valid] = sig[valid] / w[valid]
            sf.write(mix_out_dir / f"speaker_{sid}_stitched.wav", out_sig, sr)

        chunk_csv = mix_out_dir / "chunk_mappings.csv"
        _write_csv(
            chunk_csv,
            chunk_rows,
            fieldnames=[
                "chunk_id",
                "start_sample",
                "end_sample",
                "est_stream",
                "speaker_id",
                "identity_confidence",
                "stream_rms",
                "new_speakers",
                "retired_speakers",
                "stream_to_speaker",
                "sep_ms",
                "identity_ms",
            ],
        )

        activity_csv = mix_out_dir / "speaker_activity.csv"
        activity_rows = [
            {
                "speaker_id": sid,
                "active_chunks": int(speaker_activity_chunks[sid]),
            }
            for sid in stitched_ids
        ]
        _write_csv(activity_csv, activity_rows, fieldnames=["speaker_id", "active_chunks"])

        summary = {
            "mixture_index": mix_idx,
            "mixture_id": mixture_id,
            "mixture_path": str(mix_path),
            "num_samples": n_samples,
            "sample_rate": sr,
            "num_gt_sources": len(gt_sources),
            "num_chunks": chunk_id,
            "chunk_samples": chunk_samples,
            "hop_samples": hop_samples,
            "stitched_speaker_ids": stitched_ids,
            "stream_label_flip_events": int(stream_label_flip_events),
            "mapping_pattern_changes": int(mapping_pattern_changes),
            "avg_sep_ms_per_chunk": float(np.mean(sep_times_ms)) if sep_times_ms else 0.0,
            "avg_identity_ms_per_chunk": float(np.mean(identity_times_ms)) if identity_times_ms else 0.0,
            "avg_total_ms_per_chunk": float(np.mean(np.asarray(sep_times_ms) + np.asarray(identity_times_ms)))
            if sep_times_ms
            else 0.0,
            "p95_total_ms_per_chunk": float(np.percentile(np.asarray(sep_times_ms) + np.asarray(identity_times_ms), 95))
            if sep_times_ms
            else 0.0,
        }

        with (mix_out_dir / "reconstruction_summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        run_rows.append(
            {
                "mixture_index": mix_idx,
                "mixture_id": mixture_id,
                "out_dir": str(mix_out_dir),
                "num_chunks": summary["num_chunks"],
                "num_stitched_speakers": len(stitched_ids),
                "avg_sep_ms_per_chunk": summary["avg_sep_ms_per_chunk"],
                "avg_identity_ms_per_chunk": summary["avg_identity_ms_per_chunk"],
                "avg_total_ms_per_chunk": summary["avg_total_ms_per_chunk"],
            }
        )

    _write_csv(
        out_root / "index.csv",
        run_rows,
        fieldnames=[
            "mixture_index",
            "mixture_id",
            "out_dir",
            "num_chunks",
            "num_stitched_speakers",
            "avg_sep_ms_per_chunk",
            "avg_identity_ms_per_chunk",
            "avg_total_ms_per_chunk",
        ],
    )

    run_summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "config": vars(args),
        "identity_config": asdict(identity_cfg),
        "metadata_csv": str(csv_path),
        "num_mixtures": len(run_rows),
        "index_csv": str(out_root / "index.csv"),
    }

    with (out_root / "run_summary.json").open("w", encoding="utf-8") as f:
        json.dump(run_summary, f, indent=2)

    print("Reconstruction export complete")
    print(f"  Output root: {out_root}")
    print(f"  Run summary: {out_root / 'run_summary.json'}")
    print(f"  Index: {out_root / 'index.csv'}")

    return run_summary


def _build_parser() -> argparse.ArgumentParser:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    p = argparse.ArgumentParser(description="Export stitched chunked ConvTasNet examples by speaker_id")

    p.add_argument("--librimix-root", type=str, default=str(LIBRIMIX_PATH))
    p.add_argument("--mix", type=str, default="Libri2Mix", choices=["Libri2Mix", "Libri3Mix"])
    p.add_argument("--sample-rate", type=int, default=16000, choices=[8000, 16000])
    p.add_argument("--mode", type=str, default="min", choices=["min", "max"])
    p.add_argument("--subset", type=str, default="test", choices=["test", "dev", "train-100", "train-360"])

    p.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME)
    p.add_argument("--device", type=str, default="cpu")

    p.add_argument("--chunk-ms", type=float, default=200.0)
    p.add_argument("--hop-ratio", type=float, default=0.5)
    p.add_argument("--num-mixtures", type=int, default=10)
    p.add_argument("--seed", type=int, default=7)

    p.add_argument("--identity-vad-rms", type=float, default=0.01)
    p.add_argument("--match-threshold", type=float, default=0.82)
    p.add_argument("--ema-alpha", type=float, default=0.1)
    p.add_argument("--max-speakers", type=int, default=8)
    p.add_argument("--retire-after-chunks", type=int, default=25)
    p.add_argument("--new-speaker-confidence", type=float, default=0.5)

    p.add_argument("--out-dir", type=str, default=f"speaker_identity_grouping/output/reconstruct_{ts}")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    run_reconstruction(args)


if __name__ == "__main__":
    main()

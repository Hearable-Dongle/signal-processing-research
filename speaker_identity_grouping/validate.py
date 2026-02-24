from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.optimize import linear_sum_assignment

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


def _cosine_similarity_matrix(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    # a: [N, T], b: [M, T]
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + eps)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + eps)
    return a_norm @ b_norm.T


def _resolve_metadata_csv(root: Path, mix: str, sample_rate: int, mode: str, subset: str) -> Path:
    sr_tag = "wav16k" if sample_rate == 16000 else "wav8k"
    csv_path = root / mix / sr_tag / mode / "metadata" / f"mixture_{subset}_mix_clean.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Metadata csv not found: {csv_path}")
    return csv_path


def _load_rows(csv_path: Path, max_rows: int, seed: int) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if max_rows > 0 and len(rows) > max_rows:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(rows), size=max_rows, replace=False)
        idx.sort()
        rows = [rows[int(i)] for i in idx]
    return rows


def _source_cols(row: dict[str, str]) -> list[str]:
    cols = [k for k in row.keys() if k.startswith("source_") and k.endswith("_path")]
    cols.sort(key=lambda x: int(x.split("_")[1]))
    return cols


def _align_estimates_to_sources(
    est_chunk: np.ndarray,
    src_chunk: np.ndarray,
    est_rms_gate: float,
    src_rms_gate: float,
    min_oracle_similarity: float,
) -> tuple[dict[int, int | None], dict[int, float]]:
    """
    Returns:
      est_to_src: estimated stream index -> oracle source index or None
      est_to_oracle_similarity: estimated stream index -> cosine similarity to assigned source (0 for None)
    """
    n_est, _ = est_chunk.shape
    n_src, _ = src_chunk.shape

    est_to_src: dict[int, int | None] = {i: None for i in range(n_est)}
    est_to_sim: dict[int, float] = {i: 0.0 for i in range(n_est)}

    if n_est == 0 or n_src == 0:
        return est_to_src, est_to_sim

    est_rms = np.sqrt(np.mean(est_chunk**2, axis=1))
    src_rms = np.sqrt(np.mean(src_chunk**2, axis=1))
    src_active = src_rms >= src_rms_gate

    sim = _cosine_similarity_matrix(est_chunk, src_chunk)

    rows, cols = linear_sum_assignment(-sim)
    for r, c in zip(rows.tolist(), cols.tolist()):
        score = float(sim[r, c])
        if est_rms[r] < est_rms_gate:
            continue
        if not src_active[c]:
            continue
        if score < min_oracle_similarity:
            continue
        est_to_src[r] = c
        est_to_sim[r] = score

    return est_to_src, est_to_sim


def _mode(xs: list[int]) -> int | None:
    if not xs:
        return None
    return Counter(xs).most_common(1)[0][0]


def _compute_metrics(pair_rows: list[dict]) -> dict[str, float | int]:
    """pair_rows fields: chunk_id, oracle_source, pred_speaker"""
    if not pair_rows:
        return {
            "num_pairs": 0,
            "majority_vote_accuracy": 0.0,
            "switch_rate": 0.0,
            "unique_oracle_sources": 0,
            "unique_pred_speakers": 0,
            "speaker_count_ratio": 0.0,
        }

    by_oracle: dict[int, list[dict]] = defaultdict(list)
    for r in pair_rows:
        by_oracle[int(r["oracle_source"])].append(r)

    # Majority-vote identity consistency per oracle source.
    correct = 0
    total = 0
    total_switches = 0
    total_transitions = 0

    pred_speaker_set = set()
    for src, rows in by_oracle.items():
        rows_sorted = sorted(rows, key=lambda x: int(x["chunk_id"]))
        seq = [int(r["pred_speaker"]) for r in rows_sorted]
        pred_speaker_set.update(seq)

        maj = _mode(seq)
        if maj is not None:
            correct += sum(1 for x in seq if x == maj)
            total += len(seq)

        for i in range(1, len(seq)):
            total_transitions += 1
            if seq[i] != seq[i - 1]:
                total_switches += 1

    unique_oracle = len(by_oracle)
    unique_pred = len(pred_speaker_set)

    return {
        "num_pairs": len(pair_rows),
        "majority_vote_accuracy": float(correct / total) if total > 0 else 0.0,
        "switch_rate": float(total_switches / total_transitions) if total_transitions > 0 else 0.0,
        "unique_oracle_sources": unique_oracle,
        "unique_pred_speakers": unique_pred,
        "speaker_count_ratio": float(unique_pred / unique_oracle) if unique_oracle > 0 else 0.0,
    }


def _load_model(model_name: str, device: str):
    import torch
    from asteroid.models import ConvTasNet

    model = ConvTasNet.from_pretrained(model_name)
    model = model.eval().to(torch.device(device))
    return model


def run_validation(args: argparse.Namespace) -> dict:
    import torch

    librimix_root = Path(args.librimix_root).expanduser().resolve()
    csv_path = _resolve_metadata_csv(
        root=librimix_root,
        mix=args.mix,
        sample_rate=args.sample_rate,
        mode=args.mode,
        subset=args.subset,
    )

    rows = _load_rows(csv_path, max_rows=args.max_mixtures, seed=args.seed)
    if not rows:
        raise RuntimeError(f"No rows found in {csv_path}")

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    model = _load_model(args.model_name, args.device)

    cfg = IdentityConfig(
        sample_rate_hz=args.sample_rate,
        chunk_duration_ms=args.chunk_ms,
        vad_rms_threshold=args.identity_vad_rms,
        match_threshold=args.match_threshold,
        ema_alpha=args.ema_alpha,
        max_speakers=args.max_speakers,
        retire_after_chunks=args.retire_after_chunks,
        new_speaker_confidence=args.new_speaker_confidence,
    )

    chunk_samples = max(1, int(args.chunk_ms * args.sample_rate / 1000))
    sample_pair_rows: list[dict] = []
    per_mixture_summary: list[dict] = []

    all_chunk_times_ms: list[float] = []
    all_sep_times_ms: list[float] = []
    all_identity_times_ms: list[float] = []

    for mix_idx, row in enumerate(rows):
        mixture_id = row["mixture_ID"]
        mixture_path = Path(row["mixture_path"])
        src_paths = [Path(row[c]) for c in _source_cols(row)]

        mix_wav, sr = _read_mono(mixture_path)
        if sr != args.sample_rate:
            raise ValueError(f"Expected sample rate {args.sample_rate}, got {sr} for {mixture_path}")

        src_wavs = []
        for p in src_paths:
            x, xsr = _read_mono(p)
            if xsr != sr:
                raise ValueError(f"Sample rate mismatch for {p}: {xsr} != {sr}")
            src_wavs.append(x)

        n_samples = min([mix_wav.shape[0], *[s.shape[0] for s in src_wavs]])
        mix_wav = mix_wav[:n_samples]
        src_wavs = [s[:n_samples] for s in src_wavs]

        grouper = SpeakerIdentityGrouper(cfg)

        mixture_pair_rows: list[dict] = []
        mixture_new_speakers = 0
        mixture_retired_speakers = 0

        chunk_id = 0
        for start in range(0, n_samples, chunk_samples):
            end = min(n_samples, start + chunk_samples)

            mix_chunk = _pad_to_len(mix_wav[start:end], chunk_samples)
            src_chunk = np.stack([_pad_to_len(s[start:end], chunk_samples) for s in src_wavs], axis=0)

            t0 = time.perf_counter()
            with torch.no_grad():
                in_tensor = torch.tensor(mix_chunk, dtype=torch.float32, device=args.device).unsqueeze(0)
                est = model.separate(in_tensor)
                est_chunk = est.squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
            t1 = time.perf_counter()

            if est_chunk.ndim == 1:
                est_chunk = est_chunk[None, :]

            t2 = time.perf_counter()
            identity_out = grouper.update(
                IdentityChunkInput(
                    chunk_id=chunk_id,
                    timestamp_ms=1000.0 * start / sr,
                    sample_rate_hz=sr,
                    streams=[est_chunk[i] for i in range(est_chunk.shape[0])],
                )
            )
            t3 = time.perf_counter()

            est_to_src, est_to_oracle_sim = _align_estimates_to_sources(
                est_chunk=est_chunk,
                src_chunk=src_chunk,
                est_rms_gate=args.oracle_est_rms,
                src_rms_gate=args.oracle_src_rms,
                min_oracle_similarity=args.min_oracle_similarity,
            )

            for est_idx, src_idx in est_to_src.items():
                pred_id = identity_out.stream_to_speaker.get(est_idx)
                if src_idx is None or pred_id is None:
                    continue
                rec = {
                    "mixture_index": mix_idx,
                    "mixture_id": mixture_id,
                    "chunk_id": chunk_id,
                    "est_stream": int(est_idx),
                    "oracle_source": int(src_idx),
                    "pred_speaker": int(pred_id),
                    "oracle_similarity": float(est_to_oracle_sim[est_idx]),
                    "identity_confidence": float(identity_out.per_stream_confidence.get(est_idx, 0.0)),
                }
                mixture_pair_rows.append(rec)
                sample_pair_rows.append(rec)

            mixture_new_speakers += len(identity_out.new_speakers)
            mixture_retired_speakers += len(identity_out.retired_speakers)

            sep_ms = (t1 - t0) * 1000.0
            ident_ms = (t3 - t2) * 1000.0
            all_sep_times_ms.append(sep_ms)
            all_identity_times_ms.append(ident_ms)
            all_chunk_times_ms.append(sep_ms + ident_ms)

            chunk_id += 1

        mm = _compute_metrics(mixture_pair_rows)
        mm.update(
            {
                "mixture_index": mix_idx,
                "mixture_id": mixture_id,
                "num_chunks": chunk_id,
                "num_new_speaker_events": mixture_new_speakers,
                "num_retired_speaker_events": mixture_retired_speakers,
            }
        )
        per_mixture_summary.append(mm)

    overall = _compute_metrics(sample_pair_rows)
    overall.update(
        {
            "num_mixtures": len(rows),
            "avg_sep_ms_per_chunk": float(np.mean(all_sep_times_ms)) if all_sep_times_ms else 0.0,
            "avg_identity_ms_per_chunk": float(np.mean(all_identity_times_ms)) if all_identity_times_ms else 0.0,
            "avg_total_ms_per_chunk": float(np.mean(all_chunk_times_ms)) if all_chunk_times_ms else 0.0,
            "p95_total_ms_per_chunk": float(np.percentile(all_chunk_times_ms, 95)) if all_chunk_times_ms else 0.0,
            "chunk_target_ms": float(args.chunk_ms),
            "realtime_factor_total": float(np.mean(all_chunk_times_ms) / args.chunk_ms) if all_chunk_times_ms else 0.0,
        }
    )

    run_summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "config": vars(args),
        "identity_config": asdict(cfg),
        "metadata_csv": str(csv_path),
        "overall_metrics": overall,
        "num_pair_rows": len(sample_pair_rows),
    }

    summary_json = out_dir / "summary.json"
    per_mix_csv = out_dir / "per_mixture_metrics.csv"
    pairs_csv = out_dir / "pair_rows.csv"

    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(run_summary, f, indent=2)

    with per_mix_csv.open("w", encoding="utf-8", newline="") as f:
        if per_mixture_summary:
            writer = csv.DictWriter(f, fieldnames=list(per_mixture_summary[0].keys()))
            writer.writeheader()
            writer.writerows(per_mixture_summary)

    with pairs_csv.open("w", encoding="utf-8", newline="") as f:
        if sample_pair_rows:
            writer = csv.DictWriter(f, fieldnames=list(sample_pair_rows[0].keys()))
            writer.writeheader()
            writer.writerows(sample_pair_rows)

    print("Validation complete")
    print(f"  Summary: {summary_json}")
    print(f"  Per-mixture: {per_mix_csv}")
    print(f"  Pair rows: {pairs_csv}")
    print("Overall metrics:")
    for k, v in overall.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.6f}")
        else:
            print(f"  {k}: {v}")

    return run_summary


def _build_arg_parser() -> argparse.ArgumentParser:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description="Validate speaker identity grouping using ConvTasNet + LibriMix")

    parser.add_argument("--librimix-root", type=str, default=str(LIBRIMIX_PATH))
    parser.add_argument("--mix", type=str, default="Libri2Mix", choices=["Libri2Mix", "Libri3Mix"])
    parser.add_argument("--sample-rate", type=int, default=16000, choices=[8000, 16000])
    parser.add_argument("--mode", type=str, default="min", choices=["min", "max"])
    parser.add_argument("--subset", type=str, default="test", choices=["test", "dev", "train-100", "train-360"])

    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--device", type=str, default="cpu")

    parser.add_argument("--chunk-ms", type=float, default=200.0)
    parser.add_argument("--max-mixtures", type=int, default=25, help="<=0 means use all rows")
    parser.add_argument("--seed", type=int, default=7)

    parser.add_argument("--identity-vad-rms", type=float, default=0.01)
    parser.add_argument("--match-threshold", type=float, default=0.82)
    parser.add_argument("--ema-alpha", type=float, default=0.1)
    parser.add_argument("--max-speakers", type=int, default=8)
    parser.add_argument("--retire-after-chunks", type=int, default=25)
    parser.add_argument("--new-speaker-confidence", type=float, default=0.5)

    parser.add_argument("--oracle-est-rms", type=float, default=0.005)
    parser.add_argument("--oracle-src-rms", type=float, default=0.005)
    parser.add_argument("--min-oracle-similarity", type=float, default=0.1)

    parser.add_argument(
        "--out-dir",
        type=str,
        default=f"speaker_identity_grouping/output/validation_{ts}",
    )

    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    run_validation(args)


if __name__ == "__main__":
    main()

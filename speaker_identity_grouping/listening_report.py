from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf


@dataclass
class MixtureReport:
    mixture_index: int
    mixture_id: str
    mixture_dir: Path
    num_chunks: int
    num_stitched_speakers: int
    flip_rate: float
    mapping_change_rate: float
    unassigned_rate: float
    mean_confidence_assigned: float
    instability_score: float


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _safe_float(s: str, default: float = 0.0) -> float:
    try:
        return float(s)
    except Exception:
        return default


def _safe_int(s: str, default: int = 0) -> int:
    try:
        return int(s)
    except Exception:
        return default


def _score(
    flip_rate: float,
    mapping_change_rate: float,
    unassigned_rate: float,
    mean_confidence_assigned: float,
) -> float:
    # Fixed v1 weights from planning.
    return (
        3.0 * flip_rate
        + 2.0 * mapping_change_rate
        + 2.0 * unassigned_rate
        + 1.0 * (1.0 - mean_confidence_assigned)
    )


def _parse_mapping(mapping_json: str) -> dict[int, int | None]:
    try:
        raw = json.loads(mapping_json)
        out: dict[int, int | None] = {}
        for k, v in raw.items():
            kk = int(k)
            out[kk] = None if v is None else int(v)
        return out
    except Exception:
        return {}


def _compute_mixture_metrics(mixture_dir: Path) -> tuple[MixtureReport, list[dict]]:
    summary_path = mixture_dir / "reconstruction_summary.json"
    chunk_path = mixture_dir / "chunk_mappings.csv"

    summary = _read_json(summary_path)
    rows = _read_csv(chunk_path)

    num_chunks = int(summary.get("num_chunks", 0))
    num_stitched_speakers = len(summary.get("stitched_speaker_ids", []))

    mapping_changes = int(summary.get("mapping_pattern_changes", 0))
    flips = int(summary.get("stream_label_flip_events", 0))

    total_rows = len(rows)
    if total_rows == 0:
        unassigned_rate = 1.0
        mean_conf = 0.0
    else:
        unassigned = sum(1 for r in rows if (r.get("speaker_id", "").strip() == ""))
        unassigned_rate = float(unassigned / total_rows)

        conf_vals = [_safe_float(r.get("identity_confidence", "0"), 0.0) for r in rows if r.get("speaker_id", "").strip() != ""]
        mean_conf = float(np.mean(conf_vals)) if conf_vals else 0.0

    denom_chunks = max(1, num_chunks)
    flip_rate = float(flips / denom_chunks)
    mapping_change_rate = float(mapping_changes / denom_chunks)

    report = MixtureReport(
        mixture_index=int(summary.get("mixture_index", -1)),
        mixture_id=str(summary.get("mixture_id", mixture_dir.name)),
        mixture_dir=mixture_dir,
        num_chunks=num_chunks,
        num_stitched_speakers=num_stitched_speakers,
        flip_rate=flip_rate,
        mapping_change_rate=mapping_change_rate,
        unassigned_rate=unassigned_rate,
        mean_confidence_assigned=mean_conf,
        instability_score=_score(flip_rate, mapping_change_rate, unassigned_rate, mean_conf),
    )

    return report, rows


def _collect_events(rows: list[dict[str, str]], min_conf_highlight: float) -> list[dict]:
    events: list[dict] = []
    prev_mapping: str | None = None
    prev_stream_speaker: dict[int, int | None] = {}

    # Process rows in chunk/stream order.
    rows_sorted = sorted(
        rows,
        key=lambda r: (_safe_int(r.get("chunk_id", "0"), 0), _safe_int(r.get("est_stream", "0"), 0)),
    )

    for r in rows_sorted:
        chunk_id = _safe_int(r.get("chunk_id", "0"), 0)
        est_stream = _safe_int(r.get("est_stream", "0"), 0)
        start_sample = _safe_int(r.get("start_sample", "0"), 0)
        end_sample = _safe_int(r.get("end_sample", "0"), start_sample)
        conf = _safe_float(r.get("identity_confidence", "0"), 0.0)

        sid_raw = r.get("speaker_id", "").strip()
        sid = None if sid_raw == "" else _safe_int(sid_raw, 0)

        mapping = r.get("stream_to_speaker", "")
        if prev_mapping is not None and mapping != prev_mapping:
            events.append(
                {
                    "event_type": "mapping_change",
                    "chunk_id": chunk_id,
                    "est_stream": est_stream,
                    "start_sample": start_sample,
                    "end_sample": end_sample,
                    "severity": 1.0,
                    "confidence": conf,
                }
            )
        prev_mapping = mapping

        prev_sid = prev_stream_speaker.get(est_stream)
        if prev_sid is not None and sid is not None and prev_sid != sid:
            events.append(
                {
                    "event_type": "stream_flip",
                    "chunk_id": chunk_id,
                    "est_stream": est_stream,
                    "start_sample": start_sample,
                    "end_sample": end_sample,
                    "severity": 2.0,
                    "confidence": conf,
                }
            )

        if sid is None and conf < min_conf_highlight:
            events.append(
                {
                    "event_type": "unassigned_low_conf",
                    "chunk_id": chunk_id,
                    "est_stream": est_stream,
                    "start_sample": start_sample,
                    "end_sample": end_sample,
                    "severity": 1.5 + (min_conf_highlight - conf),
                    "confidence": conf,
                }
            )

        prev_stream_speaker[est_stream] = sid

    events.sort(key=lambda e: (e["severity"], -e["confidence"]), reverse=True)
    return events


def _slice_and_write(wav: np.ndarray, sr: int, start: int, end: int, out_path: Path) -> None:
    start = max(0, start)
    end = max(start + 1, min(len(wav), end))
    sf.write(out_path, wav[start:end], sr)


def _render_snippets(
    mixture_dir: Path,
    events: list[dict],
    snippets_per_mixture: int,
    snippet_seconds: float,
    out_snip_dir: Path,
) -> list[dict]:
    mix_wav_path = mixture_dir / "mixture.wav"
    mix_wav, sr = sf.read(str(mix_wav_path), dtype="float32")
    if mix_wav.ndim > 1:
        mix_wav = np.mean(mix_wav, axis=1)

    speaker_paths = sorted(mixture_dir.glob("speaker_*_stitched.wav"))
    speaker_audio: dict[str, np.ndarray] = {}
    for p in speaker_paths:
        x, xsr = sf.read(str(p), dtype="float32")
        if x.ndim > 1:
            x = np.mean(x, axis=1)
        if xsr != sr:
            continue
        speaker_audio[p.stem] = x

    out_snip_dir.mkdir(parents=True, exist_ok=True)

    half = int((snippet_seconds * sr) / 2.0)
    manifests: list[dict] = []

    for i, event in enumerate(events[:snippets_per_mixture], start=1):
        c_start = int(event["start_sample"])
        c_end = int(event["end_sample"])
        center = (c_start + c_end) // 2
        s0 = center - half
        s1 = center + half

        tag = f"event_{i:03d}_{event['event_type']}_chunk_{int(event['chunk_id']):04d}"

        mix_out = out_snip_dir / f"{tag}_mixture.wav"
        _slice_and_write(mix_wav, sr, s0, s1, mix_out)

        speaker_outs: list[str] = []
        for stem, wav in speaker_audio.items():
            outp = out_snip_dir / f"{tag}_{stem}.wav"
            _slice_and_write(wav, sr, s0, s1, outp)
            speaker_outs.append(str(outp))

        manifests.append(
            {
                "event": event,
                "start_sample": max(0, s0),
                "end_sample": min(len(mix_wav), s1),
                "mixture_snippet": str(mix_out),
                "speaker_snippets": speaker_outs,
            }
        )

    return manifests


def run_report(args: argparse.Namespace) -> dict:
    reconstruct_dir = Path(args.reconstruct_dir).expanduser().resolve()
    if not reconstruct_dir.exists():
        raise FileNotFoundError(f"Reconstruction dir not found: {reconstruct_dir}")

    index_path = reconstruct_dir / "index.csv"
    if not index_path.exists():
        raise FileNotFoundError(f"Missing index.csv: {index_path}")

    out_dir = reconstruct_dir / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    mixture_dirs = sorted([p for p in reconstruct_dir.iterdir() if p.is_dir() and (p / "chunk_mappings.csv").exists()])

    reports: list[MixtureReport] = []
    rows_by_mix: dict[str, list[dict]] = {}

    for mdir in mixture_dirs:
        try:
            rep, rows = _compute_mixture_metrics(mdir)
            reports.append(rep)
            rows_by_mix[rep.mixture_id] = rows
        except Exception as exc:
            print(f"Warning: skipping {mdir.name}: {exc}")

    reports.sort(key=lambda r: r.instability_score, reverse=True)

    ranked_rows: list[dict] = []
    ranked_json: list[dict] = []

    snippets_root = out_dir / "snippets"

    for rank, rep in enumerate(reports, start=1):
        rows = rows_by_mix.get(rep.mixture_id, [])
        events = _collect_events(rows, min_conf_highlight=args.min_confidence_highlight)
        snip_dir = snippets_root / rep.mixture_id
        snippet_manifest = _render_snippets(
            mixture_dir=rep.mixture_dir,
            events=events,
            snippets_per_mixture=args.snippets_per_mixture,
            snippet_seconds=args.snippet_seconds,
            out_snip_dir=snip_dir,
        )

        speaker_wavs = sorted(rep.mixture_dir.glob("speaker_*_stitched.wav"))
        primary_speaker_wav = str(speaker_wavs[0]) if speaker_wavs else ""

        ranked_rows.append(
            {
                "rank": rank,
                "mixture_index": rep.mixture_index,
                "mixture_id": rep.mixture_id,
                "mixture_dir": str(rep.mixture_dir),
                "instability_score": rep.instability_score,
                "flip_rate": rep.flip_rate,
                "mapping_change_rate": rep.mapping_change_rate,
                "unassigned_rate": rep.unassigned_rate,
                "mean_confidence_assigned": rep.mean_confidence_assigned,
                "num_chunks": rep.num_chunks,
                "num_stitched_speakers": rep.num_stitched_speakers,
                "listen_path_mixture": str(rep.mixture_dir / "mixture.wav"),
                "listen_path_primary_speaker": primary_speaker_wav,
                "snippet_dir": str(snip_dir),
            }
        )

        ranked_json.append(
            {
                **ranked_rows[-1],
                "snippet_manifest": snippet_manifest,
                "top_events": events[: args.snippets_per_mixture],
            }
        )

    csv_path = out_dir / "ranked_examples.csv"
    json_path = out_dir / "ranked_examples.json"
    txt_path = out_dir / "top_listen_first.txt"

    fieldnames = [
        "rank",
        "mixture_index",
        "mixture_id",
        "mixture_dir",
        "instability_score",
        "flip_rate",
        "mapping_change_rate",
        "unassigned_rate",
        "mean_confidence_assigned",
        "num_chunks",
        "num_stitched_speakers",
        "listen_path_mixture",
        "listen_path_primary_speaker",
        "snippet_dir",
    ]
    _write_csv(csv_path, ranked_rows, fieldnames)

    payload = {
        "reconstruct_dir": str(reconstruct_dir),
        "weights": {
            "flip_rate": 3.0,
            "mapping_change_rate": 2.0,
            "unassigned_rate": 2.0,
            "one_minus_mean_confidence": 1.0,
        },
        "args": vars(args),
        "ranked": ranked_json,
    }
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    top_n = min(args.top_n, len(ranked_rows))
    with txt_path.open("w", encoding="utf-8") as f:
        f.write("Listen-First Ranking\n")
        f.write("====================\n\n")
        for row in ranked_rows[:top_n]:
            reason = (
                f"score={float(row['instability_score']):.4f}, "
                f"flip_rate={float(row['flip_rate']):.4f}, "
                f"mapping_change_rate={float(row['mapping_change_rate']):.4f}, "
                f"unassigned_rate={float(row['unassigned_rate']):.4f}, "
                f"mean_conf={float(row['mean_confidence_assigned']):.4f}"
            )
            f.write(f"#{int(row['rank'])} {row['mixture_id']}\n")
            f.write(f"  {reason}\n")
            f.write(f"  mixture: {row['listen_path_mixture']}\n")
            f.write(f"  primary speaker: {row['listen_path_primary_speaker']}\n")
            f.write(f"  snippets: {row['snippet_dir']}\n\n")

    print("Listening report complete")
    print(f"  CSV: {csv_path}")
    print(f"  JSON: {json_path}")
    print(f"  Text: {txt_path}")
    print(f"  Snippets: {snippets_root}")

    return payload


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Rank reconstructed examples by identity instability for listening")
    p.add_argument("--reconstruct-dir", type=str, required=True)
    p.add_argument("--top-n", type=int, default=10)
    p.add_argument("--snippets-per-mixture", type=int, default=3)
    p.add_argument("--snippet-seconds", type=float, default=1.2)
    p.add_argument("--min-confidence-highlight", type=float, default=0.6)
    p.add_argument("--out-subdir", type=str, default="listening_report")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    run_report(args)


if __name__ == "__main__":
    main()

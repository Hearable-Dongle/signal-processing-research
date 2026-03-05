from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from beamforming.benchmark.metrics import compute_metric_bundle, load_audio_mono
from realtime_pipeline.simulation_runner import run_simulation_pipeline
from simulation.simulation_config import SimulationConfig
from simulation.simulator import run_simulation
from simulation.target_policy import iter_target_source_indices


def _slug(v: str) -> str:
    return v.lower().replace(" ", "_").replace("/", "_")


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({k for r in rows for k in r.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def _load_ref_and_raw(scene_path: Path) -> tuple[np.ndarray, np.ndarray, int]:
    cfg = SimulationConfig.from_file(scene_path)
    mic_audio, _mic_pos, source_signals = run_simulation(cfg)
    fs = int(cfg.audio.fs)
    n = int(mic_audio.shape[0])
    ref = np.zeros(n, dtype=np.float64)
    for idx in iter_target_source_indices(cfg):
        sig = np.asarray(source_signals[idx], dtype=np.float64).reshape(-1)
        if sig.shape[0] < n:
            sig = np.pad(sig, (0, n - sig.shape[0]))
        ref += sig[:n]
    raw = np.mean(np.asarray(mic_audio, dtype=np.float64), axis=1)
    return ref, raw, fs


def _plot_method_bars(summary_rows: list[dict], out_dir: Path) -> None:
    if not summary_rows:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    methods = [r["method"] for r in summary_rows]
    dsii = [float(r["delta_sii_mean"]) for r in summary_rows]
    dsdr = [float(r["delta_si_sdr_db_mean"]) for r in summary_rows]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(methods, dsii, color="#3572A5")
    ax.set_title("Method Ranking by Delta SII")
    ax.set_ylabel("delta_sii")
    ax.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(out_dir / "method_ranking_delta_sii.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(methods, dsdr, color="#3D9970")
    ax.set_title("Method Ranking by Delta SI-SDR (dB)")
    ax.set_ylabel("delta_si_sdr_db")
    ax.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(out_dir / "method_ranking_delta_si_sdr.png", dpi=160)
    plt.close(fig)


def _plot_waveforms(scene_name: str, ref: np.ndarray, raw: np.ndarray, method_to_audio: dict[str, np.ndarray], fs: int, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    take = min(len(raw), int(2.0 * fs))
    t = np.arange(take) / float(fs)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(t, raw[:take], label="raw_mean", linewidth=1.0, alpha=0.85)
    for method, audio in method_to_audio.items():
        ax.plot(t, audio[:take], label=method, linewidth=1.0, alpha=0.85)
    ax.plot(t, ref[:take], label="clean_ref", linewidth=1.2, alpha=0.9, linestyle="--")
    ax.set_title(f"{scene_name}: first 2s waveform overlay")
    ax.set_xlabel("seconds")
    ax.set_ylabel("amplitude")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_dir / f"{_slug(scene_name)}_waveform_overlay.png", dpi=160)
    plt.close(fig)


def _plot_spectrogram(scene_name: str, method: str, audio: np.ndarray, fs: int, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.specgram(audio, NFFT=512, Fs=fs, noverlap=256, cmap="viridis")
    ax.set_title(f"{scene_name}: {method} spectrogram")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("frequency (Hz)")
    fig.tight_layout()
    fig.savefig(out_dir / f"{_slug(scene_name)}_{_slug(method)}_spectrogram.png", dpi=160)
    plt.close(fig)


def _clip_rate(x: np.ndarray, threshold: float = 0.99) -> float:
    y = np.asarray(x, dtype=np.float64)
    if y.size == 0:
        return 0.0
    return float(np.mean(np.abs(y) >= threshold))


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Compare realtime beamforming methods with metrics + visualizations.")
    p.add_argument("--scene-config-dir", default="simulation/simulations/configs/library_scene")
    p.add_argument("--max-scenes", type=int, default=3)
    p.add_argument("--methods", nargs="+", default=["mvdr_fd", "gsc_fd", "delay_sum"])
    p.add_argument("--out-dir", default="beamforming/benchmark/realtime_method_compare")
    p.add_argument("--real-separation", action="store_true")
    p.add_argument("--disable-output-normalization", action="store_true")
    p.add_argument("--allow-output-amplification", action="store_true")
    return p


def main() -> None:
    args = _build_arg_parser().parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    scene_dir = Path(args.scene_config_dir)
    scene_paths = sorted(scene_dir.glob("*.json"))
    if args.max_scenes is not None:
        scene_paths = scene_paths[: int(args.max_scenes)]

    rows: list[dict] = []
    sanity_rows: list[dict] = []
    per_scene_outputs: dict[str, dict[str, np.ndarray]] = defaultdict(dict)

    for scene_path in scene_paths:
        scene_name = scene_path.stem
        ref, raw, fs = _load_ref_and_raw(scene_path)
        for method in args.methods:
            run_dir = out_dir / "runs" / _slug(method) / scene_name
            summary = run_simulation_pipeline(
                scene_config_path=scene_path,
                out_dir=run_dir,
                use_mock_separation=not args.real_separation,
                beamforming_mode=method,
                output_normalization_enabled=not args.disable_output_normalization,
                output_allow_amplification=bool(args.allow_output_amplification),
            )
            proc, sr = load_audio_mono(str(run_dir / "enhanced_fast_path.wav"))
            n = min(len(ref), len(raw), len(proc))
            bundle = compute_metric_bundle(
                clean_ref=ref[:n],
                raw_audio=raw[:n],
                processed_audio=proc[:n],
                sample_rate=int(sr if sr > 0 else fs),
            )
            per_scene_outputs[scene_name][method] = proc[:n]
            rows.append(
                {
                    "scene": scene_name,
                    "method": method,
                    "delta_sii": float(bundle.delta_sii),
                    "delta_stoi": float(bundle.delta_stoi),
                    "delta_snr_db": float(bundle.delta_snr_db),
                    "delta_si_sdr_db": float(bundle.delta_si_sdr_db),
                    "sii_processed": float(bundle.sii_processed),
                    "stoi_processed": float(bundle.stoi_processed),
                    "snr_processed_db": float(bundle.snr_db_processed),
                    "si_sdr_processed_db": float(bundle.si_sdr_db_processed),
                    "fast_rtf": float(summary["fast_rtf"]),
                    "slow_rtf": float(summary["slow_rtf"]),
                }
            )
            sanity_rows.append(
                {
                    "scene": scene_name,
                    "method": method,
                    "clip_rate_ge_0p99": _clip_rate(proc[:n]),
                    "signal_rms": float(np.sqrt(np.mean(np.asarray(proc[:n], dtype=np.float64) ** 2) + 1e-12)),
                    "finite_audio": bool(np.all(np.isfinite(proc[:n]))),
                    "finite_metrics": bool(
                        np.isfinite(bundle.delta_sii)
                        and np.isfinite(bundle.delta_snr_db)
                        and np.isfinite(bundle.delta_si_sdr_db)
                    ),
                }
            )
            _plot_spectrogram(scene_name, method, proc[:n], int(sr if sr > 0 else fs), out_dir / "visualizations")

        _plot_waveforms(
            scene_name=scene_name,
            ref=ref,
            raw=raw,
            method_to_audio=per_scene_outputs[scene_name],
            fs=fs,
            out_dir=out_dir / "visualizations",
        )

    _write_csv(out_dir / "scene_metrics.csv", rows)
    _write_csv(out_dir / "sanity_checks.csv", sanity_rows)

    by_method: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_method[r["method"]].append(r)

    summary_rows: list[dict] = []
    for method, items in by_method.items():
        summary_rows.append(
            {
                "method": method,
                "n_scenes": len(items),
                "delta_sii_mean": float(np.mean([float(v["delta_sii"]) for v in items])),
                "delta_stoi_mean": float(np.mean([float(v["delta_stoi"]) for v in items])),
                "delta_snr_db_mean": float(np.mean([float(v["delta_snr_db"]) for v in items])),
                "delta_si_sdr_db_mean": float(np.mean([float(v["delta_si_sdr_db"]) for v in items])),
                "fast_rtf_mean": float(np.mean([float(v["fast_rtf"]) for v in items])),
                "slow_rtf_mean": float(np.mean([float(v["slow_rtf"]) for v in items])),
            }
        )
    summary_rows.sort(key=lambda x: (x["delta_sii_mean"], x["delta_si_sdr_db_mean"]), reverse=True)
    _write_csv(out_dir / "summary_by_method.csv", summary_rows)
    _plot_method_bars(summary_rows, out_dir / "visualizations")

    report = {
        "scenes": [str(p) for p in scene_paths],
        "methods": list(args.methods),
        "output_normalization_enabled": bool(not args.disable_output_normalization),
        "output_allow_amplification": bool(args.allow_output_amplification),
        "summary_rows": summary_rows,
        "sanity_failures": [
            r for r in sanity_rows if (not bool(r["finite_audio"])) or (not bool(r["finite_metrics"])) or float(r["clip_rate_ge_0p99"]) > 0.02
        ],
    }
    with (out_dir / "report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps({"out_dir": str(out_dir), "methods": list(args.methods), "n_scenes": len(scene_paths)}, indent=2))


if __name__ == "__main__":
    main()

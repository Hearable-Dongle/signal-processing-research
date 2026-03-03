from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from simulation.simulation_config import SimulationConfig
from simulation.simulator import run_simulation
from simulation.target_policy import iter_target_source_indices

from .contracts import MetricRecord, SanityArtifactRecord, SubsystemVerificationResult
from .sii_utils import compute_delta_sii


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _build_ref_audio(sim_cfg: SimulationConfig, source_signals: list[np.ndarray]) -> np.ndarray:
    min_len = min(len(s) for s in source_signals) if source_signals else 0
    ref = np.zeros(min_len, dtype=float)
    for idx in iter_target_source_indices(sim_cfg):
        s = source_signals[idx]
        ref += s[:min_len]
    return ref


def verify_beamforming(
    out_root: Path,
    scene_config: str = "simulation/simulations/configs/library_scene/library_k2_scene00.json",
) -> SubsystemVerificationResult:
    out_dir = out_root / "beamforming"
    out_dir.mkdir(parents=True, exist_ok=True)

    scene_cfg_path = Path(scene_config)
    run_dir = out_dir / "single_scene"

    cmd = [
        sys.executable,
        "-m",
        "beamforming.main",
        "--simulation-scene-file",
        str(scene_cfg_path),
        "--output",
        str(run_dir),
        "--steering-source",
        "both",
        "--steering-time",
        "both",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        return SubsystemVerificationResult(
            subsystem="beamforming",
            status="error",
            details={"stderr": proc.stderr[-4000:], "stdout": proc.stdout[-4000:]},
        )

    steering_csv = run_dir / "steering_comparison.csv"
    if not steering_csv.exists():
        return SubsystemVerificationResult(subsystem="beamforming", status="error", details={"error": "missing steering_comparison.csv"})

    rows = _read_csv(steering_csv)
    # mean deltas from raw baseline (already exported by beamforming script)
    delta_sdr = []
    delta_snr = []
    for r in rows:
        try:
            if r.get("beamformer") == "Raw Audio (Mean)":
                continue
            if r.get("delta_si_sdr_db_raw", "") != "":
                delta_sdr.append(float(r["delta_si_sdr_db_raw"]))
            if r.get("delta_snr_db_raw", "") != "":
                delta_snr.append(float(r["delta_snr_db_raw"]))
        except Exception:
            pass

    # SII from scenario audio folders (best-effort).
    sim_cfg = SimulationConfig.from_file(scene_cfg_path)
    mic_audio, _mic_pos, source_signals = run_simulation(sim_cfg)
    ref = _build_ref_audio(sim_cfg, source_signals)
    raw = np.mean(mic_audio, axis=1)[: len(ref)]

    sii_deltas = []
    audio_dirs = list(run_dir.glob("*/audio")) + list(run_dir.glob("*/*/audio"))
    seen = set()
    for ad in audio_dirs:
        if ad in seen:
            continue
        seen.add(ad)
        for p in ad.glob("*.wav"):
            name = p.name
            if "raw_audio_mean" in name or name.endswith("_norm_to_ref.wav"):
                continue
            x, sr = sf.read(str(p), dtype="float32")
            if x.ndim > 1:
                x = np.mean(x, axis=1)
            d = compute_delta_sii(ref, raw, x[: len(ref)], sr)
            sii_deltas.append(float(d["delta_sii"]))

    mean_sdr = float(np.mean(delta_sdr)) if delta_sdr else 0.0
    mean_snr = float(np.mean(delta_snr)) if delta_snr else 0.0
    med_sii = float(np.median(sii_deltas)) if sii_deltas else 0.0

    metrics = [
        MetricRecord("delta_sii_median", med_sii, True, 0.03, med_sii > 0.03),
        MetricRecord("delta_si_sdr_db_mean", mean_sdr, True, 1.0, mean_sdr > 1.0),
        MetricRecord("delta_snr_db_mean", mean_snr, True, 1.0, mean_snr > 1.0),
    ]

    artifacts = [
        SanityArtifactRecord("csv", str(steering_csv)),
        SanityArtifactRecord("audio_dir", str(run_dir)),
    ]

    status = "pass" if all(m.passed for m in metrics) else "warn"
    return SubsystemVerificationResult(
        subsystem="beamforming",
        status=status,
        metrics=metrics,
        artifacts=artifacts,
        details={
            "results_dir": str(run_dir.resolve()),
            "num_rows": len(rows),
            "num_sii_audio": len(sii_deltas),
        },
    )

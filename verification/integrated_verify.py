from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from simulation.simulation_config import SimulationConfig
from simulation.simulator import run_simulation
from simulation.target_policy import iter_target_source_indices

from .contracts import MetricRecord, SanityArtifactRecord, SubsystemVerificationResult
from .sii_utils import compute_delta_sii


def _make_ref(sim_cfg: SimulationConfig, srcs: list[np.ndarray]) -> np.ndarray:
    min_len = min(len(s) for s in srcs) if srcs else 0
    ref = np.zeros(min_len, dtype=float)
    for idx in iter_target_source_indices(sim_cfg):
        ref += srcs[idx][:min_len]
    return ref


def verify_integrated(
    out_root: Path,
    scene_config: str = "simulation/simulations/configs/library_scene/library_k2_scene00.json",
) -> SubsystemVerificationResult:
    out_dir = out_root / "integrated"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Use realtime simulation runner in mock mode for stable full-stack integration signal.
    from realtime_pipeline.simulation_runner import run_simulation_pipeline

    summary = run_simulation_pipeline(
        scene_config_path=scene_config,
        out_dir=out_dir,
        use_mock_separation=True,
    )

    enh_path = out_dir / "enhanced_fast_path.wav"
    if not enh_path.exists():
        return SubsystemVerificationResult(subsystem="integrated", status="error", details={"error": "missing enhanced_fast_path.wav"})

    sim_cfg = SimulationConfig.from_file(scene_config)
    mic_audio, _mic_pos, srcs = run_simulation(sim_cfg)
    ref = _make_ref(sim_cfg, srcs)
    raw = np.mean(mic_audio, axis=1)[: len(ref)]
    enh, sr = sf.read(str(enh_path), dtype="float32")
    if enh.ndim > 1:
        enh = np.mean(enh, axis=1)
    enh = enh[: len(ref)]

    sii = compute_delta_sii(ref, raw, enh, sr)
    delta = float(sii["delta_sii"])

    # proxy scenario placeholders for future ablation ladder.
    scenario_scores = {
        "full_realtime_mock": delta,
    }
    with (out_dir / "ablation_scores.json").open("w", encoding="utf-8") as f:
        json.dump(scenario_scores, f, indent=2)

    metrics = [
        MetricRecord("delta_sii_full", delta, True, 0.02, delta > 0.02),
        MetricRecord("fast_frames", float(summary.get("fast_frames", 0)), True, 1.0, float(summary.get("fast_frames", 0)) > 0),
        MetricRecord("slow_chunks", float(summary.get("slow_chunks", 0)), True, 1.0, float(summary.get("slow_chunks", 0)) > 0),
        MetricRecord("speaker_map_updates", float(summary.get("speaker_map_updates", 0)), True, 1.0, float(summary.get("speaker_map_updates", 0)) > 0),
    ]

    artifacts = [
        SanityArtifactRecord("json", str(out_dir / "summary.json")),
        SanityArtifactRecord("audio", str(enh_path)),
        SanityArtifactRecord("json", str(out_dir / "ablation_scores.json"), "Ablation ladder placeholder (to expand with real backends)")
    ]

    status = "pass" if all(m.passed for m in metrics) else "warn"
    return SubsystemVerificationResult(
        subsystem="integrated",
        status=status,
        metrics=metrics,
        artifacts=artifacts,
        details={"results_dir": str(out_dir.resolve()), "sii": sii},
    )

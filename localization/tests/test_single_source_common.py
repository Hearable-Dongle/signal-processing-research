from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from localization.benchmark.algo_runner import _build_algorithm
from localization.tuning.single_source_common import (
    aggregate_mixed_dataset_rows,
    load_real_recording_jobs,
)


def _write_recording(root: Path, *, recording_id: str, metadata: dict, duration_s: float = 0.2, fs: int = 48000) -> None:
    recording_dir = root / "recordings" / recording_id
    raw_dir = recording_dir / "raw"
    raw_dir.mkdir(parents=True)
    t = np.arange(int(duration_s * fs), dtype=np.float32) / float(fs)
    tone = 0.05 * np.sin(2.0 * np.pi * 440.0 * t).astype(np.float32)
    for idx in range(4):
        sf.write(raw_dir / f"channel_{idx:03d}.wav", tone, fs)
    (recording_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def test_build_algorithm_supports_single_source_backend_ids() -> None:
    mic_pos_rel = np.zeros((3, 4), dtype=np.float64)
    srp = _build_algorithm("srp_phat_localization", mic_pos_rel, 16000, 1, {"nfft": 512, "freq_range": [200, 3000]})
    capon = _build_algorithm("capon_1src", mic_pos_rel, 16000, 1, {"nfft": 512, "freq_range": [200, 3000]})
    refine = _build_algorithm("capon_mvdr_refine_1src", mic_pos_rel, 16000, 1, {"nfft": 512, "freq_range": [200, 3000]})
    assert srp.__class__.__name__ == "SRPPHATLocalization"
    assert capon.__class__.__name__ == "CaponLocalization"
    assert refine.__class__.__name__ == "CaponMVDRRefineLocalization"


def test_load_real_recording_jobs_discovers_collection_and_preserves_profile(tmp_path: Path) -> None:
    root = tmp_path / "gym"
    root.mkdir()
    (root / "collection.json").write_text(json.dumps({"collectionId": "gym"}), encoding="utf-8")
    _write_recording(
        root,
        recording_id="recording-a",
        metadata={
            "recordingId": "recording-a",
            "micArrayProfile": "respeaker_xvf3800_0650",
            "speakers": [
                {
                    "speakerName": "matthew",
                    "speakingPeriods": [{"startSec": 0, "endSec": 0, "directionDeg": 45}],
                }
            ],
        },
    )

    jobs = load_real_recording_jobs("gym", root)

    assert len(jobs) == 1
    assert jobs[0].dataset_name == "gym"
    assert jobs[0].mic_array_profile == "respeaker_xvf3800_0650"
    assert jobs[0].metadata_path is not None


def test_aggregate_mixed_dataset_rows_penalizes_infeasible_trials() -> None:
    rows = [
        {
            "status": "ok",
            "dataset_name": "gym",
            "domain_type": "real",
            "mae_deg": 10.0,
            "acc_at_10": 0.9,
            "acc_at_25": 0.95,
            "coverage_rate": 1.0,
            "accepted_rate": 0.9,
            "hold_rate": 0.0,
            "abstain_rate": 0.0,
            "runtime_ms_mean": 40.0,
            "rtf": 0.8,
            "direction_jump_p95_deg": 5.0,
            "confidence_high_mae_deg": 8.0,
            "confidence_low_mae_deg": 20.0,
        },
        {
            "status": "ok",
            "dataset_name": "sim_silence_gaps",
            "domain_type": "simulation",
            "mae_deg": 12.0,
            "acc_at_10": 0.85,
            "acc_at_25": 0.9,
            "coverage_rate": 1.0,
            "accepted_rate": 0.85,
            "hold_rate": 0.0,
            "abstain_rate": 0.0,
            "runtime_ms_mean": 40.0,
            "rtf": 0.8,
            "direction_jump_p95_deg": 6.0,
            "confidence_high_mae_deg": 9.0,
            "confidence_low_mae_deg": 22.0,
        },
    ]

    agg = aggregate_mixed_dataset_rows(rows, dataset_weights={"gym": 0.5, "sim_silence_gaps": 0.5})

    assert agg["feasible_under_rtf"] is False
    assert agg["weighted_dataset_score"] < 0.0
    assert "latency_budget_violation" in agg["failure_reasons"]

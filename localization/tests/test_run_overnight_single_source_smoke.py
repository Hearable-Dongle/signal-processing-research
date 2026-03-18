from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf


def _write_dataset(root: Path, *, collection_id: str, recording_id: str, metadata: dict, duration_s: float = 0.2, fs: int = 48000) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "collection.json").write_text(json.dumps({"collectionId": collection_id}), encoding="utf-8")
    recording_dir = root / "recordings" / recording_id
    raw_dir = recording_dir / "raw"
    raw_dir.mkdir(parents=True)
    t = np.arange(int(duration_s * fs), dtype=np.float32) / float(fs)
    tone = 0.05 * np.sin(2.0 * np.pi * 440.0 * t).astype(np.float32)
    for idx in range(4):
        sf.write(raw_dir / f"channel_{idx:03d}.wav", tone, fs)
    (recording_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def test_run_overnight_single_source_smoke(tmp_path: Path) -> None:
    gym_root = tmp_path / "gym"
    _write_dataset(
        gym_root,
        collection_id="gym",
        recording_id="recording-gym",
        metadata={
            "recordingId": "recording-gym",
            "micArrayProfile": "respeaker_xvf3800_0650",
            "speakers": [{"speakerName": "matthew", "speakingPeriods": [{"startSec": 0, "endSec": 0, "directionDeg": 0}]}],
        },
    )

    kitchener_root = tmp_path / "kitchener"
    _write_dataset(
        kitchener_root,
        collection_id="kitchener",
        recording_id="recording-kpl",
        metadata={
            "recordingId": "recording-kpl",
            "micArrayProfile": "respeaker_xvf3800_0650",
            "speakers": [
                {"speakerName": "matthew", "speakingPeriods": [{"startSec": 0, "endSec": 0.1, "directionDeg": 45}]},
                {"speakerName": "lucas", "speakingPeriods": [{"startSec": 0.1, "endSec": 0.2, "directionDeg": 315}]},
            ],
        },
    )

    out_root = tmp_path / "overnight_optuna"
    cmd = [
        sys.executable,
        "-m",
        "localization.tuning.run_overnight_single_source_optimization",
        "--methods",
        "srp_phat_localization",
        "--duration-min",
        "0.03",
        "--poll-interval-sec",
        "1",
        "--gym-root",
        str(gym_root),
        "--kitchener-root",
        str(kitchener_root),
        "--max-real-recordings-per-dataset",
        "1",
        "--max-sim-scenes-per-dataset",
        "1",
        "--trial-job-workers",
        "1",
        "--srp-workers",
        "1",
        "--top-n-full-eval",
        "1",
        "--out-root",
        str(out_root),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = "." if not env.get("PYTHONPATH") else f".:{env['PYTHONPATH']}"
    subprocess.run(cmd, check=True, cwd=Path(__file__).resolve().parents[2], env=env)

    latest = out_root / "latest"
    assert latest.exists()
    run_dir = latest.resolve()
    assert (run_dir / "latest_recommendation.json").exists()
    assert (run_dir / "latest_recommendation.md").exists()
    assert (run_dir / "monitor.log").exists()

    payload = json.loads((run_dir / "latest_recommendation.json").read_text(encoding="utf-8"))
    assert payload["current_winning_backend"] == "srp_phat_localization"
    assert "gym" in payload["per_dataset_summary"]
    assert "kitchener" in payload["per_dataset_summary"]

    method_dir = run_dir / "srp_phat_localization"
    assert (method_dir / "study.db").exists()
    assert (method_dir / "best_params.json").exists()
    assert (method_dir / "best_timeline_rows.csv").exists()

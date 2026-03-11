from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_framewise_localization_benchmark_smoke(tmp_path: Path) -> None:
    out_root = tmp_path / "framewise_benchmark"
    cmd = [
        sys.executable,
        "-m",
        "realtime_pipeline.run_framewise_localization_benchmark",
        "--max-scenes",
        "1",
        "--workers",
        "1",
        "--strategies",
        "weighted_srp_dp",
        "--out-root",
        str(out_root),
        "--eval-mode",
        "both",
    ]
    proc = subprocess.run(cmd, cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr[-4000:]
    latest = out_root / "latest"
    assert latest.exists()
    run_dir = latest.resolve()
    payload = json.loads((run_dir / "benchmark_results.json").read_text(encoding="utf-8"))
    assert payload["n_scenes"] == 1
    assert payload["eval_mode"] == "both"
    assert "active_speaker_per_frame" in payload["aggregates"]
    assert "top_k_per_frame" in payload["aggregates"]
    assert (run_dir / "benchmark_results.txt").exists()
    assert (run_dir / "per_frame_predictions").exists()
    assert (run_dir / "scene_summaries").exists()
    assert (run_dir / "visualizations").exists()

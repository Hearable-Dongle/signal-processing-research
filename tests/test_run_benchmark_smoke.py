from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_run_benchmark_smoke(tmp_path: Path) -> None:
    out_root = tmp_path / "benchmark"
    cmd = [
        sys.executable,
        "run_benchmark.py",
        "--max-scenes",
        "1",
        "--workers",
        "1",
        "--skip-ipd-train",
        "--strategies",
        "weighted_srp_dp",
        "snr_weighted_srp_phat",
        "peak_confidence_srp_phat",
        "particle_filter_tracker",
        "neural_mask_gcc_phat",
        "--out-root",
        str(out_root),
    ]
    proc = subprocess.run(cmd, cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr[-4000:]
    latest = out_root / "latest"
    assert latest.exists()
    run_dir = latest.resolve()
    result_json = run_dir / "localization_benchmark_results.json"
    result_txt = run_dir / "localization_benchmark_results.txt"
    assert result_json.exists()
    assert result_txt.exists()
    payload = json.loads(result_json.read_text(encoding="utf-8"))
    assert payload["n_scenes"] == 1
    assert len(payload["strategies"]) == 6
    assert (run_dir / "mae_by_strategy.png").exists()
    assert (run_dir / "timeline_worst_scenes").exists()

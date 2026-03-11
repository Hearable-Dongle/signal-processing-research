from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_localization_benchmark_rejects_testing_specific_angles(tmp_path: Path) -> None:
    cfg = {
        "scene_roots": {
            "testing_specific_angles": "simulation/simulations/configs/testing_specific_angles",
        },
        "presets": {
            "quick": {"sample_per_bucket": 1},
            "full": {"sample_per_bucket": None},
        },
        "methods": {
            "SRP-PHAT": {
                "nfft": 512,
                "overlap": 0.5,
                "freq_range": [200, 3000],
            }
        },
    }
    cfg_path = tmp_path / "guard_config.json"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "localization.benchmark.run",
            "--config",
            str(cfg_path),
            "--preset",
            "quick",
            "--max-scenes",
            "1",
        ],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode != 0
    assert "NotImplementedError" in proc.stderr
    assert "run_framewise_localization_benchmark" in proc.stderr

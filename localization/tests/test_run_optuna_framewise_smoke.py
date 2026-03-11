from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def test_run_optuna_framewise_smoke(tmp_path: Path) -> None:
    out_root = tmp_path / "framewise_optuna"
    cmd = [
        sys.executable,
        "-m",
        "localization.tuning.run_optuna_framewise_respeaker_v3",
        "--methods",
        "SRP-PHAT",
        "--duration-min",
        "0.02",
        "--subset-per-bucket",
        "1",
        "--max-scenes",
        "1",
        "--top-n-full-eval",
        "1",
        "--srp-workers",
        "1",
        "--trial-scene-workers",
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

    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["methods"] == ["SRP-PHAT"]
    assert manifest["max_scenes"] == 1

    dashboard_text = (run_dir / "dashboard_commands.txt").read_text(encoding="utf-8")
    assert "optuna-dashboard sqlite:///" in dashboard_text

    method_dir = run_dir / "srp_phat"
    assert (method_dir / "study.db").exists()
    assert (method_dir / "best_params.json").exists()
    assert (method_dir / "best_full_eval.json").exists()
    assert (method_dir / "best_timeline_rows.csv").exists()

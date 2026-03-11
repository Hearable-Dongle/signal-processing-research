from __future__ import annotations

import argparse
import json

from realtime_pipeline.simulation_runner import run_simulation_pipeline


def run_backend_cli(backend_name: str) -> None:
    parser = argparse.ArgumentParser(description=f"Run localization backend {backend_name} on one scene.")
    parser.add_argument("--scene-config", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--real-separation", action="store_true")
    args = parser.parse_args()
    summary = run_simulation_pipeline(
        scene_config_path=args.scene_config,
        out_dir=args.out_dir,
        use_mock_separation=not bool(args.real_separation),
        capture_trace=True,
        robust_mode=True,
        localization_backend=backend_name,
        control_mode="spatial_peak_mode",
        fast_path_reference_mode="srp_peak",
        direction_long_memory_enabled=False,
    )
    print(json.dumps(summary, indent=2))

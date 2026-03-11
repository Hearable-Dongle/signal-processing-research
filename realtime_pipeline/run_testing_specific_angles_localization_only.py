from __future__ import annotations

import argparse
import csv
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

from realtime_pipeline.compare_realtime_methods import METHOD_LOCALIZATION_ONLY, run_comparison_suite


DEFAULT_SCENES_ROOT = Path("simulation/simulations/configs/testing_specific_angles")
DEFAULT_ASSETS_ROOT = Path("simulation/simulations/assets/testing_specific_angles")
DEFAULT_OUT_ROOT = Path("realtime_pipeline/output/testing_specific_angles_localization_only_postloc")


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for row in rows for k in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _load_scene_meta(scene_path: Path) -> dict:
    meta_path = DEFAULT_ASSETS_ROOT / scene_path.stem / "scenario_metadata.json"
    if not meta_path.exists():
        return {}
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _stage_scene(scene_path: Path, staging_root: Path) -> Path:
    stage_dir = staging_root / scene_path.stem
    stage_dir.mkdir(parents=True, exist_ok=True)
    staged_scene = stage_dir / scene_path.name
    shutil.copy2(scene_path, staged_scene)

    asset_dir = DEFAULT_ASSETS_ROOT / scene_path.stem
    for filename in ("scenario_metadata.json", "metrics_summary.json", "frame_ground_truth.csv"):
        src = asset_dir / filename
        if src.exists():
            shutil.copy2(src, stage_dir / filename)
    return staged_scene


def _extract_method_row(scene_path: Path, comparison_summary: dict) -> dict:
    method_info = comparison_summary["methods"][0]
    meta = _load_scene_meta(scene_path)
    return {
        "scene_id": scene_path.stem,
        "main_angle_deg": meta.get("main_angle_deg"),
        "secondary_angle_deg": meta.get("secondary_angle_deg"),
        "noise_layout_type": meta.get("noise_layout_type"),
        "noise_angles_deg": meta.get("noise_angles_deg"),
        "method_id": method_info["method_id"],
        "fast_rtf": method_info["fast_rtf"],
        "slow_rtf": method_info["slow_rtf"],
        "final_speakers": method_info["final_speakers"],
        "summary_path": method_info["summary_path"],
        "enhanced_wav": method_info["enhanced_wav"],
        "comparison_summary_path": str((Path(method_info["summary_path"]).parent.parent / "comparison_summary.json").resolve()),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run localization_only on testing_specific_angles scenes.")
    parser.add_argument("--scenes-root", default=str(DEFAULT_SCENES_ROOT))
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--mock-separation", action="store_true")
    parser.add_argument("--max-scenes", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_id = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.out_root) / run_id
    out_root.mkdir(parents=True, exist_ok=True)
    staging_root = out_root / "_staged_scenes"

    scene_paths = sorted(Path(args.scenes_root).glob("*.json"))
    if args.max_scenes is not None:
        scene_paths = scene_paths[: int(args.max_scenes)]
    if not scene_paths:
        raise FileNotFoundError(f"No scenes found under {args.scenes_root}")

    rows: list[dict] = []
    for scene_path in scene_paths:
        staged_scene = _stage_scene(scene_path, staging_root)
        scene_out = out_root / scene_path.stem
        comparison_summary = run_comparison_suite(
            out_dir=scene_out,
            methods=[METHOD_LOCALIZATION_ONLY],
            scene_config_path=staged_scene,
            use_mock_separation=bool(args.mock_separation),
        )
        rows.append(_extract_method_row(scene_path, comparison_summary))

    _write_csv(out_root / "summary.csv", rows)
    (out_root / "summary.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "out_root": str(out_root.resolve()),
                "n_scenes": len(rows),
                "rows": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    latest = Path(args.out_root) / "latest"
    if latest.exists() or latest.is_symlink():
        latest.unlink()
    latest.symlink_to(out_root.resolve(), target_is_directory=True)
    print(json.dumps({"run_id": run_id, "out_root": str(out_root.resolve()), "n_scenes": len(rows)}, indent=2))


if __name__ == "__main__":
    main()

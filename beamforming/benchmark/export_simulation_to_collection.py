from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf

from simulation.simulation_config import SimulationConfig
from simulation.simulator import run_simulation


DEFAULT_INPUT_PATH = Path("simulation/simulations/configs/testing_specific_angles")
DEFAULT_OUT_DIR = Path("beamforming/benchmark/simulated_collection_export")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _select_scene_paths(root: Path, scene_ids: list[str] | None, max_scenes: int | None) -> list[Path]:
    if root.is_file():
        return [root]
    if scene_ids:
        paths = [root / f"{scene_id}.json" for scene_id in scene_ids]
    else:
        paths = sorted(root.glob("*.json"))
    paths = [path for path in paths if path.exists()]
    if max_scenes is not None:
        paths = paths[: int(max_scenes)]
    return paths


def export_simulation_collection(
    *,
    input_path: str | Path = DEFAULT_INPUT_PATH,
    out_dir: str | Path = DEFAULT_OUT_DIR,
    scene_ids: list[str] | None = None,
    max_scenes: int | None = None,
) -> dict:
    input_path = Path(input_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    recordings_dir = out_dir / "recordings"
    recordings_dir.mkdir(parents=True, exist_ok=True)

    scene_paths = _select_scene_paths(input_path, scene_ids, max_scenes)
    if not scene_paths:
        raise FileNotFoundError(f"No scene configs found under {input_path}")

    collection_rows: list[dict] = []
    export_rows: list[dict] = []
    started_at = _utc_now_iso()

    for idx, scene_path in enumerate(scene_paths):
        recording_id = f"recording-{scene_path.stem}"
        recording_dir = recordings_dir / recording_id
        raw_dir = recording_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)

        sim_cfg = SimulationConfig.from_file(scene_path)
        mic_audio, _mic_pos, _sources = run_simulation(sim_cfg)
        mic_audio = np.asarray(mic_audio, dtype=np.float32)
        sample_rate_hz = int(sim_cfg.audio.fs)

        channel_rows: list[dict] = []
        for ch_idx in range(mic_audio.shape[1]):
            filename = f"channel_{ch_idx:03d}.wav"
            sf.write(raw_dir / filename, mic_audio[:, ch_idx], sample_rate_hz)
            channel_rows.append({"channelIndex": int(ch_idx), "filename": filename})

        metadata = {
            "recordingId": recording_id,
            "sceneConfigPath": str(scene_path.resolve()),
            "durationSec": float(sim_cfg.audio.duration),
            "sampleRateHz": sample_rate_hz,
            "channelCount": int(mic_audio.shape[1]),
        }
        (recording_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        collection_rows.append(
            {
                "recordingId": recording_id,
                "sessionId": f"sim-{idx:04d}",
                "startedAtIso": started_at,
                "stoppedAtIso": _utc_now_iso(),
                "status": "ready",
                "deviceName": "Simulation",
                "error": None,
                "sampleRateHz": sample_rate_hz,
                "channels": channel_rows,
            }
        )
        export_rows.append(
            {
                "recording_id": recording_id,
                "scene_id": scene_path.stem,
                "scene_config_path": str(scene_path.resolve()),
                "recording_dir": str(recording_dir.resolve()),
                "raw_dir": str(raw_dir.resolve()),
            }
        )

    collection = {
        "collectionId": out_dir.name,
        "title": f"Simulated export from {input_path}",
        "notes": "Generated from simulation scene configs for data_collection_benchmark compatibility.",
        "createdAtIso": started_at,
        "deviceName": "Simulation",
        "recordings": collection_rows,
    }
    (out_dir / "collection.json").write_text(json.dumps(collection, indent=2), encoding="utf-8")
    (out_dir / "export_manifest.json").write_text(
        json.dumps({"input_path": str(input_path.resolve()), "recordings": export_rows}, indent=2),
        encoding="utf-8",
    )
    return {"out_dir": str(out_dir.resolve()), "n_recordings": len(export_rows)}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render simulation scenes into a Data Collection-style recording export.")
    parser.add_argument("--input-path", default=str(DEFAULT_INPUT_PATH))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--scene-ids", nargs="*", default=None)
    parser.add_argument("--max-scenes", type=int, default=None)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    result = export_simulation_collection(
        input_path=args.input_path,
        out_dir=args.out_dir,
        scene_ids=list(args.scene_ids) if args.scene_ids else None,
        max_scenes=args.max_scenes,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

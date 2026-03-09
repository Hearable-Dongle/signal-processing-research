from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from sim.realistic_conversations.config import build_preset
from sim.realistic_conversations.generator import generate_scenario


DEFAULT_CONFIG_ROOT = Path("simulation/simulations/configs/restaurant_meeting_scene")
DEFAULT_ASSET_ROOT = Path("simulation/simulations/assets/restaurant_meeting_scene")
DEFAULT_K_VALUES = (2, 3, 4, 5)
DEFAULT_SCENES_PER_K = 40
DEFAULT_SEED = 42
DEFAULT_MAX_OVERLAP_RATIO = 0.18
DEFAULT_MAX_MEAN_SNR_DB = 9.0
DEFAULT_MAX_ATTEMPTS = 12


def _scene_id(k: int, scene_idx: int) -> str:
    return f"restaurant_meeting_k{k}_scene{scene_idx:02d}"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _minimum_duration_for_k(cfg, speaker_count: int) -> float:
    tt = cfg.turn_taking
    intro_span = float(speaker_count) * float(tt.utterance_sec_range[0])
    pause_span = float(max(0, speaker_count - 1)) * float(tt.pause_sec_range[0])
    return 0.35 + intro_span + pause_span + 0.75


def _scene_is_acceptable(
    metrics: dict,
    *,
    speaker_count: int,
    expected_speaker_count: int,
    realized_speech_speakers: int,
    max_overlap_ratio: float,
    max_mean_snr_db: float,
) -> bool:
    if int(speaker_count) != int(expected_speaker_count):
        return False
    if int(realized_speech_speakers) != int(expected_speaker_count):
        return False
    overlap_ratio = float(metrics.get("overlap_ratio", 1.0))
    mean_snr_db = float(metrics.get("snr_distribution_db", {}).get("mean", 1e9))
    return overlap_ratio <= float(max_overlap_ratio) and mean_snr_db <= float(max_mean_snr_db)


def generate_restaurant_meeting_dataset(
    *,
    config_root: str | Path = DEFAULT_CONFIG_ROOT,
    asset_root: str | Path = DEFAULT_ASSET_ROOT,
    seed: int = DEFAULT_SEED,
    scenes_per_k: int = DEFAULT_SCENES_PER_K,
    k_values: tuple[int, ...] = DEFAULT_K_VALUES,
    duration_sec: float | None = None,
    sample_rate: int | None = None,
    frame_ms: int | None = None,
    manifest_path: str | Path | None = None,
    export_audio: bool = False,
    max_overlap_ratio: float = DEFAULT_MAX_OVERLAP_RATIO,
    max_mean_snr_db: float = DEFAULT_MAX_MEAN_SNR_DB,
    max_attempts_per_scene: int = DEFAULT_MAX_ATTEMPTS,
) -> list[dict]:
    config_root = Path(config_root)
    asset_root = Path(asset_root)
    config_root.mkdir(parents=True, exist_ok=True)
    asset_root.mkdir(parents=True, exist_ok=True)

    generated: list[dict] = []
    attempt_seed = int(seed)
    for k in k_values:
        for scene_idx in range(int(scenes_per_k)):
            accepted = False
            scene_name = _scene_id(int(k), int(scene_idx))
            for _attempt in range(int(max_attempts_per_scene)):
                cfg = build_preset("restaurant_meeting")
                cfg.turn_taking.min_speakers = int(k)
                cfg.turn_taking.max_speakers = int(k)
                effective_duration_sec = duration_sec
                min_duration_sec = _minimum_duration_for_k(cfg, int(k))
                if effective_duration_sec is None or float(effective_duration_sec) < float(min_duration_sec):
                    effective_duration_sec = float(min_duration_sec)
                result = generate_scenario(
                    preset="restaurant_meeting",
                    out_dir=asset_root,
                    seed=attempt_seed,
                    duration_sec=effective_duration_sec,
                    sample_rate=sample_rate,
                    frame_ms=frame_ms,
                    manifest_path=manifest_path,
                    export_audio=bool(export_audio),
                    scene_name=scene_name,
                    config_override=cfg,
                )
                metrics = _load_json(result.metrics_path)
                metadata = _load_json(result.metadata_path)
                realized_speech_speakers = len(
                    {
                        int(row["speaker_id"])
                        for row in metadata.get("assets", {}).get("render_segments", [])
                        if row.get("classification") == "speech" and "speaker_id" in row
                    }
                )
                if _scene_is_acceptable(
                    metrics,
                    speaker_count=int(metadata.get("speaker_count", 0)),
                    expected_speaker_count=int(k),
                    realized_speech_speakers=realized_speech_speakers,
                    max_overlap_ratio=max_overlap_ratio,
                    max_mean_snr_db=max_mean_snr_db,
                ):
                    config_out = config_root / f"{scene_name}.json"
                    shutil.copy2(result.scene_config_path, config_out)
                    generated.append(
                        {
                            "scene_name": scene_name,
                            "scene_config_path": str(config_out.resolve()),
                            "scene_dir": str(result.scene_dir.resolve()),
                            "seed": int(attempt_seed),
                            "speaker_count": int(metadata["speaker_count"]),
                            "overlap_ratio": float(metrics["overlap_ratio"]),
                            "mean_snr_db": float(metrics["snr_distribution_db"]["mean"]),
                        }
                    )
                    accepted = True
                    attempt_seed += 1
                    break
                shutil.rmtree(result.scene_dir, ignore_errors=True)
                attempt_seed += 1
            if not accepted:
                raise RuntimeError(
                    f"Failed to generate acceptable restaurant_meeting scene for k={k}, scene_idx={scene_idx:02d} "
                    f"within {max_attempts_per_scene} attempts"
                )
    summary_path = asset_root / "generation_summary.json"
    summary_path.write_text(json.dumps({"scenes": generated}, indent=2), encoding="utf-8")
    return generated


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate restaurant meeting conversation-style simulation configs")
    parser.add_argument("--config-root", default=str(DEFAULT_CONFIG_ROOT))
    parser.add_argument("--asset-root", default=str(DEFAULT_ASSET_ROOT))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--scenes-per-k", type=int, default=DEFAULT_SCENES_PER_K)
    parser.add_argument("--k-values", nargs="+", type=int, default=list(DEFAULT_K_VALUES))
    parser.add_argument("--duration-sec", type=float, default=None)
    parser.add_argument("--sample-rate", type=int, default=None)
    parser.add_argument("--frame-ms", type=int, default=None)
    parser.add_argument("--asset-manifest", type=str, default=None)
    parser.add_argument("--export-audio", action="store_true")
    parser.add_argument("--max-overlap-ratio", type=float, default=DEFAULT_MAX_OVERLAP_RATIO)
    parser.add_argument("--max-mean-snr-db", type=float, default=DEFAULT_MAX_MEAN_SNR_DB)
    parser.add_argument("--max-attempts-per-scene", type=int, default=DEFAULT_MAX_ATTEMPTS)
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    generated = generate_restaurant_meeting_dataset(
        config_root=args.config_root,
        asset_root=args.asset_root,
        seed=args.seed,
        scenes_per_k=args.scenes_per_k,
        k_values=tuple(int(v) for v in args.k_values),
        duration_sec=args.duration_sec,
        sample_rate=args.sample_rate,
        frame_ms=args.frame_ms,
        manifest_path=args.asset_manifest,
        export_audio=bool(args.export_audio),
        max_overlap_ratio=float(args.max_overlap_ratio),
        max_mean_snr_db=float(args.max_mean_snr_db),
        max_attempts_per_scene=int(args.max_attempts_per_scene),
    )
    print(json.dumps({"num_scenes": len(generated), "scenes": generated}, indent=2))


if __name__ == "__main__":
    main()

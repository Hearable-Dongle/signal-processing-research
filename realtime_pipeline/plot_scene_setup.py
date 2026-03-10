from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from simulation.simulation_config import SimulationConfig


def _normalize_angle_deg(v: float) -> float:
    return float(v % 360.0)


def _relative_label(angle_deg: float) -> str:
    a = _normalize_angle_deg(angle_deg)
    if a < 22.5 or a >= 337.5:
        return "front"
    if a < 67.5:
        return "front-left"
    if a < 112.5:
        return "left"
    if a < 157.5:
        return "back-left"
    if a < 202.5:
        return "back"
    if a < 247.5:
        return "back-right"
    if a < 292.5:
        return "right"
    return "front-right"


def _load_json(path: Path | None) -> dict:
    if path is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def render_scene_setup(
    *,
    scene_config_path: str | Path,
    out_path: str | Path,
    scenario_metadata_path: str | Path | None = None,
    comparison_summary_path: str | Path | None = None,
) -> dict:
    scene_path = Path(scene_config_path).resolve()
    out_path = Path(out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sim_cfg = SimulationConfig.from_file(scene_path)
    scenario_meta = _load_json(None if scenario_metadata_path is None else Path(scenario_metadata_path).resolve())
    comparison = _load_json(None if comparison_summary_path is None else Path(comparison_summary_path).resolve())

    room_xy = np.asarray(sim_cfg.room.dimensions[:2], dtype=float)
    mic_center = np.asarray(sim_cfg.microphone_array.mic_center[:2], dtype=float)
    explicit_positions = sim_cfg.microphone_array.mic_positions
    if explicit_positions:
        mic_positions = np.asarray(explicit_positions, dtype=float)[:, :2]
    else:
        mic_count = int(sim_cfg.microphone_array.mic_count)
        radius = float(sim_cfg.microphone_array.mic_radius)
        thetas = np.linspace(0.0, 2.0 * np.pi, num=mic_count, endpoint=False)
        mic_positions = np.stack(
            [mic_center[0] + radius * np.cos(thetas), mic_center[1] + radius * np.sin(thetas)],
            axis=1,
        )

    speaker_labels = {
        int(row.get("speaker_id")): str(row.get("speaker_label"))
        for row in scenario_meta.get("speakers", [])
        if row.get("speaker_id") is not None
    }
    speech_events = scenario_meta.get("assets", {}).get("speech_events", [])
    first_turn_index = {}
    for idx, ev in enumerate(sorted(speech_events, key=lambda row: float(row.get("start_sec", 0.0))), start=1):
        sid = ev.get("speaker_id")
        if sid is None:
            continue
        first_turn_index.setdefault(int(sid), idx)

    source_rows: list[dict] = []
    for idx, src in enumerate(sim_cfg.audio.sources):
        loc_xy = np.asarray(src.loc[:2], dtype=float)
        delta = loc_xy - mic_center
        angle_deg = _normalize_angle_deg(np.degrees(np.arctan2(delta[1], delta[0])))
        distance_m = float(np.linalg.norm(delta))
        cls = (src.classification or "").strip().lower() or "unknown"
        row = {
            "index": int(idx),
            "classification": cls,
            "loc_xy": loc_xy,
            "angle_deg": angle_deg,
            "distance_m": distance_m,
        }
        if cls == "speech":
            speech_id = sum(1 for prev in sim_cfg.audio.sources[: idx + 1] if (prev.classification or "").strip().lower() == "speech") - 1
            row["speaker_id"] = int(speech_id)
            row["speaker_label"] = speaker_labels.get(int(speech_id), f"spk {speech_id}")
            row["turn_order"] = first_turn_index.get(int(speech_id))
        source_rows.append(row)

    fig = plt.figure(figsize=(13, 7.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.3, 1.0])
    ax_room = fig.add_subplot(gs[0, 0])
    ax_polar = fig.add_subplot(gs[0, 1], projection="polar")

    ax_room.add_patch(plt.Rectangle((0.0, 0.0), room_xy[0], room_xy[1], fill=False, linewidth=2.0, edgecolor="#444"))
    ax_room.scatter(mic_positions[:, 0], mic_positions[:, 1], c="#111111", s=45, label="Mic capsules", zorder=4)
    ax_room.scatter([mic_center[0]], [mic_center[1]], c="#2b6cb0", s=150, marker="*", label="User / mic center", zorder=5)

    speech_color = "#d94841"
    noise_color = "#7a8f00"
    for row in source_rows:
        loc_xy = np.asarray(row["loc_xy"], dtype=float)
        cls = row["classification"]
        color = speech_color if cls == "speech" else noise_color
        marker = "o" if cls == "speech" else "^"
        label = None
        if cls == "speech":
            sid = int(row["speaker_id"])
            turn = row.get("turn_order")
            turn_suffix = "" if turn is None else f", turn {turn}"
            label = f"spk {sid} ({row['speaker_label']}{turn_suffix})"
        else:
            label = f"noise {int(row['index'])}"
        ax_room.scatter([loc_xy[0]], [loc_xy[1]], c=color, s=110, marker=marker, zorder=4)
        ax_room.plot([mic_center[0], loc_xy[0]], [mic_center[1], loc_xy[1]], color=color, alpha=0.35, linewidth=1.5, zorder=2)
        ax_room.text(loc_xy[0] + 0.08, loc_xy[1] + 0.08, label, fontsize=8)

    ax_room.set_xlim(-0.1, room_xy[0] + 0.1)
    ax_room.set_ylim(-0.1, room_xy[1] + 0.1)
    ax_room.set_aspect("equal")
    ax_room.set_title("Top-down room layout")
    ax_room.set_xlabel("x (m)")
    ax_room.set_ylabel("y (m)")
    ax_room.grid(True, alpha=0.2)

    ax_polar.set_theta_zero_location("E")
    ax_polar.set_theta_direction(1)
    max_radius = max((float(row["distance_m"]) for row in source_rows), default=1.0)
    ax_polar.set_rmax(max(1.0, max_radius + 0.4))
    ax_polar.set_title("User-centric azimuth / range")
    for row in source_rows:
        theta = np.deg2rad(float(row["angle_deg"]))
        radius = float(row["distance_m"])
        cls = row["classification"]
        color = speech_color if cls == "speech" else noise_color
        marker = "o" if cls == "speech" else "^"
        ax_polar.scatter([theta], [radius], c=color, s=95, marker=marker)
        if cls == "speech":
            sid = int(row["speaker_id"])
            txt = f"spk {sid}\n{row['angle_deg']:.0f} deg"
        else:
            txt = f"n{int(row['index'])}\n{row['angle_deg']:.0f} deg"
        ax_polar.text(theta, radius + 0.15, txt, ha="center", va="bottom", fontsize=8)

    scene_metrics = comparison.get("scene_metrics", scenario_meta.get("summary_metrics", {}))
    speech_rows = [row for row in source_rows if row["classification"] == "speech"]
    lines = []
    for row in sorted(speech_rows, key=lambda item: (999 if item.get("turn_order") is None else int(item["turn_order"]), int(item["speaker_id"]))):
        sid = int(row["speaker_id"])
        turn = row.get("turn_order")
        turn_prefix = "" if turn is None else f"turn {turn}: "
        lines.append(
            f"{turn_prefix}spk {sid} at {row['angle_deg']:.0f} deg, {row['distance_m']:.2f} m, {_relative_label(row['angle_deg'])}"
        )
    noise_rows = [row for row in source_rows if row["classification"] == "noise"]
    if noise_rows:
        mean_noise_dist = float(np.mean([row["distance_m"] for row in noise_rows]))
        lines.append(f"{len(noise_rows)} noise sources ring the user at about {mean_noise_dist:.2f} m")
    if scene_metrics:
        overlap = scene_metrics.get("overlap_ratio")
        snr_mean = scene_metrics.get("snr_distribution_db", {}).get("mean")
        if overlap is not None:
            lines.append(f"overlap ratio {float(overlap):.2f}")
        if snr_mean is not None:
            lines.append(f"mean frame SNR {float(snr_mean):.2f} dB")

    fig.text(
        0.57,
        0.06,
        "How this maps to the user:\n" + "\n".join(f"- {line}" for line in lines),
        fontsize=10,
        family="monospace",
        va="bottom",
    )

    fig.suptitle(f"Scene setup: {scene_path.stem}", fontsize=14)
    fig.tight_layout(rect=(0.0, 0.11, 1.0, 0.95))
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

    summary = {
        "scene_config": str(scene_path),
        "output_image": str(out_path),
        "sources": [
            {
                "index": int(row["index"]),
                "classification": str(row["classification"]),
                "angle_deg": float(row["angle_deg"]),
                "distance_m": float(row["distance_m"]),
                "speaker_id": None if row.get("speaker_id") is None else int(row["speaker_id"]),
                "relative_label": _relative_label(float(row["angle_deg"])),
            }
            for row in source_rows
        ],
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a top-down and user-centric view of a simulation scene.")
    parser.add_argument("--scene-config", required=True)
    parser.add_argument("--out-path", required=True)
    parser.add_argument("--scenario-metadata", default=None)
    parser.add_argument("--comparison-summary", default=None)
    args = parser.parse_args()

    summary = render_scene_setup(
        scene_config_path=args.scene_config,
        out_path=args.out_path,
        scenario_metadata_path=args.scenario_metadata,
        comparison_summary_path=args.comparison_summary,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

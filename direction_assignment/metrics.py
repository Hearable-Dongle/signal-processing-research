from __future__ import annotations

from collections import defaultdict

import numpy as np


def angular_distance_deg(a: float, b: float) -> float:
    return float(abs((a - b + 180.0) % 360.0 - 180.0))


def compute_error_summary(errors_deg: list[float]) -> dict[str, float]:
    if not errors_deg:
        return {
            "mae_deg": 0.0,
            "rmse_deg": 0.0,
            "acc_within_5deg": 0.0,
            "acc_within_10deg": 0.0,
            "acc_within_15deg": 0.0,
            "num_points": 0.0,
        }

    e = np.asarray(errors_deg, dtype=float)
    return {
        "mae_deg": float(np.mean(np.abs(e))),
        "rmse_deg": float(np.sqrt(np.mean(e**2))),
        "acc_within_5deg": float(np.mean(e <= 5.0)),
        "acc_within_10deg": float(np.mean(e <= 10.0)),
        "acc_within_15deg": float(np.mean(e <= 15.0)),
        "num_points": float(e.size),
    }


def compute_id_switch_rate(oracle_to_sid_seq: dict[int, list[int | None]]) -> float:
    switches = 0
    transitions = 0
    for seq in oracle_to_sid_seq.values():
        prev = None
        for sid in seq:
            if sid is None:
                continue
            if prev is not None:
                transitions += 1
                if sid != prev:
                    switches += 1
            prev = sid
    if transitions == 0:
        return 0.0
    return float(switches / transitions)


def compute_track_jump_rate(track_history: dict[int, list[float]], jump_threshold_deg: float = 30.0) -> float:
    jumps = 0
    transitions = 0
    for vals in track_history.values():
        for i in range(1, len(vals)):
            transitions += 1
            if angular_distance_deg(vals[i], vals[i - 1]) > jump_threshold_deg:
                jumps += 1
    if transitions == 0:
        return 0.0
    return float(jumps / transitions)


def compute_crossing_swap_rate(
    oracle_doa_chunks: list[dict[int, float]],
    est_doa_chunks: list[dict[int, float]],
    crossing_threshold_deg: float = 15.0,
) -> float:
    swaps = 0
    checks = 0
    for o_map, e_map in zip(oracle_doa_chunks, est_doa_chunks):
        labels = sorted(set(o_map.keys()) & set(e_map.keys()))
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                li, lj = labels[i], labels[j]
                if angular_distance_deg(o_map[li], o_map[lj]) > crossing_threshold_deg:
                    continue
                checks += 1
                o_rel = ((o_map[li] - o_map[lj] + 180) % 360) - 180
                e_rel = ((e_map[li] - e_map[lj] + 180) % 360) - 180
                if o_rel * e_rel < 0:
                    swaps += 1
    if checks == 0:
        return 0.0
    return float(swaps / checks)


def per_speaker_error_rows(errors_by_oracle: dict[int, list[float]]) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for sid in sorted(errors_by_oracle.keys()):
        s = compute_error_summary(errors_by_oracle[sid])
        rows.append({"oracle_speaker": float(sid), **s})
    return rows


def summarize_scene_metrics(
    errors_on: list[float],
    errors_off: list[float],
    errors_by_oracle_on: dict[int, list[float]],
    oracle_to_sid_seq: dict[int, list[int | None]],
    track_history_on: dict[int, list[float]],
    oracle_doa_chunks: list[dict[int, float]],
    est_doa_chunks_on: list[dict[int, float]],
    runtime_ms_per_chunk: list[float],
    chunk_ms: float,
) -> dict[str, float]:
    on = compute_error_summary(errors_on)
    off = compute_error_summary(errors_off)

    out = {
        "overall_mae_deg": on["mae_deg"],
        "overall_rmse_deg": on["rmse_deg"],
        "acc_within_5deg": on["acc_within_5deg"],
        "acc_within_10deg": on["acc_within_10deg"],
        "acc_within_15deg": on["acc_within_15deg"],
        "num_points": on["num_points"],
        "prior_on_mae_deg": on["mae_deg"],
        "prior_off_mae_deg": off["mae_deg"],
        "prior_delta_mae_deg": on["mae_deg"] - off["mae_deg"],
        "id_switch_rate": compute_id_switch_rate(oracle_to_sid_seq),
        "track_jump_rate": compute_track_jump_rate(track_history_on),
        "swap_rate": compute_crossing_swap_rate(oracle_doa_chunks, est_doa_chunks_on),
        "avg_runtime_ms_per_chunk": float(np.mean(runtime_ms_per_chunk)) if runtime_ms_per_chunk else 0.0,
        "realtime_factor": (float(np.mean(runtime_ms_per_chunk)) / float(chunk_ms)) if runtime_ms_per_chunk else 0.0,
    }

    # Extra scene-level indicator: mean per-speaker MAE spread.
    ps = per_speaker_error_rows(errors_by_oracle_on)
    if ps:
        maes = [r["mae_deg"] for r in ps]
        out["per_speaker_mae_std"] = float(np.std(maes))
    else:
        out["per_speaker_mae_std"] = 0.0

    return out

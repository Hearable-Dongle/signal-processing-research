import argparse
import csv
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from numpy.typing import NDArray
from scipy.signal import istft

from beamforming.algo.beamformer import apply_beamformer_stft
from beamforming.beamformer_core import BeamformerConfig, compute_beamforming_weights
from beamforming.localization_bridge import (
    DEFAULT_METHOD_CONFIGS,
    estimate_doas_with_fallback,
    oracle_target_doas_deg,
    oracle_target_locations,
)
from beamforming.util.compare import align_signals, calc_rmse, calc_si_sdr, calc_snr, match_rms_to_reference
from beamforming.util.configure import Audio_Sources
from beamforming.util.visualize import plot_beam_pattern, plot_history, plot_mic_pos, plot_room_pos
from simulation.simulation_config import SimulationAudio, SimulationConfig
from simulation.simulator import run_simulation
from simulation.target_policy import is_speech_target, iter_target_source_indices


BEAMFORMER_STFT_KEYS = {
    "MVDR (Iterative Steepest)": "steepest_stft",
    "MVDR (Iterative Newton)": "newton_stft",
    "MVDR (Neural)": "mvdr_stft",
    "LCMV (Closed Form)": "lcmv_stft",
    "GSC (Closed Form)": "gsc_stft",
    "GSC (Iterative)": "gsc_iterative_stft",
    "LCMV (Weighted)": "lcmv_weighted_stft",
    "GSC (Weighted)": "gsc_weighted_stft",
    "GSC (Iterative Weighted)": "gsc_iterative_weighted_stft",
}


@dataclass
class SteeringDecision:
    chunk_start_s: float
    chunk_end_s: float
    steering_source: str
    requested_method: str
    method_used: str
    fallback_used: bool
    confidence: float
    doas_deg: str
    note: str


def reconstruct_audio(stft_data: NDArray, fs: int, window_params: tuple, target_length: int) -> NDArray:
    win_size, hop, window = window_params
    _, time_signal = istft(
        stft_data,
        fs=fs,
        nperseg=win_size,
        noverlap=win_size - hop,
        window=window,
    )
    time_signal = np.real(time_signal)
    if len(time_signal) > target_length:
        return time_signal[:target_length]
    return np.pad(time_signal, (0, target_length - len(time_signal)))


def _compute_ref_audio(sim_config: SimulationConfig, source_signals: list[np.ndarray], min_samples: int) -> NDArray:
    ref_audio = np.zeros(min_samples, dtype=float)
    for idx in iter_target_source_indices(sim_config):
        sig = source_signals[idx]
        if len(sig) > min_samples:
            sig = sig[:min_samples]
        else:
            sig = np.pad(sig, (0, min_samples - len(sig)))
        ref_audio += sig
    return ref_audio


def _parse_target_weights(raw: object) -> list[float] | None:
    if raw is None:
        return None
    if isinstance(raw, list):
        vals = [float(v) for v in raw]
        return vals if vals else None
    if isinstance(raw, dict):
        if "weights" in raw and isinstance(raw["weights"], list):
            vals = [float(v) for v in raw["weights"]]
            return vals if vals else None
        by_idx = []
        for key in sorted(raw.keys(), key=lambda x: int(x) if str(x).isdigit() else 10**9):
            val = raw[key]
            if isinstance(val, (int, float)):
                by_idx.append(float(val))
        return by_idx if by_idx else None
    return None


def _load_target_weights(weights_file: Path | None, full_config_data: dict) -> list[float] | None:
    if weights_file is not None:
        with weights_file.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        return _parse_target_weights(payload)

    bf_data = full_config_data.get("beamforming", {})
    return _parse_target_weights(bf_data.get("target_weights"))


def _build_target_weights(base_weights: list[float] | None, n_targets: int, mode: str) -> NDArray | None:
    if n_targets <= 0:
        return None
    if mode == "equal" or base_weights is None:
        return np.ones(n_targets, dtype=float)

    vals = np.asarray(base_weights, dtype=float).reshape(-1)
    if vals.size < n_targets:
        vals = np.pad(vals, (0, n_targets - vals.size), constant_values=1.0)
    elif vals.size > n_targets:
        vals = vals[:n_targets]

    vals = np.maximum(vals, 0.0)
    if float(np.sum(vals)) <= 1e-12:
        return np.ones(n_targets, dtype=float)
    return vals


def _compose_method_order(primary_method: str, fallbacks: list[str]) -> list[str]:
    order = [primary_method]
    for m in fallbacks:
        if m not in order:
            order.append(m)
    return order


def _build_results_packet(results: dict, mic_pos_rel: NDArray) -> dict:
    stft_tensor = results["stft_tensor"]
    packet = {
        "params": results["params"],
        "fvec": results["fvec"],
        "mic_pos": mic_pos_rel,
        "steepest_weights": results["steepest"][0],
        "steepest_hist": results["steepest"][1],
        "steepest_stft": apply_beamformer_stft(stft_tensor, results["steepest"][0]),
        "newton_weights": results["newton"][0],
        "newton_hist": results["newton"][1],
        "newton_stft": apply_beamformer_stft(stft_tensor, results["newton"][0]),
        "mvdr_weights": results["mvdr"][0],
        "mvdr_stft": apply_beamformer_stft(stft_tensor, results["mvdr"][0]),
        "lcmv_weights": results["lcmv"][0],
        "lcmv_stft": apply_beamformer_stft(stft_tensor, results["lcmv"][0]),
        "gsc_weights": results["gsc"][0],
        "gsc_stft": apply_beamformer_stft(stft_tensor, results["gsc"][0]),
        "gsc_iterative_weights": results["gsc_iterative"][0],
        "gsc_iterative_hist": results["gsc_iterative"][1],
        "gsc_iterative_stft": apply_beamformer_stft(stft_tensor, results["gsc_iterative"][0]),
        "lcmv_weighted_weights": results["lcmv_weighted"][0],
        "lcmv_weighted_stft": apply_beamformer_stft(stft_tensor, results["lcmv_weighted"][0]),
        "gsc_weighted_weights": results["gsc_weighted"][0],
        "gsc_weighted_stft": apply_beamformer_stft(stft_tensor, results["gsc_weighted"][0]),
        "gsc_iterative_weighted_weights": results["gsc_iterative_weighted"][0],
        "gsc_iterative_weighted_hist": results["gsc_iterative_weighted"][1],
        "gsc_iterative_weighted_stft": apply_beamformer_stft(stft_tensor, results["gsc_iterative_weighted"][0]),
        "target_weights": [float(v) for v in results.get("target_weights", [])],
    }
    return packet


def evaluate_results(
    output_dir: Path,
    fs: int,
    log: logging.Logger,
    mic_audio: NDArray,
    results_dict: dict,
    ref_audio: NDArray,
    sound_speed: float,
) -> list[dict]:
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    sf.write(audio_dir / "mic_raw_audio.wav", mic_audio, fs)

    reconstructed_signals: dict[str, NDArray] = {}
    for label, key in BEAMFORMER_STFT_KEYS.items():
        if key in results_dict:
            time_sig = reconstruct_audio(results_dict[key], fs, results_dict["params"], len(ref_audio))
            reconstructed_signals[label] = time_sig
            filename = label.lower().replace(" ", "_").replace("(", "").replace(")", "") + ".wav"
            sf.write(audio_dir / filename, time_sig, fs)

    metric_rows: list[dict] = []
    log.info("-" * 50)
    for label, pred_sig in {"Raw Audio (Mean)": np.mean(mic_audio, axis=1), **reconstructed_signals}.items():
        aligned_ref = align_signals(ref_audio, pred_sig)
        pred_norm, gain_db = match_rms_to_reference(pred_sig, aligned_ref)

        rmse_raw, _ = calc_rmse(aligned_ref, pred_sig)
        snr_raw = calc_snr(aligned_ref, pred_sig)
        sdr_raw = calc_si_sdr(aligned_ref, pred_sig)

        rmse_norm, _ = calc_rmse(aligned_ref, pred_norm)
        snr_norm = calc_snr(aligned_ref, pred_norm)
        sdr_norm = calc_si_sdr(aligned_ref, pred_norm)

        base = label.lower().replace(" ", "_").replace("(", "").replace(")", "")
        if label == "Raw Audio (Mean)":
            sf.write(audio_dir / "raw_audio_mean.wav", pred_sig, fs)
            sf.write(audio_dir / "raw_audio_mean_norm_to_ref.wav", pred_norm, fs)
        else:
            sf.write(audio_dir / f"{base}_norm_to_ref.wav", pred_norm, fs)

        log.info(
            f"{label: <30}: RAW RMSE={rmse_raw:.4f}, SNR={snr_raw:.4f}dB, SI-SDR={sdr_raw:.4f}dB | "
            f"NORM RMSE={rmse_norm:.4f}, SNR={snr_norm:.4f}dB, SI-SDR={sdr_norm:.4f}dB, gain={gain_db:.2f}dB"
        )
        metric_rows.append(
            {
                "beamformer": label,
                # Backward-compat aliases (raw metrics)
                "rmse": float(rmse_raw),
                "snr_db": float(snr_raw),
                "si_sdr_db": float(sdr_raw),
                # Explicit metric modes
                "rmse_raw": float(rmse_raw),
                "snr_db_raw": float(snr_raw),
                "si_sdr_db_raw": float(sdr_raw),
                "rmse_norm": float(rmse_norm),
                "snr_db_norm": float(snr_norm),
                "si_sdr_db_norm": float(sdr_norm),
                "gain_to_ref_db": float(gain_db),
            }
        )
    log.info("-" * 50)

    hist_data = {}
    if "steepest_hist" in results_dict:
        hist_data["MVDR Steepest"] = (np.mean(results_dict["steepest_hist"], axis=0), {"color": "blue", "alpha": 0.5})
    if "newton_hist" in results_dict:
        hist_data["MVDR Newton"] = (np.mean(results_dict["newton_hist"], axis=0), {"color": "green", "alpha": 0.5})
    if "gsc_iterative_hist" in results_dict:
        hist_data["GSC Iterative"] = (np.mean(results_dict["gsc_iterative_hist"], axis=0), {"color": "red", "alpha": 0.5})
    if "gsc_iterative_weighted_hist" in results_dict:
        hist_data["GSC Iterative Weighted"] = (
            np.mean(results_dict["gsc_iterative_weighted_hist"], axis=0),
            {"color": "purple", "alpha": 0.5},
        )
    if hist_data:
        plot_history(hist_data, output_dir)

    fvec = results_dict["fvec"]
    bin_idx = int(np.argmin(np.abs(fvec - 4400.0)))
    patterns_to_plot = {
        "MVDR_Steepest": "steepest_weights",
        "MVDR_Newton": "newton_weights",
        "MVDR_Neural": "mvdr_weights",
        "LCMV": "lcmv_weights",
        "GSC": "gsc_weights",
        "GSC_Iterative": "gsc_iterative_weights",
        "LCMV_Weighted": "lcmv_weighted_weights",
        "GSC_Weighted": "gsc_weighted_weights",
        "GSC_Iterative_Weighted": "gsc_iterative_weighted_weights",
    }
    for name, weight_key in patterns_to_plot.items():
        if weight_key in results_dict:
            plot_beam_pattern(
                f"beam_pattern_{name}",
                results_dict[weight_key][bin_idx, :],
                results_dict["mic_pos"],
                fvec[bin_idx],
                sound_speed,
                output_dir,
            )

    return metric_rows


def _localize_with_tracking(
    *,
    method_order: list[str],
    mic_signals: NDArray,
    mic_pos_rel: NDArray,
    fs: int,
    n_targets_hint: int,
) -> tuple[list[float], str, bool, float, str]:
    cfg_map = {m: DEFAULT_METHOD_CONFIGS.get(m, {}) for m in method_order}
    selected, attempted = estimate_doas_with_fallback(
        methods=method_order,
        mic_signals=mic_signals,
        mic_pos_rel=mic_pos_rel,
        fs=fs,
        n_targets_hint=n_targets_hint,
        method_cfg_map=cfg_map,
    )

    fallback_used = len(attempted) > 1 and attempted[0].method != selected.method
    note = ""
    if not selected.doas_deg:
        note = "no_doa_from_localization"
    elif selected.error:
        note = selected.error

    return selected.doas_deg, selected.method, fallback_used, float(selected.confidence), note


def _run_fixed_scenario(
    *,
    output_dir: Path,
    fs: int,
    log: logging.Logger,
    mic_audio: NDArray,
    mic_noise: NDArray,
    mic_pos_rel: NDArray,
    ref_audio: NDArray,
    bf_config: BeamformerConfig,
    source_locations: NDArray | None,
    source_azimuths_deg: NDArray | None,
    target_weights: NDArray | None,
) -> tuple[list[dict], dict]:
    results = compute_beamforming_weights(
        audio_input=mic_audio,
        source_locations=source_locations,
        source_azimuths_deg=source_azimuths_deg,
        noise_audio=mic_noise,
        config=bf_config,
        target_weights=target_weights,
    )
    packet = _build_results_packet(results, mic_pos_rel)
    metrics = evaluate_results(output_dir, fs, log, mic_audio, packet, ref_audio, bf_config.sound_speed)
    metadata = {
        "target_weights": packet.get("target_weights", []),
        "n_targets": int(len(source_azimuths_deg) if source_azimuths_deg is not None else (len(source_locations) if source_locations is not None else 0)),
    }
    return metrics, metadata


def _run_dynamic_scenario(
    *,
    output_dir: Path,
    fs: int,
    log: logging.Logger,
    mic_audio: NDArray,
    mic_noise: NDArray,
    mic_pos_rel: NDArray,
    ref_audio: NDArray,
    bf_config: BeamformerConfig,
    steering_source: str,
    requested_method: str,
    localization_method_order: list[str],
    oracle_locations: NDArray,
    oracle_doas: list[float],
    chunk_seconds: float,
    base_target_weights: list[float] | None,
    target_weight_mode: str,
) -> tuple[list[dict], list[SteeringDecision], dict]:
    chunk_size = max(1, int(chunk_seconds * fs))
    n_samples = mic_audio.shape[0]
    combined: dict[str, list[np.ndarray]] = {k: [] for k in BEAMFORMER_STFT_KEYS.keys()}
    decisions: list[SteeringDecision] = []

    prev_doas = oracle_doas[:] if oracle_doas else []
    used_weight_snapshots: list[list[float]] = []

    for start in range(0, n_samples, chunk_size):
        end = min(n_samples, start + chunk_size)
        chunk_mix = mic_audio[start:end, :]
        chunk_noise = mic_noise[start:end, :]

        if steering_source == "oracle":
            source_locations = oracle_locations
            source_azimuths = None
            used_doas = oracle_doas
            method_used = "oracle"
            fallback_used = False
            confidence = 1.0
            note = ""
        else:
            estimated, method_used, fallback_used, confidence, note = _localize_with_tracking(
                method_order=localization_method_order,
                mic_signals=chunk_mix.T,
                mic_pos_rel=mic_pos_rel,
                fs=fs,
                n_targets_hint=max(1, len(oracle_doas)),
            )
            if not estimated:
                estimated = prev_doas
                if estimated:
                    note = "reuse_previous_chunk_doa"
            prev_doas = estimated[:] if estimated else prev_doas

            source_locations = None
            source_azimuths = np.asarray(estimated, dtype=float) if estimated else None
            used_doas = estimated

        decisions.append(
            SteeringDecision(
                chunk_start_s=start / fs,
                chunk_end_s=end / fs,
                steering_source=steering_source,
                requested_method=requested_method,
                method_used=method_used,
                fallback_used=fallback_used,
                confidence=confidence,
                doas_deg=";".join(f"{d:.2f}" for d in used_doas),
                note=note,
            )
        )

        if source_locations is None and (source_azimuths is None or len(source_azimuths) == 0):
            for key in combined:
                combined[key].append(np.zeros(end - start))
            continue

        n_targets = len(used_doas) if source_azimuths is not None else len(source_locations)
        target_weights = _build_target_weights(base_target_weights, n_targets, target_weight_mode)
        if target_weights is not None:
            used_weight_snapshots.append([float(v) for v in target_weights.tolist()])

        results = compute_beamforming_weights(
            audio_input=chunk_mix,
            source_locations=source_locations,
            source_azimuths_deg=source_azimuths,
            noise_audio=chunk_noise,
            config=bf_config,
            target_weights=target_weights,
        )
        packet = _build_results_packet(results, mic_pos_rel)

        for label, stft_key in BEAMFORMER_STFT_KEYS.items():
            chunk_sig = reconstruct_audio(packet[stft_key], fs, packet["params"], end - start)
            combined[label].append(chunk_sig)

    track_path = output_dir / "doa_tracking.csv"
    with track_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = list(asdict(decisions[0]).keys()) if decisions else [
            "chunk_start_s",
            "chunk_end_s",
            "steering_source",
            "requested_method",
            "method_used",
            "fallback_used",
            "confidence",
            "doas_deg",
            "note",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for d in decisions:
            writer.writerow(asdict(d))

    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    sf.write(audio_dir / "mic_raw_audio.wav", mic_audio, fs)

    metric_rows: list[dict] = []
    for label, parts in combined.items():
        pred = np.concatenate(parts)[:n_samples] if parts else np.zeros(n_samples)
        filename = label.lower().replace(" ", "_").replace("(", "").replace(")", "") + ".wav"
        sf.write(audio_dir / filename, pred, fs)
        aligned_ref = align_signals(ref_audio, pred)
        pred_norm, gain_db = match_rms_to_reference(pred, aligned_ref)
        sf.write(audio_dir / filename.replace(".wav", "_norm_to_ref.wav"), pred_norm, fs)

        rmse_raw, _ = calc_rmse(aligned_ref, pred)
        snr_raw = calc_snr(aligned_ref, pred)
        sdr_raw = calc_si_sdr(aligned_ref, pred)
        rmse_norm, _ = calc_rmse(aligned_ref, pred_norm)
        snr_norm = calc_snr(aligned_ref, pred_norm)
        sdr_norm = calc_si_sdr(aligned_ref, pred_norm)

        log.info(
            f"[dynamic] {label: <30}: RAW RMSE={rmse_raw:.4f}, SNR={snr_raw:.4f}dB, SI-SDR={sdr_raw:.4f}dB | "
            f"NORM RMSE={rmse_norm:.4f}, SNR={snr_norm:.4f}dB, SI-SDR={sdr_norm:.4f}dB, gain={gain_db:.2f}dB"
        )
        metric_rows.append(
            {
                "beamformer": label,
                "rmse": float(rmse_raw),
                "snr_db": float(snr_raw),
                "si_sdr_db": float(sdr_raw),
                "rmse_raw": float(rmse_raw),
                "snr_db_raw": float(snr_raw),
                "si_sdr_db_raw": float(sdr_raw),
                "rmse_norm": float(rmse_norm),
                "snr_db_norm": float(snr_norm),
                "si_sdr_db_norm": float(sdr_norm),
                "gain_to_ref_db": float(gain_db),
            }
        )

    metadata = {
        "target_weights_snapshots": used_weight_snapshots,
        "num_chunks": len(decisions),
        "fallback_rate": float(sum(1 for d in decisions if d.fallback_used) / max(1, len(decisions))),
    }
    return metric_rows, decisions, metadata


def _write_summary_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _load_scene_inputs(args: argparse.Namespace, config_data: dict) -> list[tuple[str, SimulationConfig, str]]:
    if args.simulation_scene_file is not None:
        path = args.simulation_scene_file
        cfg = SimulationConfig.from_file(path)
        return [(path.stem, cfg, str(path))]

    if args.simulation_scene_dir is not None:
        paths = sorted(args.simulation_scene_dir.glob("*.json"))
        if args.max_scenes is not None:
            paths = paths[: args.max_scenes]
        return [(p.stem, SimulationConfig.from_file(p), str(p)) for p in paths]

    sim_cfg = SimulationConfig.from_dict(config_data["simulation"])
    return [("config_scene", sim_cfg, "")]


def _build_noise_only_config(sim_config: SimulationConfig) -> SimulationConfig:
    noise_sources = [s for s in sim_config.audio.sources if not is_speech_target(s)]
    if not noise_sources:
        # If a scene has only speech-labeled sources, keep previous behavior by allowing
        # a later fallback to mixture-as-noise.
        return SimulationConfig(
            room=sim_config.room,
            microphone_array=sim_config.microphone_array,
            audio=SimulationAudio(sources=[], duration=sim_config.audio.duration, fs=sim_config.audio.fs),
        )

    return SimulationConfig(
        room=sim_config.room,
        microphone_array=sim_config.microphone_array,
        audio=SimulationAudio(sources=noise_sources, duration=sim_config.audio.duration, fs=sim_config.audio.fs),
    )


def _run_scene(
    *,
    scene_name: str,
    scene_path: str,
    sim_config: SimulationConfig,
    bf_config_template: BeamformerConfig,
    output_dir: Path,
    log: logging.Logger,
    args: argparse.Namespace,
    base_target_weights: list[float] | None,
) -> list[dict]:
    scene_out_dir = output_dir / scene_name
    scene_out_dir.mkdir(parents=True, exist_ok=True)

    fs = sim_config.audio.fs
    print(f"Running Simulation (Mixture) for {scene_name}...")
    mic_audio, mic_pos_abs, source_signals = run_simulation(sim_config)
    min_samples = mic_audio.shape[0]

    mic_array_center = np.array(sim_config.microphone_array.mic_center)
    mic_pos_rel = mic_pos_abs - mic_array_center.reshape(3, 1)

    bf_config = BeamformerConfig(**vars(bf_config_template))
    bf_config.mic_array_center = mic_array_center
    bf_config.mic_geometry = mic_pos_rel

    plot_sources = [
        Audio_Sources(input=s.audio_path, loc=s.loc, classification=s.classification)
        for s in sim_config.audio.sources
    ]
    plot_mic_pos(mic_pos_rel, scene_out_dir)
    plot_room_pos(np.array(sim_config.room.dimensions), mic_array_center, plot_sources, scene_out_dir)

    print(f"Running Simulation (Noise Only) for {scene_name}...")
    noise_cfg = _build_noise_only_config(sim_config)
    try:
        mic_noise, _, _ = run_simulation(noise_cfg)
    except ValueError:
        log.warning(f"No explicit noise sources found for {scene_name}; using mixture audio as noise reference.")
        mic_noise = mic_audio.copy()
    if mic_noise.shape[0] > min_samples:
        mic_noise = mic_noise[:min_samples, :]
    elif mic_noise.shape[0] < min_samples:
        mic_noise = np.pad(mic_noise, ((0, min_samples - mic_noise.shape[0]), (0, 0)))

    ref_audio = _compute_ref_audio(sim_config, source_signals, min_samples)
    mic_signals = mic_audio.T
    oracle_locations = oracle_target_locations(sim_config)
    oracle_doas = oracle_target_doas_deg(sim_config)
    target_idxs = list(iter_target_source_indices(sim_config))
    source_debug = [
        {
            "index": i,
            "audio_path": s.audio_path,
            "classification": s.classification,
            "is_target": i in target_idxs,
        }
        for i, s in enumerate(sim_config.audio.sources)
    ]

    steering_sources = ["oracle", "localized"] if args.steering_source == "both" else [args.steering_source]
    time_modes = ["fixed", "dynamic"] if args.steering_time == "both" else [args.steering_time]

    if args.localization_methods:
        localized_method_modes = args.localization_methods
    else:
        localized_method_modes = [args.steering_localization_default]

    summary_rows: list[dict] = []

    for time_mode in time_modes:
        for steering_source in steering_sources:
            methods = [None] if steering_source == "oracle" else localized_method_modes

            for method in methods:
                requested = method if method else args.steering_localization_default
                method_order = _compose_method_order(requested, args.steering_localization_fallbacks)

                scenario_name = f"{time_mode}_{steering_source}" + (f"_{requested}" if steering_source == "localized" else "")
                scenario_dir = scene_out_dir / scenario_name
                scenario_dir.mkdir(parents=True, exist_ok=True)

                scenario_meta = {
                    "scene": scene_name,
                    "scene_path": scene_path,
                    "time_mode": time_mode,
                    "steering_source": steering_source,
                    "requested_method": requested if steering_source == "localized" else "",
                    "method_order": method_order if steering_source == "localized" else [],
                    "metric_modes": ["raw", "norm_to_ref"],
                    "target_selection_debug": {
                        "target_indices": target_idxs,
                        "n_targets": len(target_idxs),
                        "sources": source_debug,
                    },
                }

                if time_mode == "fixed":
                    decisions: list[SteeringDecision] = []
                    if steering_source == "oracle":
                        doas_for_weight = oracle_doas
                        metric_rows, fixed_meta = _run_fixed_scenario(
                            output_dir=scenario_dir,
                            fs=fs,
                            log=log,
                            mic_audio=mic_audio,
                            mic_noise=mic_noise,
                            mic_pos_rel=mic_pos_rel,
                            ref_audio=ref_audio,
                            bf_config=bf_config,
                            source_locations=oracle_locations,
                            source_azimuths_deg=None,
                            target_weights=_build_target_weights(base_target_weights, len(oracle_doas), args.target_weight_mode),
                        )
                        doas_desc = ";".join(f"{d:.2f}" for d in oracle_doas)
                        decisions.append(
                            SteeringDecision(
                                chunk_start_s=0.0,
                                chunk_end_s=min_samples / fs,
                                steering_source="oracle",
                                requested_method="",
                                method_used="oracle",
                                fallback_used=False,
                                confidence=1.0,
                                doas_deg=doas_desc,
                                note="",
                            )
                        )
                    else:
                        est_doas, method_used, fallback_used, confidence, note = _localize_with_tracking(
                            method_order=method_order,
                            mic_signals=mic_signals,
                            mic_pos_rel=mic_pos_rel,
                            fs=fs,
                            n_targets_hint=max(1, len(oracle_doas)),
                        )
                        doas_for_weight = est_doas
                        if not est_doas:
                            log.warning(f"Skipping scenario {scenario_name}: no DOAs estimated")
                            continue
                        metric_rows, fixed_meta = _run_fixed_scenario(
                            output_dir=scenario_dir,
                            fs=fs,
                            log=log,
                            mic_audio=mic_audio,
                            mic_noise=mic_noise,
                            mic_pos_rel=mic_pos_rel,
                            ref_audio=ref_audio,
                            bf_config=bf_config,
                            source_locations=None,
                            source_azimuths_deg=np.asarray(est_doas, dtype=float),
                            target_weights=_build_target_weights(base_target_weights, len(est_doas), args.target_weight_mode),
                        )
                        doas_desc = ";".join(f"{d:.2f}" for d in est_doas)
                        decisions.append(
                            SteeringDecision(
                                chunk_start_s=0.0,
                                chunk_end_s=min_samples / fs,
                                steering_source="localized",
                                requested_method=requested,
                                method_used=method_used,
                                fallback_used=fallback_used,
                                confidence=confidence,
                                doas_deg=doas_desc,
                                note=note,
                            )
                        )

                    scenario_meta.update(
                        {
                            "doas": doas_for_weight,
                            "target_weights": fixed_meta.get("target_weights", []),
                            "steering_decisions": [asdict(d) for d in decisions],
                        }
                    )
                else:
                    metric_rows, decisions, dyn_meta = _run_dynamic_scenario(
                        output_dir=scenario_dir,
                        fs=fs,
                        log=log,
                        mic_audio=mic_audio,
                        mic_noise=mic_noise,
                        mic_pos_rel=mic_pos_rel,
                        ref_audio=ref_audio,
                        bf_config=bf_config,
                        steering_source=steering_source,
                        requested_method=requested,
                        localization_method_order=method_order,
                        oracle_locations=oracle_locations,
                        oracle_doas=oracle_doas,
                        chunk_seconds=args.dynamic_chunk_seconds,
                        base_target_weights=base_target_weights,
                        target_weight_mode=args.target_weight_mode,
                    )
                    doas_desc = "dynamic"
                    scenario_meta.update(
                        {
                            "doas": doas_desc,
                            "target_weights_snapshots": dyn_meta.get("target_weights_snapshots", []),
                            "fallback_rate": dyn_meta.get("fallback_rate", 0.0),
                            "steering_decisions": [asdict(d) for d in decisions],
                        }
                    )

                with (scenario_dir / "scenario_metadata.json").open("w", encoding="utf-8") as f:
                    json.dump(scenario_meta, f, indent=2)

                for row in metric_rows:
                    summary_rows.append(
                        {
                            "scene": scene_name,
                            "scenario": scenario_name,
                            "time_mode": time_mode,
                            "steering_source": steering_source,
                            "localization_method": requested if steering_source == "localized" else "",
                            "doas_deg": doas_desc,
                            **row,
                        }
                    )

    _write_summary_csv(scene_out_dir / "steering_comparison.csv", summary_rows)
    return summary_rows


def main():
    parser = argparse.ArgumentParser(description="Beamforming simulation")
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "config" / "config.json")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--steering-source", choices=["oracle", "localized", "both"], default="both")
    parser.add_argument("--steering-time", choices=["fixed", "dynamic", "both"], default="both")
    parser.add_argument("--localization-methods", nargs="+", default=None)
    parser.add_argument("--steering-localization-default", default="SSZ")
    parser.add_argument("--steering-localization-fallbacks", nargs="+", default=["GMDA"])
    parser.add_argument("--dynamic-chunk-seconds", type=float, default=1.0)
    parser.add_argument("--target-weights-file", type=Path, default=None)
    parser.add_argument("--target-weight-mode", choices=["equal", "config"], default="equal")
    parser.add_argument("--simulation-scene-file", type=Path, default=None)
    parser.add_argument("--simulation-scene-dir", type=Path, default=None)
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--force-mic-count", type=int, default=None)
    parser.add_argument("--force-mic-radius", type=float, default=None)
    parser.add_argument("--causal-only", action="store_true")
    args = parser.parse_args()

    with args.config.open("r", encoding="utf-8") as f:
        full_config_data = json.load(f)

    if args.simulation_scene_file is not None and args.simulation_scene_dir is not None:
        raise ValueError("Use either --simulation-scene-file or --simulation-scene-dir, not both.")
    if args.causal_only and args.steering_time != "dynamic":
        raise ValueError("--causal-only requires --steering-time dynamic.")

    log = logging.getLogger("Beamforming")
    if not log.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
        log.addHandler(handler)
        log.setLevel(logging.INFO)

    scenes = _load_scene_inputs(args, full_config_data)
    if not scenes:
        raise RuntimeError("No simulation scenes selected.")

    template_sim_cfg = scenes[0][1]
    bf_config = BeamformerConfig.from_dict(full_config_data, fs=template_sim_cfg.audio.fs)
    if not args.localization_methods:
        args.steering_localization_default = bf_config.localization_default_method or args.steering_localization_default
        if bf_config.localization_fallback_methods:
            args.steering_localization_fallbacks = list(bf_config.localization_fallback_methods)

    base_target_weights = _load_target_weights(args.target_weights_file, full_config_data)

    output_dir = Path(args.output if args.output else bf_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    global_rows: list[dict] = []
    for scene_name, sim_cfg, scene_path in scenes:
        if args.force_mic_count is not None:
            sim_cfg.microphone_array.mic_count = int(args.force_mic_count)
        if args.force_mic_radius is not None:
            sim_cfg.microphone_array.mic_radius = float(args.force_mic_radius)
        rows = _run_scene(
            scene_name=scene_name,
            scene_path=scene_path,
            sim_config=sim_cfg,
            bf_config_template=bf_config,
            output_dir=output_dir,
            log=log,
            args=args,
            base_target_weights=base_target_weights,
        )
        global_rows.extend(rows)

    _write_summary_csv(output_dir / "steering_comparison.csv", global_rows)
    log.info(f"Beamforming simulation completed - output saved to {output_dir}")


if __name__ == "__main__":
    main()

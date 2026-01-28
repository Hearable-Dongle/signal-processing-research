import argparse
import json
import logging
from pathlib import Path

import numpy as np
import soundfile as sf
from numpy.typing import NDArray
from scipy.signal import istft

from beamforming.algo.beamformer import apply_beamformer_stft
from beamforming.util.compare import align_signals, calc_rmse, calc_si_sdr, calc_snr
from beamforming.util.visualize import plot_beam_pattern, plot_history, plot_mic_pos, plot_room_pos
from beamforming.util.configure import Audio_Sources
from beamforming.beamformer_core import compute_beamforming_weights, BeamformerConfig
from simulation.simulation_config import SimulationConfig
from simulation.simulator import run_simulation


def reconstruct_audio(
    stft_data: NDArray, 
    fs: int, 
    window_params: tuple, 
    target_length: int
) -> NDArray:
    """Helper to perform ISTFT and truncate/pad to original length."""
    win_size, hop, window = window_params
    _, time_signal = istft(
        stft_data,
        fs=fs,
        nperseg=win_size,
        noverlap=win_size - hop,
        window=window
    )
    time_signal = np.real(time_signal)
    
    if len(time_signal) > target_length:
        return time_signal[:target_length]
    else:
        return np.pad(time_signal, (0, target_length - len(time_signal)))


def evaluate_results(
    output_dir: Path, 
    fs: int, 
    log: logging.Logger, 
    mic_audio: NDArray, 
    results_dict: dict, 
    ref_audio: NDArray, 
    sound_speed: float
):
    """Calculates metrics (RMSE, SNR, SI-SDR) and plots results."""
    
    methods = {
        "MVDR (Iterative Steepest)": "steepest_stft",
        "MVDR (Iterative Newton)": "newton_stft",
        "MVDR (Neural)": "mvdr_stft",
        "LCMV (Closed Form)": "lcmv_stft",
        "GSC (Closed Form)": "gsc_stft",
        "GSC (Iterative)": "gsc_iterative_stft"
    }

    audio_dir = output_dir / "audio"
    if not audio_dir.exists():
        audio_dir.mkdir(parents=True)
        
    sf.write(audio_dir / "mic_raw_audio.wav", mic_audio, fs)
    
    reconstructed_signals = {}
    for label, key in methods.items():
        if key in results_dict:
            time_sig = reconstruct_audio(
                results_dict[key], 
                fs, 
                results_dict["params"], 
                len(ref_audio)
            )
            reconstructed_signals[label] = time_sig
            filename = label.lower().replace(" ", "_").replace("(", "").replace(")", "") + ".wav"
            sf.write(audio_dir / filename, time_sig, fs)

    def print_metrics(name, pred_sig):
        aligned_ref = align_signals(ref_audio, pred_sig)

        rmse, _mse = calc_rmse(aligned_ref, pred_sig)
        snr = calc_snr(aligned_ref, pred_sig)
        sdr = calc_si_sdr(aligned_ref, pred_sig)
        
        log.info(f"{name: <25}: RMSE={rmse:.4f}, SNR={snr:.4f}dB, SI-SDR={sdr:.4f}dB")

    log.info("-" * 50)
    print_metrics("Raw Audio (Mean)", np.mean(mic_audio, axis=1))
    for label, sig in reconstructed_signals.items():
        print_metrics(label, sig)
    log.info("-" * 50)

    # Plot Convergence History
    hist_data = {}
    if "steepest_hist" in results_dict:
        hist_data["MVDR Steepest"] = (np.mean(results_dict["steepest_hist"], axis=0), {"color": "blue", "alpha": 0.5})
    if "newton_hist" in results_dict:
        hist_data["MVDR Newton"] = (np.mean(results_dict["newton_hist"], axis=0), {"color": "green", "alpha": 0.5})
    if "gsc_iterative_hist" in results_dict:
        hist_data["GSC Iterative"] = (np.mean(results_dict["gsc_iterative_hist"], axis=0), {"color": "red", "alpha": 0.5})

    if hist_data:
        plot_history(hist_data, output_dir)

    # Beam Patterns
    fvec = results_dict["fvec"]
    bin_idx = int(np.argmin(np.abs(fvec - 4400.0)))
    
    patterns_to_plot = {
        "MVDR_Steepest": "steepest_weights",
        "MVDR_Newton": "newton_weights",
        "MVDR_Neural": "mvdr_weights",
        "LCMV": "lcmv_weights",
        "GSC": "gsc_weights",
        "GSC_Iterative": "gsc_iterative_weights"
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


def main():
    parser = argparse.ArgumentParser(description="Beamforming simulation")
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "config" / "config.json")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    # Load Configs
    with args.config.open("r") as f:
        full_config_data = json.load(f)

    # Load Simulation Config
    sim_config = SimulationConfig.from_dict(full_config_data["simulation"])
    
    # Load Beamformer Config
    bf_config = BeamformerConfig.from_dict(full_config_data, fs=sim_config.audio.fs)

    log = logging.getLogger("Beamforming")
    if not log.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        log.addHandler(handler)
        log.setLevel(logging.INFO)

    output_dir_str = args.output if args.output else bf_config.output_dir
    output_dir = Path(output_dir_str)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    sound_speed = bf_config.sound_speed
    fs = sim_config.audio.fs

    # Run Simulation for Mixed Audio
    print("Running Simulation (Mixture)...")
    mic_audio, mic_pos, source_signals = run_simulation(sim_config)
    min_samples = mic_audio.shape[0]

    # Reconstruct source objects for plot_room_pos
    plot_sources = []
    signal_locs = []

    for i, s_conf in enumerate(sim_config.audio.sources):
        cls = s_conf.classification
        plot_sources.append(Audio_Sources(input=s_conf.audio_path, loc=s_conf.loc, classification=cls))
        
        if cls == "signal":
            signal_locs.append(s_conf.loc)
    
    # Plot Setup
    # mic_pos from run_simulation is absolute
    mic_array_center = np.array(sim_config.microphone_array.mic_center)
    mic_pos_rel = mic_pos - mic_array_center.reshape(3, 1)

    plot_mic_pos(mic_pos_rel, output_dir)
    plot_room_pos(np.array(sim_config.room.dimensions), mic_array_center, plot_sources, output_dir)
    
    signal_loc = np.array(signal_locs)

    # Run Simulation for Noise Only (Ground Truth Noise)
    print("Running Simulation (Noise Only)...")
    noise_config = sim_config.create_noise_config()
    
    mic_noise, _, _ = run_simulation(noise_config)
    
    if mic_noise.shape[0] > min_samples:
        mic_noise = mic_noise[:min_samples, :]
    elif mic_noise.shape[0] < min_samples:
        pad_amt = min_samples - mic_noise.shape[0]
        mic_noise = np.pad(mic_noise, ((0, pad_amt), (0, 0)))

    # Update Beamformer Config with geometry. TODO: make this consistent across beamforming and simulation
    bf_config.mic_array_center = mic_array_center
    bf_config.mic_geometry = mic_pos_rel

    print("Computing Beamforming Weights...")
    results = compute_beamforming_weights(
        audio_input=mic_audio,
        source_locations=signal_loc,
        noise_audio=mic_noise,
        config=bf_config
    )

    stft_tensor = results["stft_tensor"]
    
    results_packet = {
        "params": results["params"],
        "fvec": results["fvec"],
        "mic_pos": mic_pos_rel,
        
        # MVDR
        "steepest_weights": results["steepest"][0],
        "steepest_hist": results["steepest"][1],
        "steepest_stft": apply_beamformer_stft(stft_tensor, results["steepest"][0]),
        
        "newton_weights": results["newton"][0],
        "newton_hist": results["newton"][1],
        "newton_stft": apply_beamformer_stft(stft_tensor, results["newton"][0]),
        
        "mvdr_weights": results["mvdr"][0],
        "mvdr_stft": apply_beamformer_stft(stft_tensor, results["mvdr"][0]),
        
        # LCMV
        "lcmv_weights": results["lcmv"][0],
        "lcmv_stft": apply_beamformer_stft(stft_tensor, results["lcmv"][0]),
        
        # GSC
        "gsc_weights": results["gsc"][0],
        "gsc_stft": apply_beamformer_stft(stft_tensor, results["gsc"][0]),
        
        "gsc_iterative_weights": results["gsc_iterative"][0],
        "gsc_iterative_hist": results["gsc_iterative"][1],
        "gsc_iterative_stft": apply_beamformer_stft(stft_tensor, results["gsc_iterative"][0]),
    }

    # Construct Ref Audio (Sum of clean signals)
    ref_audio = np.zeros(min_samples)
    for i, s_conf in enumerate(sim_config.audio.sources):
        if s_conf.classification == "signal":
            # source_signals is aligned with config.audio.sources indices from first run
            sig = source_signals[i]
            if len(sig) > min_samples: sig = sig[:min_samples]
            else: sig = np.pad(sig, (0, min_samples - len(sig)))
            ref_audio += sig

    evaluate_results(output_dir, fs, log, mic_audio, results_packet, ref_audio, sound_speed)
    
    log.info(f"Beamforming simulation completed - output saved to {output_dir}")


if __name__ == "__main__":
    main()

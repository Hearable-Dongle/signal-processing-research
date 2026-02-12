import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt
import argparse

from scipy.io import wavfile

from beamforming.util.visualize import plot_room_pos, plot_mic_pos
from beamforming.util.configure import Audio_Sources
from localization.algo import SSZLocalization, GMDALaplace, SRPPHATLocalization
from localization.visualization import plot_source_comparison
from localization.localization_config import LocalizationConfig
from simulation.simulation_config import SimulationConfig
from simulation.simulator import run_simulation

def main():
    parser = argparse.ArgumentParser(description="Run Audio Localization Simulation")
    parser.add_argument("--config", type=str, default="localization/configs/base_config.json", 
                        help="Path to the JSON configuration file")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Configuration file not found at {config_path}")
        return

    with config_path.open("r") as f:
        full_config_data = json.load(f)

    sim_config = SimulationConfig.from_dict(full_config_data["simulation"])
    algo_config = LocalizationConfig.from_dict(full_config_data["localization"])
    
    # Check for output_dir at top level first, then in localization config, then default
    output_dir_str = full_config_data.get("output_dir") or algo_config.output_dir or f"localization/output/{config_path.stem}"
    output_dir = Path(output_dir_str)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Configuration loaded from {config_path}")
    print("Running simulation...")
    
    mic_audio, mic_pos_abs, source_signals = run_simulation(sim_config)
    
    # Transpose mic_audio to (channels, samples) for localization algorithms
    mic_signals = mic_audio.T
    
    # Prepare relative mic positions
    mic_center = np.array(sim_config.microphone_array.mic_center).reshape(3, 1)
    mic_pos_rel = mic_pos_abs - mic_center

    fs = sim_config.audio.fs
    
    # Save simulated audio
    wav_out_path = output_dir / "simulated_mic_audio.wav"
    sig_norm = mic_signals.T
    if np.max(np.abs(sig_norm)) > 0:
        sig_norm = sig_norm / np.max(np.abs(sig_norm)) * 32767
    wavfile.write(wav_out_path, fs, sig_norm.astype(np.int16))
    print(f"Saved simulated audio to {wav_out_path}")
    
    print("Running localization algorithm...")
    
    algo_type = algo_config.algo_type
    
    if algo_type == "GMDA":
        print("Using GMDA-Laplace algorithm...")
        loc_system = GMDALaplace(
            mic_pos=mic_pos_rel,
            fs=fs,
            nfft=algo_config.nfft,
            overlap=algo_config.overlap,
            freq_range=algo_config.freq_range,
            max_sources=algo_config.max_sources,
            power_thresh_percentile=algo_config.power_thresh_percentile,
            mdl_beta=algo_config.mdl_beta
        )
    elif algo_type == "SRP-PHAT":
        print("Using SRP-PHAT (Power-Weighted) algorithm...")
        loc_system = SRPPHATLocalization(
            mic_pos=mic_pos_rel,
            fs=fs,
            nfft=algo_config.nfft,
            overlap=algo_config.overlap,
            freq_range=algo_config.freq_range,
            max_sources=algo_config.max_sources
        )
    elif algo_type == "SSZ":
        print("Using SSZ Localization algorithm...")
        loc_system = SSZLocalization(
            mic_pos=mic_pos_rel,
            fs=fs,
            nfft=algo_config.nfft,
            overlap=algo_config.overlap,
            epsilon=algo_config.epsilon,
            d_freq=algo_config.d_freq,
            freq_range=algo_config.freq_range,
            max_sources=algo_config.max_sources
        )
    else:
        supported = ["SSZ", "SRP-PHAT", "GMDA"]
        raise ValueError(
            f"Unsupported localization type '{algo_type}'. "
            f"Supported types: {supported}. "
            "Note: AI localization configs are no longer supported."
        )
    
    estimated_doas_rad, histogram, ssz_history = loc_system.process(mic_signals)
    
    estimated_doas_deg = np.degrees(estimated_doas_rad)
    print("\nEstimated DOAs (Average/Global):")
    for i, ang in enumerate(estimated_doas_deg):
        print(f"  Detection {i}: {ang:.2f} deg")
        
    print(f"\nSaving visualizations to {output_dir}")
    
    # Reconstruct objects for visualization
    true_doas = []
    audio_sources_objects = []
    
    mic_center_flat = sim_config.microphone_array.mic_center
    
    for i, s_conf in enumerate(sim_config.audio.sources):
        loc = s_conf.loc
        cls = s_conf.classification
        
        # Calculate True DOA
        dx = loc[0] - mic_center_flat[0]
        dy = loc[1] - mic_center_flat[1]
        angle = np.arctan2(dy, dx)
        if angle < 0: angle += 2*np.pi
        angle_deg = np.degrees(angle)
        
        if cls == "signal":
             true_doas.append(angle_deg)
        
        as_obj = Audio_Sources(input=s_conf.audio_path, loc=loc, classification=cls)
        audio_sources_objects.append(as_obj)

    plot_source_comparison(true_doas, estimated_doas_deg, output_dir)
    
    plot_room_pos(
        np.array(sim_config.room.dimensions),
        np.array(mic_center_flat),
        audio_sources_objects,
        output_dir
    )
    
    plot_mic_pos(mic_pos_rel, output_dir) 
    
    plt.figure()
    plt.plot(np.linspace(0, 360, len(histogram)), histogram)
    plt.title("Angular Histogram (Pre-Matching Pursuit)")
    plt.xlabel("Angle (Degrees)")
    plt.ylabel("Count")
    plt.grid(True)
    plt.savefig(output_dir / "angular_histogram.png")
    plt.close()

    if ssz_history:
        times, angles = zip(*ssz_history)
        angles_deg = np.degrees(angles)
        plt.figure(figsize=(10, 6))
        plt.scatter(times, angles_deg, alpha=0.5, s=5)
        plt.title("Estimated DOA over Time")
        plt.xlabel("Time (s)")
        plt.ylabel("Angle (Degrees)")
        plt.ylim(0, 360)
        plt.grid(True)
        for td in true_doas:
            plt.axhline(y=td, color='r', linestyle='--', label=f'True DOA {td:.1f}')
        plt.legend()
        plt.savefig(output_dir / "doa_over_time.png")
        plt.close()
    
    print("Done.")

if __name__ == "__main__":
    main()

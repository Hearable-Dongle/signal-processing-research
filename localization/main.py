import numpy as np
from pathlib import Path
import sys

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

import matplotlib.pyplot as plt

from beamforming.util.simulate import sim_mic, sim_room, MicType
from beamforming.util.visualize import plot_room_pos, plot_mic_pos
from beamforming.util.configure import Audio_Sources

from localization.algo import SSZLocalization, GMDALaplace
from localization.visualization import plot_source_comparison
from localization.config_loader import load_config

import argparse

def main():
    parser = argparse.ArgumentParser(description="Run Audio Localization Simulation")
    parser.add_argument("--config", type=str, default="localization/configs/base_config.json", 
                        help="Path to the JSON configuration file")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Configuration file not found at {config_path}")
        return

    config = load_config(config_path)
    
    sim_config = config["simulation"]
    algo_config = config["algorithm"]

    fs = sim_config["fs"]
    room_dim = sim_config["room_dim"]
    mic_center = np.array(sim_config["mic_center"])
    mic_radius = sim_config["mic_radius"]
    mic_count = sim_config["mic_count"]
    source_locs = sim_config["source_locs"]
    duration = sim_config["duration"]
    
    output_dir = Path(sim_config.get("output_dir", f"localization/output/{config_path.stem}"))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Setting up simulation in {room_dim}m room...")
    print(f"Configuration loaded from {config_path}")
    
    mic_array_pra, mic_pos_rel = sim_mic(
        mic_count=mic_count,
        mic_loc=mic_center,
        mic_spacing=mic_radius,
        mic_type=MicType.CIRCULAR,
        mic_fs=fs
    )
    
    room = sim_room(room_dim, fs=fs, reflection_count=10)
    
    room.add_microphone_array(mic_array_pra)
    
    num_samples = int(fs * duration)
    
    # --- Source Signal Generation ---
    source_files = sim_config.get("source_files", [])
    source_signals = []
    
    from scipy.io import wavfile
    import scipy.signal as signal_ops

    for i, wav_path_str in enumerate(source_files):
        wav_path = Path(wav_path_str)
        if wav_path.exists():
            sr_wav, audio_wav = wavfile.read(wav_path)
            if sr_wav != fs:
                num_samples_wav = int(len(audio_wav) * fs / sr_wav)
                audio_wav = signal_ops.resample(audio_wav, num_samples_wav)
            
            # Normalize
            audio_wav = audio_wav.astype(float)
            if np.max(np.abs(audio_wav)) > 0:
                audio_wav /= np.max(np.abs(audio_wav))
            
            # Loop or crop
            if len(audio_wav) < num_samples:
                repeats = int(np.ceil(num_samples / len(audio_wav)))
                sig = np.tile(audio_wav, repeats)[:num_samples]
            else:
                sig = audio_wav[:num_samples]
            source_signals.append(sig)
            print(f"Loaded source {i} from {wav_path}")
        else:
            print(f"Warning: File {wav_path} not found. Using silence.")
            source_signals.append(np.zeros(num_samples))

    audio_sources_objects = []
    true_doas = []
    
    print("Adding sources:")
    for i, loc in enumerate(source_locs):
        if i < len(source_signals):
            signal = source_signals[i]
        else:
            signal = np.zeros(num_samples)
        
        room.add_source(loc, signal=signal)
        
        as_obj = Audio_Sources(input=f"Source_{i}", loc=loc, classification="signal")
        audio_sources_objects.append(as_obj)
        
        dx = loc[0] - mic_center[0]
        dy = loc[1] - mic_center[1]
        angle = np.arctan2(dy, dx)
        if angle < 0: angle += 2*np.pi
        angle_deg = np.degrees(angle)
        true_doas.append(angle_deg)
        print(f"  Source {i}: {loc} -> True DOA: {angle_deg:.2f} deg")

    print("Simulating audio...")
    room.simulate()
    
    mic_signals = room.mic_array.signals
    print(f"Captured signals shape: {mic_signals.shape}")

    # --- Add Background Noise ---
    noise_config = sim_config.get("noise", None)
    if noise_config:
        print(f"Adding background noise: {noise_config}")
        noise_signal = None
        n_samples_out = mic_signals.shape[1]
        
        if noise_config == "white_noise":
            # Add uncorrelated white noise
            noise_signal = np.random.randn(*mic_signals.shape)
        else:
            # Assume file path
            noise_path = Path(noise_config)
            if noise_path.exists():
                sr_wav, audio_noise = wavfile.read(noise_path)
                if sr_wav != fs:
                    ns_noise = int(len(audio_noise) * fs / sr_wav)
                    audio_noise = signal_ops.resample(audio_noise, ns_noise)
                
                audio_noise = audio_noise.astype(float)
                if np.max(np.abs(audio_noise)) > 0:
                    audio_noise /= np.max(np.abs(audio_noise))
                
                # Make long enough
                if len(audio_noise) < n_samples_out:
                    repeats = int(np.ceil(n_samples_out / len(audio_noise)))
                    full_noise = np.tile(audio_noise, repeats)
                else:
                    full_noise = audio_noise
                
                # Add to mics (rolling to decorrelate)
                noise_signal = np.zeros_like(mic_signals)
                for m in range(mic_signals.shape[0]):
                    # Random start index
                    start = np.random.randint(0, len(full_noise) - n_samples_out + 1) if len(full_noise) > n_samples_out else 0
                    # Or just roll
                    shift = np.random.randint(0, len(full_noise))
                    rolled = np.roll(full_noise, shift)
                    noise_signal[m, :] = rolled[:n_samples_out]
            else:
                print(f"Warning: Noise file {noise_path} not found.")

        if noise_signal is not None:
            # Scale noise. Let's assume 10% amplitude of current max signal
            current_max = np.max(np.abs(mic_signals))
            noise_max = np.max(np.abs(noise_signal))
            if noise_max > 0:
                scale_factor = (current_max * 0.1) / noise_max
                mic_signals += noise_signal * scale_factor
                print(f"Added noise with scale factor {scale_factor:.4f}")
    
    # Save simulated audio
    from scipy.io import wavfile
    wav_out_path = output_dir / "simulated_mic_audio.wav"
    sig_norm = mic_signals.T
    if np.max(np.abs(sig_norm)) > 0:
        sig_norm = sig_norm / np.max(np.abs(sig_norm)) * 32767
    wavfile.write(wav_out_path, fs, sig_norm.astype(np.int16))
    print(f"Saved simulated audio to {wav_out_path}")
    
    print("Running localization algorithm...")
    
    algo_type = algo_config.get("type", "SSZ")
    
    if algo_type == "GMDA":
        print("Using GMDA-Laplace algorithm...")
        loc_system = GMDALaplace(
            mic_pos=mic_pos_rel,
            fs=fs,
            nfft=algo_config.get("nfft", 512),
            overlap=algo_config.get("overlap", 0.5),
            freq_range=tuple(algo_config.get("freq_range", [200, 3000])),
            max_sources=algo_config.get("max_sources", 4),
            power_thresh_percentile=algo_config.get("power_thresh_percentile", 90),
            mdl_beta=algo_config.get("mdl_beta", 0.6)
        )
    else:
        print("Using SSZ Localization algorithm...")
        loc_system = SSZLocalization(
            mic_pos=mic_pos_rel,
            fs=fs,
            nfft=algo_config.get("nfft", 512),
            overlap=algo_config.get("overlap", 0.5),
            epsilon=algo_config.get("epsilon", 0.2),
            d_freq=algo_config.get("d_freq", 2),
            freq_range=tuple(algo_config.get("freq_range", [200, 3000])),
            max_sources=algo_config.get("max_sources", 4)
        )
    
    estimated_doas_rad, histogram, ssz_history = loc_system.process(mic_signals)
    
    estimated_doas_deg = np.degrees(estimated_doas_rad)
    print("\nEstimated DOAs (Average/Global):")
    for i, ang in enumerate(estimated_doas_deg):
        print(f"  Detection {i}: {ang:.2f} deg")
        
    print(f"\nSaving visualizations to {output_dir}")
    
    plot_source_comparison(true_doas, estimated_doas_deg, output_dir)
    
    plot_room_pos(
        np.array(room_dim),
        mic_center,
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

    # Plot DOA over Time
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
        # Add ground truth lines
        for td in true_doas:
            plt.axhline(y=td, color='r', linestyle='--', label=f'True DOA {td:.1f}')
        plt.legend()
        plt.savefig(output_dir / "doa_over_time.png")
        plt.close()
    
    print("Done.")

if __name__ == "__main__":
    main()
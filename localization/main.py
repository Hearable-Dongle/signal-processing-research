import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Imports from existing codebase
from beamforming.util.simulate import sim_mic, sim_room, MicType
from beamforming.util.visualize import plot_room_pos, plot_mic_pos
from beamforming.util.configure import Audio_Sources

# Import our localization system
from localization.algo import LocalizationSystem

def main():
    # --- Configuration ---
    fs = 16000
    room_dim = [5, 5, 3]
    mic_center = np.array([2.5, 2.5, 1.5])
    mic_radius = 0.1
    mic_count = 8
    
    # Sources: [x, y, z]
    source_locs = [
        [3.5, 3.5, 1.5],  # Source 1
        [1.5, 3.5, 1.5]   # Source 2
    ]
    
    output_dir = Path("localization/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Setting up simulation in {room_dim}m room...")
    
    # --- Setup Microphones ---
    # sim_mic returns (pra.MicrophoneArray, mic_pos_relative_to_array_center)
    # But wait, sim_mic in beamforming/util/simulate.py:
    # return pra.MicrophoneArray(mic_pos_abs, fs=mic_fs), mic_pos
    # mic_pos is relative to 0,0,0 usually in sim_mic if we look at implementation:
    # mic_pos_abs = mic_pos + mic_loc.reshape(3, 1)
    # So the second return is relative layout?
    # Let's check sim_mic again.
    # mic_pos is calculated centered at 0.
    # mic_pos_abs is shifted by mic_loc.
    # It returns (pra_array, mic_pos_relative).
    
    mic_array_pra, mic_pos_rel = sim_mic(
        mic_count=mic_count,
        mic_loc=mic_center,
        mic_spacing=mic_radius,
        mic_type=MicType.CIRCULAR,
        mic_fs=fs
    )
    
    # --- Setup Room ---
    # sim_room(room_dim, fs, reflection_count)
    room = sim_room(room_dim, fs=fs, reflection_count=10)
    
    # Add microphone array to room
    room.add_microphone_array(mic_array_pra)
    
    # --- Add Sources ---
    duration = 1.0
    num_samples = int(fs * duration)
    
    # Load a speech file if possible, else generate bursty noise
    try:
        from scipy.io import wavfile
        import scipy.signal as signal_ops
        
        wav_path = Path("beamforming/input/matthew_talking.wav")
        if wav_path.exists():
            sr_wav, audio_wav = wavfile.read(wav_path)
            # Resample if needed
            if sr_wav != fs:
                # simple resample (this might be slow for long files, but fine for 1s)
                # Actually, let's just use it if close, or resample.
                # calculate number of samples needed
                num_samples_wav = int(len(audio_wav) * fs / sr_wav)
                audio_wav = signal_ops.resample(audio_wav, num_samples_wav)
            
            # Normalize
            audio_wav = audio_wav.astype(float)
            audio_wav /= np.max(np.abs(audio_wav))
            
            base_signal = audio_wav
            print(f"Using speech file: {wav_path}")
        else:
            raise FileNotFoundError("Wav file not found")
    except Exception as e:
        print(f"Fallback to synthetic bursty noise: {e}")
        # Generate bursty noise (white noise modulated by low freq sine)
        t = np.linspace(0, duration, num_samples)
        envelope = np.abs(np.sin(2 * np.pi * 3 * t)) # 3Hz modulation
        base_signal = np.random.randn(num_samples) * envelope

    # Ensure base_signal is long enough
    if len(base_signal) < num_samples:
         # Loop it
         repeats = int(np.ceil(num_samples / len(base_signal)))
         base_signal = np.tile(base_signal, repeats)[:num_samples]
    else:
         base_signal = base_signal[:num_samples]

    audio_sources_objects = []
    
    print("Adding sources:")
    for i, loc in enumerate(source_locs):
        # Time shift signal for different sources to make them distinct/decorrelated
        shift = int(fs * 0.5 * i)
        signal = np.roll(base_signal, shift)
        
        # Add to room
        room.add_source(loc, signal=signal)
        
        # Store for visualization
        # Audio_Sources(input, loc, classification)
        as_obj = Audio_Sources(input=f"Source_{i}", loc=loc, classification="signal")
        audio_sources_objects.append(as_obj)
        
        # Calculate True DOA relative to mic center
        dx = loc[0] - mic_center[0]
        dy = loc[1] - mic_center[1]
        angle = np.arctan2(dy, dx)
        if angle < 0: angle += 2*np.pi
        print(f"  Source {i}: {loc} -> True DOA: {np.degrees(angle):.2f} deg")

    # --- Simulate ---
    print("Simulating audio...")
    room.simulate()
    
    # room.mic_array.signals is (M, N)
    mic_signals = room.mic_array.signals
    print(f"Captured signals shape: {mic_signals.shape}")
    
    # --- Localization ---
    print("Running localization algorithm...")
    
    # Note: LocalizationSystem needs relative mic positions to compute steering vectors usually,
    # or absolute if we handle it correctly. 
    # algo.py uses mic_pos. 
    # If we pass mic_pos_rel, we assume the algorithm treats the array center as origin.
    # This is correct for DOA estimation relative to the array.
    
    loc_system = LocalizationSystem(
        mic_pos=mic_pos_rel,
        fs=fs,
        nfft=512,
        overlap=0.5,
        epsilon=0.1, # Updated epsilon for correlation metric (1 - 0.9 = 0.1)
        d_freq=8, # Increased d_freq for better correlation estimate
        max_sources=2
    )
    
    estimated_doas_rad, histogram = loc_system.process(mic_signals)
    
    estimated_doas_deg = np.degrees(estimated_doas_rad)
    print("\nEstimated DOAs:")
    for i, ang in enumerate(estimated_doas_deg):
        print(f"  Detection {i}: {ang:.2f} deg")
        
    # --- Visualization ---
    print(f"\nSaving visualizations to {output_dir}")
    
    # Plot Room
    plot_room_pos(
        np.array(room_dim),
        mic_center,
        audio_sources_objects,
        output_dir
    )
    
    # Plot Mics (Top Down)
    plot_mic_pos(mic_pos_rel, output_dir) # Use relative or absolute? visualize.plot_mic_pos expects 2D x,y usually.
    
    # Plot Histogram
    plt.figure()
    plt.plot(np.linspace(0, 360, len(histogram)), histogram)
    plt.title("Angular Histogram (Pre-Matching Pursuit)")
    plt.xlabel("Angle (Degrees)")
    plt.ylabel("Count")
    plt.grid(True)
    plt.savefig(output_dir / "angular_histogram.png")
    plt.close()
    
    print("Done.")

if __name__ == "__main__":
    main()

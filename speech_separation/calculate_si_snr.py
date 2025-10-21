import numpy as np
from scipy.io import wavfile
import sys

def read_wav_normalize(path):
    """
    Reads a WAV file and returns the data as a float64 array.
    Handles normalization from int types to [-1.0, 1.0].
    """
    sr, data = wavfile.read(path)
    
    # Normalize based on the data type
    if data.dtype == np.int16:
        data = data.astype(np.float64) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float64) / 2147483648.0
    elif data.dtype == np.uint8: # 8-bit WAV
        data = (data.astype(np.float64) - 128) / 128.0
    elif data.dtype == np.float32:
        data = data.astype(np.float64)
    
    # Handle stereo: average to mono
    if data.ndim > 1:
        data = np.mean(data, axis=1)
        
    return sr, data

def calculate_si_snr(target_signal, estimated_signal, epsilon=1e-8):
    """
    Calculates the Scale-Invariant Signal-to-Noise Ratio (SI-SNR)
    based on the formula provided by the user.

    Args:
        target_signal (np.ndarray): The clean target signal (s).
        estimated_signal (np.ndarray): The estimated signal (ŝ).
        epsilon (float): A small value to prevent division by zero.

    Returns:
        float: The SI-SNR value in decibels (dB).
    """
    
    if len(target_signal) != len(estimated_signal):
        print("Mismatched length - truncating signal...")
        min_len = min(len(target_signal), len(estimated_signal))
        target_signal = target_signal[:min_len]
        estimated_signal = estimated_signal[:min_len]

    # Remove DC offset (mean) 
    target_signal = target_signal - np.mean(target_signal)
    estimated_signal = estimated_signal - np.mean(estimated_signal)

    # s_target = (<ŝ, s>s) / ||s||^2
    # # <ŝ, s> is the dot product np.dot(estimated_signal, target_signal)
    # # ||s||^2 is the dot product np.dot(target_signal, target_signal)
    dot_product = np.dot(estimated_signal, target_signal)
    s_target_norm_sq = np.dot(target_signal, target_signal) + epsilon
    s_target = (dot_product / s_target_norm_sq) * target_signal

    # e_noise = ŝ - s_target
    e_noise = estimated_signal - s_target

    # SI-SNR = 10 * log10(||s_target||^2 / ||e_noise||^2)
    s_target_energy = np.dot(s_target, s_target)
    e_noise_energy = np.dot(e_noise, e_noise) + epsilon
    
    si_snr = 10 * np.log10(s_target_energy / e_noise_energy)
    
    return si_snr

def main():
    if len(sys.argv) != 3:
        print("Usage: python calculate_si_snr.py <path_to_target_signal.wav> <path_to_estimated_signal.wav>")
        print("  - target_signal.wav: The clean, original signal (s)")
        print("  - estimated_signal.wav: The output of your separation algorithm (ŝ)")
        sys.exit(1)

    target_path = sys.argv[1]
    estimated_path = sys.argv[2]

    sr_target, target_data = read_wav_normalize(target_path)
    sr_est, est_data = read_wav_normalize(estimated_path)

    if sr_target != sr_est:
        print(f"Warning: Sample rates differ! Target: {sr_target} Hz, Estimated: {sr_est} Hz.")
        print("The calculation will proceed, but this may indicate an issue in your processing pipeline.")

    si_snr_value = calculate_si_snr(target_data, est_data)

    print(f"Target file:    {target_path}")
    print(f"Estimated file: {estimated_path}")
    print("---------------------------------")
    print(f"SI-SNR:         {si_snr_value:.2f} dB")


if __name__ == "__main__":
    main()

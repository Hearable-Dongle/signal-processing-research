import argparse
from pathlib import Path
from typing import Tuple, List, Optional

import librosa
import numpy as np
import soundfile as sf
from numpy.typing import NDArray
from scipy.signal import istft

from algo.beamformer import apply_beamformer_stft
from algo.noise_estimation import estimate_Rnn, reduce_Rnn, regularize_Rnn
from util.compare import align_signals, calc_rmse, calc_si_sdr, calc_snr
from util.configure import Config
from util.simulate import MicType, sim_mic, sim_room
from util.visualize import plot_beam_pattern, plot_history, plot_mic_pos, plot_room_pos

from beamformer_core import compute_beamforming_weights, BeamformerConfig


def simulate_environment(config: Config) -> Tuple[NDArray, NDArray, NDArray, int]:
    """
    Sets up the room, microphones, and signal sources, then runs the simulation.
    Returns:
        mic_audio: Simulated audio from microphones (samples, channels)
        mic_pos: Microphone positions
        signal_loc: Locations of signal sources
        min_samples: The minimum sample count (for truncation)
    """
    room = sim_room(config.room_dim.tolist(), config.fs, config.reflection_count)
    mic, mic_pos = sim_mic(
        config.mic_count,
        config.mic_loc,
        config.mic_spacing,
        getattr(MicType, config.mic_type.upper()),
        config.fs,
    )
    room.add_microphone_array(mic)

    signal_sources = [s for s in config.sources if s.classification == "signal"]
    if not signal_sources:
        raise ValueError("No signal sources are defined")

    signal_loc = np.array([source.loc for source in signal_sources])
    
    # Determine simulation duration based on sources
    min_sample_count = config.fs * 60
    for source in config.sources:
        audio, fs = librosa.load(source.input, sr=config.fs)
        min_sample_count = min(min_sample_count, len(audio))
        
        if np.any(audio):
            audio /= np.max(np.abs(audio))
        
        room.add_source(source.loc, signal=audio)

    plot_mic_pos(mic_pos, config.output_dir)
    plot_room_pos(config.room_dim, config.mic_loc, config.sources, config.output_dir)

    # Run Simulation
    room.simulate()
    
    mic_audio = np.array(room.mic_array.signals).T

    # Normalize to prevent clipping artifacts
    max_val = np.max(np.abs(mic_audio))
    if max_val > 1.0:
        mic_audio = mic_audio / max_val

    # Truncate to valid length
    if mic_audio.shape[0] > min_sample_count:
        mic_audio = mic_audio[:min_sample_count, :]
    
    return mic_audio, mic_pos, signal_loc, min_sample_count


def get_noise_audio(config: Config, min_samples: int) -> NDArray:
    """
    Simulates the noise environment and returns the raw noise audio.
    Returns:
        mic_noise: (samples, channels)
    """
    mic_count = config.mic_count

    # Default to silence/small noise if no method selected
    if config.noise_estimation_method != "ground_truth":
         # If using 'predict' or other methods, we might assume 
         # the noise is just the quiet parts of the main audio.
         # For now, return a quiet noise floor if not simulating ground truth.
        return np.random.randn(min_samples, mic_count) * 1e-6

    config.log.info("Using ground truth simulation for noise audio")
    noise_sources = [s for s in config.sources if s.classification == "noise"]

    if not noise_sources:
        config.log.warning("No noise sources found. Returning silence.")
        return np.zeros((min_samples, mic_count))

    # Create a fresh room/mic setup just for noise
    noise_room = sim_room(config.room_dim.tolist(), config.fs, config.reflection_count)
    mic, _ = sim_mic(
        config.mic_count, config.mic_loc, config.mic_spacing,
        getattr(MicType, config.mic_type.upper()), config.fs
    )
    noise_room.add_microphone_array(mic)

    for source in noise_sources:
        audio, _ = librosa.load(source.input, sr=config.fs)
        if np.any(audio):
            audio /= np.max(np.abs(audio))
        noise_room.add_source(source.loc, signal=audio)

    noise_room.simulate()
    mic_noise = np.array(noise_room.mic_array.signals).T

    # Match lengths (Padding/Truncating)
    if mic_noise.shape[0] > min_samples:
        mic_noise = mic_noise[:min_samples, :]
    elif mic_noise.shape[0] < min_samples:
        pad_amt = min_samples - mic_noise.shape[0]
        mic_noise = np.pad(mic_noise, ((0, pad_amt), (0, 0)))

    return mic_noise

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


def evaluate_results(config: Config, mic_audio: NDArray, results_dict: dict, ref_audio: NDArray):
    """Calculates metrics (RMSE, SNR, SI-SDR) and plots results."""
    
    time_steepest = reconstruct_audio(
        results_dict["steepest_stft"], 
        config.fs, 
        results_dict["params"], 
        len(ref_audio)
    )
    time_newton = reconstruct_audio(
        results_dict["newton_stft"], 
        config.fs, 
        results_dict["params"], 
        len(ref_audio)
    )

    audio_dir = config.output_dir / "audio"
    if not audio_dir.exists():
        audio_dir.mkdir(parents=True)
        
    sf.write(audio_dir / "mic_raw_audio.wav", mic_audio, config.fs)
    sf.write(audio_dir / "mic_steepest_filtered_audio.wav", time_steepest, config.fs)
    sf.write(audio_dir / "mic_newton_filtered_audio.wav", time_newton, config.fs)

    def print_metrics(name, pred_sig):
        aligned_ref = align_signals(ref_audio, pred_sig)

        rmse, _mse = calc_rmse(aligned_ref, pred_sig)
        snr = calc_snr(aligned_ref, pred_sig)
        sdr = calc_si_sdr(aligned_ref, pred_sig)
        
        config.log.info(f"{name}: RMSE={rmse:.4f}, SNR={snr:.4f}dB, SI-SDR={sdr:.4f}dB")

    print_metrics("Raw Audio", np.mean(mic_audio, axis=1))
    print_metrics("Steepest Descent", time_steepest)
    print_metrics("Newton", time_newton)

    plot_history(
        {
            "Steepest Descent": (np.mean(results_dict["steepest_hist"], axis=0), {"color": "blue", "alpha": 0.5}),
            "Newton": (np.mean(results_dict["newton_hist"], axis=0), {"color": "green", "alpha": 0.5}),
        },
        config.output_dir,
    )

    # Beam Patterns
    # Find bin for target freq (e.g. 4400Hz)
    fvec = results_dict["fvec"]
    bin_idx = int(np.argmin(np.abs(fvec - 4400.0)))
    
    for name, weights in [("steepest", results_dict["steepest_weights"]), ("newton", results_dict["newton_weights"])]:
        plot_beam_pattern(
            f"beam_pattern_{name}",
            weights[bin_idx, :],
            results_dict["mic_pos"], 
            fvec[bin_idx],
            config.sound_speed,
            config.output_dir,
        )


def main():
    parser = argparse.ArgumentParser(description="Beamforming simulation")
    parser.add_argument("--config", type=Path, default=Path("config") / "config.json")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    config = Config(config_path=args.config, output_path=args.output)

    mic_audio, mic_pos, signal_loc, min_samples = simulate_environment(config)
    noise_audio = get_noise_audio(config, min_samples)

    bf_config = BeamformerConfig(
        fs=config.fs,
        frame_duration_ms=config.frame_duration,
        sound_speed=config.sound_speed,
        gamma_dB=15,
        iterations=20,
        mic_array_center=config.mic_loc,
        mic_geometry=mic_pos
    )

    results = compute_beamforming_weights(
        audio_input=mic_audio,
        source_locations=signal_loc,
        noise_audio=noise_audio,
        config=bf_config
    )

    freq_steepest = apply_beamformer_stft(results["stft_tensor"], results["steepest"][0])
    freq_newton = apply_beamformer_stft(results["stft_tensor"], results["newton"][0])

    results_packet = {
        "params": results["params"],
        "fvec": results["fvec"],
        "mic_pos": mic_pos,
        "steepest_weights": results["steepest"][0],
        "steepest_hist": results["steepest"][1],
        "steepest_stft": freq_steepest,
        "newton_weights": results["newton"][0],
        "newton_hist": results["newton"][1],
        "newton_stft": freq_newton,
    }

    ref_audio = np.zeros(min_samples)
    for source in [s for s in config.sources if s.classification == "signal"]:
        audio, _ = librosa.load(source.input, sr=config.fs)
        if len(audio) > min_samples: audio = audio[:min_samples]
        else: audio = np.pad(audio, (0, min_samples - len(audio)))
        ref_audio += audio

    evaluate_results(config, mic_audio, results_packet, ref_audio)
    
    config.log.info(f"Beamforming simulation completed - output saved to {config.output_dir}")


if __name__ == "__main__":
    main()
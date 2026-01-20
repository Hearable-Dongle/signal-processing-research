import argparse
from pathlib import Path
import librosa
import numpy as np
import soundfile as sf
from numpy.typing import NDArray
from scipy.signal import istft, stft  # type: ignore[reportUnknownMemberType]

from algo.beamformer import (
    apply_beamformer_stft,
    compute_steering_vector,
    wng_mvdr_newton,
    wng_mvdr_steepest,
)
from algo.noise_estimation import estimate_Rnn, reduce_Rnn, regularize_Rnn
from util.compare import calc_rmse, calc_si_sdr, calc_snr
from util.configure import Config
from util.simulate import MicType, sim_mic, sim_room
from util.visualize import plot_beam_pattern, plot_history, plot_mic_pos, plot_room_pos


def compute_wng_mvdr_weights(solver_fn, Rnn, steering_vecs, gamma, mu, iterations):
    """
    Applies a specific WNG-MVDR solver function across all frequency bins.
    
    Args:
        solver_fn: Callable (wng_mvdr_steepest or wng_mvdr_newton)
        Rnn: Noise covariance matrix (passed to solver)
        steering_vecs: List of steering vector arrays [Shape: (n_freq, n_mics)]
        gamma, mu, iterations: Solver hyperparameters
        
    Returns:
        weights: NDArray (n_freq, n_mics)
        power_history: List of history arrays
    """
    freq_bin_count = steering_vecs[0].shape[0]
    
    def process_bin(kf):
        a_vecs_bin = [sv[kf, :].reshape(-1, 1) for sv in steering_vecs]
        w_bin, p_hist = solver_fn(Rnn, a_vecs_bin, gamma, mu, iterations)
        
        return w_bin[:, 0], p_hist

    weights_list, power_histories = zip(*[process_bin(kf) for kf in range(freq_bin_count)])
    
    return np.array(weights_list, dtype=complex), list(power_histories)


def main():
    parser = argparse.ArgumentParser(description="Beamforming simulation")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config") / "config.json",
        help="Path to the configuration file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to the output directory",
    )
    args = parser.parse_args()

    config = Config(config_path=args.config, output_path=args.output)

    room = sim_room(config.room_dim.tolist(), config.fs, config.reflection_count)
    mic, mic_pos = sim_mic(
        config.mic_count,
        config.mic_loc,
        config.mic_spacing,
        getattr(MicType, config.mic_type.upper()),
        config.fs,
    )
    room.add_microphone_array(mic)  

    signal_sources = [
        source for source in config.sources if source.classification == "signal"
    ]

    if not signal_sources:
        raise ValueError("No signal sources are defined")

    signal_loc = np.array([source.loc for source in signal_sources])

    # Define minimum audio source duration, with one minute maximum
    min_sample_count = config.fs * 60

    # Add audio sources to the room
    for source in config.sources:
        source_audio, fs = librosa.load(source.input, sr=config.fs)

        min_sample_count = min(min_sample_count, len(source_audio))

        if fs != config.fs:
            raise ValueError("Room and audio source sampling rates do not match")

        # Normalize audio to prevent clipping and ensure consistent volume
        if np.any(source_audio):
            source_audio /= np.max(np.abs(source_audio))

        room.add_source(source.loc, signal=source_audio)  # type: ignore[reportUnknownMemberType]

    # Visualize 
    plot_mic_pos(mic_pos, config.output_dir)
    plot_room_pos(config.room_dim, config.mic_loc, config.sources, config.output_dir)

    # Run simulation
    room.simulate()

    mic_audio: NDArray[np.float64] = np.array(
        room.mic_array.signals
    ).T

    audio_dir = config.output_dir / "audio"
    if not audio_dir.exists():
        audio_dir.mkdir(parents=True)

    # Match minimum audio source duration
    if mic_audio.shape[0] > min_sample_count:
        mic_audio = mic_audio[:min_sample_count, :]

    sf.write(
        audio_dir / "mic_raw_audio.wav", mic_audio, config.fs
    )

    sample_count, mic_count = mic_audio.shape

    if mic_count != config.mic_count:
        raise ValueError("Microphone count of recorded audio does not match configuration")

    if config.noise_estimation_method == "ground_truth":
        config.log.info("Using ground truth for noise covariance estimation")
        noise_sources = [s for s in config.sources if s.classification == "noise"]

        if noise_sources:
            noise_room = sim_room(config.room_dim.tolist(), config.fs, config.reflection_count)
            noise_room.add_microphone_array(mic)

            for source in noise_sources:
                source_audio, _ = librosa.load(source.input, sr=config.fs)
                if np.any(source_audio):
                    source_audio /= np.max(np.abs(source_audio))
                noise_room.add_source(source.loc, signal=source_audio)

            noise_room.simulate()
            mic_noise = np.array(noise_room.mic_array.signals).T
            
            if mic_noise.shape[0] > min_sample_count:
                mic_noise = mic_noise[:min_sample_count, :]
            elif mic_noise.shape[0] < min_sample_count:
                mic_noise = np.pad(mic_noise, ((0, min_sample_count - mic_noise.shape[0]), (0, 0)))

            Rnn = estimate_Rnn(mic_noise)
        else:
            config.log.warning("'ground_truth' selected for noise estimation, but no noise sources found. Using identity matrix.")
            Rnn = np.eye(mic_count) * 1e-6
            mic_noise = np.zeros((0, mic_count))

    elif config.noise_estimation_method == "predict":
        raise NotImplementedError("Predictive noise estimation is not yet implemented")
    else:
        raise ValueError(f"Unknown noise_estimation_method: {config.noise_estimation_method}")

    # Apply principal component analysis
    if config.noise_pc_count > 0 and mic_noise.shape[0] > 0:
        Rnn = reduce_Rnn(Rnn, config.noise_pc_count)
        config.log.info(f"Component Count for PCA: {config.noise_pc_count}")

    # Apply regularization
    if config.noise_reg_factor > 0:
        Rnn = regularize_Rnn(Rnn, config.noise_reg_factor)
        config.log.info(f"Noise Regularization Factor: {config.noise_reg_factor}")

    stft_window_size = int(config.fs * config.frame_duration / 1000)

    if stft_window_size > sample_count:
        raise ValueError(
            "STFT window size is larger than signal. "
            "Try reducing the frame_duration in the config file."
        )

    # 50% overlap with Hann window
    hop = stft_window_size // 2
    window = np.hanning(stft_window_size)

    def create_stft_matrix(mic_channel) -> NDArray[np.complex128]:
        fvec, _, stft_matrix = stft(  
            mic_channel,
            fs=config.fs,
            nperseg=stft_window_size,
            noverlap=stft_window_size - hop,
            window=window,  
            padded=True,
            return_onesided=True,
        ) 
        stft_matrix = np.asarray(stft_matrix, dtype=np.complex128)

        return stft_matrix, fvec

    stft_list, fvecs = zip(*map(lambda mic_index: create_stft_matrix(mic_audio[:, mic_index]), range(config.mic_count)))
    fvec = fvecs[0]


    stft_output = np.stack(stft_list, axis=-1)

    freq_bin_count, _time_frame_count, _ = stft_output.shape

    # Compute steering vector, take first entry for now
    steering_vecs = compute_steering_vector(
        mic_pos,
        config.mic_loc,
        fvec,
        signal_loc,
        config.sound_speed,
    )

    # Define optimization parameters
    gamma_dB = 15
    gamma = 10 ** (gamma_dB / 10)
    mu = 0.01 / np.trace(Rnn.real)
    iteration_count = 20

    weights_steepest, power_history_steepest = compute_wng_mvdr_weights(
        solver_fn=wng_mvdr_steepest, 
        Rnn=Rnn, 
        steering_vecs=steering_vecs, 
        gamma=gamma, 
        mu=mu, 
        iterations=iteration_count
    )

    weights_newton, power_history_newton = compute_wng_mvdr_weights(
        solver_fn=wng_mvdr_newton, 
        Rnn=Rnn, 
        steering_vecs=steering_vecs, 
        gamma=gamma, 
        mu=mu, 
        iterations=iteration_count
    )

    plot_history(
        {
            "Steepest Descent": (
                np.mean(np.asarray(power_history_steepest), axis=0),
                {
                    "marker": ".",
                    "color": "blue",
                    "alpha": 0.5, 
                }
                ),
            "Newton": (
                np.mean(np.asarray(power_history_newton), axis=0),
                {
                    "marker": "s",
                    "color": "green",
                    "alpha": 0.5, 
                }
            ),
        },
        config.output_dir,
    )


    # Apply beamformer
    freq_steepest = apply_beamformer_stft(stft_output, weights_steepest)
    freq_newton = apply_beamformer_stft(stft_output, weights_newton)

    # Plot beam pattern
    target_freq = 4400.0
    bin_idx = int(
        np.argmin(np.abs(fvec - target_freq))
    ) 
    plot_beam_pattern(
        "beam_pattern_steepest",
        weights_steepest[bin_idx, :],
        mic_pos,
        fvec[bin_idx],  
        config.sound_speed,
        config.output_dir,
    )
    plot_beam_pattern(
        "beam_pattern_newton",
        weights_newton[bin_idx, :],
        mic_pos,
        fvec[bin_idx],  
        config.sound_speed,
        config.output_dir,
    )

    # Apply Inverse STFT
    _, time_steepest = istft(  
        freq_steepest,
        fs=config.fs,
        nperseg=stft_window_size,
        noverlap=stft_window_size - hop,
        window=window, 
    )
    _, time_newton = istft(  
        freq_newton,
        fs=config.fs,
        nperseg=stft_window_size,
        noverlap=stft_window_size - hop,
        window=window,  
    )

    time_steepest = np.real(time_steepest)
    time_newton = np.real(time_newton)

    # Match original audio length
    if len(time_steepest) > sample_count:
        time_steepest = time_steepest[:sample_count]
    else:
        time_steepest = np.pad(time_steepest, (0, sample_count - len(time_steepest)))
    if len(time_newton) > sample_count:
        time_newton = time_newton[:sample_count]
    else:
        time_newton = np.pad(time_newton, (0, sample_count - len(time_newton)))

    sf.write(
        audio_dir / "mic_steepest_filtered_audio.wav", time_steepest, config.fs
    )  
    sf.write(
        audio_dir / "mic_newton_filtered_audio.wav", time_newton, config.fs
    )  

    config.log.info("Using combined signal sources as reference audio")
    ref_audio = np.zeros(min_sample_count)
    for signal_source in signal_sources:
        audio, _ = librosa.load(signal_source.input, sr=config.fs)

        if len(audio) > min_sample_count:
            audio = audio[:min_sample_count]
        else:
            audio = np.pad(audio, (0, min_sample_count - len(audio)))

        ref_audio += audio

    # Compare optimization methods with RMSE and MSE
    rmse_raw, mse_raw = calc_rmse(ref_audio, np.mean(mic_audio, axis=1))
    rmse_steepest, mse_steepest = calc_rmse(ref_audio, time_steepest)
    rmse_newton, mse_newton = calc_rmse(ref_audio, time_newton)

    config.log.info(f"Raw Audio: {rmse_raw:.4f} RMSE, {mse_raw:.4f} MSE")
    config.log.info(f"Steepest Descent: {rmse_steepest:.4f} RMSE, {mse_steepest:.4f} MSE")
    config.log.info(f"Newton: {rmse_newton:.4f} RMSE, {mse_newton:.4f} MSE")

    # Compare optimization methods with SNR
    snr_raw = calc_snr(ref_audio, np.mean(mic_audio, axis=1))
    snr_steepest = calc_snr(ref_audio, time_steepest)
    snr_newton = calc_snr(ref_audio, time_newton)

    config.log.info(f"Raw Audio: {snr_raw:.4f} dB SNR")
    config.log.info(f"Steepest Descent: {snr_steepest:.4f} dB SNR")
    config.log.info(f"Newton: {snr_newton:.4f} dB SNR")

    # Compare optimization methods with SI SDR
    si_sdr_raw = calc_si_sdr(ref_audio, np.mean(mic_audio, axis=1))
    si_sdr_steepest = calc_si_sdr(ref_audio, time_steepest)
    si_sdr_newton = calc_si_sdr(ref_audio, time_newton)

    config.log.info(f"Raw Audio: {si_sdr_raw:.4f} dB SI SDR")
    config.log.info(f"Steepest Descent: {si_sdr_steepest:.4f} dB SI SDR")
    config.log.info(f"Newton: {si_sdr_newton:.4f} dB SI SDR")

    config.log.info(f"Beamforming simulation completed - output saved to {config.output_dir}")


if __name__ == "__main__":
    main()

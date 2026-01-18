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

# Define configuration
config = Config()

# Simulate room
room = sim_room(config.room_dim.tolist(), config.fs, config.reflection_count)

# Simulate microphone array
mic, mic_pos = sim_mic(
    config.mic_count,
    config.mic_loc,
    config.mic_spacing,
    getattr(MicType, config.mic_type.upper()),
    config.fs,
)

# Add microphone array to room
room.add_microphone_array(mic)  # type: ignore[reportUnknownMemberType]

# Define signal sources
signal_sources = [source for source in config.sources if source.classification == "signal"]

# Verify only one signal is defined
if not signal_sources:
    # Print error message
    msg = "No signal sources are defined"
    raise ValueError(msg)

# Extract signal location and reference audio
signal_loc = np.array([source.loc for source in signal_sources])
ref_audio, _ = librosa.load(signal_sources[0].input, sr=config.fs)

# Define minimum audio source duration, with one minute maximum
min_sample_count = config.fs * 60

# Iterate through sources
for source in config.sources:
    # Load audio source
    source_audio, fs = librosa.load(source.input, sr=config.fs)

    # Update minimum audio source duration
    min_sample_count = min(min_sample_count, len(source_audio))

    # Verify sampling rate is correct
    if fs != config.fs:
        # Print error message
        msg = "Room and audio source sampling rates do not match"
        raise ValueError(msg)

    # Add audio source to room
    room.add_source(source.loc, signal=source_audio)  # type: ignore[reportUnknownMemberType]

# Visualize microphone and room layouts
plot_mic_pos(mic_pos, config.output_dir)
plot_room_pos(config.room_dim, config.mic_loc, config.sources, config.output_dir)

# Run simulation
room.simulate()  # type: ignore[reportUnknownMemberType]

# Record simulated microphone signals
mic_audio: NDArray[np.float64] = np.array(room.mic_array.signals)  # type: ignore[reportUnknownMemberType]

# Create image directory if it does not exist
audio_dir = config.output_dir / "audio"
if not audio_dir.exists():
    audio_dir.mkdir(parents=True)

# Match minimum audio source duration
if mic_audio[0].size > min_sample_count:
    mic_audio = mic_audio[:, :min_sample_count]

# Write recorded audio to wavefile
sf.write(audio_dir / "mic_raw_audio.wav", mic_audio.T, config.fs)  # type: ignore[reportUnknownMemberType]

# Determine audio dimensions
sample_count, mic_count = mic_audio.T.shape

# Verify audio dimensions
if mic_count != config.mic_count:
    # Print error message
    msg = "Microphone count of recorded audio does not match configuration"
    raise ValueError(msg)

# Select noise-only segment from microphone audio, future work will implement VAD
noise_start_time = 19.0
noise_end_time = 22.0
noise_start_idx = int(noise_start_time * config.fs)
noise_end_idx = int(noise_end_time * config.fs)
mic_noise = mic_audio.T[noise_start_idx:noise_end_idx, :]

# Compute noise covariance for noise-only segment
Rnn = estimate_Rnn(mic_noise)

# Apply principal component analysis
if config.noise_pc_count > 0:
    Rnn = reduce_Rnn(Rnn, config.noise_pc_count)

    # Print to log
    config.log.info(f"Component Count for PCA: {config.noise_pc_count}")

# Apply regularization
if config.noise_reg_factor > 0:
    Rnn = regularize_Rnn(Rnn, config.noise_reg_factor)

    # Print to log
    config.log.info(f"Noise Regularization Factor: {config.noise_reg_factor}")

# Choose STFT window size
Nfft = int(config.fs * config.frame_duration / 1000)

# Use 50% overlap with Hann window
hop = Nfft // 2
window = np.hanning(Nfft)

# Initialize list to store the STFT matrices of each channel
stft_list: list[NDArray[np.complex128]] = list()
fvec: NDArray[np.float64] = np.empty(0, dtype=np.float64)

# Iterate over microphone channels
for mic_idx in range(config.mic_count):
    # Extract time-domain signal for one microphone channel
    mic_signal: NDArray[np.float64] = mic_audio[:, mic_idx]

    # Compute STFT
    fvec, _, stft_matrix = stft(  # type: ignore[reportUnknownMemberType]
        mic_audio.T[:, mic_idx],
        fs=config.fs,
        nperseg=Nfft,
        noverlap=Nfft - hop,
        window=window,  # type: ignore[reportUnknownMemberType]
        padded=True,
        return_onesided=True,
    )

    # Cast matrix type to ensure compatibility
    stft_matrix = np.asarray(stft_matrix, dtype=np.complex128)

    # Append STFT of microphone channel
    stft_list.append(stft_matrix)

# Convert list to array, all channel samples from time step grouped into same array
stft_output = np.stack(stft_list, axis=-1)

# Extract dimensional information
freq_bin_count, time_frame_count, _ = stft_output.shape


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
mu = 0.01
iteration_count = 20

# Initialize list for storing noise power history
power_history_steepest: list[NDArray[np.float64]] = list()
power_history_newton: list[NDArray[np.float64]] = list()

# Compute WNG-MVDR weights for each frequency bin with steepest descent method
weights_steepest = np.zeros((freq_bin_count, config.mic_count), dtype=complex)
for kf in range(freq_bin_count):
    a_vecs = [steering_vec[kf, :].reshape(-1, 1) for steering_vec in steering_vecs]
    freq_bin_weights, freq_bin_power_history = wng_mvdr_steepest(
        Rnn, a_vecs, gamma, mu, iteration_count
    )
    weights_steepest[kf, :] = freq_bin_weights[:, 0]

    # Append power history
    power_history_steepest.append(freq_bin_power_history)

# Compute WNG-MVDR weights for each frequency bin with Newton's method
weights_newton = np.zeros((freq_bin_count, config.mic_count), dtype=complex)
for kf in range(freq_bin_count):
    a_vecs = [steering_vec[kf, :].reshape(-1, 1) for steering_vec in steering_vecs]
    freq_bin_weights, freq_bin_power_history = wng_mvdr_newton(
        Rnn, a_vecs, gamma, mu, iteration_count
    )
    weights_newton[kf, :] = freq_bin_weights[:, 0]

    # Append power history
    power_history_newton.append(freq_bin_power_history)

# Plot convergence history
plot_history(
    {
        "Steepest Descent": np.mean(np.asarray(power_history_steepest), axis=0),
        "Newton": np.mean(np.asarray(power_history_newton), axis=0),
    },
    config.output_dir,
)

# Apply beamformer
freq_steepest = apply_beamformer_stft(stft_output, weights_steepest)
freq_newton = apply_beamformer_stft(stft_output, weights_newton)

# Plot beam pattern
target_freq = 4400.0
bin_idx = int(np.argmin(np.abs(fvec - target_freq)))  # type: ignore[reportUnknownMemberType]
plot_beam_pattern(
    "beam_pattern_steepest",
    weights_steepest[bin_idx, :],
    mic_pos,
    fvec[bin_idx],  # type: ignore[reportUnknownMemberType]
    config.sound_speed,
    config.output_dir,
)
plot_beam_pattern(
    "beam_pattern_newton",
    weights_newton[bin_idx, :],
    mic_pos,
    fvec[bin_idx],  # type: ignore[reportUnknownMemberType]
    config.sound_speed,
    config.output_dir,
)

# Apply Inverse STFT
_, time_steepest = istft(  # type: ignore[reportUnknownMemberType]
    freq_steepest,
    fs=config.fs,
    nperseg=Nfft,
    noverlap=Nfft - hop,
    window=window,  # type: ignore[reportUnknownMemberType]
)
_, time_newton = istft(  # type: ignore[reportUnknownMemberType]
    freq_newton,
    fs=config.fs,
    nperseg=Nfft,
    noverlap=Nfft - hop,
    window=window,  # type: ignore[reportUnknownMemberType]
)

# Ensure audio is real before writing to wav file
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

# Write filtered audio to wavefiles
sf.write(audio_dir / "mic_steepest_filtered_audio.wav", time_steepest, config.fs)  # type: ignore[reportUnknownMemberType]
sf.write(audio_dir / "mic_newton_filtered_audio.wav", time_newton, config.fs)  # type: ignore[reportUnknownMemberType]

# Compare optimization methods with RMSE and MSE
rmse_raw, mse_raw = calc_rmse(ref_audio, np.mean(mic_audio, axis=0))
rmse_steepest, mse_steepest = calc_rmse(ref_audio, time_steepest)
rmse_newton, mse_newton = calc_rmse(ref_audio, time_newton)

config.log.info(f"Raw Audio: {rmse_raw:.4f} RMSE, {mse_raw:.4f} MSE")
config.log.info(f"Steepest Descent: {rmse_steepest:.4f} RMSE, {mse_steepest:.4f} MSE")
config.log.info(f"Newton: {rmse_newton:.4f} RMSE, {mse_newton:.4f} MSE")

# Compare optimization methods with SNR
snr_raw = calc_snr(ref_audio, np.mean(mic_audio, axis=0))
snr_steepest = calc_snr(ref_audio, time_steepest)
snr_newton = calc_snr(ref_audio, time_newton)

config.log.info(f"Raw Audio: {snr_raw:.4f} dB SNR")
config.log.info(f"Steepest Descent: {snr_steepest:.4f} dB SNR")
config.log.info(f"Newton: {snr_newton:.4f} dB SNR")

# Compare optimization methods with SI SDR
si_sdr_raw = calc_si_sdr(ref_audio, np.mean(mic_audio, axis=0))
si_sdr_steepest = calc_si_sdr(ref_audio, time_steepest)
si_sdr_newton = calc_si_sdr(ref_audio, time_newton)

config.log.info(f"Raw Audio: {si_sdr_raw:.4f} dB SI SDR")
config.log.info(f"Steepest Descent: {si_sdr_steepest:.4f} dB SI SDR")
config.log.info(f"Newton: {si_sdr_newton:.4f} dB SI SDR")

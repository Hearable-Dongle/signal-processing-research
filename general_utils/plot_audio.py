import argparse
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
import os

def plot_audio(wav_files, cut_to_shortest=False, title=None):
    """
    Plots the amplitude and Fourier transform of one or more .wav files on the same graphs.

    Args:
        wav_files (list of str): A list of paths to the .wav files.
        cut_to_shortest (bool): If True, all audio files are truncated to the length of the shortest one.
        title (str, optional): The title of the plot.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    if title:
        fig.suptitle(title)

    all_data = []
    samplerates = []
    min_len = float('inf')

    for wav_file in wav_files:
        samplerate, data = wavfile.read(wav_file)
        if data.ndim > 1:
            data = data[:, 0]
        all_data.append(data)
        samplerates.append(samplerate)
        if len(data) < min_len:
            min_len = len(data)

    if cut_to_shortest:
        all_data = [data[:min_len] for data in all_data]

    for i, data in enumerate(all_data):
        wav_file = wav_files[i]
        samplerate = samplerates[i]

        # Create the time axis
        time = np.arange(len(data)) / samplerate

        # Calculate the Fourier transform
        fft_data = np.fft.fft(data)
        fft_freq = np.fft.fftfreq(len(data), 1 / samplerate)

        # Only plot the positive frequencies
        positive_freq_indices = np.where(fft_freq >= 0)
        fft_freq = fft_freq[positive_freq_indices]
        fft_data = np.abs(fft_data[positive_freq_indices])

        # Get the filename for the legend
        filename = os.path.basename(wav_file)

        # Plot the amplitude in time scale
        ax1.plot(time, data, label=filename)

        # Plot the Fourier transform
        ax2.plot(fft_freq, fft_data, label=filename)

    # Configure the amplitude plot
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude")
    ax1.set_title("Amplitude in Time Scale")
    ax1.legend()

    # Configure the Fourier transform plot
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Magnitude")
    ax2.set_title("Fourier Transform")
    ax2.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot the amplitude and Fourier transform of one or more .wav files.")
    parser.add_argument("wav_files", nargs='+', help="The path(s) to the .wav file(s).")
    parser.add_argument("--cut-to-shortest", action="store_true", help="Cut all audio files to the length of the shortest file.")
    parser.add_argument("--title", help="Set the title for the plot.")
    args = parser.parse_args()

    plot_audio(args.wav_files, args.cut_to_shortest, args.title)
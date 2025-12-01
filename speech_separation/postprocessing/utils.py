import torch
import matplotlib.pyplot as plt
import torchaudio.transforms as T

def plot_signal_differences(target, preds, diff, metric_val: float, metric_name: str = "SI-SNR", units: str = "dB"):
    """
    Plots the waveform and spectrogram for the target, predicted, and difference signals.
    """
    print("Generating plots...")
    n_fft = 1024
    hop_length = 512
    spec_transform = T.Spectrogram(n_fft=n_fft, hop_length=hop_length)
    db_transform = T.AmplitudeToDB()

    fig, axs = plt.subplots(3, 2, figsize=(15, 12), constrained_layout=True)

    fig.suptitle(f"{metric_name}: {metric_val:.2f} {units}", fontsize=16)

    def plot_row(ax_row, signal, title):
        # Waveform
        ax_row[0].plot(signal)
        ax_row[0].set_title(f"{title} - Waveform")
        ax_row[0].set_xlabel("Time (samples)")
        ax_row[0].set_ylabel("Amplitude")

        # Spectrogram
        spec = spec_transform(torch.from_numpy(signal))
        db_spec = db_transform(spec)
        img = ax_row[1].imshow(db_spec, aspect='auto', origin='lower', 
                              interpolation='none')
        ax_row[1].set_title(f"{title} - Spectrogram (dB)")
        ax_row[1].set_xlabel("Time (frames)")
        ax_row[1].set_ylabel("Frequency (bins)")
        fig.colorbar(img, ax=ax_row[1], format="%+2.0f dB")

    plot_row(axs[0], target, "Target Signal")
    plot_row(axs[1], preds, "Predicted Signal")
    plot_row(axs[2], diff, "Difference (Target - Preds)")

    plt.show() 
import argparse

import torchaudio
from torchmetrics.audio import ScaleInvariantSignalNoiseRatio

from utils import plot_signal_differences


def main(target_path, preds_path, visualize):
    target_signal, sr_target = torchaudio.load(target_path)
    preds_signal, sr_preds = torchaudio.load(preds_path)

    # Ensure the signals have the same length
    min_len = min(target_signal.shape[1], preds_signal.shape[1])
    target_signal = target_signal[:, :min_len]
    preds_signal = preds_signal[:, :min_len]

    if sr_target != sr_preds:
        print(f"Warning: Sample rates differ! Target: {sr_target} Hz, Preds: {sr_preds} Hz.")
        print("The calculation will proceed, but this may indicate an issue in your processing pipeline.")

    si_snr_metric = ScaleInvariantSignalNoiseRatio()
    si_snr_value = si_snr_metric(preds_signal, target_signal)

    print(f"Target file:    {target_path}")
    print(f"Predicted file: {preds_path}")
    print("---------------------------------")
    print(f"SI-SNR: {si_snr_value:.2f} dB")

    if visualize:
        plot_signal_differences(
            target_signal.cpu().numpy()[0], preds_signal.cpu().numpy()[0], (target_signal- preds_signal).cpu().numpy()[0], si_snr_value)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate SI-SNR between a target signal and a predicted signal.")
    parser.add_argument("--target-path", required=True, help="The ground truth audio (clean speech)")
    parser.add_argument("--preds-path", required=True, help="The model's output audio (denoised/separated speech)")
    parser.add_argument("--visualize", action="store_true", help="Visualize the audio signals.")
      
    args = parser.parse_args()
    main(target_path=args.target_path, preds_path=args.preds_path, visualize=args.visualize)
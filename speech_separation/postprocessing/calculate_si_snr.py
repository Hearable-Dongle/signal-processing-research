import torch
import torchaudio
import argparse

from torchmetrics.audio import ScaleInvariantSignalNoiseRatio


def main():
    parser = argparse.ArgumentParser(description="Calculate SI-SNR between a target signal and a predicted signal.")
    parser.add_argument("--target-path", required=True, help="The ground truth audio (clean speech)")
    parser.add_argument("--preds-path", required=True, help="The model's output audio (denoised/separated speech)")
    args = parser.parse_args()

    target_signal, sr_target = torchaudio.load(args.target_path)
    preds_signal, sr_preds = torchaudio.load(args.preds_path)

    # Ensure the signals have the same length
    min_len = min(target_signal.shape[1], preds_signal.shape[1])
    target_signal = target_signal[:, :min_len]
    preds_signal = preds_signal[:, :min_len]

    if sr_target != sr_preds:
        print(f"Warning: Sample rates differ! Target: {sr_target} Hz, Preds: {sr_preds} Hz.")
        print("The calculation will proceed, but this may indicate an issue in your processing pipeline.")

    # `preds` is the first argument, `target` is the second.
    si_snr_metric = ScaleInvariantSignalNoiseRatio()
    si_snr_value = si_snr_metric(preds_signal, target_signal)

    print(f"Target file:    {args.target_path}")
    print(f"Predicted file: {args.preds_path}")
    print("---------------------------------")
    print(f"SI-SNR: {si_snr_value:.2f} dB")


if __name__ == "__main__":
    main()
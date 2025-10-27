import torch
import torchaudio
import argparse

from torchmetrics.audio import ScaleInvariantSignalNoiseRatio


def main():
    parser = argparse.ArgumentParser(description="Calculate SI-SNR for a given pair of audio files.")
    parser.add_argument("--target-path", help="The clean, original signal (s)")
    parser.add_argument("--estimated-path", help="The output of your separation algorithm (ŝ)")
    args = parser.parse_args()

    target_signal, sr_target = torchaudio.load(args.target_path)
    est_signal, sr_est = torchaudio.load(args.estimated_path)

    min_len = min(target_signal.shape[1], est_signal.shape[1])
    target_signal = target_signal[:, :min_len]
    est_signal = est_signal[:, :min_len]

    if sr_target != sr_est:
        print(f"Warning: Sample rates differ! Target: {sr_target} Hz, Estimated: {sr_est} Hz.")
        print("The calculation will proceed, but this may indicate an issue in your processing pipeline.")


    si_snr_metric = ScaleInvariantSignalNoiseRatio()
    si_snr_value = si_snr_metric(est_signal, target_signal)

    print(f"Target file:    {args.target_path}")
    print(f"Estimated file: {args.estimated_path}")
    print("---------------------------------")
    print(f"SI-SNR:         {si_snr_value:.2f} dB")


if __name__ == "__main__":
    main()

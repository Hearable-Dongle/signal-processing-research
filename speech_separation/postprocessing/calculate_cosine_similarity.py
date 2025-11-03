import torch
import torchaudio
import argparse
import torch.nn.functional as F

def main():
    parser = argparse.ArgumentParser(description="Calculate cosine similarity for a given pair of audio files.")
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

    # Ensure tensors are on the same device and dtype
    est_signal = est_signal.to(target_signal.device, dtype=target_signal.dtype)

    cosine_sim = F.cosine_similarity(target_signal.squeeze(), est_signal.squeeze(), dim=0)

    print(f"Target file:    {args.target_path}")
    print(f"Estimated file: {args.estimated_path}")
    print("---------------------------------")
    print(f"Cosine Similarity: {cosine_sim.item():.4f}")


if __name__ == "__main__":
    main()

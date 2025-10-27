import time
import torch
from siamese_unet import SiameseUnet

# --- Configuration ---
NUM_INFERENCES = 200
# Note: The input to SiameseUnet is an STFT, not raw audio.
# We define a representative input size that is compatible with the model's architecture.
# The model requires input dimensions to be divisible by 128.
INPUT_FREQ_BINS = 256
INPUT_TIME_FRAMES = 256
# The user requested a check for 10ms of audio. The relationship between audio
# duration and STFT frames is complex (depends on n_fft, hop_length).
# This configuration uses a fixed, compatible STFT size for benchmarking.
CLIP_DURATION_S = 0.01 

def main():
    """
    Measures the inference time of the SiameseUnet model on a dummy STFT clip.
    """
    print("Loading SiameseUnet model...")
    model = SiameseUnet()
    model.eval()

    # Shape is (Batch, Channels, Freq, Time)
    # Channels = 2 for Real and Imaginary components
    input_shape = (1, 2, INPUT_FREQ_BINS, INPUT_TIME_FRAMES)
    dummy_mixture = torch.rand(*input_shape)
    dummy_reference = torch.rand(*input_shape)

    print(f"Running {NUM_INFERENCES} inferences on a dummy STFT clip of size {INPUT_FREQ_BINS}x{INPUT_TIME_FRAMES}...")

    inference_times = []

    # Warm-up inference
    with torch.no_grad():
        _ = model(dummy_mixture, dummy_reference)

    for i in range(NUM_INFERENCES):
        start_time = time.perf_counter()
        with torch.no_grad():
            _ = model(dummy_mixture, dummy_reference)
        end_time = time.perf_counter()
        inference_times.append(end_time - start_time)
        print(f"Inference {i + 1}/{NUM_INFERENCES}", end='\r')

    print("\nInference testing complete.")

    avg_inference_time_ms = (sum(inference_times) / NUM_INFERENCES) * 1000
    min_inference_time_ms = min(inference_times) * 1000
    max_inference_time_ms = max(inference_times) * 1000
    inferences_per_second = 1 / (sum(inference_times) / NUM_INFERENCES)

    print("\n--- PyTorch Inference Time Statistics ---")
    print(f"Model: SiameseUnet")
    print(f"Input Shape (Mixture & Reference): {list(input_shape)}")
    print(f"Number of inferences: {NUM_INFERENCES}")
    print(f"Average inference time: {avg_inference_time_ms:.2f} ms")
    print(f"Fastest inference time: {min_inference_time_ms:.2f} ms")
    print(f"Slowest inference time: {max_inference_time_ms:.2f} ms")
    print(f"Inferences per second (IPS): {inferences_per_second:.2f}")

if __name__ == "__main__":
    main()

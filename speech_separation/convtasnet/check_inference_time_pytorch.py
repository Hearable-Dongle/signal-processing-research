import time
import torch
import numpy as np
from asteroid.models import ConvTasNet

# --- Configuration ---
NUM_INFERENCES = 200
SAMPLE_RATE = 16000  # 16 kHz
CLIP_DURATION_S = 0.01  # 10 milliseconds

def main():
    """
    Measures the inference time of a pretrained ConvTasNet model on a short audio clip.
    """
    print("Loading pretrained ConvTasNet model...")
    # Using a model trained on 16kHz data (WHAMR!)
    model_name = "JorisCos/ConvTasNet_Libri2Mix_sepnoisy_16k"
    print(f"Loading pre-trained model: {model_name}...")
    model = ConvTasNet.from_pretrained(model_name)
    model.eval()

    input_length = int(CLIP_DURATION_S * SAMPLE_RATE)
    input_shape = (1, input_length) 
    dummy_input = torch.rand(*input_shape)

    print(f"Running {NUM_INFERENCES} inferences on a {CLIP_DURATION_S * 1000:.0f} ms clip ({input_length} samples)...")

    inference_times = []

    with torch.no_grad():
        _ = model(dummy_input)

    for i in range(NUM_INFERENCES):
        start_time = time.perf_counter()
        with torch.no_grad():
            _ = model(dummy_input)
        end_time = time.perf_counter()
        inference_times.append(end_time - start_time)
        print(f"Inference {i + 1}/{NUM_INFERENCES}", end='\r')

    print("\nInference testing complete.")

    avg_inference_time_ms = (sum(inference_times) / NUM_INFERENCES) * 1000
    min_inference_time_ms = min(inference_times) * 1000
    max_inference_time_ms = max(inference_times) * 1000
    inferences_per_second = 1 / (sum(inference_times) / NUM_INFERENCES)

    print("\n--- PyTorch Inference Time Statistics ---")
    print(f"Model: ConvTasNet ({model_name})")
    print(f"Input Shape: {list(input_shape)}")
    print(f"Number of inferences: {NUM_INFERENCES}")
    print(f"Average inference time: {avg_inference_time_ms:.2f} ms")
    print(f"Fastest inference time: {min_inference_time_ms:.2f} ms")
    print(f"Slowest inference time: {max_inference_time_ms:.2f} ms")
    print(f"Inferences per second (IPS): {inferences_per_second:.2f}")

if __name__ == "__main__":
    main()

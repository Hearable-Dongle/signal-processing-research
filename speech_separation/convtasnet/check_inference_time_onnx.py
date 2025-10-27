import time
import torch
import numpy as np
import onnxruntime as ort
import os
from asteroid.models import ConvTasNet

# --- Configuration ---
NUM_INFERENCES = 200
SAMPLE_RATE = 16000  # 16 kHz
CLIP_DURATION_S = 0.01  # 10 milliseconds
ONNX_MODEL_PATH = "convtasnet.onnx"

def main():
    """
    Converts a pretrained ConvTasNet model to ONNX and measures its inference time.
    """
    # --- 1. Load PyTorch Model ---
    print("Loading pretrained ConvTasNet model for conversion...")
    # Using a model trained on 16kHz data (WHAMR!)
    model_name = "JorisCos/ConvTasNet_Libri2Mix_sepnoisy_16k"
    print(f"Loading pre-trained model: {model_name}...")
    model = ConvTasNet.from_pretrained(model_name)
    model.eval()
    # --- 2. Prepare Dummy Input ---
    input_length = int(CLIP_DURATION_S * SAMPLE_RATE)
    input_shape = (1, input_length)  # Shape for ConvTasNet is (batch, samples)
    dummy_input = torch.rand(*input_shape)

    # --- 3. Convert to ONNX ---
    print(f"Converting model to ONNX format at '{ONNX_MODEL_PATH}'...")
    try:
        torch.onnx.export(model,
                          dummy_input,
                          ONNX_MODEL_PATH,
                          export_params=True,
                          opset_version=11,
                          do_constant_folding=True,
                          input_names=['input'],
                          output_names=['outputs'],
                          dynamic_axes={'input': {1: 'length'},
                                        'outputs': {2: 'length'}})
        print("Model conversion successful.")
    except Exception as e:
        print(f"Error during ONNX conversion: {e}")
        return

    # --- 4. Run ONNX Inference Benchmark ---
    print(f"Running {NUM_INFERENCES} inferences on {ONNX_MODEL_PATH}...")
    
    session = ort.InferenceSession(ONNX_MODEL_PATH)
    input_name = session.get_inputs()[0].name
    dummy_input_np = dummy_input.numpy().astype(np.float32)
    
    inference_times = []

    # First inference for warm-up
    _ = session.run(None, {input_name: dummy_input_np})

    # Benchmark loop
    for i in range(NUM_INFERENCES):
        start_time = time.perf_counter()
        _ = session.run(None, {input_name: dummy_input_np})
        end_time = time.perf_counter()
        inference_times.append(end_time - start_time)
        print(f"Inference {i + 1}/{NUM_INFERENCES}", end='\r')

    print("\nInference testing complete.")

    # --- 5. Calculate and Display Statistics ---
    avg_inference_time_ms = (sum(inference_times) / NUM_INFERENCES) * 1000
    min_inference_time_ms = min(inference_times) * 1000
    max_inference_time_ms = max(inference_times) * 1000
    inferences_per_second = 1 / (sum(inference_times) / NUM_INFERENCES)

    print("\n--- ONNX Runtime Inference Time Statistics ---")
    print(f"Model: {ONNX_MODEL_PATH} (from mpariente/ConvTasNet_WHAMR_sepclean)")
    print(f"Input Shape: {list(input_shape)}")
    print(f"Number of inferences: {NUM_INFERENCES}")
    print(f"Average inference time: {avg_inference_time_ms:.2f} ms")
    print(f"Fastest inference time: {min_inference_time_ms:.2f} ms")
    print(f"Slowest inference time: {max_inference_time_ms:.2f} ms")
    print(f"Inferences per second (IPS): {inferences_per_second:.2f}")


if __name__ == "__main__":
    main()

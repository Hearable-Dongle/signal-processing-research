import sys
import time
import numpy as np
import onnxruntime as ort
import torch


NUM_INFERENCES = 200

def main():
    if len(sys.argv) < 2:
        print("Usage: python check_inference_time_siamese_unet.py <onnx_model_path>")
        sys.exit(1)

    onnx_model_path = sys.argv[1]
    
    # Hardcoded input shapes for the Siamese Unet model
    batch_size = 4
    channels = 2
    freq_bins = 128
    time_frames = 128
    
    input_shape_mixture = (batch_size, channels, freq_bins, time_frames)
    input_shape_reference = (batch_size, channels, freq_bins, time_frames)

    inference_times = []

    # Create an ONNX runtime session
    session = ort.InferenceSession(onnx_model_path)
    input_names = [input.name for input in session.get_inputs()]
    
    # Create dummy input tensors
    dummy_input_mixture = torch.rand(*input_shape_mixture)
    dummy_input_mixture_np = dummy_input_mixture.numpy().astype(np.float32)
    
    dummy_input_reference = torch.rand(*input_shape_reference)
    dummy_input_reference_np = dummy_input_reference.numpy().astype(np.float32)

    inputs = {
        input_names[0]: dummy_input_mixture_np,
        input_names[1]: dummy_input_reference_np
    }

    print(f"Running {NUM_INFERENCES} inferences on {onnx_model_path} with input shapes {input_shape_mixture} and {input_shape_reference}...")

    # Warm-up run
    session.run(None, inputs)

    for i in range(NUM_INFERENCES):
        start_time = time.perf_counter()
        session.run(None, inputs)
        end_time = time.perf_counter()
        inference_times.append(end_time - start_time)
        print(f"Inference {i+1}/{NUM_INFERENCES}", end='\r')

    print("\nInference testing complete.")

    avg_inference_time_ms = (sum(inference_times) / NUM_INFERENCES) * 1000
    min_inference_time_ms = min(inference_times) * 1000
    max_inference_time_ms = max(inference_times) * 1000
    inferences_per_second = 1 / (sum(inference_times) / NUM_INFERENCES)


    print("\n--- Inference Time Statistics ---")
    print(f"Number of inferences: {NUM_INFERENCES}")
    print(f"Average inference time: {avg_inference_time_ms:.2f} ms")
    print(f"Fastest inference time: {min_inference_time_ms:.2f} ms")
    print(f"Slowest inference time: {max_inference_time_ms:.2f} ms")
    print(f"Inferences per second (IPS): {inferences_per_second:.2f}")


if __name__ == "__main__":
    main()

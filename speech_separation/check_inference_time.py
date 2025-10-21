import sys
import time
import numpy as np
import onnxruntime as ort
import torch


NUM_INFERENCES = 200

def main():
    onnx_model_path = sys.argv[1]
    input_shape = [1, 1, int(sys.argv[2])] if len(sys.argv) <=3 else [int(num) for num in sys.argv[2:]]

    inference_times = []

    session = ort.InferenceSession(onnx_model_path)
    input_name = session.get_inputs()[0].name

    dummy_input = torch.rand(*input_shape)
    dummy_input_np = dummy_input.numpy().astype(np.float32)

    print(f"Running {NUM_INFERENCES} inferences on {onnx_model_path} with input shape {input_shape}...")

    session.run(None, {input_name: dummy_input_np})

    for i in range(NUM_INFERENCES):
        start_time = time.perf_counter()
        session.run(None, {input_name: dummy_input_np})
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

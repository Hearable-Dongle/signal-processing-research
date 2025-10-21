import time
import numpy as np
import onnxruntime as ort

NUM_INFERENCES = 200

def main():
    """
    Measures the inference speed of the full TF-MLP model pipeline.
    This involves running the encoder, the recurrent part in a loop, and the decoder.
    """
    # --- Model Paths ---
    encoder_path = "encoder_simplified.onnx"
    recurrent_step_path = "recurrent_step_simplified.onnx"
    decoder_path = "decoder_simplified.onnx"

    # --- Model Configuration (hardcoded from tf-mlp/to_onnx.py) ---
    latent_channels = 64
    num_blocks = 4
    win_length = 512
    num_freqs = win_length // 2 + 1
    batch_size = 1
    num_frames = 100

    print("--- Model Configuration ---")
    print(f"Latent Channels: {latent_channels}")
    print(f"Num Blocks: {num_blocks}")
    print(f"Window Length: {win_length}")
    print(f"Num Frequencies: {num_freqs}")
    print(f"Batch Size: {batch_size}")
    print(f"Num Frames: {num_frames}")
    print("--------------------------")

    # --- Create ONNX runtime sessions ---
    encoder_session = ort.InferenceSession(encoder_path)
    recurrent_session = ort.InferenceSession(recurrent_step_path)
    decoder_session = ort.InferenceSession(decoder_path)

    # --- Get input/output names ---
    encoder_input_name = encoder_session.get_inputs()[0].name
    encoder_output_name = encoder_session.get_outputs()[0].name

    recurrent_input_names = [inp.name for inp in recurrent_session.get_inputs()]
    recurrent_output_names = [out.name for out in recurrent_session.get_outputs()]

    decoder_input_name = decoder_session.get_inputs()[0].name
    decoder_output_name = decoder_session.get_outputs()[0].name

    # --- Prepare dummy inputs ---
    dummy_spec_ri = np.random.randn(batch_size, 2, num_freqs, num_frames).astype(np.float32)
    
    inference_times = []

    print(f"Running {NUM_INFERENCES} inferences for the full TF-MLP model...")

    # --- Warm-up run ---
    h_states = np.zeros((num_blocks, batch_size, latent_channels, num_freqs), dtype=np.float32)
    c_states = np.zeros((num_blocks, batch_size, latent_channels, num_freqs), dtype=np.float32)
    encoded_spec = encoder_session.run([encoder_output_name], {encoder_input_name: dummy_spec_ri})[0]
    processed_frames = []
    for i in range(encoded_spec.shape[3]):
        frame = encoded_spec[:, :, :, i]
        recurrent_inputs = {
            recurrent_input_names[0]: frame,
            recurrent_input_names[1]: h_states,
            recurrent_input_names[2]: c_states
        }
        out_frame, h_states, c_states = recurrent_session.run(recurrent_output_names, recurrent_inputs)
        processed_frames.append(out_frame)
    processed_spec = np.stack(processed_frames, axis=3)
    decoder_session.run([decoder_output_name], {decoder_input_name: processed_spec})

    # --- Main Inference Loop ---
    for i in range(NUM_INFERENCES):
        start_time = time.perf_counter()

        # 1. Encoder
        encoded_spec = encoder_session.run([encoder_output_name], {encoder_input_name: dummy_spec_ri})[0]

        # 2. Recurrent steps
        # Reset states for each inference run
        h_states = np.zeros((num_blocks, batch_size, latent_channels, num_freqs), dtype=np.float32)
        c_states = np.zeros((num_blocks, batch_size, latent_channels, num_freqs), dtype=np.float32)
        processed_frames = []
        for frame_idx in range(encoded_spec.shape[3]):
            frame = encoded_spec[:, :, :, frame_idx]
            recurrent_inputs = {
                recurrent_input_names[0]: frame,
                recurrent_input_names[1]: h_states,
                recurrent_input_names[2]: c_states
            }
            out_frame, h_states, c_states = recurrent_session.run(recurrent_output_names, recurrent_inputs)
            processed_frames.append(out_frame)

        # 3. Decoder
        processed_spec = np.stack(processed_frames, axis=3)
        decoder_session.run([decoder_output_name], {decoder_input_name: processed_spec})

        end_time = time.perf_counter()
        inference_times.append(end_time - start_time)
        print(f"Inference {i+1}/{NUM_INFERENCES}", end='\r')

    print("\nInference testing complete.")

    avg_inference_time_ms = (sum(inference_times) / NUM_INFERENCES) * 1000
    min_inference_time_ms = min(inference_times) * 1000
    max_inference_time_ms = max(inference_times) * 1000
    inferences_per_second = 1 / (sum(inference_times) / NUM_INFERENCES)

    print("\n--- TF-MLP Full Model Inference Time Statistics ---")
    print(f"Number of inferences: {NUM_INFERENCES}")
    print(f"Input shape (spectrogram): {(batch_size, 2, num_freqs, num_frames)}")
    print(f"Average inference time: {avg_inference_time_ms:.2f} ms")
    print(f"Fastest inference time: {min_inference_time_ms:.2f} ms")
    print(f"Slowest inference time: {max_inference_time_ms:.2f} ms")
    print(f"Inferences per second (IPS): {inferences_per_second:.2f}")


if __name__ == "__main__":
    main()

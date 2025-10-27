import time
import numpy as np
import onnxruntime as ort
import torch
import os
from tf_mlp import Encoder, Decoder, TFMLPNet, DEFAULT_CONFIG

# --- Configuration ---
NUM_INFERENCES = 50 # Reduced for this complex loop

# --- Model & Input Parameters (Hard-coded as requested) ---
CONFIG = DEFAULT_CONFIG
LATENT_CHANNELS = CONFIG["latent_channels"]
NUM_BLOCKS = CONFIG["num_blocks"]
MIXER_REPETITIONS = CONFIG["mixer_repetitions"]
NUM_SPEAKERS = CONFIG["num_speakers"]
WIN_LENGTH = CONFIG["win_length"]
NUM_FREQS = WIN_LENGTH // 2 + 1
BATCH_SIZE = 1
NUM_FRAMES = 100  # Example number of time frames

# --- ONNX File Paths ---
ENCODER_ONNX = "encoder.onnx"
RECURRENT_STEP_ONNX = "recurrent_step.onnx"
DECODER_ONNX = "decoder.onnx"

def generate_onnx_models():
    """Generates the three required ONNX models.
    This is adapted from to_onnx.py to make the script self-contained.
    """
    print("Generating temporary ONNX models...")
    # 1. Encoder
    encoder = Encoder(LATENT_CHANNELS, WIN_LENGTH)
    encoder.eval()
    dummy_spec_ri = torch.randn(BATCH_SIZE, 2, NUM_FREQS, NUM_FRAMES)
    torch.onnx.export(encoder, dummy_spec_ri, ENCODER_ONNX, opset_version=11,
                        input_names=['spec_ri'], output_names=['encoded_spec'])

    # 2. Recurrent Step
    recurrent_step = TFMLPNet(LATENT_CHANNELS, NUM_BLOCKS, MIXER_REPETITIONS, NUM_FREQS)
    recurrent_step.eval()
    dummy_frame = torch.randn(BATCH_SIZE, LATENT_CHANNELS, NUM_FREQS)
    h_states = torch.randn(NUM_BLOCKS, BATCH_SIZE, LATENT_CHANNELS, NUM_FREQS)
    c_states = torch.randn(NUM_BLOCKS, BATCH_SIZE, LATENT_CHANNELS, NUM_FREQS)
    torch.onnx.export(recurrent_step, (dummy_frame, h_states, c_states), RECURRENT_STEP_ONNX, opset_version=11,
                        input_names=['frame', 'h_states', 'c_states'], 
                        output_names=['out_frame', 'next_h', 'next_c'])

    # 3. Decoder
    decoder = Decoder(LATENT_CHANNELS, NUM_SPEAKERS)
    decoder.eval()
    dummy_processed_spec = torch.randn(BATCH_SIZE, LATENT_CHANNELS, NUM_FREQS, NUM_FRAMES)
    torch.onnx.export(decoder, dummy_processed_spec, DECODER_ONNX, opset_version=11,
                        input_names=['processed_spec'], output_names=['mask'])
    print("ONNX models generated.")

def main():
    onnx_files = [ENCODER_ONNX, RECURRENT_STEP_ONNX, DECODER_ONNX]
    try:
        # --- 1. Generate and Load ONNX Models ---
        generate_onnx_models()
        encoder_sess = ort.InferenceSession(ENCODER_ONNX)
        recurrent_sess = ort.InferenceSession(RECURRENT_STEP_ONNX)
        decoder_sess = ort.InferenceSession(DECODER_ONNX)

        # --- 2. Prepare Dummy Inputs ---
        spec_ri_np = np.random.randn(BATCH_SIZE, 2, NUM_FREQS, NUM_FRAMES).astype(np.float32)
        h_states_np = np.zeros((NUM_BLOCKS, BATCH_SIZE, LATENT_CHANNELS, NUM_FREQS), dtype=np.float32)
        c_states_np = np.zeros((NUM_BLOCKS, BATCH_SIZE, LATENT_CHANNELS, NUM_FREQS), dtype=np.float32)

        print(f"\nRunning {NUM_INFERENCES} inferences for the full TF-MLP model...")
        print(f"Input Spec Shape: {spec_ri_np.shape}")

        # --- 3. Run Inference Benchmark ---
        inference_times = []
        
        # Warm-up run
        _ = encoder_sess.run(None, {'spec_ri': spec_ri_np})[0]

        for i in range(NUM_INFERENCES):
            start_time = time.perf_counter()

            # 1. Run Encoder
            encoded_spec = encoder_sess.run(None, {'spec_ri': spec_ri_np})[0]

            # 2. Run Recurrent Loop
            processed_frames = []
            h, c = h_states_np, c_states_np
            for frame_idx in range(NUM_FRAMES):
                frame = encoded_spec[:, :, :, frame_idx]
                # The model expects (batch, channels, freqs), so we might need to squeeze
                if frame.ndim == 4 and frame.shape[2] == 1:
                    frame = frame.squeeze(2)
                
                out_frame, h, c = recurrent_sess.run(None, {'frame': frame, 'h_states': h, 'c_states': c})
                processed_frames.append(out_frame)
            
            # Stack frames to form the full spectrogram for the decoder
            processed_spec = np.stack(processed_frames, axis=-1)

            # 3. Run Decoder
            _ = decoder_sess.run(None, {'processed_spec': processed_spec})[0]

            end_time = time.perf_counter()
            inference_times.append(end_time - start_time)
            print(f"Inference {i + 1}/{NUM_INFERENCES}", end='\r')

        print("\nInference testing complete.")

        # --- 4. Calculate and Display Statistics ---
        avg_inference_time_ms = (sum(inference_times) / NUM_INFERENCES) * 1000
        min_inference_time_ms = min(inference_times) * 1000
        max_inference_time_ms = max(inference_times) * 1000
        inferences_per_second = 1 / (sum(inference_times) / NUM_INFERENCES)

        print("\n--- Full TF-MLP (ONNX) Inference Time Statistics ---")
        print(f"Number of inferences: {NUM_INFERENCES}")
        print(f"Input frames: {NUM_FRAMES}")
        print(f"Average inference time: {avg_inference_time_ms:.2f} ms")
        print(f"Fastest inference time: {min_inference_time_ms:.2f} ms")
        print(f"Slowest inference time: {max_inference_time_ms:.2f} ms")
        print(f"Inferences per second (IPS): {inferences_per_second:.2f}")

    finally:
        # --- 5. Cleanup ---
        print("\nCleaning up temporary ONNX files...")
        for f in onnx_files:
            if os.path.exists(f):
                os.remove(f)

if __name__ == "__main__":
    main()
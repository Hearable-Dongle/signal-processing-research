import torch
import onnx
from tf_mlp import Encoder, Decoder, TFMLPNet, DEFAULT_CONFIG

def main():
    # --- Model Configuration ---
    config = DEFAULT_CONFIG
    latent_channels = config["latent_channels"]
    num_blocks = config["num_blocks"]
    mixer_repetitions = config["mixer_repetitions"]
    num_speakers = config["num_speakers"]
    win_length = config["win_length"]
    
    # Derived parameters
    num_freqs = win_length // 2 + 1
    
    # Dummy input dimensions
    batch_size = 1
    num_frames = 100 # Example number of time frames

    # --- 1. Export Encoder ---
    encoder = Encoder(latent_channels, win_length)
    encoder.to('cpu').eval()
    dummy_spec_ri = torch.randn(batch_size, 2, num_freqs, num_frames)
    
    torch.onnx.export(
        encoder,
        dummy_spec_ri,
        "encoder.onnx",
        opset_version=9,
        input_names=['spec_ri'],
        output_names=['encoded_spec'],
        dynamic_axes={'spec_ri': {0: 'batch_size', 3: 'num_frames'},
                        'encoded_spec': {0: 'batch_size', 3: 'num_frames'}}
    )
    print("Encoder exported to encoder.onnx")

    # --- 2. Export Recurrent Step (TFMLPNet) ---
    # Note: num_freqs for TFMLPNet is the encoded frequency dimension, which is the same as input here
    recurrent_step = TFMLPNet(latent_channels, num_blocks, mixer_repetitions, num_freqs)
    recurrent_step.to('cpu').eval()

    # Dummy inputs for a single time step
    dummy_frame = torch.randn(batch_size, latent_channels, num_freqs)
    h_states = torch.randn(num_blocks, batch_size, latent_channels, num_freqs)
    c_states = torch.randn(num_blocks, batch_size, latent_channels, num_freqs)

    torch.onnx.export(
        recurrent_step,
        (dummy_frame, h_states, c_states),
        "recurrent_step.onnx",
        opset_version=9,
        input_names=['frame', 'h_states', 'c_states'],
        output_names=['out_frame', 'next_h', 'next_c'],
        dynamic_axes={'frame': {0: 'batch_size'},
                        'h_states': {1: 'batch_size'},
                        'c_states': {1: 'batch_size'},
                        'out_frame': {0: 'batch_size'},
                        'next_h': {1: 'batch_size'},
                        'next_c': {1: 'batch_size'}}
    )
    print("Recurrent step exported to recurrent_step.onnx")

    # --- 3. Export Decoder ---
    decoder = Decoder(latent_channels, num_speakers)
    decoder.to('cpu').eval()
    dummy_processed_spec = torch.randn(batch_size, latent_channels, num_freqs, num_frames)

    torch.onnx.export(
        decoder,
        dummy_processed_spec,
        "decoder.onnx",
        opset_version=9,
        input_names=['processed_spec'],
        output_names=['mask'],
        dynamic_axes={'processed_spec': {0: 'batch_size', 3: 'num_frames'},
                        'mask': {0: 'batch_size', 3: 'num_frames'}}
    )
    print("Decoder exported to decoder.onnx")
    
    # --- 4. Export Simplified Decoder (Optional but good practice) ---
    # This version is for a single frame, useful for some runtimes
    decoder_simplified = Decoder(latent_channels, num_speakers)
    decoder_simplified.to('cpu').eval()
    dummy_processed_frame = torch.randn(batch_size, latent_channels, num_freqs, 1)

    torch.onnx.export(
        decoder_simplified,
        dummy_processed_frame,
        "decoder_simplified.onnx",
        opset_version=9,
        input_names=['processed_frame'],
        output_names=['mask_frame'],
        dynamic_axes={'processed_frame': {0: 'batch_size'},
                        'mask_frame': {0: 'batch_size'}}
    )
    print("Simplified Decoder exported to decoder_simplified.onnx")


if __name__ == "__main__":
    main()
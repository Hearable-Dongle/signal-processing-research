import onnx
import torch
import torch.nn as nn

from typing import List, Tuple

from tf_mlp import TFMLPNet, MLPNetBlock, MLPMixerModule, ConvBatchedLSTM, MLP

# Define Encoder and Decoder as separate modules for ONNX export
class Encoder(nn.Module):
    def __init__(self, latent_channels: int, win_length: int):
        super().__init__()
        self.num_freqs = win_length // 2 + 1
        self.encoder_layers = nn.Sequential(
            nn.ConstantPad2d((0, 0, 2, 0), 0.0), # Padding for encoder convolution
            nn.Conv2d(in_channels=2, out_channels=latent_channels, kernel_size=3)
        )
    def forward(self, spec_ri: torch.Tensor) -> torch.Tensor:
        return self.encoder_layers(spec_ri)

class Decoder(nn.Module):
    def __init__(self, latent_channels: int, num_speakers: int):
        super().__init__()
        self.decoder_layers = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=latent_channels,
                out_channels=2 * num_speakers,
                kernel_size=3
            ),
            nn.ConstantPad2d((0, 0, -2, 0), 0.0) # Trim padding from ConvTranspose2d
        )
    def forward(self, processed_spec: torch.Tensor) -> torch.Tensor:
        return self.decoder_layers(processed_spec)

def main():
    # --- Model Configuration ---
    fs = 16000
    duration_s = 4
    batch_size = 2
    
    full_config = {
        "win_length": 512,
        "hop_length": 128,
        "latent_channels": 64,
        "num_blocks": 4,
        "mixer_repetitions": 2,
        "num_speakers": 2
    }

    recurrent_config = {
        "latent_channels": full_config["latent_channels"],
        "num_blocks": full_config["num_blocks"],
        "mixer_repetitions": full_config["mixer_repetitions"],
        "num_freqs": full_config["win_length"] // 2 + 1
    }

    # Create model instances
    encoder_model = Encoder(full_config["latent_channels"], full_config["win_length"])
    recurrent_step_model = TFMLPNet(**recurrent_config)
    decoder_model = Decoder(full_config["latent_channels"], full_config["num_speakers"])

    # Move to CPU and set to eval mode
    encoder_model.to(device='cpu').eval()
    recurrent_step_model.to(device='cpu').eval()
    decoder_model.to(device='cpu').eval()

    # JIT compile the recurrent step model
    scripted_recurrent_step_model = torch.jit.script(recurrent_step_model)
    scripted_recurrent_step_model.eval()

    # --- Create Dummy Inputs ---
    num_freqs = full_config["win_length"] // 2 + 1
    num_frames = (fs * duration_s) // full_config["hop_length"] + 1

    # Dummy input for Encoder
    dummy_spec_ri_input = torch.randn(batch_size, 2, num_freqs, num_frames)

    # Dummy input for Recurrent Step Model
    # frame: (B, C, F')
    # h_states, c_states: (num_blocks, B, C, F')
    dummy_frame_input = torch.randn(batch_size, full_config["latent_channels"], num_freqs)
    dummy_h_states_input = torch.randn(full_config["num_blocks"], batch_size, full_config["latent_channels"], num_freqs)
    dummy_c_states_input = torch.randn(full_config["num_blocks"], batch_size, full_config["latent_channels"], num_freqs)

    # Dummy input for Decoder
    # processed_spec: (B, C, F', 1) - for a single frame
    dummy_processed_spec_input = torch.randn(batch_size, full_config["latent_channels"], num_freqs, 1)

    # --- Export Models to ONNX ---
    opset_version = 9 # nncase v0.2.0-beta4 works best with opset 9

    # Export Encoder
    encoder_output_file = "encoder.onnx"
    torch.onnx.export(
        encoder_model,
        dummy_spec_ri_input,
        encoder_output_file,
        export_params=True,
        opset_version=opset_version,
        input_names=['spec_ri_input'],
        output_names=['encoded_spec'],
        verbose=True
    )
    print(f"Exported {encoder_output_file}")

    # Export Recurrent Step Model
    recurrent_output_file = "recurrent_step.onnx"
    torch.onnx.export(
        scripted_recurrent_step_model,
        (dummy_frame_input, dummy_h_states_input, dummy_c_states_input),
        recurrent_output_file,
        export_params=True,
        opset_version=opset_version,
        input_names=['frame_input', 'h_states_input', 'c_states_input'],
        output_names=['processed_frame', 'next_h_states', 'next_c_states'],
        verbose=True
    )
    print(f"Exported {recurrent_output_file}")

    # Export Decoder
    decoder_output_file = "decoder.onnx"
    torch.onnx.export(
        decoder_model,
        dummy_processed_spec_input,
        decoder_output_file,
        export_params=True,
        opset_version=opset_version,
        input_names=['processed_spec_input'],
        output_names=['mask_output'],
        verbose=True
    )
    print(f"Exported {decoder_output_file}")

if __name__ == "__main__":
    main()
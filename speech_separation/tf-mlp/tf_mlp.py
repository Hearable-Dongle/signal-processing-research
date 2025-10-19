import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class MLP(nn.Module):
    """A simple two-layer MLP with a residual connection."""
    def __init__(self, dim: int, expansion_factor: float = 4.0):
        super().__init__()
        hidden_dim = int(dim * expansion_factor)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x + residual

class MLPMixerModule(nn.Module):
    """
    MLP-Mixer module that applies MLPs alternately along frequency and channel dimensions.
    This corresponds to the 'Spectral Stage' in the paper.
    """
    def __init__(self,
                 num_channels: int,
                 num_freqs: int,
                 expansion_factor: float = 4.0,
                 num_repetitions: int = 2):
        super().__init__()
        self.mixers = nn.ModuleList()
        for _ in range(num_repetitions):
            # Frequency Mixing
            self.mixers.append(MLP(num_freqs, expansion_factor))
            # Channel Mixing
            self.mixers.append(MLP(num_channels, expansion_factor))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: (Batch, Channels, Freqs)
        for i, mixer in enumerate(self.mixers):
            if i % 2 == 0: # Frequency mixing
                x = mixer(x)
            else: # Channel mixing
                x = x.transpose(1, 2) # (B, F, C)
                x = mixer(x)
                x = x.transpose(1, 2) # (B, C, F)
        return x

class ConvBatchedLSTM(nn.Module):
    """
    An LSTM-like module that processes frequency bins in parallel using 1D convolutions.
    This corresponds to the 'Temporal Stage' in the paper.
    """
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        # A single Conv1d computes all 4 gates for x_t and h_{t-1}
        self.gates_conv = nn.Conv1d(
            in_channels=input_size + hidden_size,
            out_channels=4 * hidden_size,
            kernel_size=1
        )

    def forward(self, x: torch.Tensor, state: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        h_prev, c_prev = state
        # Input x shape: (Batch, Channels, Freqs)
        # Concatenate input and previous hidden state along the channel dimension
        combined = torch.cat((x, h_prev), dim=1) # (B, C_in + H, F)

        # Compute all gates in parallel using convolution
        gates = self.gates_conv(combined) # (B, 4*H, F)

        # Split the gates
        i, f, g, o = torch.split(gates, self.hidden_size, dim=1)

        # Apply activations
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)

        # Calculate new cell state and hidden state
        c_t = f * c_prev + i * g
        h_t = o * torch.tanh(c_t)

        return h_t, (h_t, c_t)

class MLPNetBlock(nn.Module):
    """A single TF-MLPNet block combining spectral and temporal stages."""
    def __init__(self,
                 num_channels: int,
                 num_freqs: int,
                 mixer_repetitions: int = 2,
                 expansion_factor: float = 4.0):
        super().__init__()
        self.spectral_stage = MLPMixerModule(
            num_channels, num_freqs, expansion_factor, mixer_repetitions
        )
        self.temporal_stage = ConvBatchedLSTM(num_channels, num_channels)

        # Use 1x1 Convs for the Fully Connected (FC) layers in the diagram
        self.fc_spectral = nn.Conv1d(num_channels, num_channels, 1)
        self.fc_temporal = nn.Conv1d(num_channels, num_channels, 1)

    def forward(self, x: torch.Tensor, state: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # Input x shape: (Batch, Channels, Freqs)
        residual_spectral = x
        x_spectral = self.spectral_stage(x)
        x_spectral = self.fc_spectral(x_spectral)
        x = x_spectral + residual_spectral

        residual_temporal = x
        x_temporal, new_state = self.temporal_stage(x, state)
        x_temporal = self.fc_temporal(x_temporal)
        x = x_temporal + residual_temporal

        return x, new_state


class TFMLPNet(nn.Module):
    """
    The TF-MLPNet model refactored to represent a single recurrent step for ONNX export.
    It processes one time frame of the encoded spectrogram and updates the hidden states.
    """
    def __init__(self,
                 latent_channels: int,
                 num_blocks: int,
                 mixer_repetitions: int,
                 num_freqs: int): # Renamed from encoded_freqs for clarity
        super().__init__()
        self.num_blocks = num_blocks
        self.latent_channels = latent_channels
        self.num_freqs = num_freqs # This is the F' dimension

        self.blocks = nn.ModuleList([
            MLPNetBlock(latent_channels, num_freqs, mixer_repetitions)
            for _ in range(num_blocks)
        ])

    def forward(self,
                frame: torch.Tensor, # (B, C, F')
                h_states: torch.Tensor, # (num_blocks, B, C, F')
                c_states: torch.Tensor  # (num_blocks, B, C, F')
               ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # frame shape: (B, C, F') - this is a single time slice from encoded_spec

        next_h_states_list = []
        next_c_states_list = []
        for i, block in enumerate(self.blocks):
            h_prev = h_states[i]
            c_prev = c_states[i]
            
            frame, (h, c) = block(frame, (h_prev, c_prev))
            
            next_h_states_list.append(h)
            next_c_states_list.append(c)
        
        next_h_states = torch.stack(next_h_states_list, dim=0)
        next_c_states = torch.stack(next_c_states_list, dim=0)
        
        # The output of this single step is the processed frame
        # The decoder will be applied outside this single step model
        return frame, next_h_states, next_c_states

DEFAULT_CONFIG = {
        "win_length": 512,        # Corresponds to 32 ms window
        "hop_length": 128,        # Corresponds to 8 ms hop (75% overlap)
        "latent_channels": 64,    # Number of channels after encoder
        "num_blocks": 4,          # Number of MLPNet blocks (B)
        "mixer_repetitions": 2,   # MLP-Mixer repetitions (M)
        "num_speakers": 2         # S=2 for Blind Source Separation (BSS)
    }


# Example Usage
if __name__ == '__main__':
    # Define Encoder and Decoder as separate modules for demonstration
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

    # --- Model Configuration ---
    # These parameters match common settings for 16kHz audio
    fs = 16000
    duration_s = 4
    batch_size = 2
    
    full_config = {
        "win_length": 512,        # Corresponds to 32 ms window
        "hop_length": 128,        # Corresponds to 8 ms hop (75% overlap)
        "latent_channels": 64,    # Number of channels after encoder
        "num_blocks": 4,          # Number of MLPNet blocks (B)
        "mixer_repetitions": 2,   # MLP-Mixer repetitions (M)
        "num_speakers": 2         # S=2 for Blind Source Separation (BSS)
    }

    # Recurrent step model configuration
    recurrent_config = {
        "latent_channels": full_config["latent_channels"],
        "num_blocks": full_config["num_blocks"],
        "mixer_repetitions": full_config["mixer_repetitions"],
        "num_freqs": full_config["win_length"] // 2 + 1 # F' dimension
    }

    # Create model instances
    encoder_model = Encoder(full_config["latent_channels"], full_config["win_length"])
    recurrent_step_model = TFMLPNet(**recurrent_config)
    decoder_model = Decoder(full_config["latent_channels"], full_config["num_speakers"])

    print(f"Recurrent Step Model created with {sum(p.numel() for p in recurrent_step_model.parameters())/1e6:.2f}M parameters.")

    # --- Create Dummy Input ---
    # Original waveform input
    dummy_wav_input = torch.randn(batch_size, fs * duration_s)
    print(f"Dummy Waveform Input shape: {dummy_wav_input.shape}")

    # Simulate STFT to get complex spectrogram
    n_fft = full_config["win_length"]
    hop_length = full_config["hop_length"]
    window = torch.hann_window(n_fft)
    spec = torch.stft(
        dummy_wav_input,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=window,
        return_complex=True
    )
    spec_ri_input = torch.stack([spec.real, spec.imag], dim=1) # (B, 2, F, T)
    print(f"Simulated Spectrogram Input shape: {spec_ri_input.shape}")

    # --- Run Full Inference Pipeline (simulated) ---
    with torch.no_grad():
        encoder_model.eval()
        recurrent_step_model.eval()
        decoder_model.eval()

        # 1. Encoder
        encoded_spec = encoder_model(spec_ri_input) # (B, C, F', T')
        B, C, F_prime, T_prime = encoded_spec.shape
        print(f"Encoded Spec shape: {encoded_spec.shape}")

        # Initialize states for the recurrent step
        h_states = torch.zeros(recurrent_config["num_blocks"], B, C, F_prime, device=encoded_spec.device)
        c_states = torch.zeros(recurrent_config["num_blocks"], B, C, F_prime, device=encoded_spec.device)

        outputs_over_time = []
        for t in range(T_prime):
            frame = encoded_spec[..., t] # (B, C, F')
            processed_frame, h_states, c_states = recurrent_step_model(frame, h_states, c_states)
            outputs_over_time.append(processed_frame)

        processed_spec = torch.stack(outputs_over_time, dim=3) # (B, C, F', T')
        print(f"Processed Spec shape: {processed_spec.shape}")

        # 3. Decoder
        mask = decoder_model(processed_spec) # (B, 2*S, F, T)
        print(f"Output Mask shape: {mask.shape}")

    # Expected output shape: (batch_size, num_speakers * 2, num_freqs, num_frames)
    num_freqs_orig = full_config["win_length"] // 2 + 1
    num_frames_orig = (fs * duration_s) // full_config["hop_length"] + 1
    assert mask.shape == (batch_size, full_config["num_speakers"] * 2, num_freqs_orig, num_frames_orig)
    print("Full inference pipeline simulation successful. Output shape is correct.")

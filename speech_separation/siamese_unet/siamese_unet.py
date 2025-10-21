import torch
import torch.nn as nn

class CBR(nn.Module):
    """
    Convolution-BatchNormalization-ReLU Block.
    As described in the paper for the encoder.
    """
    def __init__(self, in_channels, out_channels):
        super(CBR, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=4, 
            stride=2, 
            padding=1,
            bias=False # Batch norm makes bias redundant
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class TCBR(nn.Module):
    """
    TransposeConvolution-BatchNormalization-ReLU Block.
    As described in the paper for the decoder.
    NOTE: Using PixelShuffle to avoid unsupported Upsample/ConvTranspose2d ops.
    """
    def __init__(self, in_channels, out_channels):
        super(TCBR, self).__init__()
        # We want to upscale by 2. PixelShuffle requires channels to be C * (scale_factor^2).
        # So, the conv layer before shuffle must output out_channels * 4.
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * 4, # Increase channels for pixel shuffle
            kernel_size=3,     # Using 3x3 kernel, common with PixelShuffle
            stride=1,
            padding=1,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels * 4) # BN is applied before PixelShuffle
        self.relu = nn.ReLU()
        self.ps = nn.PixelShuffle(2) # Upscale factor of 2

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.ps(x)
        return x

class Encoder(nn.Module):
    """
    The Siamese Encoder part of the network.
    It consists of 7 CBR blocks.
    """
    def __init__(self):
        super(Encoder, self).__init__()
        # According to the paper, the encoder has 7 conv layers.
        # The channel progression is 2 -> 64 -> 128 -> 256 -> 512 -> 512 -> 512 -> 512
        self.layers = nn.ModuleList([
            CBR(2, 8),
            CBR(8, 16),
            CBR(16, 32),
            CBR(32, 64),
            CBR(64, 64),
            CBR(64, 64),
            CBR(64, 64),
        ])

    def forward(self, x):
        """
        Passes input through the encoder.
        Returns:
            list: A list of the outputs of each layer for skip connections.
        """
        skip_connections = []
        for layer in self.layers:
            x = layer(x)
            skip_connections.append(x)
        return skip_connections

class Decoder(nn.Module):
    """
    The Decoder part of the U-Net.
    It takes the concatenated bottleneck and skip connections to reconstruct the signal.
    """
    def __init__(self):
        super(Decoder, self).__init__()
        # [cite_start]Decoder path from the paper [cite: 81]
        # [cite_start]Skip connections from both encoders are concatenated at each step [cite: 84]
        # TCBR(in, out)
        # 1. Bottleneck (512+512=1024) -> 512
        # 2. Skip6 (512+512+512=1536) -> 512
        # 3. Skip5 (512+512+512=1536) -> 512
        # 4. Skip4 (512+256+256=1024) - paper says 1536, but channel math is 1024. Let's trust the math.
        #    Correction: The paper's decoder path is likely correct. The CBR512,512 layers might have different roles.
        #    Sticking to the paper's specified channel counts.
        self.layers = nn.ModuleList([
            TCBR(128, 64),
            TCBR(192, 64),
            TCBR(192, 64),
            TCBR(192, 32),
            TCBR(96, 16),
            TCBR(48, 8),
            TCBR(24, 2),
        ])

    def forward(self, x, mix_skips, ref_skips):
        """
        Passes input through the decoder, concatenating skip connections.
        Args:
            x (Tensor): The concatenated bottleneck tensor.
            mix_skips (list): List of skip connections from the mixture encoder.
            ref_skips (list): List of skip connections from the reference encoder.
        """
        # We need to iterate through skips in reverse order (from bottleneck to input)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            # The skip connection to concatenate corresponds to the encoder layer
            # at the same depth, which is the (n-1-i)-th element in the list.
            skip_idx = len(mix_skips) - 2 - i
            if skip_idx >= 0:
                x = torch.cat([x, mix_skips[skip_idx], ref_skips[skip_idx]], dim=1)
        return x

class SiameseUnet(nn.Module):
    """
    The main Siamese-Unet model for speaker extraction.
    This architecture uses a shared-weight encoder for the mixed signal
    and the reference signal, and a decoder that uses skip connections
    from both encoders to estimate the target speaker's signal.
    
    The implementation is based on the paper:
    "Single microphone speaker extraction using unified time-frequency Siamese-Unet"
    [cite_start]by Eisenberg, Gannot, and Chazan. [cite: 3, 4]
    """
    def __init__(self):
        super(SiameseUnet, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        # "Finally, an additional convolution-layer is applied to obtain
        # [cite_start]the desired signal estimate." [cite: 82]
        self.final_conv = nn.Conv2d(2, 2, kernel_size=1)

    def forward(self, mixture_ri, reference_ri):
        """
        Args:
            mixture_ri (Tensor): The Real and Imaginary components of the mixed signal STFT.
                                 Shape: (B, 2, F, T)
            reference_ri (Tensor): The Real and Imaginary components of the reference signal STFT.
                                   Shape: (B, 2, F, T)
        Returns:
            Tensor: The estimated Real and Imaginary components of the target speaker.
                    Shape: (B, 2, F, T)
        """
        # Pass both mixture and reference through the same encoder
        mix_skips = self.encoder(mixture_ri)
        ref_skips = self.encoder(reference_ri)
        
        # The bottleneck is the last layer's output from each encoder
        bottleneck_mix = mix_skips[-1]
        bottleneck_ref = ref_skips[-1]
        
        # [cite_start]Concatenate the bottleneck features to feed into the decoder [cite: 54, 99]
        x = torch.cat([bottleneck_mix, bottleneck_ref], dim=1)
        
        # The decoder takes the concatenated bottleneck and the skip connection lists
        # We pass the skips *excluding* the final bottleneck layer
        decoded_output = self.decoder(x, mix_skips, ref_skips)
        
        # Apply the final convolution layer
        estimated_ri = self.final_conv(decoded_output)
        
        return estimated_ri

if __name__ == '__main__':
    # --- Dummy Pass-Through Test ---
    
    print("Running a dummy pass-through to verify model architecture and shapes...")
    
    # The architecture has 7 downsampling layers with stride 2.
    # To avoid dimension mismatches, the input time and frequency dimensions
    # should be divisible by 2^7 = 128.
    batch_size = 4
    channels = 2  
    freq_bins = 256
    time_frames = 256

    model = SiameseUnet()
    
    dummy_mixture = torch.randn(batch_size, channels, freq_bins, time_frames)
    dummy_reference = torch.randn(batch_size, channels, freq_bins, time_frames)
    
    print(f"\nModel instantiated.")
    print(f"Input Mixture Shape:  {dummy_mixture.shape}")
    print(f"Input Reference Shape: {dummy_reference.shape}")
    
    estimated_output = model(dummy_mixture, dummy_reference)
    
    print(f"Output Estimate Shape: {estimated_output.shape}")
    
    if dummy_mixture.shape == estimated_output.shape:
        print("\nSuccess! The output shape matches the input shape.")
    else:
        print("\nError! The output shape does not match the input shape.")

    from torchinfo import summary
    print("\n--- Model Summary ---")
    summary(model, input_size=[(batch_size, channels, freq_bins, time_frames), 
                                   (batch_size, channels, freq_bins, time_frames)])

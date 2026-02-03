import torch
import torch.nn as nn
from typing import List
import os
from asteroid.models import ConvTasNet
from asteroid.masknn import norms

# --- MONKEY PATCH ---
def _glob_norm_onnx_safe(x, eps: float = 1e-8):
    # For ConvTasNet, the tensor is 3D (batch, feats, time).
    # We hardcode normalization over the last two dimensions.
    dims: List[int] = [1, 2]
    return norms.z_norm(x, dims, eps)

norms._glob_norm = _glob_norm_onnx_safe
# --------------------

class HailoExportWrapper(nn.Module):
    def __init__(self, original_model):
        super().__init__()
        self.encoder = original_model.encoder
        self.masker = original_model.masker
        self.decoder = original_model.decoder
        self.enc_activation = original_model.enc_activation

    def forward(self, wav):
        # 1. FORCE FIXED INPUT SHAPE
        # We assume wav is pre-processed to be [1, 1, Fixed_Time]
        # Remove _unsqueeze_to_3d logic, assume input is correct.
        
        # Encoder
        tf_rep = self.encoder(wav)
        tf_rep = self.enc_activation(tf_rep)
        
        # Masker
        est_masks = self.masker(tf_rep)
        
        # Application (Element-wise multiplication is supported by Hailo)
        # Note: We use unsqueeze(1) here to broadcast mask to sources
        masked_tf_rep = est_masks * tf_rep.unsqueeze(1)
        
        # Decoder
        decoded = self.decoder(masked_tf_rep)
        
        # STOP HERE.
        # Do NOT call pad_x_to_y.
        # Do NOT call _shape_reconstructed.
        
        return decoded

def main():
    # 1. Load your trained asteroid model
    # Using the model specified for the Hailo export
    pretrained_model = ConvTasNet.from_pretrained("mpariente/ConvTasNet_WHAM!_sepclean")
    
    # 2. Wrap it
    export_model = HailoExportWrapper(pretrained_model)
    export_model.eval()
    
    # 3. Create Dummy Input with FIXED size
    # Hailo needs a specific input size (e.g., 2 seconds of audio @ 8kHz)
    # 8000 Hz * 2 sec = 16000 samples
    dummy_input = torch.randn(1, 1, 16000) 
    
    output_path = "hailo/convtas_hailo_ready.onnx"
    
    # 4. Export
    print(f"Exporting model to {output_path}...")
    torch.onnx.export(
        export_model,
        dummy_input,
        output_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=12,
        do_constant_folding=True,
        # CRITICAL: Do NOT use dynamic axes for Hailo if possible. 
        # It prefers fixed sizes. If you must, only make the time dimension dynamic,
        # but fixed is much safer for the initial compilation.
    )
    print(f"Export done to {output_path}")

if __name__ == "__main__":
    main()
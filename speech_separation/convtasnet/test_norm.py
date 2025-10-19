import onnx
import torch
from asteroid.models import ConvTasNet
import asteroid.masknn.norms as norms

# --- The Final K210-Safe Normalization Function ---
def _glob_norm_onnx_safe_k210(x, eps: float = 1e-8):
    """
    Performs z-normalization by manually calculating standard deviation
    using only basic math operations compatible with the ncc compiler.
    """
    # The dimensions to normalize over
    dims: list[int] = [1, 2] 
    
    # Calculate the mean, keeping the dimensions
    mean = torch.mean(x, dim=dims, keepdim=True)
    
    # Manually calculate standard deviation using multiplication
    mean_sq = torch.mean(x * x, dim=dims, keepdim=True)
    var = mean_sq - (mean * mean)
    std = torch.sqrt(var + eps)
    
    # Apply the normalization
    return (x - mean) / std
# ---------------------------------------------------------------

# --- MONKEY PATCH with the new, working function ---
norms._glob_norm = _glob_norm_onnx_safe_k210
# ---------------------------------------------------

def main():
    # model_name = "JorisCos/ConvTasNet_Libri2Mix_sepnoisy_16k"
    # print(f"Loading pre-trained model: {model_name}...")
    # model = ConvTasNet.from_pretrained(model_name)
    model = ConvTasNet(
        n_src=2,
        n_feats=64,   # Drastically reduce the number of features/filters
        n_hid=64,     # Drastically reduce the hidden size
        n_layers=4,   # Reduce the number of layers
        n_blocks=2,   # Reduce the number of blocks
        # ... keep other params like kernel_size or use defaults ...
    )

    model.to(device='cpu')
    model.eval()

    output_file = "test_convtas.onnx"

    # Define a fixed-shape input tensor
    # The length (16000) can be adjusted, but it must be a fixed size
    # single_clip = torch.randn(1, 1, 16000)
    single_clip = torch.randn(1, 1, 4096)

    print("Exporting model to ONNX...")
    torch.onnx.export(
        model,
        single_clip,
        output_file,
        export_params=True,
        opset_version=9, # Keep this at 9 for the old compiler
        # opset_version=9, # Keep this at 9 for the old compiler
        input_names=['input'],
        output_names=['output'],
        verbose=False
    )
    print(f"✅ Export successful. Model saved to {output_file}")


if __name__ == "__main__":
    main()

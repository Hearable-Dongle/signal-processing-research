import onnx
import torch

from asteroid.models import ConvTasNet
import asteroid.masknn.norms as norms
from typing import List

# --- MONKEY PATCH ---
def _glob_norm_onnx_safe(x, eps: float = 1e-8):
    # For ConvTasNet, the tensor is 3D (batch, feats, time).
    # We hardcode normalization over the last two dimensions.
    dims: List[int] = [1, 2]
    return norms.z_norm(x, dims, eps)

norms._glob_norm = _glob_norm_onnx_safe
# --------------------

EXPECTED_SAMPLE_RATE = 16000


def main():
    model_name = "JorisCos/ConvTasNet_Libri2Mix_sepnoisy_16k"
    print(f"Loading pre-trained model: {model_name}...")
    model = ConvTasNet.from_pretrained(model_name)
    model.to(device='cpu')
    model.eval()

    print(f"Model created with {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters.")

    # TODO: actually add weights
    # file_path = '.'
    # state_dict = torch.load(file_path)
    output_file = "convtask.onnx"    

    # single_clip = torch.randn(1, 1, 16000)
    single_clip = torch.randn(1, 1, 4096)
    output = model(single_clip)

    print("Called model on a single input")
    print(f"Input shape: {single_clip.shape}")
    print(f"Output shape: {output.shape}")

    torch.onnx.export(
        model,
        single_clip,
        output_file,
        export_params=True,
        opset_version=9,          # nncase v0.2.0-beta4 works best with opset 9
        input_names=['input'],
        output_names=['output'],
        verbose=False
    )
    print("Export done")


if __name__ == "__main__":
    main()

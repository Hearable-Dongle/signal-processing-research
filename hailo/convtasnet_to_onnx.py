import torch
from typing import List
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
    import os
    output_file = os.path.join(os.path.dirname(__file__), "convtask.onnx")

    # single_clip = torch.randn(1, 1, 16000)
    single_clip = torch.randn(1, 1, 4096)
    output = model(single_clip)

    print("Called model on a single input")
    print(f"Input shape: {single_clip.shape}")
    print(f"Output shape: {output.shape}")
    torch.onnx.export(
        model,
        single_clip,
        "hailo/convtas.onnx",
        opset_version=11,          # <--- Critical: PyTorch 1.13 handles this natively
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size', 2: 'time'},
            'output': {0: 'batch_size', 2: 'time'}
        }
    )
    print(f"Export done to {output_file}")


if __name__ == "__main__":
    main()

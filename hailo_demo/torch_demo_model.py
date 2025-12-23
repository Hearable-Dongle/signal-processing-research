import torch
import torch.nn as nn
import onnx


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 112 * 112, 10)
        )

    def forward(self, x):
        return self.conv(x)

model = SimpleCNN().eval()
dummy_input = torch.randn(1, 3, 224, 224)
onnx_path = "model_static.onnx"

print("Exporting to ONNX...")
torch.onnx.export(
    model, 
    dummy_input, 
    onnx_path,
    export_params=True,
    opset_version=15, # Supported by Hailo
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    # Force the legacy exporter to prevent automatic upgrade to 18
    operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
    # Explicitly disable the new dynamo-based exporter
    dynamo=False 
)


model = onnx.load(onnx_path)
print(f"Verified Opset Version: {model.opset_import[0].version}")


import torch
import torch.nn as nn
import torch.onnx

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

torch.onnx.export(model, dummy_input, "model.onnx", 
                  input_names=['input'], 
                  output_names=['output'],
                  export_params=True,
                  opset_version=11,
                  do_constant_folding=True
                  )
print("Model exported to model.onnx")

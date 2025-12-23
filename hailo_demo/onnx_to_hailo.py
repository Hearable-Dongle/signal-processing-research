
from hailo_sdk_client import ClientRunner

onnx_path = "model_static.onnx"
model_name = "simple_cnn"
har_path = f"{model_name}.har"
hef_path = f"{model_name}.hef"

runner = ClientRunner()

# Stage 1: Translation [cite: 18, 19]
print("Starting Translation...")
runner.translate_onnx_model(
    onnx_path, 
    model_name,
    start_node_names=['input'],
    end_node_names=['output']
)
runner.save_har(har_path) # Saves the Hailo Archive [cite: 174, 175]

# Stage 2: Optimization (Quantization) [cite: 18, 27]
print("Starting Optimization...")
runner.optimize(calib_data=None) 

# Stage 3: Compilation [cite: 18, 47]
print("Starting Compilation...")
hef = runner.compile() # Transitions to Compiled Model state [cite: 48]

with open(hef_path, "wb") as f:
    f.write(hef)

print(f"Success! HEF saved to: {hef_path}")
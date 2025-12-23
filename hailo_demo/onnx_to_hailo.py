
from hailo_sdk_client import ClientRunner

onnx_path = "model_static.onnx"
model_name = "simple_cnn"
har_path = f"{model_name}.har"
hef_path = f"{model_name}.hef"

runner = ClientRunner()

print("Starting Translation...")
runner.translate_onnx_model(
    onnx_path, 
    model_name,
    start_node_names=['input'],
    end_node_names=['output']
)
runner.save_har(har_path) 

# TODO: Here and on is not working
print("Starting Optimization...")
runner.optimize(calib_data=None) 


print("Starting Compilation...")
hef = runner.compile() 

with open(hef_path, "wb") as f:
    f.write(hef)

print(f"Success! HEF saved to: {hef_path}")
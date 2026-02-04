import numpy as np
import argparse
import os
from hailo_sdk_client import ClientRunner

SAMPLE_RATE = 16000


def main():
    parser = argparse.ArgumentParser(description="Convert ONNX to HEF using Hailo SDK")
    parser.add_argument("onnx_path", nargs="?", default="hailo/convtas_hailo_ready_patched.onnx", help="Input ONNX file path")
    parser.add_argument("hef_path", nargs="?", default="hailo/convtas.hef", help="Output HEF file path")
    parser.add_argument("--har_path", default=None, help="Output HAR file path")
    parser.add_argument("--model_name", default="convtas", help="Model name for Hailo SDK")
    
    args = parser.parse_args()
    
    onnx_path = args.onnx_path
    hef_path = args.hef_path
    model_name = args.model_name
    
    if args.har_path:
        har_path = args.har_path
    else:
        # derive from hef_path
        base, _ = os.path.splitext(hef_path)
        har_path = base + ".har"

    print(f"Loading ONNX model from {onnx_path}")
    
    runner = ClientRunner()
    
    print("Starting Translation...")

    try:
        runner.translate_onnx_model(
            onnx_path, 
            model_name,
            start_node_names=['input'],
            net_input_shapes={'input': [1, 1, 1, SAMPLE_RATE]} 
        )
    except Exception as e:
        print(f"Translation failed with error: {e}")
        raise e
    
    runner.save_har(har_path)
    print(f"HAR saved to {har_path}")

    print("Starting Optimization (with random calibration data)...")

    # 8 samples of random noise
    calib_data = [np.random.randn(1, 1, 1, SAMPLE_RATE).astype(np.float32) for _ in range(8)]
    
    print(f"Calib data type: {type(calib_data)}")
    if isinstance(calib_data, list):
        print(f"Calib data len: {len(calib_data)}")
        if len(calib_data) > 0:
            print(f"Calib data item type: {type(calib_data[0])}")
            print(f"Calib data item shape: {calib_data[0].shape}")

    # Use dict to be explicit, and convert list to numpy array
    # Input name is mapped to convtas/input_layer1
    # Transpose NCHW [1, 1, 1, SAMPLE_RATE] -> NHWC [1, 1, SAMPLE_RATE, 1]
    # And remove batch dim -> [1, SAMPLE_RATE, 1]
    calib_data_transposed = [d.transpose(0, 2, 3, 1)[0] for d in calib_data]
    
    # Note: Input layer name might depend on the model name. 
    # Usually it's {model_name}/input_layer1 if not explicitly renamed, 
    # but let's check what the SDK expects or use the auto-assigned name if possible.
    # The previous code hardcoded 'convtas/input_layer1'.
    # If model_name changes, this key might need to change.
    # However, since we define model_name in translate_onnx_model, it should match.
    input_layer_name = f"{model_name}/input_layer1"
    calib_data_dict = {input_layer_name: np.array(calib_data_transposed)}

    runner.optimize(calib_data=calib_data_dict)

    print("Starting Compilation...")
    hef = runner.compile() 

    with open(hef_path, "wb") as f:
        f.write(hef)

    print(f"Success! HEF saved to: {hef_path}")

if __name__ == "__main__":
    main()
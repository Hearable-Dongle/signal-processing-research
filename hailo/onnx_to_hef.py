import os
import numpy as np
from hailo_sdk_client import ClientRunner

SAMPLE_RATE = 16000


def main():
    onnx_path = "hailo/convtas_hailo_ready_patched.onnx"
    model_name = "convtas"
    har_path = f"hailo/{model_name}.har"
    hef_path = f"hailo/{model_name}.hef"

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
    calib_data_dict = {'convtas/input_layer1': np.array(calib_data_transposed)}

    runner.optimize(calib_data=calib_data_dict)

    print("Starting Compilation...")
    hef = runner.compile() 

    with open(hef_path, "wb") as f:
        f.write(hef)

    print(f"Success! HEF saved to: {hef_path}")

if __name__ == "__main__":
    main()
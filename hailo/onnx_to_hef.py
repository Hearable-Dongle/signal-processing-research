import os
import numpy as np
from hailo_sdk_client import ClientRunner

def main():
    # onnx_path = "hailo/convtas_fixed.onnx"
    onnx_path = "hailo/convtas_hailo_ready_fixed.onnx"
    model_name = "convtas"
    har_path = f"hailo/{model_name}.har"
    hef_path = f"hailo/{model_name}.hef"

    print(f"Loading ONNX model from {onnx_path}")
    
    runner = ClientRunner()
    
    print("Starting Translation...")
    
    # The compiler suggested these nodes in your error log. 
    # This cuts off the unsupported Expand/Reshape layers.
    # suggested_end_nodes = [
    #     "node_var", 
    #     "node_unsqueeze", 
    #     "node_Sub_29", 
    #     "node_Concat_823", 
    #     "node_Concat_828", 
    #     "node_Concat_818", 
    #     "node_Concat_36", 
    #     "node_Concat_833", 
    #     "node_Concat_41"
    # ]
    suggested_end_nodes = [
        "node_Sub_10", 
        "node_unsqueeze", 
        "node_var"
    ]

    try:
        runner.translate_onnx_model(
            onnx_path, 
            model_name,
            start_node_names=['input'],
            end_node_names=suggested_end_nodes,
            net_input_shapes={'input': [1, 1, 16000]} 
        )
    except Exception as e:
        print(f"Translation failed with error: {e}")
        raise e
    
    runner.save_har(har_path)
    print(f"HAR saved to {har_path}")

    print("Starting Optimization (with random calibration data)...")
    # 8 samples of random noise
    calib_data = [np.random.randn(1, 1, 16000).astype(np.float32) for _ in range(8)]
    
    runner.optimize(calib_data=calib_data)

    print("Starting Compilation...")
    hef = runner.compile() 

    with open(hef_path, "wb") as f:
        f.write(hef)

    print(f"Success! HEF saved to: {hef_path}")

if __name__ == "__main__":
    main()
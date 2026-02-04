import numpy as np
import argparse
import os
from hailo_sdk_client import ClientRunner

SAMPLE_RATE = 16000


def main():
    parser = argparse.ArgumentParser(description="Convert ONNX to HAR using Hailo SDK")
    parser.add_argument("onnx_path", nargs="?", default="hailo/convtas_hailo_ready_patched.onnx", help="Input ONNX file path")
    parser.add_argument("har_path", nargs="?", default="hailo/convtas.har", help="Output HAR file path")
    parser.add_argument("--model_name", default="convtas", help="Model name for Hailo SDK")
    
    args = parser.parse_args()
    
    onnx_path = args.onnx_path
    har_path = args.har_path
    model_name = args.model_name
    
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
    print(f"Success! HAR saved to {har_path}")

if __name__ == "__main__":
    main()
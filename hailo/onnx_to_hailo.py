import numpy as np
import argparse
import os
import onnx
from hailo_sdk_client import ClientRunner

SAMPLE_RATE = 16000


def _infer_input_shape_from_onnx(onnx_path: str, input_name: str = "input"):
    model = onnx.load(onnx_path)
    selected = None
    for value in model.graph.input:
        if value.name == input_name:
            selected = value
            break
    if selected is None and model.graph.input:
        selected = model.graph.input[0]
    if selected is None:
        return [1, 1, 1, SAMPLE_RATE], input_name

    dims = selected.type.tensor_type.shape.dim
    shape = []
    for d in dims:
        if d.dim_value and d.dim_value > 0:
            shape.append(int(d.dim_value))
        else:
            shape.append(1)

    if len(shape) != 4:
        shape = [1, 1, 1, SAMPLE_RATE]
    return shape, selected.name


def main():
    parser = argparse.ArgumentParser(description="Convert ONNX to HAR using Hailo SDK")
    parser.add_argument("onnx_path", nargs="?", default="hailo/convtas_hailo_ready_patched.onnx", help="Input ONNX file path")
    parser.add_argument("har_path", nargs="?", default="hailo/convtas.har", help="Output HAR file path")
    parser.add_argument("--model_name", default="convtas", help="Model name for Hailo SDK")
    parser.add_argument("--hw_arch", choices=["hailo8", "hailo8l", "hailo8r"], default="hailo8", help="Target hardware")
    
    args = parser.parse_args()
    
    onnx_path = args.onnx_path
    har_path = args.har_path
    model_name = args.model_name
    
    print(f"Loading ONNX model from {onnx_path}")
    
    runner = ClientRunner(hw_arch=args.hw_arch)
    
    print("Starting Translation...")
    input_shape, input_name = _infer_input_shape_from_onnx(onnx_path, input_name="input")
    print(f"Using ONNX input {input_name} shape {input_shape}")

    try:
        runner.translate_onnx_model(
            onnx_path, 
            model_name,
            start_node_names=[input_name],
            net_input_shapes={input_name: input_shape}
        )
    except Exception as e:
        print(f"Translation failed with error: {e}")
        raise e
    
    runner.save_har(har_path)
    print(f"Success! HAR saved to {har_path}")

if __name__ == "__main__":
    main()

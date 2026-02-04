import onnx
from onnx import helper

input_path = "hailo/convtas_hailo_ready.onnx"
output_path = "hailo/convtas_hailo_ready_patched.onnx"

print(f"Loading {input_path}...")
model = onnx.load(input_path)

# Build a map of initializers for quick lookup
initializers = {init.name: init for init in model.graph.initializer}

count = 0
for node in model.graph.node:
    if node.op_type in ["Conv", "ConvTranspose"]:
        if count == 0:
            print(f"Inspecting first Conv node: {node.name}")
            print(f"Inputs: {node.input}")
            for attr in node.attribute:
                print(f"Attr: {attr.name} = {attr}")

        # Check if kernel_shape exists
        exists = any(a.name == "kernel_shape" for a in node.attribute)
        if exists:
            continue
            
        # Try to find weights
        if len(node.input) < 2:
            print(f"Node {node.name} has < 2 inputs, skipping.")
            continue
            
        weight_name = node.input[1]
        if weight_name in initializers:
            init = initializers[weight_name]
            dims = init.dims
            # Assume 1D convolution converted to 2D Horizontal
            # Dims: [Out, In, 1, K] or [Out, In, K]
            
            kernel_size = []
            if len(dims) == 3:
                # [Out, In, K] -> [1, K]
                kernel_size = [1, dims[-1]]
            elif len(dims) == 4:
                # [Out, In, 1, K] -> [1, K]
                # Check if dim 2 is 1?
                if dims[2] == 1:
                    kernel_size = [1, dims[3]]
                else:
                    # Maybe [Out, In, K, 1]? (Vertical)
                    # But we switched to Horizontal.
                    # Let's assume standard behavior based on convert script.
                    # We expect [Out, In, 1, K]
                    kernel_size = [dims[2], dims[3]]
            else:
                 print(f"Node {node.name}: Weight dims {dims} unexpected. Skipping.")
                 continue

            attr = helper.make_attribute("kernel_shape", kernel_size)
            node.attribute.append(attr)
            print(f"Node {node.name}: Added kernel_shape={kernel_size}")
            count += 1
        else:
            print(f"Node {node.name}: Weights {weight_name} not found in initializers.")

print(f"Patched {count} nodes.")
print(f"Saving to {output_path}...")
onnx.save(model, output_path)
print("Done.")
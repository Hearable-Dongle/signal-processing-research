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
            # Assume 1D convolution for this model (3 dims: Out, In, K)
            # The kernel size is the last dimension.
            # If it were 2D, it would be last 2. But we know this is ConvTasNet (audio).
            # Safety check:
            if len(dims) == 3:
                kernel_size = [dims[-1]]
            elif len(dims) == 4:
                # Could be 2D conv? e.g. [Out, In, H, W]
                # But wait, checking node_conv1d_1: [128, 512, 1]
                # If we encounter [Out, In, H, W], we probably want [H, W]
                kernel_size = [d for d in dims[2:]]
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
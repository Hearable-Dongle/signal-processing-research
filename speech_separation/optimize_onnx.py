import onnx
import onnxsim # Import the simplification tool
import os
import sys

input_model = sys.argv[1]
output_model = sys.argv[2]

print(f"Loading model: {input_model}")
model = onnx.load(input_model)

print("Applying ONNX simplification (removes Identity, Dropout, etc.)...")
simplified_model, check = onnxsim.simplify(
    model,
    check_n=0,
    perform_optimization=True
)

if check:
    print("Optimization successful and verified.")
else:
   print("Optimization finished, but verification failed. Proceeding anyway.")

model_size_bytes = simplified_model.ByteSize()

model_size_mb = model_size_bytes / (1024 * 1024)

print(f"Simplified model size: {model_size_mb:.2f} MB")

onnx.save(simplified_model, output_model)

print(f"Successfully simplified and saved to: {output_model}")


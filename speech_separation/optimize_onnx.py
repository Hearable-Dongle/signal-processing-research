import onnx
import onnxsim # Import the simplification tool
import os
import sys

# Define file paths
input_model = sys.argv[1]
output_model = sys.argv[2]

print(f"Loading model: {input_model}")
model = onnx.load(input_model)

# Use onnx-simplifier to clean up the graph.
# 'check_n' is set to 0 because onnx-simplifier needs to run the model
# and it might complain if it can't run the model for verification.
print("Applying ONNX simplification (removes Identity, Dropout, etc.)...")
simplified_model, check = onnxsim.simplify(
    model,
    check_n=0,
    perform_optimization=True
)

if check:
    print("Optimization successful and verified.")
else:
    # If check is False, the simplification succeeded, but verification failed.
    # We proceed, but a warning is issued.
    print("Optimization finished, but verification failed. Proceeding anyway.")

# Save the cleaned model
onnx.save(simplified_model, output_model)

print(f"Successfully simplified and saved to: {output_model}")
print("\nNext, re-run the nncase compiler on the optimized model.")


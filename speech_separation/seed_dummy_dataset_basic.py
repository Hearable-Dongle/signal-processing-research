import os
import numpy as np

# --- Configuration ---
# Directory to save the calibration files
OUTPUT_DIR = "dummy_dataset_basic"
# Number of sample files to generate per input
NUM_FILES = 20
# Input tensor shape parameters (from basicNN.py)
# NOTE: Batch size is 1 for calibration datasets
BATCH_SIZE = 1
CHANNELS = 2
FREQ_BINS = 128 # Must be divisible by 8 (2^3)
TIME_FRAMES = 128 # Must be divisible by 8 (2^3)
# The names of the ONNX model's inputs
INPUT_NAMES = ['mixture_ri', 'reference_ri']

def generate_dataset():
    """
    Generates a calibration dataset with random data matching the BasicSiameseUnet input shapes.
    """
    if not os.path.exists(OUTPUT_DIR):
        print(f"Creating base directory: {OUTPUT_DIR}")
        os.makedirs(OUTPUT_DIR)
    else:
        print(f"Base directory '{OUTPUT_DIR}' already exists. Subdirectories may be overwritten.")

    # --- Loop through each model input ---
    for input_name in INPUT_NAMES:
        input_dir = os.path.join(OUTPUT_DIR, input_name)
        if not os.path.exists(input_dir):
            print(f"Creating subdirectory: {input_dir}")
            os.makedirs(input_dir)

        print(f"\nGenerating {NUM_FILES} sample files for input '{input_name}'...")

        # --- Loop to create each sample file for the current input ---
        for i in range(NUM_FILES):
            # Generate random data with a normal distribution
            # Shape: (batch_size, channels, freq_bins, time_frames)
            # Data type is float32, as is common for models.
            random_data = np.random.randn(BATCH_SIZE, CHANNELS, FREQ_BINS, TIME_FRAMES).astype(np.float32)

            # Define the output file path
            file_path = os.path.join(input_dir, f"sample_{i:02d}.npy")

            # Save the numpy array to a .npy file
            np.save(file_path, random_data)

    print("-" * 30)
    print("✅ Dummy dataset generation complete!")
    print(f"Saved {NUM_FILES} files for each of the {len(INPUT_NAMES)} inputs in the '{OUTPUT_DIR}' directory.")
    print("You can now use this directory for calibration during compilation.")
    print("-" * 30)


if __name__ == "__main__":
    generate_dataset()

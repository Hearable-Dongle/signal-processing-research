import os
import numpy as np

# --- Configuration ---
# Directory to save the calibration files
OUTPUT_DIR = "dummy_data"
# Number of sample files to generate
NUM_FILES = 20
# Number of samples per file (to match model input)
NUM_SAMPLES = 4096
# The sample rate isn't needed for .npy, but for context, it's 16kHz
SAMPLE_RATE = 16000

def generate_dataset():
    """
    Generates a calibration dataset with random audio-like data.
    """
    # 1. Create the output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        print(f"Creating directory: {OUTPUT_DIR}")
        os.makedirs(OUTPUT_DIR)
    else:
        print(f"Directory '{OUTPUT_DIR}' already exists. Files may be overwritten.")

    print(f"Generating {NUM_FILES} sample files...")

    # 2. Loop to create each file
    for i in range(NUM_FILES):
        # Generate random data with a normal distribution (mean 0, variance 1)
        # The shape (1, 1, NUM_SAMPLES) matches the model's expected input:
        # (batch_size, num_channels, num_samples)
        # Data type is float32, as specified in the compile script.
        random_data = np.random.randn(1, 1, NUM_SAMPLES).astype(np.float32)

        # Define the output file path
        file_path = os.path.join(OUTPUT_DIR, f"sample_{i:02d}.npy")

        # Save the numpy array to a .npy file
        np.save(file_path, random_data)

    print("-" * 30)
    print("✅ Dataset generation complete!")
    print(f"Saved {NUM_FILES} files in the '{OUTPUT_DIR}' directory.")
    print("You can now run the 'to_kmodel.sh' script.")
    print("-" * 30)


if __name__ == "__main__":
    generate_dataset()


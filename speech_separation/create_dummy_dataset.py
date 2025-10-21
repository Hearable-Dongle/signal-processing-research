import os
import numpy as np

OUTPUT_DIR = "dummy_data"
NUM_FILES = 20
NUM_SAMPLES = 4096
SAMPLE_RATE = 16000

def generate_dataset():
    """
    Generates a calibration dataset with random audio-like data.
    """
    if not os.path.exists(OUTPUT_DIR):
        print(f"Creating directory: {OUTPUT_DIR}")
        os.makedirs(OUTPUT_DIR)
    else:
        print(f"Directory '{OUTPUT_DIR}' already exists. Files may be overwritten.")

    print(f"Generating {NUM_FILES} sample files...")

    for i in range(NUM_FILES):
       # (batch_size, num_channels, num_samples)
        random_data = np.random.randn(1, 1, NUM_SAMPLES).astype(np.float32)

        file_path = os.path.join(OUTPUT_DIR, f"sample_{i:02d}.npy")
        np.save(file_path, random_data)

    print("-" * 30)
    print("Dataset generation complete!")
    print(f"Saved {NUM_FILES} files in the '{OUTPUT_DIR}' directory.")
    print("You can now run the 'to_kmodel.sh' script.")
    print("-" * 30)


if __name__ == "__main__":
    generate_dataset()


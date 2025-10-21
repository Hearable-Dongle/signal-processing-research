import os
import numpy as np

OUTPUT_DIR = "dummy_dataset"
NUM_FILES = 20
BATCH_SIZE = 1
CHANNELS = 2
FREQ_BINS = 128
TIME_FRAMES = 128
INPUT_NAMES = ['mixture_ri', 'reference_ri']

def generate_dataset():
    """
    Generates a calibration dataset with random data matching the SiameseUnet input shapes.
    """
    if not os.path.exists(OUTPUT_DIR):
        print(f"Creating base directory: {OUTPUT_DIR}")
        os.makedirs(OUTPUT_DIR)
    else:
        print(f"Base directory '{OUTPUT_DIR}' already exists. Subdirectories may be overwritten.")

    for input_name in INPUT_NAMES:
        input_dir = os.path.join(OUTPUT_DIR, input_name)
        if not os.path.exists(input_dir):
            print(f"Creating subdirectory: {input_dir}")
            os.makedirs(input_dir)

        print(f"\nGenerating {NUM_FILES} sample files for input '{input_name}'...")

        for i in range(NUM_FILES):
           random_data = np.random.randn(BATCH_SIZE, CHANNELS, FREQ_BINS, TIME_FRAMES).astype(np.float32)

            file_path = os.path.join(input_dir, f"sample_{i:02d}.npy")

            np.save(file_path, random_data)

    print("-" * 30)
    print("Dummy dataset generation complete!")
    print(f"Saved {NUM_FILES} files for each of the {len(INPUT_NAMES)} inputs in the '{OUTPUT_DIR}' directory.")
    print("You can now use this directory for calibration during compilation.")
    print("-" * 30)


if __name__ == "__main__":
    generate_dataset()

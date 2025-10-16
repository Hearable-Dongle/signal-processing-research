import argparse
import os
import torch
import soundfile as sf
from asteroid.models import ConvTasNet

def separate_audio(input_file, output_dir):
    """
    Loads a pre-trained Conv-TasNet model to separate audio sources from a given file.
    """
    # 1. Load the pre-trained model from Hugging Face
    # This model was trained on 16kHz audio, so our input must match.
    model_name = "JorisCos/ConvTasNet_Libri2Mix_sepnoisy_16k"
    print(f"Loading pre-trained model: {model_name}...")
    try:
        model = ConvTasNet.from_pretrained(model_name)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure you have an internet connection and the necessary libraries are installed (`pip install asteroid torch`).")
        return

    # Use a GPU if available, otherwise default to CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Model loaded successfully on '{device}'.")

    # 2. Load the input audio file
    print(f"Loading audio from: {input_file}")
    try:
        mixed_audio, sample_rate = sf.read(input_file, dtype='float32')
    except Exception as e:
        print(f"Error reading input file: {e}")
        return

    # --- CRITICAL: Check and enforce the sample rate ---
    # The model expects 16kHz audio. If the input is different, it will not work correctly.
    expected_sr = 16000
    if sample_rate != expected_sr:
        print(f"Warning: Input audio sample rate is {sample_rate}Hz, but the model expects {expected_sr}Hz.")
        print("Results may be suboptimal. Please resample your audio to 16kHz for best performance.")
        # For a production script, you would add resampling logic here, e.g., using `librosa.resample`.

    # Convert NumPy array to PyTorch tensor
    mixed_tensor = torch.tensor(mixed_audio, device=device)

    # If the audio is stereo, convert it to mono by averaging the channels.
    if mixed_tensor.ndim > 1:
        print("Input audio is stereo. Converting to mono by averaging channels.")
        mixed_tensor = mixed_tensor.mean(dim=1)

    # The model expects a batch dimension, so we add one: [num_samples] -> [1, num_samples]
    mixed_tensor = mixed_tensor.unsqueeze(0)

    # 3. Perform the separation
    print("Performing source separation...")
    with torch.no_grad():
        # The `separate` method returns a tensor of separated sources
        separated_tensors = model.separate(mixed_tensor)
    print("Separation complete.")

    # 4. Save the separated audio files
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # The output shape is (batch, n_sources, n_samples). We remove the batch dimension.
    separated_tensors = separated_tensors.squeeze(0) # -> [n_sources, n_samples]

    # Move the tensor to the CPU and convert to a NumPy array for saving
    separated_numpy = separated_tensors.cpu().numpy()

    num_sources = separated_numpy.shape[0]
    print(f"Found {num_sources} sources. Saving to '{output_dir}'...")

    for i, source_audio in enumerate(separated_numpy):
        output_filename = os.path.join(output_dir, f"separated_source_{i + 1}.wav")
        # Save each source as a new WAV file
        sf.write(output_filename, source_audio, sample_rate)
        print(f"Successfully saved: {output_filename}")


def main():
    """
    Main function to parse command-line arguments and run the separation process.
    """
    parser = argparse.ArgumentParser(
        description="Separate audio sources from a mixed WAV file using a pre-trained Conv-TasNet model."
    )
    parser.add_argument(
        '--input-file',
        type=str,
        required=True,
       help="Path to the mixed input WAV file (e.g., 'mixed.wav')."
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help="Directory to save the separated output WAV files (e.g., './separated_output')."
    )

    args = parser.parse_args()

    separate_audio(args.input_file, args.output_dir)


if __name__ == "__main__":
    main()

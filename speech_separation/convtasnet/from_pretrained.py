import argparse
import os
import torch
import soundfile as sf
from asteroid.models import ConvTasNet

EXPECTED_SAMPLE_RATE = 16000

def separate_audio(input_file, output_dir):
    """
    Loads a pre-trained Conv-TasNet model to separate audio sources from a given file.
    """
    model_name = "JorisCos/ConvTasNet_Libri2Mix_sepnoisy_16k"
    print(f"Loading pre-trained model: {model_name}...")
    try:
        model = ConvTasNet.from_pretrained(model_name)
    except Exception as e:
        print(f"Error loading model: {e}")
        return


    print(f"Loading audio from: {input_file}")
    try:
        mixed_audio, sample_rate = sf.read(input_file, dtype='float32')
    except Exception as e:
        print(f"Error reading input file: {e}")
        return

    if sample_rate != EXPECTED_SAMPLE_RATE:
        print(f"Warning: Input audio sample rate is {sample_rate}Hz, but the model expects {expected_sr}Hz.")
        print("Results may be suboptimal. Please resample your audio to 16kHz for best performance.")

    mixed_tensor = torch.tensor(mixed_audio, device=device)

    if mixed_tensor.ndim > 1:
        print("Input audio is stereo. Converting to mono by averaging channels.")
        mixed_tensor = mixed_tensor.mean(dim=1)

    mixed_tensor = mixed_tensor.unsqueeze(0)

    print("Performing source separation...")
    with torch.no_grad():
        separated_tensors = model.separate(mixed_tensor)
    print("Separation complete.")

    os.makedirs(output_dir, exist_ok=True)

    separated_tensors = separated_tensors.squeeze(0) # -> [n_sources, n_samples]

    separated_numpy = separated_tensors.cpu().numpy()

    num_sources = separated_numpy.shape[0]
    print(f"Found {num_sources} sources. Saving to '{output_dir}'...")

    for i, source_audio in enumerate(separated_numpy):
        output_filename = os.path.join(output_dir, f"separated_source_{i + 1}.wav")
        sf.write(output_filename, source_audio, sample_rate)
        print(f"Successfully saved: {output_filename}")


def main():
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

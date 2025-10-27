import argparse
from pydub import AudioSegment
import os

def convert_opus_to_wav(input_file, output_file):
    """Converts an audio file from .opus to .wav format."""
    if not os.path.exists(input_file):
        print(f"Error: Input file not found at '{input_file}'")
        return

    audio = AudioSegment.from_file(input_file, format="ogg")
    audio.export(output_file, format="wav", parameters=["-ac", "1", "-ar", "16000"])
    print("Conversion successful!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert .opus audio files to .wav format.")
    parser.add_argument("input_file", help="Path to the input .opus file.")
    parser.add_argument("-o", "--output_file", 
                        help="Path for the output .wav file. (Optional: defaults to the same name with a .wav extension)")

    args = parser.parse_args()

    if not args.output_file:
        base_name = os.path.splitext(args.input_file)[0]
        args.output_file = base_name + ".wav"

    convert_opus_to_wav(args.input_file, args.output_file)

import sys
import torch
import torchaudio
import argparse
from pathlib import Path

def resample_audio_file(input_path: str, output_path: str, new_sample_rate: int):
    """
    Loads a WAV file, resamples it to a new sample rate, and saves it to a new file.

    Args:
        input_path (str): Path to the input WAV file.
        output_path (str): Path to save the resampled output WAV file.
        new_sample_rate (int): The target sample rate to resample to.
    Returns:
        None
    """
    print(f"Loading input file: {input_path}")
    try:
        waveform, sr = torchaudio.load(input_path)
    except Exception as e:
        print(f"Error loading audio file: {e}", file=sys.stderr)
        return

    if sr == new_sample_rate:
        print(f"Input sample rate ({sr} Hz) is already the same as the target rate. Copying file.")
        resampled_waveform = waveform
    else:
        print(f"Resampling from {sr} Hz to {new_sample_rate} Hz.")
        resampler = torchaudio.transforms.Resample(
            orig_freq=sr,
            new_freq=new_sample_rate,
        )
        resampled_waveform = resampler(waveform)

    try:
        torchaudio.save(
            output_path, 
            resampled_waveform, 
            new_sample_rate,
            format="wav"
        )
        print(f"Success: Resampled audio saved to: {output_path}")
    except Exception as e:
        print(f"Error saving audio file: {e}", file=sys.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Resample an input WAV file to a new sample rate."
    )

    parser.add_argument(
        '--input-path',
        type=str,
        required=True,
        help='The path to the input WAV file.'
    )

    parser.add_argument(
        '--output-path',
        type=str,
        required=True,
        help='The path to the output WAV file where the resampled audio will be written.'
    )

    parser.add_argument(
        '--sample-rate',
        type=int,
        required=True,
        help='The target sample rate (e.g., 16000, 44100).'
    )
    
    args = parser.parse_args()

    resample_audio_file(
        input_path=args.input_path, 
        output_path=args.output_path,
        new_sample_rate=args.sample_rate
    )

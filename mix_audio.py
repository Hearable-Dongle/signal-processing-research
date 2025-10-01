
import sys
import torch
import torchaudio
import argparse
from pathlib import Path

DEFAULT_SAMPLE_RATE = 16000

def mix_audio_files(input_paths: list[str], output_path: str):
    """
    Loads multiple WAV files, ensures they have the same sample rate,
    concatenates their waveforms, and saves the result to a new file.

    Args:
        input_paths (list[str]): List of paths to input WAV files.
        output_path (str): Path to save the concatenated output WAV file.
    Returns:
        None
    """
    if len(input_paths) < 2:
        print("Error: Must provide at least two input files for concatenation.", file=sys.stderr)
        return

    waveforms = []
    sample_rate = None
    max_length = 0

    for i, file_path in enumerate(input_paths):
        print(f"Loading input file {i+1}/{len(input_paths)}: {file_path}")
        waveform, sr = torchaudio.load(file_path)

        # Convert to stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        if not sample_rate:
            # Take first sample rate
            sample_rate = sr
        elif sr != sample_rate:
            print(f"Warning: Resampling '{Path(file_path).name}' from {sr} Hz to {sample_rate} Hz.")
            resampler = torchaudio.transforms.Resample(
                orig_freq=sr,
                new_freq=sample_rate,
            )
            waveform = resampler(waveform)

        current_length = waveform.shape[1]
        max_length = max(current_length, max_length)

        waveforms.append(waveform)

    if not waveforms:
        print("No valid waveforms loaded. Exiting.")
        return

    padded_waveforms = []
    for waveform in waveforms:
        current_length = waveform.shape[1]
        padding_needed = max_length - current_length
        if padding_needed > 0:
            # (num_channels, num_samples) -> pad on the sample dimension (dim=1)
            padding = torch.zeros((waveform.shape[0], padding_needed))
            padded_waveform = torch.cat([waveform, padding], dim=1)
        else:
            padded_waveform = waveform
        
        padded_waveforms.append(padded_waveform)

    mixed_waveform = torch.sum(torch.stack(padded_waveforms), dim=0)

    torchaudio.save(
        output_path, 
        mixed_waveform, 
        sample_rate,
        format="wav"
    )
    print(f"Success: Mixed audio saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Concatenate multiple input WAV files into a single output WAV file."
    )

    parser.add_argument(
        '--input-paths',
        nargs='+',  
        type=str,
        required=True,
        help='A space-separated list of input waveform paths (e.g., file1.wav file2.wav).'
    )

    parser.add_argument(
        '--output-path',
        type=str,
        required=True,
        help='The path to the output WAV file where the concatenated audio will be written.'
    )
    
    args = parser.parse_args()

    mix_audio_files(
        input_paths=args.input_paths, 
        output_path=args.output_path
    )
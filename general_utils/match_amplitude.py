import sys
import torch
import torchaudio
import argparse
from pathlib import Path

def match_amplitude(input_path: str, reference_path: str, output_path: str):
    """
    Loads an input audio file and a reference audio file, and scales the
    amplitude of the input to match the RMS amplitude of the reference.

    Args:
        input_path (str): Path to the input WAV file to modify.
        reference_path (str): Path to the reference WAV file for amplitude matching.
        output_path (str): Path to save the modified output WAV file.
    """
    print(f"Loading input file: {input_path}")
    input_waveform, sr_in = torchaudio.load(input_path)

    print(f"Loading reference file: {reference_path}")
    ref_waveform, sr_ref = torchaudio.load(reference_path)

    # Convert to mono
    if input_waveform.shape[0] > 1:
        input_waveform = torch.mean(input_waveform, dim=0, keepdim=True)
    if ref_waveform.shape[0] > 1:
        ref_waveform = torch.mean(ref_waveform, dim=0, keepdim=True)

    # Resample reference to match input's sample rate if they differ
    if sr_in != sr_ref:
        print(f"Warning: Resampling reference audio from {sr_ref} Hz to {sr_in} Hz.")
        resampler = torchaudio.transforms.Resample(orig_freq=sr_ref, new_freq=sr_in)
        ref_waveform = resampler(ref_waveform)

    # Calculate RMS for both waveforms
    rms_in = torch.sqrt(torch.mean(input_waveform.pow(2)))
    rms_ref = torch.sqrt(torch.mean(ref_waveform.pow(2)))

    if rms_in == 0:
        print("Input audio is silent. Cannot scale amplitude. Output will also be silent.")
        scaled_waveform = input_waveform
    else:
        scaling_factor = rms_ref / rms_in
        print(f"Scaling input audio by a factor of {scaling_factor:.4f}")
        scaled_waveform = input_waveform * scaling_factor

    # Prevent clipping by normalizing if peak exceeds 1.0
    max_amp = torch.max(torch.abs(scaled_waveform))
    if max_amp > 1.0:
        scaled_waveform /= max_amp
        print("Warning: Clipping detected after scaling. The output audio has been normalized to prevent distortion.")

    torchaudio.save(
        output_path,
        scaled_waveform,
        sr_in,
        format="wav"
    )
    print(f"Success: Amplitude-matched audio saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Matches the amplitude of an input WAV file to a reference WAV file."
    )

    parser.add_argument(
        '--input-path',
        type=str,
        required=True,
        help='The path to the input WAV file to modify.'
    )

    parser.add_argument(
        '--reference-path',
        type=str,
        required=True,
        help='The path to the reference WAV file for amplitude matching.'
    )

    parser.add_argument(
        '--output-path',
        type=str,
        required=True,
        help='The path for the output WAV file.'
    )

    args = parser.parse_args()

    match_amplitude(
        input_path=args.input_path,
        reference_path=args.reference_path,
        output_path=args.output_path
    )

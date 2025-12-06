import sys
import torch
import torchaudio
import argparse
from pathlib import Path

def concatenate_audio_files(input_paths: list[str], output_path: str):
    """
    Loads multiple WAV files, normalizes them to the same average volume,
    ensures they have the same sample rate, concatenates them sequentially, 
    and saves the result to a new file.

    Args:
        input_paths (list[str]): List of paths to input WAV files in order.
        output_path (str): Path to save the concatenated output WAV file.
    Returns:
        None
    """
    if len(input_paths) < 2:
        print("Error: Must provide at least two input files for concatenation.", file=sys.stderr)
        return

    waveforms = []
    sample_rate = None

    # --- 1. Load and Resample ---
    for i, file_path in enumerate(input_paths):
        print(f"Loading input file {i+1}/{len(input_paths)}: {file_path}")
        try:
            waveform, sr = torchaudio.load(file_path)
        except Exception as e:
            print(f"Error loading {file_path}: {e}", file=sys.stderr)
            return

        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        if not sample_rate:
            # Set target sample rate to the first file's rate
            sample_rate = sr
        elif sr != sample_rate:
            print(f"Warning: Resampling '{Path(file_path).name}' from {sr} Hz to {sample_rate} Hz.")
            resampler = torchaudio.transforms.Resample(
                orig_freq=sr,
                new_freq=sample_rate,
            )
            waveform = resampler(waveform)

        waveforms.append(waveform)

    if not waveforms:
        print("No valid waveforms loaded. Exiting.")
        return

    # Normalize volume to prevent jarring volume jumps between different files
    rms_values = [torch.sqrt(torch.mean(wf.pow(2))) for wf in waveforms]
    max_rms = max(rms for rms in rms_values if rms > 0) if any(rms > 0 for rms in rms_values) else 0

    normalized_waveforms = []
    if max_rms > 0:
        for i, waveform in enumerate(waveforms):
            if rms_values[i] > 0:
                scaling_factor = max_rms / rms_values[i]
                scaled_waveform = waveform * scaling_factor
                
                # Prevent clipping if normalization pushes peak > 1.0
                max_amp = torch.max(torch.abs(scaled_waveform))
                if max_amp > 1.0:
                    scaled_waveform /= max_amp
                
                normalized_waveforms.append(scaled_waveform)
            else:
                normalized_waveforms.append(waveform) 
    else:
        normalized_waveforms = waveforms

    if not normalized_waveforms:
        print("No waveforms to process. Exiting.")
        return

    print("Concatenating waveforms...")
    concatenated_waveform = torch.cat(normalized_waveforms, dim=1)

    torchaudio.save(
        output_path, 
        concatenated_waveform, 
        sample_rate,
        format="wav"
    )
    print(f"Success: Concatenated audio saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Concatenate multiple input WAV files into a single output WAV file."
    )

    parser.add_argument(
        '--input-paths',
        nargs='+',  
        type=str,
        required=True,
        help='A space-separated list of input waveform paths (e.g., intro.wav content.wav outro.wav).'
    )

    parser.add_argument(
        '--output-path',
        type=str,
        required=True,
        help='The path to the output WAV file where the concatenated audio will be written.'
    )
    
    args = parser.parse_args()

    concatenate_audio_files(
        input_paths=args.input_paths, 
        output_path=args.output_path
    )
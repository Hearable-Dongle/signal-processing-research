
import sys
import torch
import torchaudio
import argparse
from pathlib import Path

DEFAULT_SAMPLE_RATE = 16000

def mix_audio_files(input_paths: list[str], output_path: str, cut_to_shortest: bool = False):
    """
    Loads multiple WAV files, normalizes them to the same average volume,
    ensures they have the same sample rate, mixes them, and saves the
    result to a new file.

    Args:
        input_paths (list[str]): List of paths to input WAV files.
        output_path (str): Path to save the mixed output WAV file.
        cut_to_shortest (bool): If True, truncate all samples to the shortest sample length.
                                Otherwise, pad all samples to the longest sample length.
    Returns:
        None
    """
    if len(input_paths) < 2:
        print("Error: Must provide at least two input files for mixing.", file=sys.stderr)
        return

    waveforms = []
    sample_rate = None

    for i, file_path in enumerate(input_paths):
        print(f"Loading input file {i+1}/{len(input_paths)}: {file_path}")
        waveform, sr = torchaudio.load(file_path)

        # Convert to mono
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

        waveforms.append(waveform)

    if not waveforms:
        print("No valid waveforms loaded. Exiting.")
        return

    # Normalize volumes based on RMS to make them have similar loudness
    rms_values = [torch.sqrt(torch.mean(wf.pow(2))) for wf in waveforms]
    max_rms = max(rms for rms in rms_values if rms > 0) if any(rms > 0 for rms in rms_values) else 0

    normalized_waveforms = []
    if max_rms > 0:
        for i, waveform in enumerate(waveforms):
            if rms_values[i] > 0:
                scaling_factor = max_rms / rms_values[i]
                scaled_waveform = waveform * scaling_factor
                
                # Prevent clipping by normalizing if peak exceeds 1.0
                max_amp = torch.max(torch.abs(scaled_waveform))
                if max_amp > 1.0:
                    scaled_waveform /= max_amp
                
                normalized_waveforms.append(scaled_waveform)
            else:
                normalized_waveforms.append(waveform)  # Append silent audio as is
    else:
        normalized_waveforms = waveforms

    if not normalized_waveforms:
        print("No waveforms to process. Exiting.")
        return

    # Pad to longest or cut to shortest
    if cut_to_shortest:
        min_length = min(wf.shape[1] for wf in normalized_waveforms)
        processed_waveforms = [wf[:, :min_length] for wf in normalized_waveforms]
        print(f"Cutting all samples to shortest length: {min_length} samples.")
    else:
        max_length = max(wf.shape[1] for wf in normalized_waveforms)
        processed_waveforms = []
        for waveform in normalized_waveforms:
            padding_needed = max_length - waveform.shape[1]
            if padding_needed > 0:
                padding = torch.zeros((waveform.shape[0], padding_needed))
                padded_waveform = torch.cat([waveform, padding], dim=1)
            else:
                padded_waveform = waveform
            processed_waveforms.append(padded_waveform)
        print(f"Padding all samples to longest length: {max_length} samples.")


    # Mix by summing all waveforms
    mixed_waveform = torch.sum(torch.stack(processed_waveforms), dim=0)

    # Normalize the final mixed waveform to prevent clipping, with a little headroom
    final_peak = torch.max(torch.abs(mixed_waveform))
    if final_peak > 0.98:
        mixed_waveform *= (0.98 / final_peak)

    torchaudio.save(
        output_path, 
        mixed_waveform, 
        sample_rate,
        format="wav"
    )
    print(f"Success: Mixed audio saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Mix multiple input WAV files into a single output WAV file."
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
        help='The path to the output WAV file where the mixed audio will be written.'
    )

    parser.add_argument(
        '--cut-to-shortest',
        action='store_true',
        help='If set, truncate all audio files to the length of the shortest one instead of padding.'
    )
    
    args = parser.parse_args()

    mix_audio_files(
        input_paths=args.input_paths, 
        output_path=args.output_path,
        cut_to_shortest=args.cut_to_shortest
    )


import argparse
import glob
import shutil
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio
import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score
from general_utils.constants import LIBRIMIX_PATH

from own_voice_suppression.voice_detection import main as run_suppression

def plot_vad_results(confidence_logs, y_true, total_len_samples, sr, output_path, threshold):
    import matplotlib.pyplot as plt
    
    if not confidence_logs:
        return

    log_times = [log['time'] for log in confidence_logs]
    raw_scores = [log['score'] for log in confidence_logs]
    smoothed_scores = [log['smoothed_score'] for log in confidence_logs]
    
    fig, ax = plt.subplots(figsize=(15, 5))

    ax.plot(log_times, raw_scores, label='Raw Confidence', color='orange', alpha=0.7, linewidth=1)
    ax.plot(log_times, smoothed_scores, label='Smoothed Confidence', color='dodgerblue', linewidth=2)

    ax.axhline(y=threshold, color='r', linestyle='--', label=f'Current Threshold ({threshold:.2f})')

    frame_len_samples = sr * 20 // 1000 # 20ms frames, must match get_energy_vad
    num_frames = total_len_samples // frame_len_samples
    y_true_padded = np.zeros(num_frames)
    y_true_padded[:len(y_true)] = y_true
    
    time_axis_true = np.arange(num_frames) * (frame_len_samples / sr)
    
    ax.fill_between(time_axis_true, y_true_padded, 0, color='green', alpha=0.3, step='pre', label='Ground Truth (Target Active)')

    ax.set_title('Speaker Detection Confidence and Ground Truth')
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Confidence / Activity')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def compute_si_sdr(estimate: torch.Tensor, reference: torch.Tensor, epsilon=1e-8):
    """
    Computes Scale-Invariant Signal-to-Distortion Ratio (SI-SDR).
    Higher is better.
    """
    if estimate.shape[-1] != reference.shape[-1]:
        min_len = min(estimate.shape[-1], reference.shape[-1])
        estimate = estimate[..., :min_len]
        reference = reference[..., :min_len]

    ref_energy = torch.sum(reference ** 2, dim=-1, keepdim=True) + epsilon
    optimal_scaling = torch.sum(reference * estimate, dim=-1, keepdim=True) / ref_energy
    
    # Projection
    projection = optimal_scaling * reference
    
    # Noise (error)
    noise = estimate - projection
    
    # SI-SDR = 10 * log10( ||projection||^2 / ||noise||^2 )
    ratio = torch.sum(projection ** 2, dim=-1) / (torch.sum(noise ** 2, dim=-1) + epsilon)
    si_sdr = 10 * torch.log10(ratio + epsilon)
    
    return si_sdr.item()

def compute_suppression_db(mixture: torch.Tensor, output: torch.Tensor, target_s1: torch.Tensor):
    """
    Approximates how much of S1 (Owner) was removed in dB.
    """
    min_len = min(mixture.shape[-1], output.shape[-1], target_s1.shape[-1])
    mix = mixture[..., :min_len]
    out = output[..., :min_len]
    s1 = target_s1[..., :min_len]
    
    original_owner_energy = torch.sum(s1 ** 2)
    
    # Residual owner energy in output
    # residual = (out . s1) / ||s1||^2 * s1
    alpha = torch.sum(out * s1) / (torch.sum(s1 ** 2) + 1e-8)
    residual_component = alpha * s1
    
    residual_energy = torch.sum(residual_component ** 2)
    
    suppression = 10 * torch.log10(original_owner_energy / (residual_energy + 1e-8))
    return suppression.item()

def get_energy_vad(waveform, sr, threshold_db=-40, window_ms=20):
    """
    Creates a binary mask (1=Active, 0=Silent) based on energy.
    """
    frame_length = int(sr * window_ms / 1000)
    # Reshape to frames
    num_frames = waveform.shape[1] // frame_length
    trimmed_wave = waveform[:, :num_frames * frame_length]
    frames = trimmed_wave.view(1, num_frames, frame_length)
    
    # Calculate energy in dB per frame
    frame_energy = torch.mean(frames ** 2, dim=2)
    frame_db = 10 * torch.log10(frame_energy + 1e-9)
    
    vad_mask = (frame_db > threshold_db).int().squeeze().numpy()
    return vad_mask

def evaluate_detection_sdr(librimix_root, num_samples=10, model_type="wavlm-large", save_outputs=False, speaker_detection_threshold=0.65, smoothing_window=10):
    librimix_path = Path(librimix_root)

    search_pattern = str(librimix_path / "**" / "s1" / "*.wav")
    s1_files = glob.glob(search_pattern, recursive=True)

    if not s1_files:
        print(f"No files found in {librimix_path}. Check your path structure.")
        return 0

    s1_files = sorted(s1_files)[:num_samples]
    results = []

    print(f"\n--- Starting SDR Evaluation on {num_samples} samples ---")
    print(f"Classifier: {model_type}\n")

    base_output_dir = Path("own_voice_suppression/outputs/validation")
    if save_outputs:
        perm_out_dir = base_output_dir / f"{model_type}-{num_samples}"
        perm_out_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving outputs to {perm_out_dir}")

    else:
        perm_out_dir = None

    temp_out_dir = Path("temp_eval_outputs")
    temp_out_dir.mkdir(exist_ok=True)

    for s1_path in tqdm(s1_files, desc="Evaluating SDR"):
        s1_path = Path(s1_path)
        mix_path = Path(str(s1_path).replace("/s1/", "/mix_clean/"))
        s2_path = Path(str(s1_path).replace("/s1/", "/s2/"))

        if not mix_path.exists() or not s2_path.exists():
            print(f"Skipping {s1_path.name}: distinct mix/s2 files not found.")
            continue

        run_suppression(
            enrolment_path=s1_path,
            mixed_path=mix_path,
            output_directory=temp_out_dir,
            classifier_type=model_type,
            smoothing_window=smoothing_window,
            speaker_detection_threshold=speaker_detection_threshold
        )

        suppressed_path = temp_out_dir / "suppressed.wav"

        if not suppressed_path.exists():
            print("Error: Output file not created.")
            continue

        est_audio, _ = torchaudio.load(suppressed_path)
        ref_audio, _ = torchaudio.load(s2_path)
        mix_audio, _ = torchaudio.load(mix_path)
        s1_audio, _ = torchaudio.load(s1_path)

        si_sdr = compute_si_sdr(est_audio, ref_audio)
        suppression_db = compute_suppression_db(mix_audio, est_audio, s1_audio)

        input_si_sdr = compute_si_sdr(mix_audio, ref_audio)

        results.append({
            "file": s1_path.name,
            "input_si_sdr": input_si_sdr,
            "output_si_sdr": si_sdr,
            "improvement": si_sdr - input_si_sdr,
            "suppression_db": suppression_db
        })

        if save_outputs and perm_out_dir:
            sample_out_dir = perm_out_dir / s1_path.stem
            sample_out_dir.mkdir(exist_ok=True)
            shutil.copy(suppressed_path, sample_out_dir / "suppressed.wav")
            shutil.copy(mix_path, sample_out_dir / "original_mix.wav")
            shutil.copy(s2_path, sample_out_dir / "ground_truth_background.wav")
            shutil.copy(s1_path, sample_out_dir / "original_s1.wav")

    shutil.rmtree(temp_out_dir)

    if not results:
        print("No results were generated.")
        return 0

    avg_sdr_imp = np.mean([r['improvement'] for r in results])
    avg_supp = np.mean([r['suppression_db'] for r in results])

    print("\n" + "="*40)
    print("       SDR EVALUATION RESULTS       ")
    print("="*40)
    print(f"{'Filename':<20} | {'SI-SDR Imp (dB)':<15} | {'Suppression (dB)':<15}")
    print("-" * 56)

    for r in results:
        print(f"{r['file'][:18]:<20} | {r['improvement']:<15.2f} | {r['suppression_db']:<15.2f}")

    print("-" * 56)
    print(f"AVERAGE IMPROVEMENT: {avg_sdr_imp:.2f} dB")
    print(f"AVERAGE SUPPRESSION: {avg_supp:.2f} dB")
    print("="*40)
    
    return avg_sdr_imp


def evaluate_detection_vad(librimix_root, num_samples=10, model_type="wavlm-large", save_outputs=False, speaker_detection_threshold=0.65, smoothing_window=10, vad_threshold=-40.0):
    
    def _trim_or_pad(audio, length_samples):
        if audio.shape[1] > length_samples:
            return audio[:, :length_samples]
        elif audio.shape[1] < length_samples:
            padding = length_samples - audio.shape[1]
            return F.pad(audio, (0, padding))
        return audio

    librimix_path = Path(librimix_root)
    s1_files = glob.glob(str(librimix_path / "**" / "s1" / "*.wav"), recursive=True)
    s2_files = glob.glob(str(librimix_path / "**" / "s2" / "*.wav"), recursive=True)
    
    if not s1_files or not s2_files:
        print(f"No files found in {librimix_path}. Check your path structure.")
        return 0
        
    s1_files = sorted(s1_files)[:num_samples]
    
    metrics = {"precision": [], "recall": [], "f1": []}
    
    base_output_dir = Path("own_voice_suppression/outputs/validation")
    if save_outputs:
        perm_out_dir = base_output_dir / f"{model_type}-{num_samples}-surround"
        perm_out_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving outputs to {perm_out_dir}")
    else:
        perm_out_dir = None

    temp_out_dir = Path("temp_eval_outputs")
    temp_out_dir.mkdir(exist_ok=True)

    print(f"\n--- Evaluating Detection (VAD) on {num_samples} constructed samples ---")
    print(f"Classifier: {model_type}\n")

    for s1_path in tqdm(s1_files, desc="Evaluating VAD"):
        s1_path = Path(s1_path)
        
        # --- Construct new audio mix with overlaps ---
        s1_audio, sr = torchaudio.load(s1_path)
        
        # Find two other random speakers for padding and ensure consistent sample rate
        other_speaker_paths = np.random.choice([p for p in s2_files if p != str(s1_path).replace("/s1/", "/s2/")], 2, replace=False)
        
        other_1_audio, other_1_sr = torchaudio.load(other_speaker_paths[0])
        if other_1_sr != sr:
            other_1_audio = torchaudio.functional.resample(other_1_audio, other_1_sr, sr)

        other_2_audio, other_2_sr = torchaudio.load(other_speaker_paths[1])
        if other_2_sr != sr:
            other_2_audio = torchaudio.functional.resample(other_2_audio, other_2_sr, sr)

        # Define segment and overlap lengths
        pad_len_sec = 3
        target_len_sec = 5  # Increased for more overlap content
        overlap_sec = 1.5
        
        pad_len_samples = pad_len_sec * sr
        target_len_samples = target_len_sec * sr
        overlap_samples = int(overlap_sec * sr)

        s1_segment = _trim_or_pad(s1_audio, target_len_samples)
        pad1_segment = _trim_or_pad(other_1_audio, pad_len_samples)
        pad2_segment = _trim_or_pad(other_2_audio, pad_len_samples)

        # Calculate total length for the overlapping mix
        total_len_samples = pad_len_samples + target_len_samples + pad_len_samples - 2 * overlap_samples
        constructed_mix = torch.zeros(1, total_len_samples)

        # Place segments with overlap by adding them to the mix tensor
        s1_start = pad_len_samples - overlap_samples
        s2_start = s1_start + target_len_samples - overlap_samples

        constructed_mix[:, :pad_len_samples] += pad1_segment
        constructed_mix[:, s1_start : s1_start + target_len_samples] += s1_segment
        constructed_mix[:, s2_start : s2_start + pad_len_samples] += pad2_segment
        
        # Normalize the final mix to prevent clipping
        max_abs = torch.max(torch.abs(constructed_mix))
        if max_abs > 1.0:
            constructed_mix /= max_abs
        
        # --- Create ground truth VAD for the constructed mix ---
        frame_len_samples = sr * 20 // 1000  # 20ms frames for VAD
        total_frames = total_len_samples // frame_len_samples
        y_true = np.zeros(total_frames)

        s1_vad = get_energy_vad(s1_segment, sr, threshold_db=vad_threshold)
        
        s1_vad_start_frame = s1_start // frame_len_samples
        s1_vad_len_frames = len(s1_vad)

        # Ensure VAD mask is not placed out of bounds
        end_frame = min(s1_vad_start_frame + s1_vad_len_frames, total_frames)
        s1_vad_len_frames = end_frame - s1_vad_start_frame
        
        if s1_vad_len_frames > 0:
            y_true[s1_vad_start_frame : end_frame] = s1_vad[:s1_vad_len_frames]

        # --- Run suppression on the new mix ---
        mix_path = temp_out_dir / "constructed_mix.wav"
        torchaudio.save(mix_path, constructed_mix, sr)

        confidence_logs = run_suppression(
            enrolment_path=s1_path, # Enrol with original clean s1
            mixed_path=mix_path,
            output_directory=temp_out_dir,
            classifier_type=model_type,
            smoothing_window=smoothing_window, 
            speaker_detection_threshold=speaker_detection_threshold
        )
        
        suppressed_path = temp_out_dir / "suppressed.wav"
        if not suppressed_path.exists(): continue
        
        out_audio, _ = torchaudio.load(suppressed_path)
        
        removed_signal = constructed_mix[:, :out_audio.shape[1]] - out_audio
        y_pred = get_energy_vad(removed_signal, sr, threshold_db=-50)
        
        min_len = min(len(y_true), len(y_pred))
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]
        
        p = precision_score(y_true, y_pred, zero_division=0)
        r = recall_score(y_true, y_pred, zero_division=0)
        f = f1_score(y_true, y_pred, zero_division=0)
        
        metrics["precision"].append(p)
        metrics["recall"].append(r)
        metrics["f1"].append(f)

        if save_outputs and perm_out_dir:
            sample_out_dir = perm_out_dir / s1_path.stem
            sample_out_dir.mkdir(exist_ok=True)
            
            # Save main output files
            shutil.copy(suppressed_path, sample_out_dir / "suppressed.wav")
            shutil.copy(mix_path, sample_out_dir / "constructed_mix.wav")
            shutil.copy(s1_path, sample_out_dir / "original_enrolment.wav")
            
            # Create a subfolder for and save the source clips
            source_clips_dir = sample_out_dir / "source_clips"
            source_clips_dir.mkdir(exist_ok=True)
            torchaudio.save(source_clips_dir / "01_target_speaker.wav", s1_segment, sr)
            torchaudio.save(source_clips_dir / "02_padding_speaker_1.wav", pad1_segment, sr)
            torchaudio.save(source_clips_dir / "03_padding_speaker_2.wav", pad2_segment, sr)

            plot_vad_results(
                confidence_logs=confidence_logs,
                y_true=y_true,
                total_len_samples=total_len_samples,
                sr=sr,
                output_path=sample_out_dir / "detection_confidence.png",
                threshold=speaker_detection_threshold
            )

    shutil.rmtree(temp_out_dir)

    if not metrics["f1"]:
        print("No metrics were generated.")
        return 0

    avg_f1 = np.mean(metrics['f1'])
    print("\n" + "="*40)
    print("      DETECTION PERFORMANCE       ")
    print("="*40)
    print(f"RECALL (Did we catch the user?): {np.mean(metrics['recall'])*100:.1f}%")
    print(f"PRECISION (Did we avoid false cuts?): {np.mean(metrics['precision'])*100:.1f}%")
    print(f"F1 SCORE: {avg_f1*100:.1f}%")
    print("="*40)
    return avg_f1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate voice suppression system.")
    parser.add_argument("--validation-mode", type=str, choices=['vad', 'sdr'], default='vad', 
                        help="Validation mode: 'vad' for voice activity detection (default) or 'sdr' for signal-based metrics.")
    parser.add_argument("--librimix-root", type=Path, default=LIBRIMIX_PATH, help="Path to LibriMix root directory")
    parser.add_argument("--samples", type=int, default=10, help="Number of files to evaluate")
    parser.add_argument("--model-type", type=str, default="wavlm-large", help="Which classifier model to use")
    parser.add_argument("--save-outputs", action='store_true', help="Save output audio files for inspection")
    parser.add_argument("--vad-threshold", type=float, default=-40.0, help="Threshold in dB for Voice Activity Detection (VAD).")
    parser.add_argument("--speaker-detection-threshold", type=float, default=0.65, help="Confidence threshold for speaker detection (0.0 to 1.0).")
    parser.add_argument("--smoothing-window", type=int, default=10, help="Number of frames for score smoothing.")

    args = parser.parse_args()

    if args.validation_mode == 'sdr':
        evaluate_detection_sdr(
            args.librimix_root, 
            args.samples, 
            args.model_type, 
            args.save_outputs, 
            speaker_detection_threshold=args.speaker_detection_threshold,
            smoothing_window=args.smoothing_window
        )
    else:
        evaluate_detection_vad(
            args.librimix_root, 
            args.samples, 
            args.model_type, 
            args.save_outputs, 
            speaker_detection_threshold=args.speaker_detection_threshold, 
            vad_threshold=args.vad_threshold,
            smoothing_window=args.smoothing_window
        )

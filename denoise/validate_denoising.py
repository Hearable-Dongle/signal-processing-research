import argparse
import glob
import shutil
from pathlib import Path
import sys
import torch
import torchaudio
import numpy as np
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))

import time

from denoise.denoise import load_enhancer, MODEL_OPTIONS
from own_voice_suppression.audio_utils import prep_audio
from own_voice_suppression.validate_voice_detection import compute_si_sdr, LIBRIMIX_PATH


def denoise_long_audio(enhancer, noisy_wav, working_sr):
    """
    Denoises a long audio file using a sliding window approach and measures latency.
    """
    window_sec = 2.0  # Window size for processing
    stride_sec = 0.5  # Overlap between windows
    
    window_samples = int(window_sec * working_sr)
    stride_samples = int(stride_sec * working_sr)
    num_samples = noisy_wav.shape[1]
    
    output_buffer = torch.zeros_like(noisy_wav)
    total_inference_time = 0
    num_chunks = 0
    
    device = noisy_wav.device
    
    current_start = 0
    while current_start + window_samples <= num_samples:
        chunk = noisy_wav[:, current_start : current_start + window_samples]
        
        start_time = time.monotonic()
        denoised_chunk = enhancer.process(chunk)
        end_time = time.monotonic()
        
        total_inference_time += (end_time - start_time)
        num_chunks += 1
        
        # Overlap-add with cross-fade
        if current_start == 0:
            output_buffer[:, 0:window_samples] = denoised_chunk
        else:
            fade_len = window_samples - stride_samples
            fade_in = torch.linspace(0, 1, fade_len).to(device)
            fade_out = torch.linspace(1, 0, fade_len).to(device)
            
            output_buffer[:, current_start : current_start + fade_len] *= fade_out
            output_buffer[:, current_start : current_start + fade_len] += denoised_chunk[:, :fade_len] * fade_in
            output_buffer[:, current_start + fade_len : current_start + window_samples] = denoised_chunk[:, fade_len:]

        current_start += stride_samples
        
    avg_latency = (total_inference_time / num_chunks) if num_chunks > 0 else 0
    
    return output_buffer, avg_latency


def evaluate_denoising(librimix_root, num_samples=10, model_type="convtasnet", save_outputs=False):
    """
    Evaluates a denoising model on the LibriMix dataset.
    Treats s2 as the target speech and s1 as the noise.
    Computes SI-SDR improvement.
    """
    librimix_path = Path(librimix_root)
    
    search_pattern = str(librimix_path / "**" / "s2" / "*.wav")
    s2_files = glob.glob(search_pattern, recursive=True)
    
    if not s2_files:
        print(f"No files found in {librimix_path}. Check your path structure.")
        return

    s2_files = sorted(s2_files)[:num_samples]
    
    results = []
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")
    
    print(f"\n--- Starting Denoising Evaluation (SI-SDR) on {num_samples} samples ---")
    print(f"Model: {model_type}\n")

    # Load model once
    enhancer = load_enhancer(model_type, device)
    working_sr = enhancer.NATIVE_SR
    print(f"Working Sample Rate: {working_sr} Hz")

    base_output_dir = Path("denoise/outputs/validation_denoising")
    if save_outputs:
        perm_out_dir = base_output_dir / f"{model_type}-{num_samples}"
        perm_out_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving outputs to {perm_out_dir}")
    else:
        perm_out_dir = None
    
    temp_out_dir = Path("temp_denoise_outputs")
    temp_out_dir.mkdir(exist_ok=True)

    for s2_path in tqdm(s2_files):
        s2_path = Path(s2_path)
        
        mix_path = Path(str(s2_path).replace("/s2/", "/mix_clean/"))
        s1_path = Path(str(s2_path).replace("/s2/", "/s1/"))
        
        if not mix_path.exists() or not s1_path.exists():
            print(f"Skipping {s2_path.name}: mix/s1 files not found.")
            continue

        noisy_wav, sr = torchaudio.load(mix_path)
        noisy_wav = prep_audio(noisy_wav, sr, working_sr).to(device)

        output_buffer, avg_latency = denoise_long_audio(enhancer, noisy_wav, working_sr)

        denoised_path = temp_out_dir / "denoised.wav"
        torchaudio.save(denoised_path, prep_audio(output_buffer, working_sr, 16000).cpu(), 16000)

        est_speech, _ = torchaudio.load(denoised_path)
        ref_speech, _ = torchaudio.load(s2_path)
        mix_audio, _ = torchaudio.load(mix_path)

        output_si_sdr = compute_si_sdr(est_speech, ref_speech)
        input_si_sdr = compute_si_sdr(mix_audio, ref_speech)
        sdr_improvement = output_si_sdr - input_si_sdr
        
        results.append({
            "file": s2_path.name,
            "input_si_sdr": input_si_sdr,
            "output_si_sdr": output_si_sdr,
            "sdr_improvement": sdr_improvement,
            "latency_ms": avg_latency * 1000
        })
        
        if save_outputs and perm_out_dir:
            sample_out_dir = perm_out_dir / s2_path.stem
            sample_out_dir.mkdir(exist_ok=True)
            shutil.copy(denoised_path, sample_out_dir / "denoised_speech.wav")
            shutil.copy(mix_path, sample_out_dir / "original_mix.wav")
            shutil.copy(s2_path, sample_out_dir / "ground_truth_speech.wav")
            shutil.copy(s1_path, sample_out_dir / "ground_truth_noise.wav")

    shutil.rmtree(temp_out_dir)

    if not results:
        print("No results generated.")
        return
        
    avg_imp = np.mean([r['sdr_improvement'] for r in results])
    avg_output_sdr = np.mean([r['output_si_sdr'] for r in results])
    avg_latency = np.mean([r['latency_ms'] for r in results])

    print("\n" + "="*95)
    print("                 DENOISING RESULTS (SI-SDR & LATENCY)                ")
    print("="*95)
    print(f"{ 'Filename':<25} | {'Input SI-SDR':<12} | {'Output SI-SDR':<13} | {'Improvement (dB)':<15} | {'Latency (ms)':<15}")
    print("-" * 95)
    for r in results:
        print(f"{r['file'][:23]:<25} | {r['input_si_sdr']:<12.2f} | {r['output_si_sdr']:<13.2f} | {r['sdr_improvement']:<15.2f} | {r['latency_ms']:.2f}")
    print("-" * 95)
    print(f"AVG OUTPUT SI-SDR:      {avg_output_sdr:.2f} dB")
    print(f"AVG SI-SDR IMPROVEMENT: {avg_imp:.2f} dB   <-- Key Metric")
    print(f"AVERAGE LATENCY:        {avg_latency:.2f} ms")
    print("="*95)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Denoising models on LibriMix.")
    parser.add_argument("--librimix-root", type=Path, default=LIBRIMIX_PATH, help="Path to LibriMix root directory")
    parser.add_argument("--samples", type=int, default=10, help="Number of files to evaluate")
    parser.add_argument("--model-type", type=str, choices=MODEL_OPTIONS, default="convtasnet", help="Which denoising model to use")
    parser.add_argument("--save-outputs", action='store_true', help="Save output audio files for inspection")
    
    args = parser.parse_args()
    
    evaluate_denoising(args.librimix_root, args.samples, args.model_type, args.save_outputs)

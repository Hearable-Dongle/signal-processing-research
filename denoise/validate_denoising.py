import argparse
import glob
import shutil
from pathlib import Path
import sys
import torch
import torchaudio
import numpy as np
from tqdm import tqdm
import time
import uuid

sys.path.append(str(Path(__file__).resolve().parents[1]))

from denoise.denoise import load_enhancer, MODEL_OPTIONS
from own_voice_suppression.audio_utils import prep_audio
from own_voice_suppression.validate_voice_detection import compute_si_sdr, LIBRIMIX_PATH
from own_voice_suppression.validate_source_separation import calculate_sii_from_audio, align_volume
from own_voice_suppression.plot_utils import plot_denoising_waveforms
from speech_separation.postprocessing.calculate_sii import sii


def denoise_long_audio(enhancer, noisy_wav, working_sr):
    """
    Denoises a long audio file using a sliding window approach and measures latency.
    """
    window_sec = 2.0
    stride_sec = 0.5
    
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
    rtf = avg_latency / window_sec if window_sec > 0 else 0
    
    return output_buffer, avg_latency, rtf

def _adjust_snr(clean_signal, noise_signal, snr_db):
    """
    Adjusts the noise level to a target SNR relative to the clean signal.
    """
    clean_len = clean_signal.shape[-1]
    noise_len = noise_signal.shape[-1]
    if clean_len > noise_len:
        repeat_factor = clean_len // noise_len + 1
        noise_signal = noise_signal.repeat(1, repeat_factor)
    noise_signal = noise_signal[..., :clean_len]
    power_clean = torch.mean(clean_signal ** 2)
    power_noise = torch.mean(noise_signal ** 2) + 1e-8
    power_noise_target = power_clean / (10**(snr_db / 10))
    scaling_factor = torch.sqrt(power_noise_target / power_noise)
    return noise_signal * scaling_factor


def evaluate_denoising(librimix_root, num_samples=10, model_type="convtasnet", save_outputs=False, background_noise_db=20.0, noise_type='wham'):
    """
    Evaluates a denoising model.
    Adds ambient or white noise to clean speech and evaluates the model's ability to remove it.
    """
    librimix_path = Path(librimix_root)
    
    search_pattern = str(librimix_path / "**" / "s2" / "*.wav")
    s2_files = sorted(glob.glob(search_pattern, recursive=True))[:num_samples]
    
    if not s2_files:
        print(f"No LibriMix s2 files found in {librimix_path}. Check path.")
        return

    noise_files = []
    if noise_type == 'wham':
        if background_noise_db is not None:
            noise_dir = librimix_path / "wham_noise"
            if noise_dir.exists():
                noise_files = glob.glob(str(noise_dir / "**" / "*.wav"), recursive=True)
            if not noise_files:
                raise RuntimeError(f"Warning: WHAM! noise directory not found or empty at {noise_dir}. Cannot use 'wham' noise type.")

    results = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")
    
    print(f"\n--- Starting Denoising Evaluation on {num_samples} samples ---")
    print(f"Model: {model_type}, Noise Type: {noise_type}, Noise Level: {background_noise_db} dB SNR\n")

    enhancer = load_enhancer(model_type, device)
    working_sr = enhancer.sr if hasattr(enhancer, 'sr') else 16000
    print(f"Working Sample Rate: {working_sr} Hz")

    base_output_dir = Path("denoise/outputs/validation_denoising")
    perm_out_dir = base_output_dir / f"{model_type}-{num_samples}_snr{int(background_noise_db)}_{noise_type}" if save_outputs else None
    if perm_out_dir:
        perm_out_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving outputs to {perm_out_dir}")
    
    temp_out_dir = Path(f"temp_denoise_outputs_{uuid.uuid4()}")
    temp_out_dir.mkdir(exist_ok=True)

    for s2_path in tqdm(s2_files):
        s2_path = Path(s2_path)
        
        ref_speech, sr = torchaudio.load(s2_path)
        ref_speech = prep_audio(ref_speech, sr, working_sr)

        if noise_type == 'wham':
            if not noise_files:
                print("Skipping sample because no WHAM! noise files are available.")
                continue
            noise_path = np.random.choice(noise_files)
            noise_wav, noise_sr = torchaudio.load(noise_path)
            noise_wav = prep_audio(noise_wav, noise_sr, working_sr)
        elif noise_type == 'white':
            noise_wav = torch.randn_like(ref_speech)
        else:
            raise ValueError(f"Unsupported noise type: {noise_type}")

        scaled_noise = _adjust_snr(ref_speech, noise_wav, background_noise_db)
        noisy_wav = (ref_speech + scaled_noise).to(device)

        output_buffer, avg_latency, rtf = denoise_long_audio(enhancer, noisy_wav, working_sr)
        
        denoised_for_metrics = prep_audio(output_buffer.cpu(), working_sr, 16000)
        ref_speech_for_metrics = prep_audio(ref_speech.cpu(), working_sr, 16000)
        noisy_for_metrics = prep_audio(noisy_wav.cpu(), working_sr, 16000)

        denoised_aligned = align_volume(denoised_for_metrics, ref_speech_for_metrics)

        output_si_sdr = compute_si_sdr(denoised_aligned, ref_speech_for_metrics)
        input_si_sdr = compute_si_sdr(noisy_for_metrics, ref_speech_for_metrics)
        
        results.append({
            "file": s2_path.name,
            "input_si_sdr": input_si_sdr,
            "output_si_sdr": output_si_sdr,
            "sdr_improvement": output_si_sdr - input_si_sdr,
            "latency_ms": avg_latency * 1000,
            "sii": calculate_sii_from_audio(ref_speech_for_metrics, denoised_aligned - ref_speech_for_metrics, 16000),
            "rtf": rtf
        })
        
        if save_outputs and perm_out_dir:
            sample_out_dir = perm_out_dir / s2_path.stem
            sample_out_dir.mkdir(parents=True, exist_ok=True)
            torchaudio.save(sample_out_dir / "denoised_speech.wav", denoised_aligned.cpu(), 16000)
            torchaudio.save(sample_out_dir / "noisy_input.wav", noisy_for_metrics.cpu(), 16000)
            torchaudio.save(sample_out_dir / "added_noise.wav", prep_audio(scaled_noise, working_sr, 16000).cpu(), 16000)
            shutil.copy(s2_path, sample_out_dir / "ground_truth_speech.wav")
            
            plot_denoising_waveforms(
                noisy_wav=noisy_for_metrics,
                denoised_wav=denoised_aligned,
                ground_truth_wav=ref_speech_for_metrics,
                sr=16000,
                output_path=sample_out_dir / "denoising_comparison.png"
            )
            
    shutil.rmtree(temp_out_dir)

    if not results:
        print("No results generated.")
        return
        
    avg_imp = np.mean([r['sdr_improvement'] for r in results])
    avg_output_sdr = np.mean([r['output_si_sdr'] for r in results])
    avg_latency = np.mean([r['latency_ms'] for r in results])
    avg_sii = np.mean([r['sii'] for r in results])
    avg_rtf = np.mean([r['rtf'] for r in results])

    print("\n" + "="*125)
    print("                 DENOISING RESULTS (SI-SDR, SII, LATENCY & RTF)                ")
    print("="*125)
    print(f"{ 'Filename':<25} | {'Input SI-SDR':<12} | {'Output SI-SDR':<13} | {'Improvement (dB)':<15} | {'Latency (ms)':<15} | {'SII':<5} | {'RTF':<5}")
    print("-" * 125)
    for r in results:
        print(f"{r['file'][:23]:<25} | {r['input_si_sdr']:<12.2f} | {r['output_si_sdr']:<13.2f} | {r['sdr_improvement']:<15.2f} | {r['latency_ms']:.2f} | {r['sii']:.3f} | {r['rtf']:.3f}")
    print("-" * 125)
    print(f"AVG OUTPUT SI-SDR:      {avg_output_sdr:.2f} dB")
    print(f"AVG SI-SDR IMPROVEMENT: {avg_imp:.2f} dB   <-- Key Metric")
    print(f"AVERAGE LATENCY:        {avg_latency:.2f} ms")
    print(f"AVERAGE SII (CLARITY):  {avg_sii:.3f}")
    print(f"AVERAGE RTF:            {avg_rtf:.3f}")
    print("="*125)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Denoising models on LibriMix.")
    parser.add_argument("--librimix-root", type=Path, default=LIBRIMIX_PATH, help="Path to LibriMix root directory")
    parser.add_argument("--samples", type=int, default=10, help="Number of files to evaluate")
    parser.add_argument("--model-type", type=str, choices=MODEL_OPTIONS, default="convtasnet", help="Which denoising model to use")
    parser.add_argument("--save-outputs", action='store_true', help="Save output audio files for inspection")
    parser.add_argument("--background-noise-db", type=float, default=20.0, help="SNR for adding background noise. Default: 20dB.")
    parser.add_argument("--noise-type", type=str, choices=['wham', 'white'], default='wham', help="Type of noise to add.")
    
    args = parser.parse_args()
    
    evaluate_denoising(args.librimix_root, args.samples, args.model_type, args.save_outputs, args.background_noise_db, args.noise_type)
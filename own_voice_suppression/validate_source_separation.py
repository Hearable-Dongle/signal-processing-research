import argparse
import glob
import shutil
from pathlib import Path
import argparse
import glob
import shutil
from pathlib import Path
import sys

import torch
import torchaudio
import numpy as np
from tqdm import tqdm

from speech_separation.postprocessing.calculate_sii import sii

from own_voice_suppression.validate_voice_detection import (
    compute_si_sdr,
    compute_suppression_db,
    run_suppression,
    LIBRIMIX_PATH
)

# ANSI S3.5-1997 third-octave bands (center frequencies from 160Hz to 8000Hz)
FREQ_BANDS = [
    (141, 178), (178, 224), (224, 281), (281, 355), (355, 447),
    (447, 562), (562, 708), (708, 891), (891, 1122), (1122, 1413),
    (1413, 1778), (1778, 2239), (2239, 2818), (2818, 3548),
    (3548, 4467), (4467, 5623), (5623, 7079), (7079, 8913)
]

def _audio_to_18_band_spectrum_level(audio, sample_rate, n_fft=4096):
    """
    Calculates the spectrum level in 18 third-octave bands for a given audio signal.
    """
    if audio.dim() > 1 and audio.shape[0] > 1:
        audio = audio.mean(dim=0)
    audio = audio.squeeze()

    win_length = n_fft
    hop_length = win_length // 4
    
    spectrogram_transform = torchaudio.transforms.Spectrogram(
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        window_fn=torch.hann_window,
        power=2.0
    )
    power_spec = spectrogram_transform(audio)
    avg_power_per_bin = power_spec.mean(dim=1)
    
    fft_freqs = torch.fft.rfftfreq(n_fft, d=1.0/sample_rate)
    
    band_powers = []
    for low_f, high_f in FREQ_BANDS:
        band_mask = (fft_freqs >= low_f) & (fft_freqs < high_f)
        if not torch.any(band_mask):
            power = 1e-12
        else:
            power = avg_power_per_bin[band_mask].sum().item()
        band_powers.append(power)
    
    band_powers = np.array(band_powers)
    band_widths = np.array([h - l for l, h in FREQ_BANDS])
    
    spectrum_level = band_powers / band_widths
    return spectrum_level

def calculate_sii_from_audio(target_audio, residual_noise_audio, sample_rate):
    """
    Calculates the Speech Intelligibility Index (SII) from raw audio tensors.
    """
    n_fft = 4096
    
    # ssl: Speech Spectrum Level (from target background audio)
    # nsl: Noise Spectrum Level (from residual user voice)
    ssl_level = _audio_to_18_band_spectrum_level(target_audio, sample_rate)
    nsl_level = _audio_to_18_band_spectrum_level(residual_noise_audio, sample_rate)
    
    epsilon = 1e-20
    ssl_db = 10 * np.log10(ssl_level + epsilon)
    nsl_db = 10 * np.log10(nsl_level + epsilon)

    # Normalize levels. Assume the peak of the calculated speech spectrum
    # corresponds to the peak of the standard 'normal' speech spectrum (~35 dB).
    # This pseudo-calibrates the levels for the SII function.
    peak_ssl_db = np.max(ssl_db)
    offset = 35 - peak_ssl_db
    
    ssl_calibrated = ssl_db + offset
    nsl_calibrated = nsl_db + offset
    
    hearing_threshold = np.zeros(18)
    
    sii_score = sii(ssl_calibrated, nsl_calibrated, hearing_threshold)
    return sii_score


def evaluate_separation(librimix_root, num_samples=10, model_type="wavlm-large", save_outputs=False):
    """
    Evaluates Source Separation quality using SI-SDR and SII.
    - SI-SDR: Checks how well the system isolates the background (Target) and suppresses the User (Interference).
    - SII: Estimates speech intelligibility of the target speech in the presence of the residual noise.
    """
    librimix_path = Path(librimix_root)
    
    search_pattern = str(librimix_path / "**" / "s1" / "*.wav")
    s1_files = glob.glob(search_pattern, recursive=True)
    
    if not s1_files:
        print(f"No files found in {librimix_path}. Check your path structure.")
        return

    s1_files = sorted(s1_files)[:num_samples]
    
    results = []
    
    print(f"\n--- Starting Source Separation Evaluation (SI-SDR & SII) on {num_samples} samples ---")
    print(f"Classifier: {model_type}\n")

    base_output_dir = Path("own_voice_suppression/outputs/validation_separation")
    if save_outputs:
        perm_out_dir = base_output_dir / f"{model_type}-{num_samples}"
        perm_out_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving outputs to {perm_out_dir}")
    else:
        perm_out_dir = None
    
    temp_out_dir = Path("temp_sep_outputs")
    temp_out_dir.mkdir(exist_ok=True)

    for s1_path in tqdm(s1_files):
        s1_path = Path(s1_path)
        
        # LibriMix mapping: 
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
            smoothing_window=10,
            speaker_detection_threshold=0.65
        )
        
        suppressed_path = temp_out_dir / "suppressed.wav"
        if not suppressed_path.exists():
            print("Error: Output file not created.")
            continue
            
        est_audio, sample_rate = torchaudio.load(suppressed_path) # The system's attempt at "Background Only"
        ref_background, _ = torchaudio.load(s2_path)    # Ground Truth Background (S2)
        ref_user, _ = torchaudio.load(s1_path)          # Ground Truth User (S1)
        mix_audio, _ = torchaudio.load(mix_path)

        # SI-SDR Metrics
        sdr_background = compute_si_sdr(est_audio, ref_background)
        input_sdr_background = compute_si_sdr(mix_audio, ref_background)
        sdr_improvement = sdr_background - input_sdr_background
        suppression_val = compute_suppression_db(mix_audio, est_audio, ref_user)
        
        # Speech Intelligibility Index (SII)
        # We model the background (s2) as the target 'speech' and the residual user voice as the 'noise'.
        residual_noise = est_audio - ref_background
        sii_score = calculate_sii_from_audio(target_audio=ref_background, residual_noise_audio=residual_noise, sample_rate=sample_rate)
        
        results.append({
            "file": s1_path.name,
            "sdr_background": sdr_background,
            "sdr_improvement": sdr_improvement,
            "suppression_db": suppression_val,
            "sii": sii_score
        })
        
        if save_outputs and perm_out_dir:
            sample_out_dir = perm_out_dir / s1_path.stem
            sample_out_dir.mkdir(exist_ok=True)
            shutil.copy(suppressed_path, sample_out_dir / "estimated_background.wav")
            shutil.copy(mix_path, sample_out_dir / "original_mix.wav")
            shutil.copy(s2_path, sample_out_dir / "ground_truth_background.wav")
            shutil.copy(s1_path, sample_out_dir / "ground_truth_user.wav")

    shutil.rmtree(temp_out_dir)

    avg_sdr = np.mean([r['sdr_background'] for r in results])
    avg_imp = np.mean([r['sdr_improvement'] for r in results])
    avg_supp = np.mean([r['suppression_db'] for r in results])
    avg_sii = np.mean([r['sii'] for r in results])
    
    print("\n" + "="*75)
    print("                SOURCE SEPARATION & INTELLIGIBILITY RESULTS                ")
    print("="*75)
    print(f"{'Filename':<25} | {'SDR (dB)':<10} | {'Imp (dB)':<10} | {'Supp (dB)':<10} | {'SII':<5}")
    print("-" * 75)
    for r in results:
        print(f"{r['file'][:23]:<25} | {r['sdr_background']:<10.2f} | {r['sdr_improvement']:<10.2f} | {r['suppression_db']:<10.2f} | {r['sii']:.3f}")
    print("-" * 75)
    print(f"AVG SDR (Quality):       {avg_sdr:.2f} dB")
    print(f"AVG IMPROVEMENT:         {avg_imp:.2f} dB   <-- Key Metric")
    print(f"AVG SUPPRESSION (User):  {avg_supp:.2f} dB")
    print(f"AVG SII (Clarity):       {avg_sii:.3f}")
    print("="*75)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Source Separation (SI-SDR & SII) on LibriMix.")
    parser.add_argument("--librimix-root", type=Path, default=LIBRIMIX_PATH, help="Path to LibriMix root directory")
    parser.add_argument("--samples", type=int, default=10, help="Number of files to evaluate")
    parser.add_argument("--model-type", type=str, default="wavlm-large", help="Which classifier model to use")
    parser.add_argument("--save-outputs", action='store_true', help="Save output audio files for inspection")
    
    args = parser.parse_args()
    
    evaluate_separation(args.librimix_root, args.samples, args.model_type, args.save_outputs)
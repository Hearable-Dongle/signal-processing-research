import argparse
import glob
import shutil
from pathlib import Path
import sys

# Add project root to allow sibling imports
sys.path.append(str(Path(__file__).resolve().parents[1]))

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


def _adjust_snr(clean_signal, noise_signal, snr_db):
    """
    Adjusts the noise level to a target SNR relative to the clean signal.
    """
    # Ensure equal length
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

def evaluate_separation(librimix_root, num_samples=10, model_type="wavlm-large", save_outputs=False, background_noise_db=None):
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
    
    # Find noise files if requested
    noise_files = []
    if background_noise_db is not None:
        noise_dir = librimix_path / "wham_noise" # Assuming noise is inside the LibriMix root
        if not noise_dir.exists():
            print(f"Warning: Noise directory not found at {noise_dir}. Cannot add background noise.")
            background_noise_db = None
        else:
            noise_files = glob.glob(str(noise_dir / "**" / "*.wav"), recursive=True)
            if not noise_files:
                print(f"Warning: No noise files found in {noise_dir}. Cannot add background noise.")
                background_noise_db = None

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
        
        mix_path = Path(str(s1_path).replace("/s1/", "/mix_clean/"))
        s2_path = Path(str(s1_path).replace("/s1/", "/s2/"))
        
        if not mix_path.exists() or not s2_path.exists():
            print(f"Skipping {s1_path.name}: distinct mix/s2 files not found.")
            continue

        mix_clean_audio, sr = torchaudio.load(mix_path)
        input_mix_path = mix_path
        input_mix_for_metrics = mix_clean_audio

        if background_noise_db is not None and noise_files:
            noise_path = np.random.choice(noise_files)
            noise_audio, noise_sr = torchaudio.load(noise_path)

            if noise_audio.shape[0] > 1:
                noise_audio = torch.mean(noise_audio, dim=0, keepdim=True)
            
            if noise_sr != sr:
                resampler = torchaudio.transforms.Resample(noise_sr, sr)
                noise_audio = resampler(noise_audio)
            
            scaled_noise = _adjust_snr(mix_clean_audio, noise_audio, background_noise_db)
            mix_noisy_audio = mix_clean_audio + scaled_noise

            noisy_mix_path = temp_out_dir / "temp_noisy_mix.wav"
            torchaudio.save(noisy_mix_path, mix_noisy_audio, sr)
            
            input_mix_path = noisy_mix_path
            input_mix_for_metrics = mix_noisy_audio

        run_suppression(
            enrolment_path=s1_path,
            mixed_path=input_mix_path,
            output_directory=temp_out_dir,
            classifier_type=model_type,
            smoothing_window=10,
            speaker_detection_threshold=0.65
        )
        
        suppressed_path = temp_out_dir / "suppressed.wav"
        if not suppressed_path.exists():
            print("Error: Output file not created.")
            continue
            
        est_audio, sample_rate = torchaudio.load(suppressed_path)
        ref_background, _ = torchaudio.load(s2_path)
        ref_user, _ = torchaudio.load(s1_path)

        # SI-SDR Metrics
        sdr_background = compute_si_sdr(est_audio, ref_background)
        input_sdr_background = compute_si_sdr(input_mix_for_metrics, ref_background)
        sdr_improvement = sdr_background - input_sdr_background
        suppression_val = compute_suppression_db(input_mix_for_metrics, est_audio, ref_user)
        
        # Speech Intelligibility Index (SII)
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
            shutil.copy(str(input_mix_path), sample_out_dir / "original_mix.wav") # Copy the potentially noisy mix
            shutil.copy(s2_path, sample_out_dir / "ground_truth_background.wav")
            shutil.copy(s1_path, sample_out_dir / "ground_truth_user.wav")

    import shutil
from pathlib import Path
import sys

# Add project root to allow sibling imports
sys.path.append(str(Path(__file__).resolve().parents[1]))

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
from own_voice_suppression.plot_utils import plot_target_presence

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


def _adjust_snr(clean_signal, noise_signal, snr_db):
    """
    Adjusts the noise level to a target SNR relative to the clean signal.
    """
    # Ensure equal length
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

def evaluate_separation(librimix_root, num_samples=10, model_type="wavlm-large", save_outputs=False, background_noise_db=None):
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
    
    # Find noise files if requested
    noise_files = []
    if background_noise_db is not None:
        noise_dir = librimix_path / "wham_noise" # Assuming noise is inside the LibriMix root
        if not noise_dir.exists():
            print(f"Warning: Noise directory not found at {noise_dir}. Cannot add background noise.")
            background_noise_db = None
        else:
            noise_files = glob.glob(str(noise_dir / "**" / "*.wav"), recursive=True)
            if not noise_files:
                print(f"Warning: No noise files found in {noise_dir}. Cannot add background noise.")
                background_noise_db = None

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
        
        mix_path = Path(str(s1_path).replace("/s1/", "/mix_clean/"))
        s2_path = Path(str(s1_path).replace("/s1/", "/s2/"))
        
        if not mix_path.exists() or not s2_path.exists():
            print(f"Skipping {s1_path.name}: distinct mix/s2 files not found.")
            continue

        mix_clean_audio, sr = torchaudio.load(mix_path)
        input_mix_path = mix_path
        input_mix_for_metrics = mix_clean_audio

        if background_noise_db is not None and noise_files:
            noise_path = np.random.choice(noise_files)
            noise_audio, noise_sr = torchaudio.load(noise_path)

            if noise_audio.shape[0] > 1:
                noise_audio = torch.mean(noise_audio, dim=0, keepdim=True)
            
            if noise_sr != sr:
                resampler = torchaudio.transforms.Resample(noise_sr, sr)
                noise_audio = resampler(noise_audio)
            
            scaled_noise = _adjust_snr(mix_clean_audio, noise_audio, background_noise_db)
            mix_noisy_audio = mix_clean_audio + scaled_noise

            noisy_mix_path = temp_out_dir / "temp_noisy_mix.wav"
            torchaudio.save(noisy_mix_path, mix_noisy_audio, sr)
            
            input_mix_path = noisy_mix_path
            input_mix_for_metrics = mix_noisy_audio

        confidence_logs = run_suppression(
            enrolment_path=s1_path,
            mixed_path=input_mix_path,
            output_directory=temp_out_dir,
            classifier_type=model_type,
            smoothing_window=10,
            speaker_detection_threshold=0.65
        )
        
        suppressed_path = temp_out_dir / "suppressed.wav"
        if not suppressed_path.exists():
            print("Error: Output file not created.")
            continue
            
        est_audio, sample_rate = torchaudio.load(suppressed_path)
        ref_background, _ = torchaudio.load(s2_path)
        ref_user, _ = torchaudio.load(s1_path)

        # SI-SDR Metrics
        sdr_background = compute_si_sdr(est_audio, ref_background)
        input_sdr_background = compute_si_sdr(input_mix_for_metrics, ref_background)
        sdr_improvement = sdr_background - input_sdr_background
        suppression_val = compute_suppression_db(input_mix_for_metrics, est_audio, ref_user)
        
        # Speech Intelligibility Index (SII)
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
            shutil.copy(str(input_mix_path), sample_out_dir / "original_mix.wav") # Copy the potentially noisy mix
            shutil.copy(s2_path, sample_out_dir / "ground_truth_background.wav")
            shutil.copy(s1_path, sample_out_dir / "ground_truth_user.wav")
            
            if confidence_logs:
                plot_path = sample_out_dir / "target_presence.png"
                plot_target_presence(confidence_logs, plot_path, model_type)

    shutil.rmtree(temp_out_dir)

    avg_sdr = np.mean([r['sdr_background'] for r in results])
    avg_imp = np.mean([r['sdr_improvement'] for r in results])
    avg_supp = np.mean([r['suppression_db'] for r in results])
    avg_sii = np.mean([r['sii'] for r in results])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Source Separation (SI-SDR & SII) on LibriMix.")
    parser.add_argument("--librimix-root", type=Path, default=LIBRIMIX_PATH, help="Path to LibriMix root directory")
    parser.add_argument("--samples", type=int, default=10, help="Number of files to evaluate")
    parser.add_argument("--model-type", type=str, default="wavlm-large", help="Which classifier model to use")
    parser.add_argument("--save-outputs", action='store_true', help="Save output audio files for inspection")
    parser.add_argument("--background-noise-db", type=float, default=None, help="Add background noise at a specific SNR (in dB) relative to the clean mix.")
    
    args = parser.parse_args()
    
    evaluate_separation(args.librimix_root, args.samples, args.model_type, args.save_outputs, args.background_noise_db)
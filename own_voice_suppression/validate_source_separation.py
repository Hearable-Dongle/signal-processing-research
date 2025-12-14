import argparse
import glob
import sys
from pathlib import Path
from os import PathLike

import numpy as np
import torch
import torchaudio
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))

from general_utils.constants import LIBRIMIX_PATH
from own_voice_suppression.plot_utils import plot_target_presence, plot_waveform_comparison
from own_voice_suppression.source_separation import (
    DETECTION_THRESHOLD, MODEL_OPTIONS, STRIDE_SEC, WAVLM_REQUIRED_SR, WINDOW_SEC,
    ModelOption, prep_audio, resample, run_separation_pipeline)
from own_voice_suppression.validate_voice_detection import (
    compute_si_sdr, compute_suppression_db)
from speech_separation.postprocessing.calculate_sii import sii

RANDOM_SEED = 42

FREQ_BANDS = [
    (141, 178), (178, 224), (224, 281), (281, 355), (355, 447),
    (447, 562), (562, 708), (708, 891), (891, 1122), (1122, 1413),
    (1413, 1778), (1778, 2239), (2239, 2818), (2818, 3548),
    (3548, 4467), (4467, 5623), (5623, 7079), (7079, 8913)
]

def _audio_to_18_band_spectrum_level(audio, sample_rate, n_fft=4096):
    if audio.dim() > 1 and audio.shape[0] > 1:
        audio = audio.mean(dim=0)
    audio = audio.squeeze()
    win_length = n_fft
    hop_length = win_length // 4
    spectrogram_transform = torchaudio.transforms.Spectrogram(
        n_fft=n_fft, win_length=win_length, hop_length=hop_length,
        window_fn=torch.hann_window, power=2.0
    )
    power_spec = spectrogram_transform(audio)
    avg_power_per_bin = power_spec.mean(dim=1)
    fft_freqs = torch.fft.rfftfreq(n_fft, d=1.0/sample_rate)
    band_powers = []
    for low_f, high_f in FREQ_BANDS:
        band_mask = (fft_freqs >= low_f) & (fft_freqs < high_f)
        power = avg_power_per_bin[band_mask].sum().item() if torch.any(band_mask) else 1e-12
        band_powers.append(power)
    band_powers = np.array(band_powers)
    band_widths = np.array([h - l for l, h in FREQ_BANDS])
    spectrum_level = band_powers / band_widths
    return spectrum_level


def align_volume(estimated: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Scales the estimated signal to optimally match the target signal 
    using Least Squares (minimizing the energy of the difference).
    """
    # Ensure 1D for dot product
    est_flat = estimated.view(-1)
    tgt_flat = target.view(-1)
    
    # Calculate optimal scaling factor alpha
    # alpha = <estimated, target> / <estimated, estimated>
    dot_prod = torch.dot(est_flat, tgt_flat)
    energy_est = torch.dot(est_flat, est_flat) + 1e-8
    
    alpha = dot_prod / energy_est
    
    return alpha * estimated


def calculate_sii_from_audio(target_audio, residual_noise_audio, sample_rate):
    n_fft = 4096
    ssl_level = _audio_to_18_band_spectrum_level(target_audio, sample_rate, n_fft)
    nsl_level = _audio_to_18_band_spectrum_level(residual_noise_audio, sample_rate, n_fft)
    
    epsilon = 1e-20
    ssl_db = 10 * np.log10(ssl_level + epsilon)
    nsl_db = 10 * np.log10(nsl_level + epsilon)
    
    peak_ssl_db = np.max(ssl_db)
    offset = 65 - peak_ssl_db  
    
    ssl_calibrated = ssl_db + offset
    nsl_calibrated = nsl_db + offset
    
    hearing_threshold = np.zeros(18)
    sii_score = sii(ssl_calibrated, nsl_calibrated, hearing_threshold)
    return sii_score


def _adjust_snr(clean_signal, noise_signal, snr_db):
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


def evaluate_separation(
    librimix_root: Path,
    num_samples: int,
    model_type: ModelOption,
    detection_threshold: float,
    save_outputs: bool,
    output_dir: PathLike,
    background_noise_db: float,
    window_sec: float = WINDOW_SEC,
    smoothing_window: int = 10,
    stride_sec: float = STRIDE_SEC
):
    output_dir = Path(output_dir)
    np.random.seed(RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    s1_files = sorted(glob.glob(str(librimix_root / "**" / "s1" / "*.wav"), recursive=True))
    s2_files = sorted(glob.glob(str(librimix_root / "**" / "s2" / "*.wav"), recursive=True))
    
    if not s1_files or not s2_files:
        print(f"No LibriMix files found in {librimix_root}. Check path.")
        return 0

    np.random.shuffle(s1_files)
    np.random.shuffle(s2_files)
    
    noise_files = []
    if background_noise_db is not None:
        noise_dir = librimix_root.parent / "wham_noise"
        if noise_dir.exists():
            noise_files = glob.glob(str(noise_dir / "**" / "*.wav"), recursive=True)
            if not noise_files:
                print(f"Warning: No noise files found in {noise_dir}. Cannot add background noise.")
        else:
            print(f"Warning: Noise directory not found at {noise_dir}. Cannot add background noise.")

    results = []
    
    print(f"\n--- Evaluating Speaker Suppression on {num_samples} synthesized samples ---")
    
    for i in tqdm(range(num_samples), desc="Evaluating Separation"):
        target_path = Path(s1_files[i % len(s1_files)])
        background_path = Path(s2_files[i % len(s2_files)])

        target_audio, sr1 = torchaudio.load(target_path)
        background_audio, sr2 = torchaudio.load(background_path)

        target_audio = prep_audio(target_audio, sr1, WAVLM_REQUIRED_SR)
        background_audio = prep_audio(background_audio, sr2, WAVLM_REQUIRED_SR)

        min_len = min(target_audio.shape[1], background_audio.shape[1])
        target_audio = target_audio[:, :min_len]
        background_audio = background_audio[:, :min_len]
        
        synthesized_mix = target_audio + background_audio
        synthesized_mix /= torch.max(torch.abs(synthesized_mix)) + 1e-8
        
        input_for_processing = synthesized_mix

        if background_noise_db is not None and noise_files:
            noise_path = np.random.choice(noise_files)
            noise_audio, noise_sr = torchaudio.load(noise_path)
            noise_audio = prep_audio(noise_audio, noise_sr, WAVLM_REQUIRED_SR)
            scaled_noise = _adjust_snr(synthesized_mix, noise_audio, background_noise_db)
            input_for_processing += scaled_noise
            input_for_processing /= torch.max(torch.abs(input_for_processing)) + 1e-8

        output_audios, logs, result_sr = run_separation_pipeline(
            mixed_audio=input_for_processing,
            orig_sr_mix=WAVLM_REQUIRED_SR,
            enrolment_audio=target_audio,
            orig_sr_enrol=WAVLM_REQUIRED_SR,
            model_type=model_type,
            device=device,
            suppress=True,
            detection_threshold=detection_threshold,
            window_sec=window_sec,
            stride_sec=stride_sec,
            smoothing_window=smoothing_window
        )
        estimated_background = output_audios[0]

        if result_sr != WAVLM_REQUIRED_SR:
            estimated_background = resample(estimated_background.cpu(), result_sr, WAVLM_REQUIRED_SR)
        else:
            estimated_background = estimated_background.cpu()

        estimated_background = align_volume(estimated_background, background_audio)

        sdr = compute_si_sdr(estimated_background, background_audio)
        input_sdr = compute_si_sdr(input_for_processing.cpu(), background_audio)
        sdr_improvement = sdr - input_sdr
        suppression_db = compute_suppression_db(input_for_processing.cpu(), estimated_background, target_audio)
        
        residual_for_sii = estimated_background - background_audio
        sii = calculate_sii_from_audio(background_audio, residual_for_sii, WAVLM_REQUIRED_SR)

        results.append({
            "file": f"mix_{i+1}",
            "sdr_improvement": sdr_improvement,
            "suppression_db": suppression_db,
            "sii": sii
        })

        if save_outputs:
            sample_out_dir = output_dir / f"{model_type}_sample_{i+1}"
            sample_out_dir.mkdir(parents=True, exist_ok=True)
            torchaudio.save(sample_out_dir / "input_mix.wav", input_for_processing.cpu(), WAVLM_REQUIRED_SR)
            torchaudio.save(sample_out_dir / "ground_truth_background.wav", background_audio.cpu(), WAVLM_REQUIRED_SR)
            torchaudio.save(sample_out_dir / "estimated_background.wav", estimated_background.cpu(), WAVLM_REQUIRED_SR)
            torchaudio.save(sample_out_dir / "ground_truth_target.wav", target_audio.cpu(), WAVLM_REQUIRED_SR)
            
            plot_waveform_comparison(
                estimated_wav=estimated_background,
                ground_truth_wav=background_audio,
                sr=WAVLM_REQUIRED_SR,
                output_path=sample_out_dir / "waveform_comparison.png"
            )

            if logs:
                plot_target_presence(logs, sample_out_dir / "target_presence.png", model_type)

    if not results:
        print("No results generated.")
        return 0

    avg_imp = np.mean([r['sdr_improvement'] for r in results])
    avg_supp = np.mean([r['suppression_db'] for r in results])
    avg_sii = np.mean([r['sii'] for r in results])

    print("\n" + "="*51)
    print("      SPEAKER SUPPRESSION EVALUATION RESULTS      ")
    print("="*51)
    print(f"{ 'Sample':<10} | { 'SDR Imp (dB)':<15} | { 'Supp (dB)':<12} | { 'SII':<5}")
    print("-" * 51)
    for r in results:
        print(f"{r['file']:<10} | {r['sdr_improvement']:<15.2f} | {r['suppression_db']:<12.2f} | {r['sii']:.3f}")
    print("-" * 51)
    print(f"AVERAGE SI-SDR IMPROVEMENT: {avg_imp:.2f} dB")
    print(f"AVERAGE TARGET SUPPRESSION: {avg_supp:.2f} dB")
    print(f"AVERAGE SII (CLARITY):    {avg_sii:.3f}")
    print("="*51)
    
    if save_outputs:
        print("\nOutputs saved to:", output_dir)

    return avg_sii


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate speaker suppression performance on synthesized mixes.")
    parser.add_argument("--librimix-root", type=Path, default=LIBRIMIX_PATH)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--model-type", type=str, default="convtasnet", choices=MODEL_OPTIONS)
    parser.add_argument("--save-outputs", action='store_true')
    parser.add_argument("--background-noise-db", type=float, default=None, help="Add background noise at a specific SNR (in dB).")
    parser.add_argument("--detection-threshold", type=float, default=DETECTION_THRESHOLD)
    parser.add_argument("--window-sec", type=float, default=WINDOW_SEC, help="Window size in seconds for processing audio chunks.")
    parser.add_argument("--stride-sec", type=float, default=STRIDE_SEC, help="Stride size in seconds for processing audio chunks.")
    parser.add_argument("--smoothing-window", type=int, default=10, help="Number of frames for score smoothing.")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    if not args.output_dir:
        args.output_dir = Path("own_voice_suppression/outputs/validation/separation") / f"{args.model_type}_thresh_{args.detection_threshold}"

    evaluate_separation(
        librimix_root=args.librimix_root,
        num_samples=args.samples,
        model_type=args.model_type,
        detection_threshold=args.detection_threshold,
        save_outputs=args.save_outputs,
        output_dir=args.output_dir,
        background_noise_db=args.background_noise_db,
        window_sec=args.window_sec,
        smoothing_window=args.smoothing_window,
        stride_sec=args.stride_sec
    )

import argparse
import glob
import shutil
from pathlib import Path
import torch
import torchaudio
import numpy as np
from tqdm import tqdm

# Reuse metrics and constants from your existing validation script
from own_voice_suppression.validate_voice_detection import (
    compute_si_sdr,
    compute_suppression_db,
    run_suppression,
    LIBRIMIX_PATH
)

def evaluate_separation(librimix_root, num_samples=10, model_type="wavlm-large", save_outputs=False):
    """
    Evaluates Source Separation quality using SI-SDR.
    Checks how well the system isolates the background (Target) and suppresses the User (Interference).
    """
    librimix_path = Path(librimix_root)
    
    search_pattern = str(librimix_path / "**" / "s1" / "*.wav")
    s1_files = glob.glob(search_pattern, recursive=True)
    
    if not s1_files:
        print(f"No files found in {librimix_path}. Check your path structure.")
        return

    s1_files = sorted(s1_files)[:num_samples]
    
    results = []
    
    print(f"\n--- Starting Source Separation Evaluation (SI-SDR) on {num_samples} samples ---")
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
            
        est_audio, _ = torchaudio.load(suppressed_path) # The system's attempt at "Background Only"
        ref_background, _ = torchaudio.load(s2_path)    # Ground Truth Background (S2)
        ref_user, _ = torchaudio.load(s1_path)          # Ground Truth User (S1)
        mix_audio, _ = torchaudio.load(mix_path)

        sdr_background = compute_si_sdr(est_audio, ref_background)
        
        # Baseline SDR: How close was the original mix to the background?
        input_sdr_background = compute_si_sdr(mix_audio, ref_background)
        
        # Improvement: The "Value Add" of your system
        sdr_improvement = sdr_background - input_sdr_background

        # Suppression: How much of S1 did we kill?
        suppression_val = compute_suppression_db(mix_audio, est_audio, ref_user)
        
        results.append({
            "file": s1_path.name,
            "sdr_background": sdr_background,
            "sdr_improvement": sdr_improvement,
            "suppression_db": suppression_val
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
    
    print("\n" + "="*60)
    print("           SOURCE SEPARATION RESULTS (SI-SDR)           ")
    print("="*60)
    print(f"{'Filename':<25} | {'SDR (dB)':<10} | {'Imp (dB)':<10} | {'Supp (dB)':<10}")
    print("-" * 65)
    for r in results:
        print(f"{r['file'][:23]:<25} | {r['sdr_background']:<10.2f} | {r['sdr_improvement']:<10.2f} | {r['suppression_db']:<10.2f}")
    print("-" * 65)
    print(f"AVG SDR (Quality):       {avg_sdr:.2f} dB")
    print(f"AVG IMPROVEMENT:         {avg_imp:.2f} dB   <-- Key Metric")
    print(f"AVG SUPPRESSION (User):  {avg_supp:.2f} dB")
    print("="*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Source Separation (SI-SDR) on LibriMix.")
    parser.add_argument("--librimix-root", type=Path, default=LIBRIMIX_PATH, help="Path to LibriMix root directory")
    parser.add_argument("--samples", type=int, default=10, help="Number of files to evaluate")
    parser.add_argument("--model-type", type=str, default="wavlm-large", help="Which classifier model to use")
    parser.add_argument("--save-outputs", action='store_true', help="Save output audio files for inspection")
    
    args = parser.parse_args()
    
    evaluate_separation(args.librimix_root, args.samples, args.model_type, args.save_outputs)
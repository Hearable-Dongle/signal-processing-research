# Description:
# This script analyzes the effect of window and stride sizes on the
# Speech Intelligibility Index (SII) of the speaker suppression system.

import argparse
from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))

from own_voice_suppression.validate_source_separation import evaluate_separation
from own_voice_suppression.source_separation import (
    MODEL_OPTIONS,
    DETECTION_THRESHOLD,
    WINDOW_SEC,
    STRIDE_SEC
)
from general_utils.constants import LIBRIMIX_PATH

def tune_parameters(
    tune_mode: str,
    constant_window: float,
    constant_stride: float,
    model_type: str,
    samples: int,
    output_dir: Path,
    librimix_root: Path,
    detection_threshold: float,
    smoothing_window: int
):
    """
    Runs the evaluation loop for a range of parameter values and plots the result.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    param_to_tune = ""
    fixed_param_name = ""
    fixed_param_val = 0
    
    if tune_mode == 'window':
        param_to_tune = "window_sec"
        fixed_param_name = "stride_sec"
        fixed_param_val = constant_stride
        # Generate a range of window sizes from 32ms up to the fixed value
        tuning_values = np.linspace(0.032, constant_window, num=15)
    elif tune_mode == 'stride':
        param_to_tune = "stride_sec"
        fixed_param_name = "window_sec"
        fixed_param_val = constant_window
        # Stride must be <= window. Generate strides from 32ms up to the window size.
        tuning_values = np.linspace(0.032, constant_window, num=15)
    else:
        raise ValueError("tune_mode must be 'window' or 'stride'")

    sii_scores = []
    
    print(f"--- Tuning {param_to_tune} ---")
    print(f"Constant {fixed_param_name}: {fixed_param_val:.3f}s")
    
    for val in tqdm(tuning_values, desc=f"Tuning {param_to_tune}"):
        
        # Ensure stride is not greater than window
        if param_to_tune == 'window_sec' and fixed_param_val > val:
            continue
        if param_to_tune == 'stride_sec' and val > fixed_param_val:
            continue

        avg_sii = evaluate_separation(
            librimix_root=librimix_root,
            num_samples=samples,
            model_type=model_type,
            detection_threshold=detection_threshold,
            save_outputs=False,
            output_dir=output_dir / "temp_outputs",
            background_noise_db=None,
            window_sec=val if param_to_tune == 'window_sec' else fixed_param_val,
            stride_sec=val if param_to_tune == 'stride_sec' else fixed_param_val,
            smoothing_window=smoothing_window
        )
        sii_scores.append(avg_sii)

    # Filter out values that were skipped
    valid_tuning_values = [v for v in tuning_values if v <= (constant_window if tune_mode == 'stride' else v)]
    if tune_mode == 'window':
        valid_tuning_values = [v for v in tuning_values if v >= constant_stride]


    # --- Plotting ---
    plt.figure(figsize=(10, 6))
    plt.plot(valid_tuning_values, sii_scores, marker='o', linestyle='-')
    
    plt.title(f'SII vs. {param_to_tune.replace("_", " ").title()} (Fixed {fixed_param_name.replace("_", " ")} = {fixed_param_val:.3f}s)')
    plt.xlabel(f'{param_to_tune.replace("_", " ").title()} (seconds)')
    plt.ylabel("Average SII Score")
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plot_filename = f"sii_vs_{param_to_tune}_const_{fixed_param_name}_{int(fixed_param_val*1000)}ms.png"
    plot_path = output_dir / plot_filename
    plt.savefig(plot_path)
    
    print(f"\nPlot saved to: {plot_path}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze the effect of window and stride size on SII for speaker suppression."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--tune-window", action='store_true', help="Tune window_sec while keeping stride constant.")
    group.add_argument("--tune-stride", action='store_true', help="Tune stride_sec while keeping window constant.")
    
    parser.add_argument("--model-type", type=str, default="convtasnet", choices=MODEL_OPTIONS)
    parser.add_argument("--samples", type=int, default=10, help="Number of samples to average SII over for each data point.")
    parser.add_argument("--output-dir", type=Path)
    
    # Arguments for the constant (non-tuned) parameter
    parser.add_argument("--constant-window-sec", type=float, default=WINDOW_SEC)
    parser.add_argument("--constant-stride-sec", type=float, default=STRIDE_SEC)

    # Arguments required by evaluate_separation
    parser.add_argument("--librimix-root", type=Path, default=LIBRIMIX_PATH)
    parser.add_argument("--detection-threshold", type=float, default=DETECTION_THRESHOLD)
    parser.add_argument("--smoothing-window", type=int, default=10)

    args = parser.parse_args()

    tune_mode = 'window' if args.tune_window else 'stride'

    if not args.output_dir:
        args.output_dir = Path(f"own_voice_suppression/outputs/tuning_results/{args.model_type}_thresh_{args.detection_threshold}_smooth_win_{args.smoothing_window}/{tune_mode}_tuning")

    tune_parameters(
        tune_mode=tune_mode,
        constant_window=args.constant_window_sec,
        constant_stride=args.constant_stride_sec,
        model_type=args.model_type,
        samples=args.samples,
        output_dir=args.output_dir,
        librimix_root=args.librimix_root,
        detection_threshold=args.detection_threshold,
        smoothing_window=args.smoothing_window
    )

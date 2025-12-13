import argparse
import optuna
from pathlib import Path
import sys

from own_voice_suppression.validate_voice_detection import evaluate_detection_vad, evaluate_detection_sdr
from own_voice_suppression.voice_detection import CLASSIFIER_OPTIONS
from general_utils.constants import LIBRIMIX_PATH


def objective(trial: optuna.Trial, args):
    """
    Optuna objective function to maximize the chosen metric by tuning hyperparameters.
    """
    threshold = trial.suggest_float("speaker_detection_threshold", 0.3, 1.0)
    smoothing_window = trial.suggest_int("smoothing_window", 1, 20)

    print(f"\n--- Trial {trial.number}: Testing threshold={threshold:.4f}, smoothing_window={smoothing_window} ---")

    if args.metric_mode == 'vad':
        metric_value = evaluate_detection_vad(
            librimix_root=args.librimix_root,
            num_samples=args.samples,
            model_type=args.model_type,
            save_outputs=False,  # Disable saving during tuning for speed
            speaker_detection_threshold=threshold,
            smoothing_window=smoothing_window
        )
        return metric_value
    else:  # 'sdr'
        metric_value = evaluate_detection_sdr(
            librimix_root=args.librimix_root,
            num_samples=args.samples,
            model_type=args.model_type,
            save_outputs=False, 
            speaker_detection_threshold=threshold,
            smoothing_window=smoothing_window
        )
        return metric_value

def main():
    parser = argparse.ArgumentParser(
        description="Tune hyperparameters for the voice suppression system using Optuna.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--model-type", type=str, default="wavlm-large", choices=CLASSIFIER_OPTIONS,
                        help="Which classifier model to use for tuning.")
    parser.add_argument("--samples", type=int, default=20,
                        help="Number of audio samples to use for evaluation in each trial.")
    parser.add_argument("--trials", type=int, default=50,
                        help="Number of optimization trials to run.")
    parser.add_argument("--metric-mode", type=str, choices=['vad', 'sdr'], default='vad',
                        help="Metric to optimize: 'vad' for F1-score or 'sdr' for SI-SDR improvement.")
    parser.add_argument("--librimix-root", type=Path, default=LIBRIMIX_PATH,
                        help="Path to the LibriMix root directory for evaluation data.")
    
    args = parser.parse_args()

    print("--- STARTING HYPERPARAMETER TUNING ---")
    print(f"Tuning for model: {args.model_type}")
    print(f"Optimizing for: {args.metric_mode.upper()} metric")
    print(f"Number of trials: {args.trials}")
    print(f"Samples per trial: {args.samples}")
    print("-" * 40)

    study = optuna.create_study(direction="maximize")
    
    study.optimize(lambda trial: objective(trial, args), n_trials=args.trials, show_progress_bar=True)

    print("\n" + "="*40)
    print("      OPTIMIZATION FINISHED      ")
    print("="*40)
    
    if not study.trials:
        print("No trials were completed. Cannot determine the best parameters.")
        return

    print(f"Best trial for '{args.metric_mode}' metric:")
    best_trial = study.best_trial
    print(f"  Value (Metric Score): {best_trial.value:.4f}")
    
    print("  Best Parameters:")
    for key, value in best_trial.params.items():
        if isinstance(value, float):
            print(f"    - {key}: {value:.4f}")
        else:
            print(f"    - {key}: {value}")
            
    print("\nRECOMMENDATION:")
    print(f"Set SPEAKER_DETECTION_THRESHOLD = {best_trial.params['speaker_detection_threshold']:.4f}")
    print(f"Set SMOOTHING_WINDOW = {best_trial.params['smoothing_window']}")
    print("="*40)

if __name__ == "__main__":
    main()

# Own Voice Suppression

This directory contains scripts and modules for performing and evaluating "own voice suppression," a task that involves separating a target speaker's voice from a mixed audio stream or suppressing it entirely.

## Core Scripts

These are the main scripts for running and validating the suppression system.

### `source_separation.py`

This script performs source separation or suppression on a given audio file.

**What it does:**
-   **Separation Mode (default):** Separates a mixed audio file into two distinct tracks based on a target speaker's enrolment audio. The system uses the enrolment to determine which voice belongs to the target speaker and isolates it from other voices or background sounds.
-   **Suppression Mode (`--suppress`):** Removes the voice of the enrolled target speaker from the mixed audio, leaving the remaining background audio.

**How to run:**

You can run this script using `python -m own_voice_suppression.source_separation`.

**Command-line arguments:**
-   `--enrolment-path`: (Required) Path to the enrolment audio file for the target speaker.
-   `--mixed-path`: (Required) Path to the mixed audio file to be processed.
-   `--model-type`: The separation model to use. Choices: `convtasnet`, `sepformer`, `diart`. Default: `convtasnet`.
-   `--suppress`: A flag to enable suppression mode. If not set, the script runs in separation mode.
-   `--output-directory`: Directory to save the output audio files. Defaults to an auto-generated path in `own_voice_suppression/outputs/`.

**Example (Suppression):**
```bash
python -m own_voice_suppression.source_separation \
    --enrolment-path path/to/your_voice.wav \
    --mixed-path path/to/meeting_audio.wav \
    --model-type sepformer \
    --suppress
```

**Example (Separation):**
```bash
python -m own_voice_suppression.source_separation \
    --enrolment-path path/to/your_voice.wav \
    --mixed-path path/to/meeting_audio.wav \
    --model-type sepformer
```

### `validate_source_separation.py`

This script evaluates the performance of the speaker suppression system using the LibriMix dataset.

**What it does:**
-   Synthesizes mixed audio samples by combining speech from two different speakers from the LibriMix dataset.
-   Optionally adds background noise from the WHAM dataset at a specified Signal-to-Noise Ratio (SNR).
-   Runs the suppression pipeline on each synthesized mix to remove the target speaker.
-   Calculates and reports performance metrics, including:
    -   **SI-SDR Improvement (dB):** How much the Signal-to-Distortion ratio of the background improves.
    -   **Target Suppression (dB):** How much the target speaker's voice was attenuated.
    -   **Speech Intelligibility Index (SII):** A measure of the clarity of the remaining background speech.

**How to run:**

You can run this script using `python -m own_voice_suppression.validate_source_separation`.

**Command-line arguments:**
-   `--librimix-root`: (Required) Path to the root of the LibriMix dataset.
-   `--samples`: The number of audio samples to generate and evaluate. Default: 20.
-   `--model-type`: The separation model to evaluate. Choices: `convtasnet`, `sepformer`, `diart`. Default: `convtasnet`.
-   `--save-outputs`: If set, saves the resulting audio files and plots for inspection.
-   `--background-noise-db`: Adds background noise at a specified SNR in dB (e.g., `10`).

**Example:**
```bash
python -m own_voice_suppression.validate_source_separation \
    --librimix-root /path/to/LibriMix \
    --model-type sepformer \
    --samples 50 \
    --save-outputs
```

## Other Files

-   `audio_utils.py`: Utility functions for audio preprocessing, such as resampling and converting to mono.
-   `blind_pipeline.py`: An alternative, experimental pipeline for "blind" source separation that does not require a separate enrolment audio.
-   `direct_extraction.py`: An alternative pipeline that uses the SpeakerBeam model to directly extract the target speaker's voice.
-   `model_sizes.py`: A utility script to analyze and report metrics (Parameters, Size, TFLOPs) for the different models used in this project.
-   `plot_utils.py`: Helper functions for creating plots with `matplotlib`, used by the validation scripts to visualize results.
-   `requirements.txt`: A list of Python packages required to run the scripts in this directory.
-   `tune_threshold.py`: A script to automatically tune the `speaker_detection_threshold` and `smoothing_window` hyperparameters using Optuna.
-   `tune_window_and_stride.py`: A script to analyze how different window and stride sizes affect the Speech Intelligibility Index (SII) metric.
-   `validate_voice_detection.py`: A validation script focused specifically on the *speaker detection* component, measuring its performance with metrics like F1-score, precision, and recall.
-   `voice_detection.py`: A script that implements speaker detection using various embedding models (e.g., WavLM, ECAPA). It generates a suppression mask based on the confidence that a target speaker is present.

# Multispeaker Separation

This directory contains prototype code for speaker-count-aware separation and counting:

- A separation entrypoint that selects a ConvTasNet model based on estimated speaker count.
- A lightweight CRNN speaker-count subsystem (`count_speakers/`) for edge-oriented deployment.
- An optional pyannote-based counter wrapper for higher-quality offline counting.

## Current Status

This is research/prototype scaffolding and not yet a production training/inference package.

## Important Checkpoint Disclaimer

The files below are **dummy placeholder checkpoints** created for integration/smoke flow:

- `models/convtasnet_1spk.pth`
- `models/convtasnet_2spk.pth`
- `models/convtasnet_3spk.pth`
- `models/convtasnet_4spk.pth`
- `models/convtasnet_5spk.pth`

They were generated via `torch.save({}, path)` in `run_system.py` and do **not** contain trained ConvTasNet weights.
They are present only so the multi-model loading path can be wired and tested.

## Files

- `run_system.py`: smoke/integration harness and dummy model bootstrap.
- `speaker_counter.py`: pyannote diarization-based counting wrapper.
- `count_speakers/model.py`: compact CRNN definition for counting.
- `count_speakers/train.py`: CRNN training pipeline.
- `count_speakers/inference.py`: CRNN inference utilities.
- `models/speaker_count_crnn.pth`, `models/speaker_count_crnn.onnx`: CRNN artifacts.

## How This Fits the Pipeline

The intended flow is:

1. Estimate number of active speakers from short audio context.
2. Route mixture audio to the matching separation model (`k`-speaker variant).
3. Emit separated streams for downstream speaker identity/grouping + direction assignment.

In the current branch, step 2 model files are placeholders and should be replaced with real trained checkpoints before quantitative separation evaluation.

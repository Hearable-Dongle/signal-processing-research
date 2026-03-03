# Signal Processing Research

Main goals are denoising, own-voice suppression, speech separation, localization, and beamforming.

## Architecture Overview

The repo is organized as composable DSP/ML modules plus integration layers:

- `simulation/`: scene generation + microphone simulation (`pyroomacoustics`), used for offline validation.
- `localization/`: DOA estimation algorithms (including SRP-PHAT).
- `multispeaker_separation/`: multi-speaker separation and speaker-counting utilities (currently incomplete in this workspace).
- `speaker_identity_grouping/`: chunk-level separated-stream -> persistent `speaker_id` mapping.
- `direction_assignment/`: combines separated streams + identity + SRP peaks into `speaker_id -> direction` + beamforming weights.
- `beamforming/`: beamformer implementations and benchmark/evaluation utilities.
- `realtime_pipeline/`: integration/orchestration layer that wires fast + slow real-time paths together.

## Realtime Pipeline

Use [`realtime_pipeline/README.md`](./realtime_pipeline/README.md) for detailed architecture, runbook commands, sanity checks, and TODOs.

At a high level:

- Fast path (~10 ms): raw multichannel audio -> SRP-PHAT peaks -> low-latency beamforming using shared mapping table.
- Slow path (~200 ms): buffered multichannel audio -> separation -> identity grouping -> direction assignment -> shared mapping table update.

## Other Major Areas

- [denoise](./denoise): denoising work.
- [own_voice_suppression](./own_voice_suppression): own-voice suppression and related pipelines.
- [general_utils](./general_utils): reusable audio/data utilities.
- [hailo_demo](./hailo_demo): model deployment experiments for Hailo hardware.

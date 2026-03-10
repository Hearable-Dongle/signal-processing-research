# Signal Processing Research

Realtime, causal speech-enhancement research for multi-speaker scenes. The main integrated path is a fast localization + beamforming thread plus a slower separation / speaker-tracking thread.

## Realtime Stack

The repo is organized around these realtime subsystem layers:

1. `simulation/`
   - Scene generation, microphone simulation, and evaluation assets.
   - Used to generate causal test scenes and validate the integrated pipeline.

2. `localization/` and `realtime_pipeline/localization_backends.py`
   - Realtime DOA observation layer.
   - Implemented realtime backends:
     - `srp_phat_legacy`
     - `weighted_srp_dp`
     - `tiny_dp_ipd`
     - `music_1src`
     - `gcc_tdoa_1src`
   - Best current realtime multi-speaker default:
     - `weighted_srp_dp + multi_peak_v2`
   - Best current realtime single-source option:
     - `gcc_tdoa_1src`
   - Notes:
     - `tiny_dp_ipd` has useful diagnostics and score-map controls, but it is not the current recommended default for general multi-speaker realtime runs.
     - `music_1src` is experimental and not the practical default in this Python path.

3. `multispeaker_separation/` and `realtime_pipeline/separation_backends.py`
   - Speaker separation layer for the slow path.
   - Implemented integrated backends:
     - `multispeaker_separation` resolver path
     - Asteroid ConvTasNet fallback
     - `MockSeparationBackend` for deterministic tests only
   - Best current integrated realtime choice:
     - use the realtime resolver order in `realtime_pipeline`: `multispeaker_separation` when available, otherwise Asteroid ConvTasNet
   - Notes:
     - `multispeaker_separation/` is still prototype scaffolding in this workspace, so the integrated runtime may fall back to Asteroid depending on environment state.

4. `speaker_identity_grouping/`
   - Separated-stream to persistent-speaker assignment.
   - Implemented methods:
     - `mfcc_legacy`
     - `speaker_embed_session`
   - Best current realtime default:
     - `mfcc_legacy`
   - Notes:
     - `speaker_embed_session` is promising, but it is not yet the safer default for general realtime use.

5. `direction_assignment/`
   - Speaker-to-direction tracking and gain-target generation.
   - Implemented control modes:
     - `spatial_peak_mode`
     - `speaker_tracking_mode`
   - Best current stable baseline:
     - `spatial_peak_mode`
   - New development path:
     - `speaker_tracking_mode`
   - Notes:
     - `speaker_tracking_mode` is identity-led and DOA-corrected.
     - Large DOA relocks are persistence-gated to reduce wrong-speaker focus jumps.

6. `beamforming/` and `realtime_pipeline/fast_path.py`
   - Low-latency enhancement / steering layer.
   - Implemented realtime methods:
     - `mvdr_fd`
     - `gsc_fd`
     - `delay_sum`
   - Best current realtime default:
     - `mvdr_fd`
   - Current selected artifact-reduction defaults:
     - WOLA-style frequency-domain processing for MVDR/GSC
     - fractional-delay alignment for delay-and-sum
     - covariance update gating during likely speech-active frames
     - mild smoothed postfilter with gain floor `0.22`
   - Notes:
     - `mvdr_fd` is the preferred current method because it reduced static/choppy artifacts while preserving intelligibility better than the alternatives.

7. `realtime_pipeline/`
   - End-to-end integration/orchestration layer.
   - Fast path:
     - realtime localization
     - beamforming
     - low-latency output
   - Slow path:
     - separation
     - speaker identity grouping
     - direction assignment / speaker tracking
     - speaker-map publishing
   - Current integrated realtime defaults:
     - localization: `weighted_srp_dp`
     - tracking: `multi_peak_v2`
     - beamforming: `mvdr_fd`
     - control mode baseline: `spatial_peak_mode`

## Other Modules

- [own_voice_suppression](./own_voice_suppression): target-speaker suppression / extraction work.
- [denoise](./denoise): denoising experiments.
- [hailo_demo](./hailo_demo): Hailo deployment experiments.
- [general_utils](./general_utils): shared utilities.

## Recommended Reading

- [realtime_pipeline/README.md](./realtime_pipeline/README.md): integrated pipeline runbook and validation commands
- [beamforming/README.md](./beamforming/README.md): beamforming benchmarks and method comparison tooling
- [localization/README.md](./localization/README.md): localization algorithms and benchmark framework
- [direction_assignment/README_validation.md](./direction_assignment/README_validation.md): speaker-to-direction validation and plots

# RNNoise Postfilter Tuning

## Summary

Best current RNNoise postfilter state combines:
- input high-pass before RNNoise
- VAD-adaptive dry/wet blend
- final output high-pass
- final low-pass
- final notch

This document records:
1. the current default path in code
2. the best completed artifact-reduction trial so far

The focused artifact tuner currently exists at:
- `beamforming/benchmark/tune_rnnoise_artifacts_behind_poster.py`

## Current Default Params

These are the current default runtime settings:

```text
rnnoise_wet_mix = 0.9
rnnoise_input_highpass_enabled = true
rnnoise_input_highpass_cutoff_hz = 80.0
rnnoise_output_highpass_enabled = true
rnnoise_output_highpass_cutoff_hz = 70.0
rnnoise_output_lowpass_cutoff_hz = 7500.0
rnnoise_output_notch_freq_hz = 500.0
rnnoise_output_notch_q = 20.0
rnnoise_vad_adaptive_blend_enabled = true
rnnoise_vad_blend_gamma = 0.5
rnnoise_vad_min_speech_preserve = 0.15
rnnoise_vad_max_speech_preserve = 0.95
rnnoise_residual_highband_enabled = false
rnnoise_residual_highband_cutoff_hz = 3000.0
rnnoise_residual_highband_gain = 0.5
rnnoise_residual_jump_limit_enabled = false
rnnoise_residual_jump_limit_band_low_hz = 3000.0
rnnoise_residual_jump_limit_rise_db_per_frame = 4.0
rnnoise_residual_ema_enabled = false
rnnoise_residual_ema_alpha = 0.0
```

## Best Completed Artifact Reducer So Far

Best completed tuner output so far:
- root:
  - `beamforming/benchmark/_sanity_rnnoise_artifact_tuner_3skyncgm`
- best trial:
  - `beamforming/benchmark/_sanity_rnnoise_artifact_tuner_3skyncgm/trials/stage3_000`

This was a one-recording behind-poster sanity run on:
- `data-collection/nano-sympoium/nano-symposium-behind-poster/recordings/recording-3skyncgm`

Best params from that completed run:

```text
rnnoise_output_highpass_cutoff_hz = 60.0
rnnoise_vad_blend_gamma = 0.5
rnnoise_vad_max_speech_preserve = 0.9
rnnoise_residual_highband_enabled = true
rnnoise_residual_highband_cutoff_hz = 2500.0
rnnoise_residual_highband_gain = 0.35
rnnoise_residual_jump_limit_enabled = true
rnnoise_residual_jump_limit_band_low_hz = 3000.0
rnnoise_residual_jump_limit_rise_db_per_frame = 3.0
```

Associated summary:
- `beamforming/benchmark/_sanity_rnnoise_artifact_tuner_3skyncgm/best_valid_config.json`

Important note:
- `valid_found = false` in that run
- so this is the best artifact reducer from the completed sweep, not a fully accepted final setting

## Practical Recommendation

If the goal is the safest currently-supported default:
- keep the code defaults listed above

If the goal is strongest artifact suppression for manual listening tests:
- start from the `stage3_000` settings above
- apply them on behind-poster first

## Useful Output Roots

Baseline-style sanity output with the current improved RNNoise path:
- `beamforming/benchmark/_sanity_rnnoise_hpf_vadblend_outputhpf70_3skyncgm`

Completed artifact tuner root:
- `beamforming/benchmark/_sanity_rnnoise_artifact_tuner_3skyncgm`

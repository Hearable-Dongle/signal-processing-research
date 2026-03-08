# Realistic Conversations Simulation

This module generates multi-speaker conversational scenes for mic-array denoising and beamforming work while remaining compatible with the existing `simulation` renderer.

## Presets

- `quiet_room`
- `office`
- `cafe`
- `moving_speaker`
- `noisy_home`

Each preset produces:

- `audio/mic_array.wav`: rendered multichannel mixture
- `audio/mix_mono.wav`: mean microphone reference
- `audio/speaker_*_dry.wav`: dry per-speaker references
- `audio/background_noise_dry.wav`: dry noise reference
- `scene_config.json`: `simulation.SimulationConfig` compatible scene config
- `scenario_metadata.json`: seed, preset, room, mic geometry, utterance schedule, overlap intervals, render assets
- `frame_ground_truth.csv` and `frame_ground_truth.json`: 20 ms frame labels with active speakers, positions, overlap flags, and frame SNR
- `metrics_summary.json`: overlap/activity/SNR summary
- `event_schedule.json`: speech/noise/transient events

## Generate

```bash
python -m sim.realistic_conversations.generator \
  --preset office \
  --num-scenes 2 \
  --out-dir sim/output/realistic_conversations
```

Optional overrides:

- `--seed`
- `--duration-sec`
- `--sample-rate`
- `--frame-ms`
- `--asset-manifest`
- `--no-audio`

By default, assets are discovered under `/home/mkeller/data/librimix`. For tests or custom datasets, pass a manifest JSON:

```json
{
  "speech": [
    {"speaker_id": "spk_a", "path": "/abs/path/speaker_a.wav"},
    {"speaker_id": "spk_b", "path": "/abs/path/speaker_b.wav"}
  ],
  "noise": [
    {"category": "noise", "path": "/abs/path/noise.wav"}
  ]
}
```

## Evaluate

```bash
python -m sim.realistic_conversations.evaluate \
  --input sim/output/realistic_conversations/office_scene00
```

Or aggregate a full output root:

```bash
python -m sim.realistic_conversations.evaluate \
  --input sim/output/realistic_conversations
```

## Notes

- The generator biases toward single-speaker turns with short transition overlaps and low-probability backchannels.
- `moving_speaker` approximates motion by splitting the moving speaker into multiple rendered source segments with updated positions.
- `scene_config.json` points at generated absolute-path WAV assets; `simulation.SimulationSource` now accepts absolute audio paths for this use case.

# Direction Assignment Validation

Run synthetic multi-speaker validation with identity conditioning, metrics, plots, and audio artifacts.

## Command

```bash
python -m direction_assignment.validate \
  --num-scenes 5 \
  --duration-sec 8 \
  --speaker-choices 2,3,4 \
  --export-audio \
  --export-plots
```

Outputs are written to:
- `direction_assignment/output/validation_<timestamp>/`

## Identity Modes

- `--identity-mode online_only`
- `--identity-mode enroll_audio --enroll-audio-manifest <json>`
- `--identity-mode seed_embeddings --seed-embeddings-npz <npz>`

## Artifacts

- `summary.json`
- `per_scene_metrics.csv`
- `per_chunk_metrics.csv`
- `per_speaker_metrics.csv`
- Per scene:
  - `scenario_metadata.json`
  - `track_assignments.csv`
  - `audio/*.wav` (if `--export-audio`)
  - `plots/*.png` (if `--export-plots`)

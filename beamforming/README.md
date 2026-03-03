# Beamforming Simulation Project

This directory contains a frequency-domain beamforming simulation pipeline.

## Pipeline

1. Simulate multichannel mixture audio (speech + noise) and noise-only audio.
2. Compute STFT features and spatial covariance matrices.
3. Build steering vectors from either:
- oracle target locations/DOAs, or
- localization-estimated DOAs.
4. Solve beamformer weights:
- MVDR (iterative steepest/newton)
- MVDR (neural-mask)
- LCMV / GSC / GSC iterative
- weighted LCMV / weighted GSC variants
5. Reconstruct waveforms and evaluate RMSE / SNR / SI-SDR.

## Run

Install dependencies:

```bash
python -m venv beamforming-env
source beamforming-env/bin/activate
pip install -r beamforming/requirements.txt
```

Default run:

```bash
python -m beamforming.main
```

## Localization Steering Options

```bash
python -m beamforming.main \
  --steering-source both \
  --steering-time both \
  --steering-localization-default SSZ \
  --steering-localization-fallbacks GMDA
```

Useful flags:
- `--simulation-scene-file <path>` run one scene config directly.
- `--simulation-scene-dir <path> --max-scenes N` run multiple scene configs.
- `--localization-methods ...` override and explicitly test methods.
- `--dynamic-chunk-seconds <float>` chunk size for dynamic steering.
- `--target-weight-mode {equal,config}` default equal weighting.
- `--target-weights-file <json>` optional per-target weights.

Outputs include:
- `beamforming/output/<run>/steering_comparison.csv`
- `beamforming/output/<run>/<scene>/<scenario>/doa_tracking.csv`
- `beamforming/output/<run>/<scene>/<scenario>/scenario_metadata.json`
- audio files in both forms:
  - raw solver output (`*.wav`)
  - loudness-matched to reference (`*_norm_to_ref.wav`)

`steering_comparison.csv` reports both raw and normalized metrics:
- raw: `rmse_raw`, `snr_db_raw`, `si_sdr_db_raw`
- normalized: `rmse_norm`, `snr_db_norm`, `si_sdr_db_norm`

## Benchmark Over Library/Restaurant Scenes

```bash
python -m beamforming.benchmark.run \
  --preset quick \
  --steering-source both \
  --steering-time both
```

Benchmark defaults are best-only:
- `--steering-source localized`
- `--steering-time fixed`
- `--steering-localization-default SSZ`
- `--steering-localization-fallbacks GMDA`

To run a full sweep explicitly:

```bash
python -m beamforming.benchmark.run \
  --preset full \
  --steering-source both \
  --steering-time both
```

Benchmark outputs:
- `beamforming/benchmark/results/<run_id>/scene_metrics.csv`
- `beamforming/benchmark/results/<run_id>/summary_by_method.csv`
- `beamforming/benchmark/results/<run_id>/README_summary.md`

Run only one scene family:

```bash
python -m beamforming.benchmark.run --preset quick --scene-types library
python -m beamforming.benchmark.run --preset quick --scene-types restaurant
```

Run all configs from one scene directory directly:

```bash
python -m beamforming.main \
  --simulation-scene-dir simulation/simulations/configs/library_scene \
  --steering-source both \
  --steering-time both
```

# Localization Benchmark Guide

This benchmark compares localization methods across simulation scenes from:

- `simulation/simulations/configs/library_scene`
- `simulation/simulations/configs/restaurant_scene`

Targets are speech-only (`LibriSpeech/...` sources). Background noise emitters (`wham_noise/...`) are excluded from ground-truth scoring.

## Methods

Default benchmark methods:

- `SSZ`
- `SRP-PHAT`
- `GMDA`

Additional available methods:

- `MUSIC`
- `NormMUSIC`
- `CSSM`
- `WAVES`

## Metrics

Per scene:

- `mae_deg_matched`: mean angular error over matched target/prediction pairs
- `rmse_deg_matched`: RMSE angular error over matched pairs
- `median_ae_deg`: median absolute angular error over matched pairs
- `acc_within_5deg`, `acc_within_10deg`, `acc_within_15deg`: matched-pair accuracy at thresholds
- `recall = matched / n_true`
- `precision = matched / n_pred`
- `f1`: harmonic mean of precision and recall
- `misses`: unmatched true targets
- `false_alarms`: unmatched predicted sources

Matching uses one-to-one minimum-cost assignment with circular distance:

- `min(|a-b|, 360-|a-b|)`

## Presets

- `quick`: stratified sample per `(scene_type, k)` bucket
- `full`: all discovered scenes

## Outputs

Each run writes to:

- `localization/benchmark/results/<run_id>/raw_results.jsonl`
- `localization/benchmark/results/<run_id>/scene_metrics.csv`
- `localization/benchmark/results/<run_id>/summary_by_method.csv`
- `localization/benchmark/results/<run_id>/summary_by_scene_type.csv`
- `localization/benchmark/results/<run_id>/summary_by_k.csv`
- `localization/benchmark/results/<run_id>/README_summary.md`
- `localization/benchmark/results/<run_id>/overall_method_comparison.png`
- `localization/benchmark/results/<run_id>/scene_type_mae_comparison.png`
- `localization/benchmark/results/<run_id>/k_trends.png`

`localization/benchmark/results/latest` symlinks to the newest run.

## Commands

Quick benchmark:

```bash
python -m localization.benchmark.run --preset quick
```

Full benchmark (all scenes):

```bash
python -m localization.benchmark.run --preset full
```

Parallel workers can be controlled explicitly:

```bash
python -m localization.benchmark.run --preset full --workers 8
```

Regenerate reports from existing raw results:

```bash
python -m localization.benchmark.report \
  --results localization/benchmark/results/<run_id>/raw_results.jsonl \
  --out localization/benchmark/results/<run_id> \
  --run-id <run_id>
```

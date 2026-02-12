# Localization

Localization algorithms and benchmarking tools for simulated multi-speaker scenes.

## What Is Here

- Runtime localization pipeline: `python -m localization.main --config ...`
- Algorithms:
  - `SSZ` (`localization.algo.SSZLocalization`)
  - `SRP-PHAT` (`localization.algo.SRPPHATLocalization`)
  - `GMDA` (`localization.algo.GMDALaplace`)
  - `MUSIC` (`localization.algo.MUSICLocalization`)
  - `NormMUSIC` (`localization.algo.NormMUSICLocalization`)
  - `CSSM` (`localization.algo.CSSMLocalization`)
  - `WAVES` (`localization.algo.WAVESLocalization`)

AI localization modules/configs were removed. Valid `localization.type` values are now: `SSZ`, `SRP-PHAT`, `GMDA`, `MUSIC`, `NormMUSIC`, `CSSM`, `WAVES`.
- Benchmark framework:
  - `python -m localization.benchmark.run`
  - `python -m localization.benchmark.report`

## Run A Single Localization Config

```bash
python -m localization.main --config localization/configs/base_config.json
```

This runs simulation + localization and writes plots/audio to `localization/output/...`.

## Benchmark Across Library + Restaurant Scenes

Scene sources:

- `simulation/simulations/configs/library_scene`
- `simulation/simulations/configs/restaurant_scene`

Default benchmark policy:

- Ground truth targets: speech sources only (`LibriSpeech/...`)
- Methods: `SSZ`, `SRP-PHAT`, `GMDA`
- Optional methods: `MUSIC`, `NormMUSIC`, `CSSM`, `WAVES`
- Presets:
  - `quick`: stratified sample per `(scene_type, k)`
  - `full`: all scenes

Quick run:

```bash
python -m localization.benchmark.run --preset quick
```

Full run:

```bash
python -m localization.benchmark.run --preset full
```

Optional controls:

```bash
python -m localization.benchmark.run \
  --preset quick \
  --methods SSZ SRP-PHAT GMDA MUSIC NormMUSIC CSSM WAVES \
  --workers 8 \
  --seed 42 \
  --max-scenes 20
```

## Benchmark Outputs

Each run writes to `localization/benchmark/results/<run_id>/`:

- `raw_results.jsonl`
- `scene_metrics.csv`
- `summary_by_method.csv`
- `summary_by_scene_type.csv`
- `summary_by_k.csv`
- `README_summary.md`
- `overall_method_comparison.png`
- `scene_type_mae_comparison.png`
- `k_trends.png`

The symlink `localization/benchmark/results/latest` points to the newest run.

## Implemented Benchmark Components

Code added under `localization/benchmark/`:

- `run.py`: benchmark orchestration CLI
- `report.py`: CSV/markdown report generation CLI
- `scene_loader.py`: scene discovery and speech-target filtering
- `algo_runner.py`: unified execution for `SSZ`, `SRP-PHAT`, `GMDA`, `MUSIC`, `NormMUSIC`, `CSSM`, `WAVES`
- `matching.py`: circular-distance assignment matching
- `metrics.py`: per-scene comparable metrics
- `configs/default.json`: default methods, scene roots, and preset behavior

Tests added under `localization/tests/`:

- `test_benchmark_matching.py`
- `test_benchmark_metrics.py`
- `test_benchmark_scene_loader.py`

## Performance Overview

Latest generated summary is in:

- `localization/benchmark/results/latest/README_summary.md`

Current smoke-run snapshot (single scene, `SSZ` only):

| method | n_scenes | MAE(deg) | RMSE(deg) | Acc@10 | Recall | Precision | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| SSZ | 1 | 0.295 | 0.295 | 1.000 | 1.000 | 1.000 | 1.000 |

For full cross-method and all-scene numbers, run `--preset full` and use the generated `README_summary.md`.

## Validation Status

Validated in this workspace:

- Benchmark smoke run:
  - `python -m localization.benchmark.run --preset quick --max-scenes 1 --methods SSZ --seed 42`
- Report regeneration:
  - `python -m localization.benchmark.report --results localization/benchmark/results/20260211_231948/raw_results.jsonl --out localization/benchmark/results/20260211_231948 --run-id 20260211_231948`
- Compile check:
  - `python -m compileall localization/benchmark localization/tests`

Note: `pytest` is not installed in the current environment, so the new unit tests are present but were not executed here.

## Benchmark Metric Definitions

See `localization/README_BENCHMARK.md`.

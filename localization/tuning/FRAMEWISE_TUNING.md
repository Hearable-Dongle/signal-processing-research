# Framewise Optuna Tuning

This tuning path targets the realtime-style framewise localization problem on
`testing_specific_angles`, not the older scene-level localization benchmark.

## What It Tunes

Supported methods:

- `SRP-PHAT`
- `GMDA`
- `SSZ`

Shared search space:

- `window_ms`: `50..300`, step `50`
- `hop_ms`: `30..100`, step `5`
- `overlap`: `0.25..0.90`, step `0.05`
- `freq_low_hz`: `100..1500`, step `50`
- `freq_high_hz`: `max(freq_low_hz + 400, 1500)..6000`, step `100`

Method-specific search space:

- `GMDA`: `power_thresh_percentile`, `mdl_beta`
- `SSZ`: `epsilon`, `d_freq`

Primary optimization target:

- weighted score dominated by lower MAE and higher `Acc@25`

## Single Runner

Run one or more methods directly:

```bash
PYTHONPATH=. python -m localization.tuning.run_optuna_framewise_respeaker_v3 \
  --methods SRP-PHAT GMDA SSZ \
  --duration-min 60 \
  --subset-per-bucket 1 \
  --top-n-full-eval 5 \
  --srp-workers 10 \
  --gmda-workers 8 \
  --ssz-workers 4 \
  --trial-scene-workers 2
```

For a quick smoke run:

```bash
PYTHONPATH=. python -m localization.tuning.run_optuna_framewise_respeaker_v3 \
  --methods SRP-PHAT \
  --duration-min 0.02 \
  --subset-per-bucket 1 \
  --max-scenes 1 \
  --top-n-full-eval 1 \
  --srp-workers 1 \
  --trial-scene-workers 1
```

## Parallel Launcher

Launch one coordinator per method:

```bash
PYTHONPATH=. python -m localization.tuning.launch_optuna_framewise_methods \
  --methods SRP-PHAT GMDA SSZ \
  --duration-min 120 \
  --subset-per-bucket 1 \
  --top-n-full-eval 5 \
  --srp-workers 10 \
  --gmda-workers 8 \
  --ssz-workers 4 \
  --trial-scene-workers 2
```

This keeps one SQLite study per method and writes `process_status.json` so you
can monitor trial counts while the coordinators run.

## Output Layout

Outputs are written under:

- `localization/tuning/results/framewise_respeaker_v3_optuna/<run_id>/`

Per run:

- `run_manifest.json`
- `dashboard_commands.txt`
- `overall_best_configs.csv`
- `overall_best_configs.json`
- `worker_activity.csv`

Per method:

- `study.db`
- `study_manifest.json`
- `trials_export.csv`
- `trials_export.json`
- `best_trials.json`
- `best_params.json`
- `best_full_eval.csv`
- `best_full_eval.json`
- `best_scene_rows.csv`
- `best_timeline_rows.csv`

The `latest` symlink in the output root points at the newest run.

## Optuna Dashboard

Each run writes ready-to-use commands to `dashboard_commands.txt`.

Example:

```bash
optuna-dashboard sqlite:////abs/path/to/localization/tuning/results/framewise_respeaker_v3_optuna/<run_id>/srp_phat/study.db
```

## Resume Behavior

Studies are resumable because they use SQLite storage. Re-running with the same
method output directory reuses the existing DB and continues the study.

## Adding A New Method

To add a new method:

1. Add the method name to `METHODS` in
   `localization/tuning/run_optuna_framewise_respeaker_v3.py`.
2. Ensure the method exists in
   `localization/benchmark/configs/default.json`.
3. Extend `_default_trial_params()` if the method needs non-default seed params.
4. Extend `_suggest_eval_config()` for method-specific hyperparameters.
5. If needed, add method-specific worker routing in the launcher.

The framewise evaluator itself lives in:

- `localization/tuning/framewise_common.py`

That module is the right place to extend scoring or export behavior.

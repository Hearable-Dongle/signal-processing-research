# Realtime Pipeline

Integration/orchestration layer for real-time multi-speaker spatial processing.

## Purpose

`realtime_pipeline` connects existing modules into a two-thread pipeline:

- Fast path (`~10 ms` frame cadence, CPU):
  - read raw mic-array frames
  - run SRP-PHAT on rolling mixed audio window
  - produce candidate DOA peaks
  - apply low-latency beamforming with per-speaker gain weights from shared state
    - `mvdr_fd` (default)
    - `gsc_fd`
    - `delay_sum`
- Slow path (`~200 ms` chunk cadence):
  - buffer raw multichannel audio
  - run source separation backend
  - run speaker identity grouping
  - run direction assignment (with SRP peaks)
  - publish updated `speaker_id -> (direction_deg, gain_weight)` map

## Package Layout

- `contracts.py`: shared dataclasses/config (`PipelineConfig`, SRP snapshot, mapping entries).
- `shared_state.py`: thread-safe snapshot state and runtime stats.
- `srp_tracker.py`: rolling-window SRP-PHAT tracker wrapper.
- `fast_path.py`: fast-thread worker and delay-and-sum frame processor.
- `slow_path.py`: slow-thread worker (separation -> identity -> direction assignment).
- `separation_backends.py`: separation backend protocol + implementations + resolver.
- `orchestrator.py`: `RealtimeSpeakerPipeline` lifecycle.
- `simulation_runner.py`: simulation E2E runner + CLI.
- `robustness_validate.py`: baseline-vs-robust validation across localization, grouping, direction assignment, and full pipeline.
- `sanity_checks.py`: compile/import/mock-E2E validation report generator.

## Backend Strategy

Real separation backend resolver order is configurable:

1. `multispeaker_separation` backend (preferred when available)
2. Asteroid ConvTasNet backend (fallback)

If neither exists, resolver raises an actionable error with probe details.

For deterministic simulation validation, use `MockSeparationBackend`.

## Runbook

Run commands from repo root.

### 1) Install test runner (current env)

```bash
python -m pip install -U pytest
```

### 2) Run realtime pipeline unit/integration tests

```bash
PYTHONPATH=. pytest -q tests/test_realtime_pipeline_shared_state.py
PYTHONPATH=. pytest -q tests/test_realtime_pipeline_backend_resolver.py
PYTHONPATH=. pytest -q tests/test_realtime_pipeline_integration_mock.py
```

### 3) Generate validation report (compile + imports + backend probe + mock E2E)

```bash
python -m realtime_pipeline.sanity_checks \
  --scene-config simulation/simulations/configs/library_scene/library_k2_scene00.json \
  --out-dir realtime_pipeline/output/validation
```

Outputs:
- `realtime_pipeline/output/validation/validation_report.json`
- `realtime_pipeline/output/validation/mock_smoke/*`

### 4) Run simulation E2E (mock backend)

```bash
python -m realtime_pipeline.simulation_runner \
  --scene-config simulation/simulations/configs/library_scene/library_k2_scene00.json \
  --out-dir realtime_pipeline/output/sim_run
```

Outputs:
- `summary.json`
- `enhanced_fast_path.wav`

`summary.json` includes realtime latency decomposition:
- `fast_rtf`, `slow_rtf`
- `fast_stage_avg_ms`: SRP, beamforming, output safety, sink, slow-queue enqueue
- `slow_stage_avg_ms`: separation, identity grouping, direction assignment, speaker-map publish

Robust tracking controls are enabled by default in this path:
- SRP peak smoothing, hysteresis, hold, and confidence gating
- identity continuity bonus, anti-switch margin, and weak-evidence carry-forward
- direction-assignment transition penalties, low-confidence hold, stale decay
- slow-path speaker-map hold/decay to avoid abrupt weak-evidence remaps

Disable them for ablation:

```bash
python -m realtime_pipeline.simulation_runner \
  --scene-config simulation/simulations/configs/library_scene/library_k2_scene00.json \
  --out-dir realtime_pipeline/output/sim_run_baseline \
  --disable-robust-mode
```

### Runtime focus/boost control

`RealtimeSpeakerPipeline` now supports atomic runtime control updates:

```python
pipe.set_focus_control(
  focused_speaker_ids=[2],      # or None
  focused_direction_deg=None,   # or angle in degrees
  user_boost_db=8.0,            # clamped to [0, config.max_user_boost_db]
)
```

Behavior:
- if `focused_speaker_ids` is set, those speakers receive focus gain and others are attenuated
- else if only `focused_direction_deg` is set, nearest tracked speaker is focused
- if neither is set, all speaker gains default to `1.0`
- fast path applies soft clipping by default and optional RMS normalization (`PipelineConfig.output_target_rms`)
  - attenuation-only normalization is default (`output_allow_amplification=False`)

### 5) Run simulation E2E with real backend resolution

```bash
python -m realtime_pipeline.simulation_runner \
  --scene-config simulation/simulations/configs/library_scene/library_k2_scene00.json \
  --out-dir realtime_pipeline/output/sim_run_real \
  --real-separation
```

Note: this requires either a working `multispeaker_separation.inference` implementation or Asteroid ConvTasNet installed/configured.

### 6) Validate-only via simulation runner CLI

```bash
python -m realtime_pipeline.simulation_runner \
  --scene-config simulation/simulations/configs/library_scene/library_k2_scene00.json \
  --out-dir realtime_pipeline/output/validation_cli \
  --validate-only
```

### 7) Focus amplification sanity check (causal realtime)

Runs a small stratified subset of benchmark scene roots (`library` + `restaurant` by default), bootstraps focus by location, then locks to tracked speaker IDs during runtime.

```bash
python -m realtime_pipeline.focus_sanity_check \
  --beamforming-config beamforming/benchmark/configs/default.json \
  --scene-types library restaurant \
  --scenes-per-type 3 \
  --scene-repeats 3 \
  --focus-ratio 2.0 \
  --out-dir realtime_pipeline/output/focus_sanity
```

Outputs:
- `summary.json`
- `per_scene_metrics.json`
- per-scene `selection_trace.csv`
- per-scene `enhanced_fast_path.wav`

### 8) Robustness benchmark by speaker-count bucket

Runs baseline vs robust comparisons for localization, speaker grouping, direction assignment, and the full realtime pipeline. Outputs are timestamped and bucketed by `1`, `2`, and `3+` active speakers.

```bash
PYTHONPATH=. python -m realtime_pipeline.robustness_validate \
  --scenes-per-bucket 1 \
  --synthetic-scenes-per-bucket 2
```

Outputs:
- `realtime_pipeline/output/robustness_validation/<run_id>/summary.json`
- `realtime_pipeline/output/robustness_validation/<run_id>/per_scene_metrics.csv`
- `realtime_pipeline/output/robustness_validation/<run_id>/summary_by_bucket.csv`
- stage folders for `localization`, `grouping`, `direction_assignment`, `pipeline`
- per-scene CSV traces and `.png` comparison plots

Catchup/adaptation metrics in per-scene summaries:
- `startup_lock_ms`: time to first lock from direction bootstrap
- `reacquire_catchup_ms_median`: median re-lock latency after timeout-based reacquire
- `nearest_change_catchup_ms_median`: latency to align lock with stable nearest-speaker changes

## Expected Sanity Signals

- `summary.json` should show:
  - `fast_frames > 0`
  - `slow_chunks > 0`
  - `speaker_map_updates > 0`
- `validation_report.json` should show:
  - `overall_ok: true`
  - `mock_e2e.ok: true`
- with focus boost enabled, `SpeakerGainDirection.gain_weight` for the focused speaker should increase measurably
- clipping should remain bounded by fast-path soft clipper (no hard overflow spikes)

## Current Limitations

- Fast path uses a simple delay-and-sum beamformer, not weighted LCMV/GSC.
- Real separation backend availability depends on local environment state.
- Tests currently rely on `PYTHONPATH=.` unless package install/import path is normalized.

## Next TODO Tasks

1. Add packaging/test-path cleanup so `pytest` runs without `PYTHONPATH=.`.
2. Add a reusable real-time audio source/sink abstraction for live hardware input.
3. Add performance gates (`p95` fast-path latency budget checks) to sanity checks.
4. Add integration tests for slow-path queue overflow and backpressure behavior.
5. Add structured logs/metrics export (per-thread timings, drop counts, mapping update lag).
6. Add real-backend smoke test target that can be toggled in CI when dependencies are present.
7. Add configurable beamforming mode switch (delay-sum vs existing weighted beamforming stack).
8. Add progress logging and resumability to the robustness benchmark runner.
9. Add end-to-end artifact index (`manifest.json`) for generated outputs.
10. Document recommended environment setup for `multispeaker_separation` backend restoration.

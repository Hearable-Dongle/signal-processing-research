# Monolithic ConvTasNet `.hef` Bring-Up Plan (RPi + Hailo-8)

## Summary
Goal: find the **first monolithic full-model ConvTasNet `.hef`** that compiles and runs on the Hailo-8 (RPi target), then validate one real audio separation pass.

Chosen strategy (locked):
- Search goal: **first compile only**
- Time budget: **2-hour focused ladder**
- Milestone acceptance: **compile + one audio run**

This plan is decision-complete for a new implementation agent.

## Helpful Context Files
Read these before implementing:
- `hailo/README.md` (current runbook and metric conventions)
- `hailo/RUNNING_TODO_CONVTAS_HEF.md` (historical status + evidence paths)
- `hailo/workarounds.md` (what has already been tried)
- `hailo/NEXT_AGENT_FEB_22.md` (allocator-failure history and interpretation)
- `hailo/hef_tiled_decoder_path.py` (runtime executor pattern for block HEFs)
- `hailo/masker_tiled_path.py` (masker decomposition + manifests)
- `hailo/hef_tiled_full_chain.py` (end-to-end stitched parity structure)
- `hailo/hailo_runtime_runner.py` (HEF runtime helper for `hailo_platform`)
- `hailo/scripts/hailo_test_hef_tiled_decoder_path.sh`
- `hailo/scripts/hailo_test_hef_tiled_masker_path.sh`
- `hailo/scripts/hailo_test_hef_tiled_full_chain.sh`
- `hailo/librimix_hailo_forward_example.py` (real LibriMix forward example flow)
- `general_utils/constants.py` (source of `LIBRIMIX_PATH`)

## Edit Guardrails (Must Follow)
Allowed to edit:
- Anything under `hailo/`
- Anything under `hailo-asteroid/` (Hailo-specific model/module work is allowed here)

Do not edit:
- Anything under `asteroid/` (baseline non-Hailo tree must remain untouched)

Additional rules:
- Keep Hailo architecture changes in `hailo_*` files/modules when modifying `hailo-asteroid/`.
- Do not refactor baseline model definitions used by non-Hailo workflows.
- Keep run artifacts in `hailo/module_runs/<run_ts>/` and avoid rewriting historical evidence.

## Scope
In scope:
- Add scripts to run deterministic monolithic sweeps (`ONNX -> HAR -> HEF`) for `hailo_convtasnet_k1`-style full graph.
- Auto-capture compile outcomes and failure signatures.
- Add one runtime execution script on RPi to run compiled monolithic `.hef` on a real LibriMix sample.

Out of scope (for this milestone):
- Achieving near-baseline model size/quality.
- Replacing stitched-block deployment path.
- Full parity guarantees for monolithic model.

## Success Criteria
A run is successful when all are true:
1. `har_success=true` for at least one monolithic config.
2. `hef_success=true` for that config.
3. Same `.hef` runs on RPi/Hailo-8 with one real LibriMix sample and outputs 2 separated sources.
4. Run artifacts are written under `hailo/module_runs/<run_ts>/` with summary TSV + logs.

## Public Interfaces / Files to Add
1. `hailo/scripts/hailo_test_monolithic_full_to_hef.sh`
- Purpose: deterministic 2-hour ladder for monolithic full-model compile.
- Inputs (env vars):
  - `HAILO_RUN_TS` optional
  - `PROFILE=focused|extended` default `focused`
  - `TIME_LIMIT_MIN` default `120`
  - `CALIB_NPZ` default `hailo/calibration_1000ms_16k_64.npz`
  - `HW_ARCH` default `hailo8`
  - `QUICK_OPT=1|0` default `1`
- Outputs:
  - `hailo/module_runs/<run_ts>/monolithic_full_hef_summary.tsv`
  - per-case `*.log`, `*_hef.log`, `*_compile_failure.txt`, `*_compile_failure.txt.json`
  - produced `*.har`, `*.hef`

2. `hailo/monolithic_full_search.py`
- Purpose: generate case matrix + execute export/compile commands + normalize outcomes.
- Interface:
  - CLI args mirror script env vars (for direct Python use).
- TSV schema:
  - `run_tag, n_filters, bn_chan, hid_chan, skip_chan, n_blocks, n_repeats, input_len, har_success, hef_success, exception_type, failure_head, har_path, hef_path, har_log, hef_log, failure_log`

3. `hailo/scripts/hailo_run_monolithic_hef_librimix.sh`
- Purpose: run one compiled monolithic `.hef` on target RPi/Hailo-8.
- Inputs:
  - `HEF_PATH` required
  - `MIX_WAV` optional (auto-pick from `LIBRIMIX_PATH` if unset)
  - `LIBRIMIX_ROOT` optional
  - `OUT_DIR` optional
- Outputs:
  - `mix.wav`, `sep_src1.wav`, `sep_src2.wav`
  - `runtime_metrics.json` (latency, rtf, output lengths, sample rate)

4. `hailo/monolithic_hef_runtime_infer.py`
- Purpose: minimal runtime inference entrypoint using `hailo_platform` (`pyhailort`) and monolithic `.hef`.
- Notes:
  - Reuse `hailo/hailo_runtime_runner.py` where possible.
  - Must support single-input/single-output or explicit stream selection flags.

5. `hailo/MONOLITHIC_HEF_NEXT_AGENT.md`
- Purpose: short execution runbook and handoff for future agents.
- Include:
  - exact commands
  - expected pass/fail signatures
  - latest passing case pointer

## Implementation Plan

### Phase 1: Monolithic Compile Sweep (Focused Ladder)
Use a fixed ladder, stop-on-first-success enabled by default.

Ordered configs (descending complexity):
1. `(n_filters=256, bn=128, hid=256, skip=128, blocks=2, repeats=1, input_len=16000)`
2. `(256, 96, 192, 96, 2, 1, 16000)`
3. `(256, 64, 128, 64, 2, 1, 16000)`
4. `(192, 64, 128, 64, 2, 1, 16000)`
5. `(128, 64, 128, 64, 2, 1, 16000)`
6. `(128, 48, 96, 48, 2, 1, 16000)`
7. `(128, 32, 64, 32, 2, 1, 16000)`
8. `(96, 32, 64, 32, 2, 1, 16000)`
9. `(96, 24, 48, 24, 1, 1, 16000)`
10. `(64, 24, 48, 24, 1, 1, 16000)`

Rules:
- `QUICK_OPT=1` during search.
- Use `--input_length_policy error`.
- Hard timeout per case (e.g., 12 min); abort case and continue.
- If `hef_success=true`, mark winner and optionally stop unless `PROFILE=extended`.

### Phase 2: Failure Classification + Next-Knob Logic
For each failing case:
- Parse `exception_type` and `failure_head`.
- Bucket:
  - `BackendAllocatorException/Agent infeasible`
  - shape issues
  - translation/export failures
- If all fail by allocator:
  - next knob order: reduce `hid_chan`, then `bn_chan/skip_chan`, then `n_filters`, then `n_blocks`.
- Persist chosen “next knob” in summary footer or sidecar JSON.

### Phase 3: RPi Runtime Validation of Winning Monolithic HEF
On target RPi:
1. Validate `hailo_platform` import and device presence.
2. Run `hailo_run_monolithic_hef_librimix.sh` with winning `.hef`.
3. Write `runtime_metrics.json` with:
- `device_info`
- `input_samples`, `output_samples`
- `latency_ms`
- `rtf`
- `ok=true/false`
4. Save separated outputs for quick listening.

## Test Cases / Scenarios

### Compile-side tests
1. Case row creation:
- Verify all configured cases are emitted in TSV even if timed out.
2. HAR pass + HEF fail:
- Ensure failure logs and JSON are generated and paths are non-empty.
3. First HEF success:
- Ensure script marks winner row and exits early in focused mode.
4. No HEF success:
- Script exits non-zero and prints “no monolithic hef success”.

### Runtime-side tests (RPi)
1. Missing runtime package:
- Friendly error: “install pyhailort/hailo_platform”.
2. HEF stream mismatch:
- Clear error listing available stream names.
3. Successful run:
- Produces `sep_src1.wav`, `sep_src2.wav`, and `runtime_metrics.json` with `ok=true`.

## Edge Cases / Failure Modes
- Calibration mismatch: case marked fail with shape signature.
- Timeout: mark `exception_type=Timeout`.
- Multiple stream names: require explicit selection fallback.
- Very long input wav: trim or process fixed window (default 1–2 s) for milestone validation.

## Reporting Format (must be printed at end)
1. `HAR status: <count_pass>/<count_total>`
2. `HEF status: <count_pass>/<count_total> ; winner=<run_tag|none>`
3. `Runtime status: pass/fail ; output_dir=<path|none>`
4. `Next knob: <single explicit change>`

## Assumptions / Defaults
- Target board has Hailo-8 connected and `hailo_platform` available.
- `general_utils/constants.py` `LIBRIMIX_PATH` is valid on RPi, otherwise `MIX_WAV` is explicitly provided.
- First milestone prioritizes feasibility over quality.
- Existing stitched HEF flow remains untouched as fallback path.

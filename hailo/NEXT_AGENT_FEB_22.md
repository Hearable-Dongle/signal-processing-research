# NEXT_AGENT_FEB_22

## Date
- February 22, 2026

## TL;DR
- `HAR` generation is working end-to-end for the staged `hailo_*` ConvTasNet path.
- Failure is at `HAR -> HEF` compile (allocator/mapping), not at ONNX export or HAR translation.
- The `dw66/conv68 Agent infeasible` spam means the compiler cannot place a feasible mapping for the graph on Hailo8 resources.

## What Has Worked (Confirmed)

### 1) Step-by-step module HAR pipeline passed
- Summary: `hailo/module_runs/20260218_223015/summary.tsv`
- All rows are `har_success=true` for stage1-5.
- Includes:
  - primitives
  - conv block
  - tdconvnet
  - encoder/decoder
  - `s5_hailo_convtasnet_k1`

### 2) Progressive full-model HAR pipeline passed
- Summary: `hailo/module_runs/fullprog_223623/summary.tsv`
- All rows are `har_success=true` for:
  - `s6_full_bypass_k0_r1b2`
  - `s6_full_bypass_k0_r2b4`
  - `s6_full_bypass_k0_r3b8`
  - `s6_full_normal_k0_r3b8`

### 3) Hailo-prefixed model/module files exist and are wired
- `hailo-asteroid/asteroid/models/hailo_conv_tasnet.py`
- `hailo-asteroid/asteroid/masknn/hailo_convolutional.py`
- `hailo-asteroid/asteroid/masknn/hailo_norms.py`
- `hailo-asteroid/asteroid/masknn/hailo_activations.py`

## What Is Failing (Current Blocker)

### HEF compile fails with allocator infeasibility
- Failure artifact: `hailo/compile_failure.txt`
- Structured failure: `hailo/compile_failure.txt.json`
- Exception class: `BackendAllocatorException`
- Tail symptom:
  - repeated `dw66 errors: Agent infeasible`
  - repeated `conv68 errors: Agent infeasible`
  - `auto_spatial_reshape_from_dw66_to_conv69_* ... Agent infeasible`
  - then `Mapping Failed (Timeout, allocation time: 31m 42s)`

Interpretation:
- This is a compiler placement/resource problem (mapping stage), not an unsupported-op translation problem.
- The graph is valid enough to translate to HAR; it becomes infeasible when quantized graph is mapped to hardware resources.

## Important Clarification (for confusion seen in terminal)
- `.har` is **not** executable on-device; it is an intermediate representation.
- `.hef` is the hardware executable package used for deployment.
- So: `ONNX -> HAR` passing does **not** guarantee `HAR -> HEF` will pass.

## Root-Cause Signals Seen So Far
- Very deep repeated DW/Conv chain with many reshape boundaries (auto spatial reshape helper failures).
- Failure cluster appears late in stack (e.g., around `dw66/conv68`, sometimes `dw72/conv74` depending on build).
- This pattern is consistent with resource fragmentation / infeasible assignment at target depth/width.

## Guardrails: What Next Agent Can/Cannot Touch

### Can touch
- `hailo-asteroid/asteroid/models/hailo_*.py`
- `hailo-asteroid/asteroid/masknn/hailo_*.py`
- `hailo/scripts/*.sh`
- `hailo/*.py` (tooling/export/compile scripts)
- New docs under `hailo/*.md`

### Cannot touch (unless explicitly requested by user)
- Non-Hailo baseline model files in `hailo-asteroid` (e.g. original non-prefixed ConvTasNet modules)
- Existing non-Hailo training/inference codepaths outside Hailo migration scope
- Historical run artifacts/logs except adding new run folders/files

Rule:
- Any architecture changes in `hailo-asteroid` must stay in new `hailo_*` files so baseline remains intact and diffable.

## Immediate Next Steps to Unblock HEF

### 1) Reproduce HEF failure on a pinned target HAR
- Target first: `s6_full_normal_k0_r3b8.har`
- Use current command pattern with `--quick_opt` to shorten iterations.
- Always save `compile_failure.txt` and `.json` per run directory (avoid overwriting single global file).

### 2) Add deterministic compile sweep script (small architecture knobs)
- Create/extend script to generate+compile a short matrix around full model:
  - `hid_chan`: `256 -> 192 -> 128`
  - `bn_chan`: `128 -> 96 -> 64`
  - `n_blocks`: `8 -> 6 -> 4`
  - keep `n_repeats=3` initially, then reduce to `2` if needed
- Goal: find first HEF-success frontier, not final best quality.

### 3) Prioritize reducing late-depth pressure
- Keep op types the same (HAR is already passing).
- Reduce channels and/or blocks before introducing new topology changes.
- Avoid adding normalization variants that expand graph complexity.

### 4) Make failure artifacts run-scoped
- Current global `hailo/compile_failure.txt` is repeatedly overwritten and causes confusion.
- Save as `<run_dir>/compile_failure.txt` and `<run_dir>/compile_failure.txt.json` for each attempt.

### 5) After first HEF-success config is found
- Freeze that as `stage7_hef_baseline`.
- Then scale one axis at a time back toward target model.

## Known Tooling Pitfall
- `har_to_hef.py` now includes input-layer auto-detection fallback.
- If auto-detection picks the wrong input in a multi-input graph, pass `--input_layer_name` explicitly.

## Minimal Runbook

### HAR stages
```bash
./hailo/scripts/hailo_test_all.sh
```

### HAR + HEF sweep (new stage7 helper)
```bash
./hailo/scripts/hailo_test_stage7_hef_sweep.sh
```

### Compile one HAR to HEF
```bash
./hailo/scripts/hailo_har_to_hef.sh \
  hailo/module_runs/fullprog_223623/s6_full_normal_k0_r3b8.har \
  hailo/module_runs/fullprog_223623/s6_full_normal_k0_r3b8.hef \
  s6_full_normal_k0_r3b8
```

### Direct Python compile (quick iteration)
```bash
hailo/to-hailo-env/bin/python -m hailo.har_to_hef \
  hailo/module_runs/fullprog_223623/s6_full_normal_k0_r3b8.har \
  hailo/module_runs/fullprog_223623/s6_full_normal_k0_r3b8.hef \
  --model_name s6_full_normal_k0_r3b8 \
  --hw_arch hailo8 \
  --calib_npz hailo/calibration_1000ms_16k_64.npz \
  --quick_opt
```

## Implemented In This Iteration
- Added `hailo/scripts/hailo_test_stage7_hef_sweep.sh`:
  - Builds HAR + attempts HEF on a deterministic config ladder.
  - Writes run-scoped outputs under `hailo/module_runs/<run_ts>/`.
  - Writes per-case failure artifacts (`*_compile_failure.txt`) and `hef_summary.tsv`.
- Updated `hailo/har_to_hef.py`:
  - Added input-layer auto-detection fallback (avoids strict dependence on `--model_name/input_layer1`).
  - Added optional `--input_layer_name` override for explicit control.
  - Ensures failure-log parent directories are created automatically.

## Reporting Format (for future updates)
For every run, report in 3 lines:
1) `HAR status:` pass/fail + summary path
2) `HEF status:` pass/fail + first failing layer family (`dwXX/convYY/...`)
3) `Next knob:` exact single change to test next

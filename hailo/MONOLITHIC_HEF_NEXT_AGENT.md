# Monolithic HEF Next Agent

## Scope
This runbook is for monolithic ConvTasNet `ONNX -> HAR -> HEF` search only.
Runtime execution is intentionally separate.

## Compile Command
```bash
PROFILE=focused \
TIME_LIMIT_MIN=120 \
CALIB_NPZ=hailo/calibration_1000ms_16k_64.npz \
HW_ARCH=hailo8 \
QUICK_OPT=1 \
./hailo/scripts/hailo_test_monolithic_full_to_hef.sh
```

Artifacts are saved at:
- `hailo/module_runs/<run_ts>/monolithic_full_hef_summary.tsv`
- per case: `*.har`, `*.hef`, `*.log`, `*_hef.log`, `*_compile_failure.txt`, `*_compile_failure.txt.json`

## Expected Signatures
- Allocator failure (common):
  - `exception_type=BackendAllocatorException`
  - `failure_head` contains `Agent infeasible` and/or `Mapping Failed`
- Timeout:
  - `exception_type=Timeout`

## Runtime Command (Target Device Only)
Run only on machine with `hailo_platform` / `pyhailort`:
```bash
HEF_PATH=<winner.hef> ./hailo/scripts/hailo_run_monolithic_hef_librimix.sh
```

## Latest Passing Case Pointer
- No winner recorded yet in this runbook.
- Read winner from: `hailo/module_runs/<run_ts>/monolithic_full_hef_summary.tsv` (`hef_success=true`).

## Prefix Ramp Isolation (Add Modules Until Failure)
Use this to identify the first monolithic failure boundary by progressively increasing active model depth.

```bash
TIME_LIMIT_MIN=60 \
CASE_TIMEOUT_MIN=10 \
N_FILTERS=256 BN_CHAN=128 HID_CHAN=256 SKIP_CHAN=128 \
N_BLOCKS=2 N_REPEATS=1 INPUT_LEN=16000 \
./hailo/scripts/hailo_test_monolithic_prefix_ramp.sh
```

Artifacts:
- `hailo/module_runs/<run_ts>/monolithic_prefix_ramp_summary.tsv`
- optional component stress attribution: `hailo/module_runs/<run_ts>/monolithic_component_attribution.tsv`

Interpretation:
- Stage A (`A_masker_depth`) sweeps active masker depth (`truncate_k_blocks=1..N`).
- Stage B (`B_block_count`) increases total block count (`n_blocks=1..4`).
- Stage C (`C_repeat_count`) increases repeats (`n_repeats=1..3`).
- First failing row is the mapping frontier.

# Running TODO: ConvTasNet -> HEF

Last updated: 2026-02-23 (run_ts=20260223_082809)
Owner: current agent/user session

## Current Goal
- [ ] Get all top-level ConvTasNet modules to `.hef` independently.
- [ ] Get smallest realistic full `hailo_convtasnet` (`n_blocks=1`, `n_repeats=1`, `time_len=16000`) to `.hef`.
- [ ] Scale up depth/channels from smallest passing full model.

## Locked Decisions
- Module gating granularity: top-level modules only.
- Smallest-full target: real-length (`time_len=16000`), not short-length shortcut.
- Evidence format: every checkmark must include concrete artifact paths.

## Scripts (Source of Truth)
- Module gate: `hailo/scripts/hailo_test_modules_to_hef.sh`
- Smallest full: `hailo/scripts/hailo_test_smallest_full_to_hef.sh`
- Scale-up ladder: `hailo/scripts/hailo_test_stage7_hef_sweep.sh`

## Module HEF Gate
Run:
```bash
./hailo/scripts/hailo_test_modules_to_hef.sh
```

Acceptance:
- `module_hef_summary.tsv` has all target rows with `har_success=true` and `hef_success=true`.

Checklist:
- [ ] `norm_affine` -> HEF
  - Evidence: `hailo/module_runs/20260223_020616/module_hef_summary.tsv` (`hef_success=false`)
  - Evidence: `hailo/module_runs/20260223_020616/norm_affine_hef.log`
  - Evidence: `hailo/module_runs/20260223_020616/norm_affine_compile_failure.txt`
- [ ] `norm_identity` -> HEF
  - Evidence: `hailo/module_runs/20260223_020616/module_hef_summary.tsv` (`hef_success=false`)
  - Evidence: `hailo/module_runs/20260223_020616/norm_identity_hef.log`
  - Evidence: `hailo/module_runs/20260223_020616/norm_identity_compile_failure.txt`
- [ ] `activation_sigmoid` -> HEF
  - Evidence: `hailo/module_runs/20260223_020616/module_hef_summary.tsv` (`hef_success=false`)
  - Evidence: `hailo/module_runs/20260223_020616/activation_sigmoid_hef.log`
  - Evidence: `hailo/module_runs/20260223_020616/activation_sigmoid_compile_failure.txt`
- [ ] `conv1d_block_skip0` -> HEF
  - Evidence: `hailo/module_runs/20260223_020616/module_hef_summary.tsv` (`hef_success=false`)
  - Evidence: `hailo/module_runs/20260223_020616/conv1d_block_skip0_hef.log`
  - Evidence: `hailo/module_runs/20260223_020616/conv1d_block_skip0_compile_failure.txt`
- [ ] `tdconvnet_k1` -> HEF
  - Evidence: `hailo/module_runs/20260223_020616/module_hef_summary.tsv` (`hef_success=false`)
  - Evidence: `hailo/module_runs/20260223_020616/tdconvnet_k1_hef.log`
  - Evidence: `hailo/module_runs/20260223_020616/tdconvnet_k1_compile_failure.txt`
- [ ] `encoder` -> HEF
  - Evidence: `hailo/module_runs/20260223_020616/module_hef_summary.tsv` (`hef_success=false`, `BackendAllocatorException`)
  - Evidence: `hailo/module_runs/20260223_020616/encoder_hef.log`
  - Evidence: `hailo/module_runs/20260223_020616/encoder_compile_failure.txt`
- [ ] `decoder_conv1x1` -> HEF
  - Evidence: `hailo/module_runs/20260223_020616/module_hef_summary.tsv` (`hef_success=false`)
  - Evidence: `hailo/module_runs/20260223_020616/decoder_conv1x1_hef.log`
  - Evidence: `hailo/module_runs/20260223_020616/decoder_conv1x1_compile_failure.txt`
- [ ] `hailo_convtasnet_k1` -> HEF
  - Evidence: `hailo/module_runs/20260223_020616/module_hef_summary.tsv` (`hef_success=false`, `BackendAllocatorException`)
  - Evidence: `hailo/module_runs/20260223_020616/hailo_convtasnet_k1_hef.log`
  - Evidence: `hailo/module_runs/20260223_020616/hailo_convtasnet_k1_compile_failure.txt`

## Smallest Full HEF
Run:
```bash
./hailo/scripts/hailo_test_smallest_full_to_hef.sh
```

Acceptance:
- At least one row in `smallest_full_hef_summary.tsv` has `hef_success=true`.

Checklist:
- [x] Baseline smallest-full attempt completed
  - Evidence: `hailo/module_runs/20260223_022045/smallest_full_hef_summary.tsv`
- [ ] At least one smallest-full configuration compiled to HEF
  - Evidence: `hailo/module_runs/20260223_022045/smallest_full_f256_bn64_hid128_skip128_compile_failure.txt`
  - Evidence: `hailo/module_runs/20260223_022045/smallest_full_f256_bn48_hid96_skip96_compile_failure.txt`
  - Evidence: `hailo/module_runs/20260223_022045/smallest_full_f256_bn32_hid64_skip64_compile_failure.txt`

## Finer Submodule Migrations (Encoder + Full-k1 internals)
Run:
```bash
./hailo/scripts/hailo_test_fine_submodules_to_hef.sh
```

Acceptance:
- `fine_submodule_hef_summary.tsv` shows whether any internal split submodule reaches HEF.

Checklist:
- [x] Encoder split submodule(s) tested to HEF
  - Evidence: `hailo/module_runs/20260223_031237/fine_submodule_hef_summary.tsv`
  - Evidence: `hailo/module_runs/20260223_031237/fine_encoder_conv_only_hef.log`
- [x] Full-k1 internal submodules tested to HEF
  - Evidence: `hailo/module_runs/20260223_031237/fine_submodule_hef_summary.tsv`
  - Evidence: `hailo/module_runs/20260223_031237/fine_full_encoder_only_hef.log`
  - Evidence: `hailo/module_runs/20260223_031237/fine_full_masker_only_hef.log`
  - Evidence: `hailo/module_runs/20260223_031237/fine_full_source_projector_only_hef.log`
  - Evidence: `hailo/module_runs/20260223_031237/fine_full_decoder_only_hef.log`
  - Evidence: `hailo/module_runs/20260223_031237/fine_full_decoder_pre_only.log` (`HARBuildFailed`, opset conversion error)

## Scale-Up After Smallest Pass
Run:
```bash
./hailo/scripts/hailo_test_stage7_hef_sweep.sh
```

Acceptance:
- First pass frontier identified and documented (largest passing config).

Checklist:
- [ ] Scale-up sweep executed
  - Evidence:
- [ ] Failure frontier documented (first failing layer family + timeout)
  - Evidence:
- [ ] Next single knob chosen (one change only)
  - Evidence:

## Current Known State
- [x] HAR staged pipeline previously passed (stage1-5).
  - Evidence: `hailo/module_runs/20260218_223015/summary.tsv`
- [x] Progressive full-model HAR pipeline previously passed.
  - Evidence: `hailo/module_runs/fullprog_223623/summary.tsv`
- [x] Full-size HEF compile previously failed with allocator infeasible.
  - Evidence: `hailo/module_runs/20260222_233627/s7_hef_full_b8_r3_bn128_hid256_compile_failure.txt`
  - Evidence: `hailo/module_runs/20260222_233627/s7_hef_full_b8_r3_bn128_hid256_compile_failure.txt.json`
- [x] Module HEF gate script executed once across all top-level module cases.
  - Evidence: `hailo/module_runs/20260223_020616/module_hef_summary.tsv`
  - Result: all module rows have `har_success=true` and `hef_success=false`
- [x] Smallest-full HEF ladder executed once (3 configs).
  - Evidence: `hailo/module_runs/20260223_022045/smallest_full_hef_summary.tsv`
  - Result: all 3 rows have `har_success=true` and `hef_success=false` (`BackendAllocatorException`)
- [x] Fine submodule migration run executed once (encoder/full-k1 internal splits).
  - Evidence: `hailo/module_runs/20260223_031237/fine_submodule_hef_summary.tsv`
  - Result: 5 submodules reached HAR but all failed HEF; 1 case failed at HAR export (`convtas_decoder_pre_only`)
- [x] Calibration-length-matched module pass executed once (`input_length_policy=error`).
  - Evidence: `hailo/module_runs/20260223_031812/module_hef_calib_match_summary.tsv`
  - Result: all module rows still failed HEF; no `BadInputsShape` failures observed
  - Conclusion: failures are allocator/mapping dominated, not calibration-length mismatch

## Reporting Template (Use Every Run)
1. `HAR status:` pass/fail + summary path
2. `HEF status:` pass/fail + first failing layer family
3. `Next knob:` exact single change next

## Guardrails
- Do not edit baseline non-Hailo model files for architecture experiments.
- Keep architecture edits in `hailo_*` model/module files only.
- Keep all failure logs run-scoped in `hailo/module_runs/<run_ts>/`.

## Allocator + Temporal Tiling Track
Goal: convert known allocator failures into passing HEF units via temporal tiling + 64-channel block decomposition, then stitch host-side.

Evidence:
- `hailo/module_runs/20260223_052929/allocator_mapping_fixes_summary.tsv`
- `hailo/workarounds.md`

Checklist:
- [x] Allocator mapping script implemented and run (`PROFILE=full`)
  - Script: `hailo/scripts/hailo_test_allocator_mapping_fixes.sh`
  - Evidence: `hailo/module_runs/20260223_052929/allocator_mapping_fixes_summary.tsv`
  - Result: 46/49 HEF passes, 3 temporal-width fails (`w2000`, `w2000`, `w1024`)
- [x] Host-side temporal tiling reconstruction script implemented for decoder path
  - Target: source projector + decoder_pre + decoder_head reconstruction from 64-channel blocks
  - Evidence: `hailo/tiled_decoder_path.py`
- [x] Tiled reconstruction parity validated against direct PyTorch forward
  - Acceptance: max absolute error < 1e-5 for conv1x1 path
  - Evidence: `hailo/module_runs/20260223_060334/tiled_decoder_path_summary.tsv`
  - Evidence: `hailo/module_runs/20260223_060334/tile_lat2000_w256.json`
  - Evidence: `hailo/module_runs/20260223_060334/tile_lat2000_w512.json`
- [x] Command wrapper added for reproducible tiled parity run
  - Evidence: `hailo/scripts/hailo_test_tiled_decoder_path.sh`
  - Evidence: `hailo/module_runs/20260223_060334/tiled_decoder_path_summary.tsv`
- [x] Next step after parity: wire HEF block outputs into same stitching path
  - Evidence: `hailo/hef_tiled_decoder_path.py`
  - Evidence: `hailo/scripts/hailo_test_hef_tiled_decoder_path.sh`
  - Evidence: `hailo/module_runs/20260223_060610/hef_tiled_decoder_path_summary.tsv`
- [x] Non-decoder masker block decomposition implemented (`bneck`, `tcn0_in`, `tcn0_depth`, `tcn0_res`, `tcn0_skip`, `head`)
  - Evidence: `hailo-asteroid/asteroid/models/hailo_conv_tasnet_submodules.py`
  - Evidence: `hailo/export_hailo_module_to_onnx.py`
  - Evidence: `hailo/scripts/hailo_test_masker_allocator_fixes.sh`
- [x] Masker allocator smoke run: temporal + block cases compile to HEF
  - Evidence: `hailo/module_runs/20260223_082226/masker_allocator_fixes_summary.tsv`
  - Result: 10/10 selected cases (`hef_success=true`)
- [x] HEF-tiled masker path parity runner implemented and validated (`torch_proxy`)
  - Evidence: `hailo/masker_tiled_path.py`
  - Evidence: `hailo/scripts/hailo_test_hef_tiled_masker_path.sh`
  - Evidence: `hailo/module_runs/20260223_082756/hef_tiled_masker_path_summary.tsv`
- [x] Full-chain tiled parity runner (masker + decoder path) implemented and validated (`torch_proxy`)
  - Evidence: `hailo/hef_tiled_full_chain.py`
  - Evidence: `hailo/scripts/hailo_test_hef_tiled_full_chain.sh`
  - Evidence: `hailo/module_runs/20260223_082809/hef_tiled_full_chain_summary.tsv`
- [ ] Runtime backend validation on target machine (`hailo_platform/pyhailort` available)
  - Note: current environment lacks runtime package, so HEF-on-device execution remains pending
  - Evidence: `hailo/module_runs/20260223_061240_runtime/hef_tiled_decoder_path_summary.tsv`
  - Evidence: `hailo/module_runs/20260223_061240_runtime/hef_tile_lat2000_w256.log`
  - Evidence: `hailo/module_runs/20260223_061240_runtime/hef_tile_lat1024_w256.log`

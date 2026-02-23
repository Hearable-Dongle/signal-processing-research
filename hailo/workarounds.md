# Hailo ConvTasNet Workarounds Log

Last updated: 2026-02-23

## Goal
Track every workaround attempt and outcome while migrating ConvTasNet submodules to `.har` / `.hef`.

## Key Result So Far
- We can reliably produce `.har` for many fine submodules.
- `.hef` still fails mainly at allocator mapping (`BackendAllocatorException`, `Agent infeasible`).
- One real bug was fixed: `decoder_pre_only` previously failed at HAR due to identity export; now it reaches HAR and fails later at allocator stage like the others.

## Chronological Attempts
1. **Baseline fine submodules run**
   - Run: `hailo/module_runs/20260223_031237/fine_submodule_hef_summary.tsv`
   - What was tried: encoder/masker/source_projector/decoder_pre/decoder split.
   - Outcome:
     - HAR mostly passed.
     - `convtas_decoder_pre_only` failed at HAR build.
     - Others failed at HEF.

2. **Added smaller masker submodules**
   - Code:
     - `hailo-asteroid/asteroid/models/hailo_conv_tasnet_submodules.py`
     - `hailo/export_hailo_module_to_onnx.py`
     - `hailo/scripts/hailo_test_fine_submodules_to_hef.sh`
   - New modules:
     - `convtas_masker_bottleneck_only`
     - `convtas_masker_tcn_block0_only`
     - `convtas_masker_mask_head_only`

3. **Identity-to-conv workaround for decoder_pre export**
   - Code:
     - `hailo-asteroid/asteroid/models/hailo_conv_tasnet_submodules.py`
     - `hailo/export_hailo_module_to_onnx.py`
   - Workaround:
     - `nn.Identity` in decoder pre-stage replaced (for export) with explicit identity `Conv2d(1x1)`.
   - Outcome:
     - `convtas_decoder_pre_only` HAR now passes.
     - Run evidence: `hailo/module_runs/20260223_032909/fine_submodule_hef_summary.tsv`

4. **Fine calib-match runner (shape-isolation pass)**
   - Code:
     - `hailo/scripts/hailo_test_fine_submodules_to_hef_calib_match.sh`
   - What was tried:
     - Per-case synthetic calibration files matching expected temporal length and channels.
   - First outcome:
     - `ValueError: Unsupported calibration shape` for multi-channel NHWC.
     - Run evidence: `hailo/module_runs/20260223_033145/fine_submodule_hef_calib_match_summary.tsv`

5. **Calibration parser fix for multi-channel NHWC**
   - Code:
     - `hailo/har_to_hef.py`
   - Workaround:
     - `_to_nhwc` now accepts `[N,1,T,C]` for `C>1` (previously effectively only `C==1`).
   - Outcome:
     - Multi-channel fine submodule calib now reaches true optimization+allocation.
     - Failures become allocator infeasible (not calibration shape parse errors).
     - Run evidence: `hailo/module_runs/20260223_033822/fine_submodule_hef_calib_match_summary.tsv`

## Latest Run Snapshot (20260223_033822)
- Summary: `hailo/module_runs/20260223_033822/fine_submodule_hef_calib_match_summary.tsv`
- Cases run:
  - `calib_match_fine_full_masker_only`: HAR pass, HEF fail (`BackendAllocatorException`, `Agent infeasible`)
  - `calib_match_fine_full_masker_bottleneck_only`: HAR pass, HEF fail (`BackendAllocatorException`, `Agent infeasible`)
  - `calib_match_fine_full_masker_tcn_block0_only`: HAR pass, HEF fail (`BackendAllocatorException`, `Agent infeasible`)
  - `calib_match_fine_full_masker_mask_head_only`: HAR pass, HEF fail (`BackendAllocatorException`, `Agent infeasible`)
  - `calib_match_fine_full_source_projector_only`: HAR pass, HEF fail (`BackendAllocatorException`, `Agent infeasible`)
  - `calib_match_fine_full_decoder_pre_only`: HAR pass, HEF fail (`BackendAllocatorException`, `Agent infeasible`)
  - `calib_match_fine_full_decoder_only`: HAR pass, HEF fail (`BackendAllocatorException`, `Agent infeasible`)

## New Micro-Split Workaround (20260223_041021 + 20260223_041356)
Implemented exact micro splits requested for the currently failing regions:

- `source_projector_only` split:
  - `convtas_source_projector_out0` (256 -> 256)
  - `convtas_source_projector_out1` (256 -> 256)
- `decoder_pre_only` split:
  - `convtas_decoder_pre_half0` (512 -> 256)
  - `convtas_decoder_pre_half1` (512 -> 256)
- `decoder_only` split:
  - `convtas_decoder_head_src0` (512 -> 1)
  - `convtas_decoder_head_src1` (512 -> 1)

Code added/updated:
- `hailo-asteroid/asteroid/models/hailo_conv_tasnet_submodules.py`
- `hailo/export_hailo_module_to_onnx.py`
- `hailo/scripts/hailo_test_micro_submodules_to_hef.sh`

Run 1:
- Summary: `hailo/module_runs/20260223_041021/micro_submodule_hef_summary.tsv`
- Results:
  - `micro_source_projector_out0`: HAR pass, HEF fail (`BackendAllocatorException`)
  - `micro_source_projector_out1`: HAR pass, HEF fail (`BackendAllocatorException`)
  - `micro_decoder_pre_half0`: HAR pass, HEF fail (`BackendAllocatorException`)
  - `micro_decoder_pre_half1`: HAR pass, HEF fail (`BackendAllocatorException`)
  - `micro_decoder_head_src0`: HAR fail (export check bug in wrapper detection)
  - `micro_decoder_head_src1`: HAR fail (same bug)

Fix applied after run 1:
- Exporter now unwraps `HailoDecoderConv1x1Head.conv` for decoder head splits.

Run 2 (decoder head retry):
- Summary: `hailo/module_runs/20260223_041356/micro_submodule_hef_summary.tsv`
- Results:
  - `micro_decoder_head_src0`: HAR pass, HEF fail (`BackendAllocatorException`)
  - `micro_decoder_head_src1`: HAR pass, HEF fail (`BackendAllocatorException`)

## What Did Not Help
- Matching calibration length alone (`W`) does not fix allocator failures.
- Splitting the masker into smaller pieces did not avoid allocator failures.
- Replacing decoder pre identity with explicit 1x1 conv fixes HAR export only, not HEF allocator.
- Splitting source projector/decoder pre/decoder head into smaller micro ops still fails at HEF allocator stage.

## Allocator Mapping Fixes Implemented (20260223_051742)
New code for allocator-targeted breakdown and sweeps:
- `hailo-asteroid/asteroid/models/hailo_conv_tasnet_submodules.py`
  - Added `HailoConv1x1PartialBlock`
- `hailo/export_hailo_module_to_onnx.py`
  - Added modules:
    - `convtas_source_projector_block`
    - `convtas_decoder_pre_block`
    - `convtas_decoder_head_block`
    - `plain_conv1x1`
  - Added block args:
    - `--block_chan`, `--in_block_idx`, `--out_block_idx`, `--proj_src_idx`, `--half_idx`, `--head_src_idx`
- `hailo/scripts/hailo_test_allocator_mapping_fixes.sh`
  - Combines temporal sweep + blockwise partial-conv tests.

Targeted run executed:
- Summary: `hailo/module_runs/20260223_051742/allocator_mapping_fixes_summary.tsv`
- Cases run (filtered) and outcome:
  - `allocfix_plain_conv1x1_64_w256`: HEF PASS
  - `allocfix_temporal_source_out0_w256`: HEF PASS
  - `allocfix_temporal_dec_head0_w256`: HEF PASS
  - `allocfix_temporal_source_out0_w128`: HEF PASS
  - `allocfix_temporal_dec_head0_w128`: HEF PASS
  - `allocfix_block_source_s0_o0_i0_w256`: HEF PASS
  - `allocfix_block_decpre_h0_o0_i0_w256`: HEF PASS
  - `allocfix_block_dechead_s0_i0_w256`: HEF PASS

Interpretation:
- Allocator infeasible was size/config dependent, not absolute.
- Reducing temporal length to latent `W<=256` and/or using 64-channel partial conv blocks enables successful `.hef`.
- This confirms a workable path to reconstruct full behavior from multiple passing micro HEFs.

Full-profile follow-up run:
- Summary: `hailo/module_runs/20260223_052929/allocator_mapping_fixes_summary.tsv`
- Aggregate:
  - total cases: 49
  - HAR pass: 49
  - HEF pass: 46
  - HEF fail: 3
- Failing temporal-width cases:
  - `allocfix_temporal_source_out0_w2000` (`BackendAllocatorException`, `Agent infeasible`)
  - `allocfix_temporal_dec_head0_w2000` (`BackendAllocatorException`, `Agent infeasible`)
  - `allocfix_temporal_dec_head0_w1024` (`BackendAllocatorException`, `Agent infeasible`)
- Key outcome:
  - Full 64-channel source-projector block grid (`PROFILE=full`) passed.

## HEF-Tiled Stitching Implementation (20260223_060610)
Implemented decoder-path stitching pipeline that consumes the allocator-fix manifest and runs tiled block composition through a backend interface.

Code:
- `hailo/hef_tiled_decoder_path.py`
  - `BlockManifest` parser from summary TSV
  - `BlockExecutor` interface
  - `TorchProxyExecutor` (validated)
  - `HailoRuntimeExecutor` scaffold (runtime dependency guard)
  - tiled reconstruction for source projector / decoder_pre / decoder_head
- `hailo/scripts/hailo_test_hef_tiled_decoder_path.sh`

Run:
- Summary: `hailo/module_runs/20260223_060610/hef_tiled_decoder_path_summary.tsv`
- Backend: `torch_proxy`
- Result: all listed tiled cases pass (`latent_w=2000,tile_w=256`; `latent_w=1024,tile_w=256`; `latent_w=2000,tile_w=512`).

Re-run evidence:
- `torch_proxy` backend:
  - Summary: `hailo/module_runs/20260223_061230_proxy/hef_tiled_decoder_path_summary.tsv`
  - Result: all listed tiled cases pass.
- `hailo_runtime` backend:
  - Summary: `hailo/module_runs/20260223_061240_runtime/hef_tiled_decoder_path_summary.tsv`
  - Result: all listed cases fail in this environment due to missing runtime package.
  - Error: `ModuleNotFoundError: No module named 'hailo_platform'`
  - Logs:
    - `hailo/module_runs/20260223_061240_runtime/hef_tile_lat2000_w256.log`
    - `hailo/module_runs/20260223_061240_runtime/hef_tile_lat1024_w256.log`

Important constraint:
- This environment has compile SDK (`hailo_sdk_client`) but not runtime package (`hailo_platform` / `pyhailort`), so direct HEF execution against device cannot be validated here.
- Runtime backend path is wired and guarded; run on target machine with runtime installed.

## Non-Decoder (Masker) Implementation + Validation (20260223_082226 -> 20260223_082809)
Implemented remaining non-decoder path block modules and validation stack.

Code:
- `hailo-asteroid/asteroid/models/hailo_conv_tasnet_submodules.py`
  - Added masker block wrappers:
    - `HailoMaskerBottleneckBlock`
    - `HailoMaskerTCN0InConvBlock`
    - `HailoMaskerTCN0DepthBlock`
    - `HailoMaskerTCN0ResBlock`
    - `HailoMaskerTCN0SkipBlock`
    - `HailoMaskerHeadBlock`
  - Added `HailoDepthwisePartialBlock`
- `hailo/export_hailo_module_to_onnx.py`
  - Added module targets:
    - `convtas_masker_bottleneck_block`
    - `convtas_masker_tcn0_inconv_block`
    - `convtas_masker_tcn0_depth_block`
    - `convtas_masker_tcn0_res_block`
    - `convtas_masker_tcn0_skip_block`
    - `convtas_masker_head_block`
  - Added `--depth_block_idx`
- `hailo/scripts/hailo_test_masker_allocator_fixes.sh`
- `hailo/masker_tiled_path.py`
- `hailo/scripts/hailo_test_hef_tiled_masker_path.sh`
- `hailo/hef_tiled_full_chain.py`
- `hailo/scripts/hailo_test_hef_tiled_full_chain.sh`

Runs:
1. Masker allocator smoke:
   - Summary: `hailo/module_runs/20260223_082226/masker_allocator_fixes_summary.tsv`
   - Result: 10/10 selected temporal+block cases pass HEF.
2. Tiled masker parity (`torch_proxy`):
   - Summary: `hailo/module_runs/20260223_082756/hef_tiled_masker_path_summary.tsv`
   - Result: all listed cases pass.
3. Full-chain tiled parity (`torch_proxy`):
   - Summary: `hailo/module_runs/20260223_082809/hef_tiled_full_chain_summary.tsv`
   - Result: all listed cases pass.

Bug fixed during implementation:
- Initial masker block exports failed because block wrappers applied full-channel norm on 64-channel block inputs.
- Fix: export linear block contributions only for bottleneck/inconv/depth; apply nonlinearity+norm after reconstruction.

## What Helped
- Submodule decomposition improved observability and isolated failure domains.
- Multi-channel calibration support fix in `har_to_hef.py` removed false-negative shape parsing failures.

## Next Experiments (Not Yet Done)
1. Try smaller channel widths for fine modules (`n_filters/bn_chan/hid_chan/skip_chan`) while keeping topology fixed.
2. Try shorter `input_len` for latent-space modules (`2000 -> 1024 -> 512`) to test if allocator pressure is primarily spatial/temporal.
3. Run focused single-case sweeps on the first failing op chain in each fine module (starting with masker-only and tcn-block0).

## Path From Micro Modules To End-to-End
1. Expand passing temporal window:
   - Run `hailo_test_allocator_mapping_fixes.sh` without filter and collect highest passing latent width (`W`) for:
     - `convtas_source_projector_out0`
     - `convtas_decoder_head_src0`
2. Build full block grid for decoder path (`PROFILE=full`):
   - Compile all required blocks for:
     - source projector (src 0/1, out blocks 0..3, in blocks 0..3)
     - decoder pre (half 0/1, out blocks 0..3, in blocks 0..7)
     - decoder head (src 0/1, in blocks 0..7)
3. Host-side reconstruction:
   - Reconstruct outputs by summing partial block outputs for each output block.
   - Concat output blocks to form full tensors.
   - Add bias only once per output block (already encoded by `include_bias` on first input block).
4. Integrate decoder path into end-to-end pipeline:
   - Keep masker bypass mode first.
   - Validate waveform parity against PyTorch reference on short clips.
5. Scale width upward gradually:
   - Increase latent `W` from passing point (e.g. 128 -> 256 -> 512 -> 1024 -> 2000) and stop at first allocator failure.

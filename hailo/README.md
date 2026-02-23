# Hailo ConvTasNet Runbook

This directory contains the ConvTasNet-to-Hailo workflow and validation scripts.

## Pipeline
`PyTorch -> ONNX -> HAR -> HEF`

For this model family, direct full-model HEF compile is allocator-sensitive. The workflow therefore compiles decomposed modules first, then validates stitched parity.

Trackers:
- `hailo/RUNNING_TODO_CONVTAS_HEF.md`
- `hailo/workarounds.md`

## Setup
Use Python 3.10.

ONNX export env:
```bash
python3.10 -m venv hailo/to-onnx-env
source hailo/to-onnx-env/bin/activate
pip install -r hailo/to-onnx-env-requirements.txt
```

Hailo compile env:
```bash
python3.10 -m venv hailo/to-hailo-env
source hailo/to-hailo-env/bin/activate
pip install <hailo_sdk_client_whl>
pip install -r hailo/to-hailo-env-requirements.txt
```

Notes:
- `hailo_sdk_client` is sufficient for ONNX->HAR->HEF.
- Real device execution of `.hef` needs runtime packages (`hailo_platform` / `pyhailort`) on the target machine.

## Quickstart (Recommended)
Run from repo root.

1. Decoder allocator + block grid:
```bash
PROFILE=full ./hailo/scripts/hailo_test_allocator_mapping_fixes.sh
```

2. Masker allocator + block grid:
```bash
PROFILE=full ./hailo/scripts/hailo_test_masker_allocator_fixes.sh
```

3. Decoder stitched parity (proxy backend):
```bash
BACKEND=torch_proxy ./hailo/scripts/hailo_test_hef_tiled_decoder_path.sh
```

4. Masker stitched parity (proxy backend):
```bash
BACKEND=torch_proxy ./hailo/scripts/hailo_test_hef_tiled_masker_path.sh
```

5. End-to-end stitched parity:
```bash
./hailo/scripts/hailo_test_hef_tiled_full_chain.sh
```

6. Real LibriMix forward pass (2-speaker separation, Hailo-format wrapper):
```bash
./hailo/scripts/hailo_run_librimix_forward_example.sh
```
Optional overrides:
```bash
MIX_WAV=/home/mkeller/data/librimix/Libri2Mix/wav8k/max/train-360/mix_clean/<file>.wav \
MODEL_ID=mpariente/ConvTasNet_WHAM!_sepclean \
./hailo/scripts/hailo_run_librimix_forward_example.sh
```

Optional runtime checks (on target with runtime installed):
```bash
BACKEND=hailo_runtime ./hailo/scripts/hailo_test_hef_tiled_decoder_path.sh
BACKEND=hailo_runtime ./hailo/scripts/hailo_test_hef_tiled_masker_path.sh
```

## Output Layout
Every script writes to:
`hailo/module_runs/<run_ts>/`

Common files:
- `*_summary.tsv`: case-level status table
- `*.log`: ONNX export/translate logs
- `*_hef.log`: HAR->HEF logs
- `*_compile_failure.txt`: normalized failure excerpt
- `*.json`: numeric parity metrics

## Metrics Reference

### Compile summary metrics (`*_summary.tsv`)
Main columns:
- `har_success`: ONNX->HAR status
- `hef_success`: HAR->HEF status
- `exception_type`: normalized top-level error class
- `failure_head`: first failure signature line

Typical interpretation:
- `har_success=true` and `hef_success=false`: model translates, but fails in quantization/allocation/compile.
- `exception_type=BackendAllocatorException` with `failure_head` containing `Agent infeasible`: allocator placement/resource infeasibility.

### Tiled path summary metrics (`hef_tiled_*_summary.tsv`)
Columns:
- `run_tag`: test case id
- `backend`: `torch_proxy` or `hailo_runtime` (not present in full-chain TSV)
- `latent_w` / `wav_t`: temporal length in latent or waveform domain
- `tile_w`: tile width used for stitched execution
- `success`: pass/fail for that case
- `output_json`: detailed per-stage numeric metrics

### Detailed parity metrics (`*.json`)
Shared keys:
- `all_close`: global numeric pass flag
- `tol`: threshold used by checker
- `*_max_abs`: per-stage maximum absolute error vs reference

Important stage keys:
- Decoder path:
  - `source_projector_max_abs`
  - `decoder_pre_max_abs`
  - `decoder_head_max_abs`
- Masker path:
  - `bottleneck_max_abs`
  - `tcn0_in_max_abs`
  - `tcn0_depth_max_abs`
  - `tcn0_residual_max_abs`
  - `tcn0_skip_max_abs`
  - `mask_score_max_abs`
  - `mask_output_max_abs`
- Full chain:
  - `masker_output_max_abs`
  - `masked_rep_max_abs`
  - `decoder_output_max_abs`
  - `final_output_max_abs`

Real forward example metrics (`librimix_hailo_forward_metrics.json`):
- `parity_max_abs_vs_reference`: max absolute diff between reference ConvTasNet and Hailo-format wrapper output.
- `si_snr_hailo_src1_vs_gt_s1_db`, `si_snr_hailo_src2_vs_gt_s2_db`: SI-SNR to LibriMix ground-truth sources (best source permutation).
- `si_snr_hailo_pair_sum_db`: sum SI-SNR across both separated sources.

Acceptance used in this repo:
- nominal tolerance is `tol=1e-5`
- passing case expects `all_close=true`
- smaller `*_max_abs` is better; values around `1e-7` to `1e-6` are strong parity.

## Known-Good References
- `hailo/module_runs/20260223_052929/allocator_mapping_fixes_summary.tsv`
- `hailo/module_runs/20260223_082226/masker_allocator_fixes_summary.tsv`
- `hailo/module_runs/20260223_082756/hef_tiled_masker_path_summary.tsv`
- `hailo/module_runs/20260223_082809/hef_tiled_full_chain_summary.tsv`

## Guardrails
- Keep Hailo-specific architecture/decomposition in `hailo_*` files/modules.
- Do not overwrite baseline non-Hailo model definitions.
- Keep run evidence under `hailo/module_runs/<run_ts>/`.

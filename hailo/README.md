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

## Asteroid Submodule
Hailo ConvTasNet uses the Hailo-modified `asteroid/` submodule. Initialize/update it before running:
```bash
git submodule update --init --recursive asteroid
```

## Runtime Payloads (On-Device)
Payloads are sent to Hailo via `hailo/hailo_runtime_runner.py`, which wraps `hailo_platform` vstreams.
The parity scripts below use `BACKEND=hailo_runtime` to run blocks on the device and send NCHW inputs.

## Quickstart: Conversion Machine (x86)
Run from repo root on the machine that can compile HAR -> HEF.

These scripts **produce** new artifacts (ONNX, HAR, HEF). They are not just runtime checks.

### A) Build decoder-path HEFs (full profile)
```bash
PROFILE=full HAILO_RUN_TS=<DECODER_RUN_TS> ./hailo/scripts/hailo_test_allocator_mapping_fixes.sh
```
What this builds:
- Decoder-side block HEFs used after encoder/masker:
  - `source_projector` blocks (`allocfix_full_source_*`)
  - `decoder_pre` blocks (`allocfix_block_decpre_*`)
  - `decoder_head` blocks (`allocfix_block_dechead_*`)
- With `PROFILE=full`, this script generates the full source-projector block grid needed by runtime stitching.

### B) Build masker-path HEFs
```bash
PROFILE=quick HAILO_RUN_TS=<MASKER_RUN_TS> ./hailo/scripts/hailo_test_masker_allocator_fixes.sh
```
What this builds:
- Masker-side block HEFs (front half of the hybrid path):
  - bottleneck (`allocfix_masker_bneck_*`)
  - first TCN block pieces (`allocfix_masker_tcn0_in_*`, `allocfix_masker_tcn0_depth_*`, `allocfix_masker_tcn0_res_*`, `allocfix_masker_tcn0_skip_*`)
  - masker head (`allocfix_masker_head_*`)
- `PROFILE=quick` builds a minimal runtime set (single block-index path) and is usually sufficient for current stitched runtime.
- Use `PROFILE=full` for masker only if you intentionally want the larger block-index sweep for broader compile coverage/debug.

Why there are two scripts:
- Decoder-path and masker-path are different subgraphs with different decomposition strategies and manifests.
- Runtime full-chain parity needs both manifests:
  - decoder summary: `allocator_mapping_fixes_summary.tsv`
  - masker summary: `masker_allocator_fixes_summary.tsv`

Outputs:
- `hailo/module_runs/<DECODER_RUN_TS>/allocator_mapping_fixes_summary.tsv`
- `hailo/module_runs/<MASKER_RUN_TS>/masker_allocator_fixes_summary.tsv`
- `.hef` files referenced by `hef_success=true` rows in those summaries.

Note:
- True hybrid parity in this repo is manifest-driven and requires multiple block HEFs, not a single monolithic HEF.

### Reproducible from your PyTorch weights
Step 1: build a Hailo-compatible state dict from an HF pretrained Asteroid model (or override `--model_name`):

```bash
hailo/to-onnx-env/bin/python -m hailo.build_hailo_state_dict_from_pretrained \
  --model_name JorisCos/ConvTasNet_Libri3Mix_sepclean_8k \
  --output /tmp/hailo_from_pretrained_state_dict.pt
```

Step 2: produce decoder/masker HEFs using that state dict:

```bash
MODEL_STATE_DICT=/tmp/hailo_from_pretrained_state_dict.pt \
PROFILE=full HAILO_RUN_TS=<DECODER_RUN_TS> \
./hailo/scripts/hailo_test_allocator_mapping_fixes.sh

MODEL_STATE_DICT=/tmp/hailo_from_pretrained_state_dict.pt \
PROFILE=quick HAILO_RUN_TS=<MASKER_RUN_TS> \
./hailo/scripts/hailo_test_masker_allocator_fixes.sh
```

Details:
- `MODEL_STATE_DICT` is forwarded to ONNX export (`--state_dict_path`) and used when modules are exported before HAR/HEF compilation.
- Loader supports common checkpoint wrappers (`state_dict`, `model_state_dict`, `model`, `net`).
- Key prefixes like `module.` and `model.` are normalized automatically.
- The helper script transfers encoder + masker weights from pretrained Asteroid ConvTasNet into `HailoConvTasNet` and initializes `conv1x1_head` decoder deterministically.

## Copy Artifacts to RPi
Create a deterministic export bundle on the conversion machine:

```bash
DEC_TS=<DECODER_RUN_TS>
MASK_TS=<MASKER_RUN_TS>

awk -F'\t' 'NR>1 && $9=="true" {print $8}' "hailo/module_runs/${DEC_TS}/allocator_mapping_fixes_summary.tsv" > /tmp/hailo_hef_files.txt
awk -F'\t' 'NR>1 && $9=="true" {print $8}' "hailo/module_runs/${MASK_TS}/masker_allocator_fixes_summary.tsv" >> /tmp/hailo_hef_files.txt

echo "hailo/module_runs/${DEC_TS}/allocator_mapping_fixes_summary.tsv" >> /tmp/hailo_hef_files.txt
echo "hailo/module_runs/${MASK_TS}/masker_allocator_fixes_summary.tsv" >> /tmp/hailo_hef_files.txt

tar -czf /tmp/hailo_true_hybrid_artifacts.tgz -T /tmp/hailo_hef_files.txt
```

Copy bundle to RPi and extract at repo root:

```bash
scp /tmp/hailo_true_hybrid_artifacts.tgz <rpi_user>@<rpi_host>:/tmp/
ssh <rpi_user>@<rpi_host> "cd /home/<user>/signal-processing-research && tar -xzf /tmp/hailo_true_hybrid_artifacts.tgz"
```

## Quickstart: RPi Runtime + Parity
Run from repo root on RPi with Hailo runtime installed.

Set explicit summaries:
```bash
export DECODER_SUMMARY_TSV="hailo/module_runs/<DECODER_RUN_TS>/allocator_mapping_fixes_summary.tsv"
export MASKER_SUMMARY_TSV="hailo/module_runs/<MASKER_RUN_TS>/masker_allocator_fixes_summary.tsv"
```

1. Decoder parity on hardware:
```bash
BACKEND=hailo_runtime SUMMARY_TSV="$DECODER_SUMMARY_TSV" ./hailo/scripts/hailo_test_hef_tiled_decoder_path.sh
```

2. Masker parity on hardware:
```bash
BACKEND=hailo_runtime SUMMARY_TSV="$MASKER_SUMMARY_TSV" ./hailo/scripts/hailo_test_hef_tiled_masker_path.sh
```

3. Full stitched-chain parity on hardware:
```bash
BACKEND=hailo_runtime DECODER_SUMMARY_TSV="$DECODER_SUMMARY_TSV" MASKER_SUMMARY_TSV="$MASKER_SUMMARY_TSV" ./hailo/scripts/hailo_test_hef_tiled_full_chain.sh
```

4. End-to-end hybrid validation (real sample, latency + WAV outputs):
```bash
BACKEND=hailo_runtime \
DECODER_SUMMARY_TSV="$DECODER_SUMMARY_TSV" \
MASKER_SUMMARY_TSV="$MASKER_SUMMARY_TSV" \
MODEL_ID=JorisCos/ConvTasNet_Libri3Mix_sepclean_8k \
MIX_WAV=hailo/sanity_librimix3/sanity_mix.wav \
N_SRC=2 SAMPLE_RATE=8000 MAX_SECONDS=4.0 \
./hailo/scripts/hailo_validate_hybrid_librimix.sh
```

Optional wrapper-only forward example:
```bash
./hailo/scripts/hailo_run_librimix_forward_example.sh
```
Optional overrides:
```bash
MIX_WAV=/home/mkeller/data/librimix/Libri2Mix/wav8k/max/train-360/mix_clean/<file>.wav \
MODEL_ID=mpariente/ConvTasNet_WHAM!_sepclean \
./hailo/scripts/hailo_run_librimix_forward_example.sh
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

Important:
- `hailo/module_runs/prefix_smoke_20260223_205551/prefixA_active1_b2_r1.hef` is useful for prefix runtime experiments, but it is not sufficient for the true hybrid parity path in this runbook.
- True hybrid parity uses manifest-driven block HEFs from allocator-fix summaries.

## Guardrails
- Keep Hailo-specific architecture/decomposition in `hailo_*` files/modules.
- Do not overwrite baseline non-Hailo model definitions.
- Keep run evidence under `hailo/module_runs/<run_ts>/`.

# FOR_RPI_AGENT

This runbook is for an agent running on Raspberry Pi with Hailo-8 hardware attached (`hailo_platform`/`pyhailort` installed).

## Goal
Validate the hybrid deployment path where heavy conv blocks run on Hailo and CPU handles orchestration/light glue.

## Prerequisites
- Hailo runtime installed and importable (`hailo_platform`)
- Repo checkout with `hailo/to-onnx-env` and `hailo/to-hailo-env`
- Access to LibriMix data path from `general_utils/constants.py`
- Network access for loading reference weights:
  - `JorisCos/ConvTasNet_Libri3Mix_sepclean_8k`

## 1) Prepare Libri3Mix Sanity Audio
Copy one Libri3Mix mixture and the 3 source voices into `hailo/sanity_librimix3/`:

```bash
./hailo/scripts/hailo_prepare_librimix3_sanity_assets.sh
```

Expected files:
- `hailo/sanity_librimix3/sanity_mix.wav`
- `hailo/sanity_librimix3/sanity_voice_1.wav`
- `hailo/sanity_librimix3/sanity_voice_2.wav`
- `hailo/sanity_librimix3/sanity_voice_3.wav`
- `hailo/sanity_librimix3/sanity_manifest.json`

## 2) Runtime Parity Sanity for Block Paths (Hailo backend)
### Decoder path
```bash
BACKEND=hailo_runtime ./hailo/scripts/hailo_test_hef_tiled_decoder_path.sh
```

### Masker path
```bash
BACKEND=hailo_runtime ./hailo/scripts/hailo_test_hef_tiled_masker_path.sh
```

### Full stitched chain
```bash
BACKEND=hailo_runtime ./hailo/scripts/hailo_test_hef_tiled_full_chain.sh
```

Collect and keep:
- `hailo/module_runs/<run_ts>/hef_tiled_decoder_path_summary.tsv`
- `hailo/module_runs/<run_ts>/hef_tiled_masker_path_summary.tsv`
- `hailo/module_runs/<run_ts>/hef_tiled_full_chain_summary.tsv`

## 3) End-to-End Hybrid Latency + WAV Output Validation
Run the real-sample hybrid validation script:

```bash
BACKEND=hailo_runtime \
MODEL_ID=JorisCos/ConvTasNet_Libri3Mix_sepclean_8k \
MIX_WAV=hailo/sanity_librimix3/sanity_mix.wav \
N_SRC=2 \
SAMPLE_RATE=8000 \
MAX_SECONDS=4.0 \
./hailo/scripts/hailo_validate_hybrid_librimix.sh
```

Outputs in:
- `hailo/module_runs/<run_ts>_hybrid_librimix/`

Expected artifacts:
- `hybrid_validation_metrics.json`
- `mix.wav`
- `hybrid_sep_src1.wav`
- `hybrid_sep_src2.wav`
- `ref_sep_src1.wav`
- `ref_sep_src2.wav`
- `ref_sep_src3.wav`
- `gt_voice_1.wav`
- `gt_voice_2.wav`
- `gt_voice_3.wav`

Notes:
- `N_SRC=2` matches the currently stitched HEF graph path.
- Reference model has 3 outputs; script records parity only when source count matches.

## 4) Acceptance Checks
- Runtime parity scripts complete and write summaries.
- `hybrid_validation_metrics.json` exists and contains:
  - `latency_ms_total`
  - `rtf_total`
  - per-stage latency fields
- Listenable WAV outputs are present for both hybrid and reference separations.

## 5) Report Back
Share these files/paths:
- Latest run dir for step 2 + summary TSVs
- Latest run dir for step 3 + `hybrid_validation_metrics.json`
- Any runtime errors from `*.log` files

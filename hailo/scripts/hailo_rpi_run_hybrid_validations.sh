#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

RUN_TS="${HAILO_RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
export HAILO_RUN_TS="$RUN_TS"

# 1) Prepare sanity assets (idempotent)
./hailo/scripts/hailo_prepare_librimix3_sanity_assets.sh

# 2) Runtime parity checks
BACKEND=hailo_runtime ./hailo/scripts/hailo_test_hef_tiled_decoder_path.sh
BACKEND=hailo_runtime ./hailo/scripts/hailo_test_hef_tiled_masker_path.sh
BACKEND=hailo_runtime ./hailo/scripts/hailo_test_hef_tiled_full_chain.sh

# 3) End-to-end hybrid sample run (latency + wav outputs)
BACKEND=hailo_runtime \
MODEL_ID="${MODEL_ID:-JorisCos/ConvTasNet_Libri3Mix_sepclean_8k}" \
MIX_WAV="${MIX_WAV:-hailo/sanity_librimix3/sanity_mix.wav}" \
N_SRC="${N_SRC:-2}" \
SAMPLE_RATE="${SAMPLE_RATE:-8000}" \
MAX_SECONDS="${MAX_SECONDS:-4.0}" \
./hailo/scripts/hailo_validate_hybrid_librimix.sh

echo "[DONE] hybrid validations run_ts=${RUN_TS}"

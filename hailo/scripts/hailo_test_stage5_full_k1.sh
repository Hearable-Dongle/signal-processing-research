#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

HAILO_RUN_TS="${HAILO_RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
export HAILO_RUN_TS
export HAILO_STAGE="stage5_full_k1"

./hailo/scripts/hailo_module_to_har.sh hailo_convtasnet s5_hailo_convtasnet_k1 \
  --n_src 2 \
  --n_filters 256 \
  --bn_chan 128 \
  --hid_chan 256 \
  --skip_chan 128 \
  --n_blocks 1 \
  --n_repeats 1 \
  --truncate_k_blocks 1 \
  --mask_mul_mode bypass \
  --skip_topology_mode project \
  --decoder_mode conv1x1_head \
  --time_len 16000

echo "[STAGE PASS] stage5_full_k1 run_ts=${HAILO_RUN_TS}"

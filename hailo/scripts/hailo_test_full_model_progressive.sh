#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

HAILO_RUN_TS="${HAILO_RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
export HAILO_RUN_TS
export HAILO_STAGE="stage6_full_model_progressive"

# 1) Baseline full topology in safer mask-bypass mode.
./hailo/scripts/hailo_module_to_har.sh hailo_convtasnet s6_full_bypass_k0_r1b2 \
  --n_src 2 --n_filters 256 --bn_chan 128 --hid_chan 256 --skip_chan 128 \
  --n_blocks 2 --n_repeats 1 --truncate_k_blocks 0 \
  --mask_mul_mode bypass --skip_topology_mode project --decoder_mode conv1x1_head --time_len 16000

# 2) Intermediate full (more blocks/repeats), still bypass.
./hailo/scripts/hailo_module_to_har.sh hailo_convtasnet s6_full_bypass_k0_r2b4 \
  --n_src 2 --n_filters 256 --bn_chan 128 --hid_chan 256 --skip_chan 128 \
  --n_blocks 4 --n_repeats 2 --truncate_k_blocks 0 \
  --mask_mul_mode bypass --skip_topology_mode project --decoder_mode conv1x1_head --time_len 16000

# 3) Target "entire model" depth: 8 blocks x 3 repeats.
./hailo/scripts/hailo_module_to_har.sh hailo_convtasnet s6_full_bypass_k0_r3b8 \
  --n_src 2 --n_filters 256 --bn_chan 128 --hid_chan 256 --skip_chan 128 \
  --n_blocks 8 --n_repeats 3 --truncate_k_blocks 0 \
  --mask_mul_mode bypass --skip_topology_mode project --decoder_mode conv1x1_head --time_len 16000

# 4) Same entire model depth with normal mask multiplication.
./hailo/scripts/hailo_module_to_har.sh hailo_convtasnet s6_full_normal_k0_r3b8 \
  --n_src 2 --n_filters 256 --bn_chan 128 --hid_chan 256 --skip_chan 128 \
  --n_blocks 8 --n_repeats 3 --truncate_k_blocks 0 \
  --mask_mul_mode normal --skip_topology_mode project --decoder_mode conv1x1_head --time_len 16000

echo "[STAGE PASS] stage6_full_model_progressive run_ts=${HAILO_RUN_TS}"
echo "Summary: hailo/module_runs/${HAILO_RUN_TS}/summary.tsv"

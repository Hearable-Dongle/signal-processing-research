#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

HAILO_RUN_TS="${HAILO_RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
export HAILO_RUN_TS
export HAILO_STAGE="stage3_tdconvnet"

./hailo/scripts/hailo_module_to_har.sh tdconvnet s3_td_k1 --in_chan 128 --out_chan 128 --n_src 2 --bn_chan 128 --hid_chan 256 --skip_chan 128 --n_blocks 1 --n_repeats 1 --truncate_k_blocks 1 --time_len 1024 --mask_act sigmoid
./hailo/scripts/hailo_module_to_har.sh tdconvnet s3_td_k2 --in_chan 128 --out_chan 128 --n_src 2 --bn_chan 128 --hid_chan 256 --skip_chan 128 --n_blocks 2 --n_repeats 1 --truncate_k_blocks 2 --time_len 1024 --mask_act sigmoid

echo "[STAGE PASS] stage3_tdconvnet run_ts=${HAILO_RUN_TS}"

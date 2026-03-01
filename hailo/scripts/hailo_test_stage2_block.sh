#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

HAILO_RUN_TS="${HAILO_RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
export HAILO_RUN_TS
export HAILO_STAGE="stage2_block"

./hailo/scripts/hailo_module_to_har.sh conv1d_block s2_block_skip0 --in_chan 128 --hid_chan 256 --skip_chan 0 --kernel_size 3 --dilation 1 --time_len 1024
./hailo/scripts/hailo_module_to_har.sh conv1d_block s2_block_skip128 --in_chan 128 --hid_chan 256 --skip_chan 128 --kernel_size 3 --dilation 1 --time_len 1024

echo "[STAGE PASS] stage2_block run_ts=${HAILO_RUN_TS}"

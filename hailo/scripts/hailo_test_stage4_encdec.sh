#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

HAILO_RUN_TS="${HAILO_RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
export HAILO_RUN_TS
export HAILO_STAGE="stage4_encdec"

./hailo/scripts/hailo_module_to_har.sh encoder s4_encoder --n_filters 256 --encdec_kernel_size 16 --encdec_stride 8 --time_len 16000
./hailo/scripts/hailo_module_to_har.sh decoder s4_decoder_conv1x1 --in_chan 256 --n_src 2 --time_len 2000

echo "[STAGE PASS] stage4_encdec run_ts=${HAILO_RUN_TS}"

#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

HAILO_RUN_TS="${HAILO_RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
export HAILO_RUN_TS
export HAILO_STAGE="stage1_primitives"

./hailo/scripts/hailo_module_to_har.sh norm s1_norm_affine --in_chan 128 --norm_mode affine --time_len 1024
./hailo/scripts/hailo_module_to_har.sh norm s1_norm_identity --in_chan 128 --norm_mode identity --time_len 1024
./hailo/scripts/hailo_module_to_har.sh activation s1_activation_sigmoid --in_chan 128 --mask_act sigmoid --time_len 1024

echo "[STAGE PASS] stage1_primitives run_ts=${HAILO_RUN_TS}"

#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

HAILO_RUN_TS="${HAILO_RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
export HAILO_RUN_TS

./hailo/scripts/hailo_test_stage1_primitives.sh
./hailo/scripts/hailo_test_stage2_block.sh
./hailo/scripts/hailo_test_stage3_tdconvnet.sh
./hailo/scripts/hailo_test_stage4_encdec.sh
./hailo/scripts/hailo_test_stage5_full_k1.sh
./hailo/scripts/hailo_test_full_model_progressive.sh

echo "[ALL PASS] run_ts=${HAILO_RUN_TS}"
echo "Summary: hailo/module_runs/${HAILO_RUN_TS}/summary.tsv"

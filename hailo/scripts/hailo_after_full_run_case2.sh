#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

RUN_TS="${1:?run_ts required}"
BASE_TAG="s7_hef_full_b8_r3_bn128_hid256"
NEXT_TAG="s7_hef_full_b8_r3_bn96_hid192"
LOG="hailo/module_runs/${RUN_TS}/after_full_case2.log"
mkdir -p "hailo/module_runs/${RUN_TS}"

exec > >(tee -a "$LOG") 2>&1

echo "[$(date -Iseconds)] waiting for base compile to finish: ${BASE_TAG}"
while true; do
  if pgrep -f "hailo\.har_to_hef .*${BASE_TAG}\.har" >/dev/null; then
    sleep 10
    continue
  fi
  break
done

echo "[$(date -Iseconds)] base compile done; starting ${NEXT_TAG}"

./hailo/scripts/hailo_module_to_har.sh hailo_convtasnet "$NEXT_TAG" \
  --n_src 2 --n_filters 256 --bn_chan 96 --hid_chan 192 --skip_chan 128 \
  --n_blocks 8 --n_repeats 3 --truncate_k_blocks 0 \
  --mask_mul_mode normal --skip_topology_mode project --decoder_mode conv1x1_head --time_len 16000

hailo/to-hailo-env/bin/python -m hailo.har_to_hef \
  "hailo/module_runs/${RUN_TS}/${NEXT_TAG}.har" \
  "hailo/module_runs/${RUN_TS}/${NEXT_TAG}.hef" \
  --model_name "$NEXT_TAG" \
  --hw_arch hailo8 \
  --calib_npz hailo/calibration_1000ms_16k_64.npz \
  --quick_opt \
  --log_failed_layers_path "hailo/module_runs/${RUN_TS}/${NEXT_TAG}_compile_failure.txt"

echo "[$(date -Iseconds)] finished ${NEXT_TAG}"

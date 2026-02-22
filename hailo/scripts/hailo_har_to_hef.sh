#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <input.har> <output.hef> [model_name]"
  exit 2
fi

HAR_PATH="$1"
HEF_PATH="$2"
MODEL_NAME="${3:-$(basename "$HAR_PATH" .har)}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

CALIB_NPZ="${CALIB_NPZ:-hailo/calibration_1000ms_16k_64.npz}"
HW_ARCH="${HW_ARCH:-hailo8}"

echo "[INFO] Using model_name=${MODEL_NAME}"

hailo/to-hailo-env/bin/python -m hailo.har_to_hef \
  "$HAR_PATH" "$HEF_PATH" \
  --model_name "$MODEL_NAME" \
  --hw_arch "$HW_ARCH" \
  --calib_npz "$CALIB_NPZ"

echo "[DONE] HEF written to $HEF_PATH"

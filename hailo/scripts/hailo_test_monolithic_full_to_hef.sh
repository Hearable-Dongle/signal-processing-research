#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

RUN_TS="${HAILO_RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
PROFILE="${PROFILE:-focused}"
TIME_LIMIT_MIN="${TIME_LIMIT_MIN:-120}"
CALIB_NPZ="${CALIB_NPZ:-hailo/calibration_1000ms_16k_64.npz}"
HW_ARCH="${HW_ARCH:-hailo8}"
QUICK_OPT="${QUICK_OPT:-1}"
CASE_TIMEOUT_MIN="${CASE_TIMEOUT_MIN:-12}"

if [[ "$PROFILE" != "focused" && "$PROFILE" != "extended" ]]; then
  echo "PROFILE must be focused|extended (got '$PROFILE')" >&2
  exit 2
fi

if [[ "$QUICK_OPT" != "0" && "$QUICK_OPT" != "1" ]]; then
  echo "QUICK_OPT must be 0|1 (got '$QUICK_OPT')" >&2
  exit 2
fi

export HAILO_RUN_TS="$RUN_TS"

hailo/to-hailo-env/bin/python -m hailo.monolithic_full_search \
  --run_ts "$RUN_TS" \
  --profile "$PROFILE" \
  --time_limit_min "$TIME_LIMIT_MIN" \
  --case_timeout_min "$CASE_TIMEOUT_MIN" \
  --calib_npz "$CALIB_NPZ" \
  --hw_arch "$HW_ARCH" \
  --quick_opt "$QUICK_OPT"

echo "[DONE] Monolithic full HEF search run_ts=${RUN_TS}"
echo "Summary: hailo/module_runs/${RUN_TS}/monolithic_full_hef_summary.tsv"

#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

RUN_TS="${HAILO_RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
TIME_LIMIT_MIN="${TIME_LIMIT_MIN:-60}"
CASE_TIMEOUT_MIN="${CASE_TIMEOUT_MIN:-10}"
CALIB_NPZ="${CALIB_NPZ:-hailo/calibration_1000ms_16k_64.npz}"
HW_ARCH="${HW_ARCH:-hailo8}"
QUICK_OPT="${QUICK_OPT:-1}"
STOP_ON_FIRST_FALSE="${STOP_ON_FIRST_FALSE:-1}"
STAGE_MODE="${STAGE_MODE:-exact_shape_incremental}"

N_FILTERS="${N_FILTERS:-256}"
BN_CHAN="${BN_CHAN:-128}"
HID_CHAN="${HID_CHAN:-256}"
SKIP_CHAN="${SKIP_CHAN:-128}"
N_BLOCKS="${N_BLOCKS:-2}"
N_REPEATS="${N_REPEATS:-1}"
INPUT_LEN="${INPUT_LEN:-16000}"

if [[ "$QUICK_OPT" != "0" && "$QUICK_OPT" != "1" ]]; then
  echo "QUICK_OPT must be 0|1 (got '$QUICK_OPT')" >&2
  exit 2
fi
if [[ "$STOP_ON_FIRST_FALSE" != "0" && "$STOP_ON_FIRST_FALSE" != "1" ]]; then
  echo "STOP_ON_FIRST_FALSE must be 0|1 (got '$STOP_ON_FIRST_FALSE')" >&2
  exit 2
fi

export HAILO_RUN_TS="$RUN_TS"

hailo/to-hailo-env/bin/python -m hailo.monolithic_prefix_ramp \
  --run_ts "$RUN_TS" \
  --time_limit_min "$TIME_LIMIT_MIN" \
  --case_timeout_min "$CASE_TIMEOUT_MIN" \
  --calib_npz "$CALIB_NPZ" \
  --hw_arch "$HW_ARCH" \
  --quick_opt "$QUICK_OPT" \
  --stage_mode "$STAGE_MODE" \
  --stop_on_first_false "$STOP_ON_FIRST_FALSE" \
  --n_filters "$N_FILTERS" \
  --bn_chan "$BN_CHAN" \
  --hid_chan "$HID_CHAN" \
  --skip_chan "$SKIP_CHAN" \
  --n_blocks "$N_BLOCKS" \
  --n_repeats "$N_REPEATS" \
  --input_len "$INPUT_LEN"

echo "[DONE] Monolithic prefix ramp run_ts=${RUN_TS}"
echo "Summary: hailo/module_runs/${RUN_TS}/monolithic_prefix_ramp_summary.tsv"

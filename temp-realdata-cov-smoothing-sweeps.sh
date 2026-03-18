#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/mkeller/miniconda3/bin/python}"
ROOT_DIR="/home/mkeller/mkeller/signal-processing-research"
cd "$ROOT_DIR"

INPUT_PATH="${1:-data-collection/gym-take-two-mar-17}"
OUT_PREFIX="${OUT_PREFIX:-beamforming/benchmark/_realdata_cov_smoothing_sweeps}"
METHOD="${METHOD:-mvdr_fd}"
TRACE_LOAD="${TRACE_LOAD:-0.01}"
IDENTITY_BLEND="${IDENTITY_BLEND:-0.0}"

CONFIGS=(
  "base 0.60 0.15 1.00"
  "smooth 0.30 0.10 0.50"
  "very_smooth 0.15 0.08 0.25"
)

for ENTRY in "${CONFIGS[@]}"; do
  read -r NAME EMA ACTIVE INACTIVE <<<"$ENTRY"
  OUT_DIR="${OUT_PREFIX}_${METHOD}_${NAME}"
  LOG_PATH="${OUT_DIR}.log"
  echo "Starting ${NAME}: ema=${EMA} active=${ACTIVE} inactive=${INACTIVE}"
  nohup env PYTHONPATH=. "$PYTHON_BIN" beamforming/benchmark/data_collection_benchmark.py \
      --input-path "$INPUT_PATH" \
      --out-dir "$OUT_DIR" \
      --methods "$METHOD" \
      --workers 1 \
      --algorithm-mode speaker_tracking_single_active \
      --localization-backend capon_1src \
      --no-localization-vad-enabled \
      --fd-noise-covariance-mode estimated_target_subtractive_frozen \
      --fd-diag-load 0.01 \
      --fd-trace-diagonal-loading-factor "$TRACE_LOAD" \
      --fd-identity-blend-alpha "$IDENTITY_BLEND" \
      --fd-cov-ema-alpha "$EMA" \
      --fd-cov-update-scale-target-active "$ACTIVE" \
      --fd-cov-update-scale-target-inactive "$INACTIVE" \
      --target-activity-rnn-update-mode estimated_target_activity \
      --target-activity-detector-mode target_blocker_calibrated \
      --target-activity-detector-backend silero_fused \
      --target-activity-update-every-n-fast-frames 2 \
      --postfilter-method off \
      --own-voice-suppression-mode off \
      > "$LOG_PATH" 2>&1 &
  echo "PID: $! LOG: $LOG_PATH"
done

echo "All covariance smoothing sweeps launched."

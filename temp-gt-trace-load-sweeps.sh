#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/mkeller/miniconda3/bin/python}"
ROOT_DIR="/home/mkeller/mkeller/signal-processing-research"
cd "$ROOT_DIR"

INPUT_PATH="${1:-data-collection/gym-take-two-mar-17}"
METHOD="${METHOD:-mvdr_fd}"
OUT_PREFIX="${OUT_PREFIX:-beamforming/benchmark/_gt_trace_load_sweeps}"

TRACE_LOADS=(0 0.001 0.01 0.05)

for LOAD in "${TRACE_LOADS[@]}"; do
  SLUG="${LOAD//./p}"
  OUT_DIR="${OUT_PREFIX}_${METHOD}_load_${SLUG}"
  LOG_PATH="${OUT_DIR}.log"

  echo "Starting METHOD=${METHOD} TRACE_LOAD=${LOAD} OUT_DIR=${OUT_DIR}"
  echo "Log: ${LOG_PATH}"

  nohup env PYTHONPATH=. "$PYTHON_BIN" beamforming/benchmark/data_collection_benchmark.py \
      --input-path "$INPUT_PATH" \
      --out-dir "$OUT_DIR" \
      --methods "$METHOD" \
      --workers 1 \
      --algorithm-mode speaker_tracking_single_active \
      --use-ground-truth-doa-override \
      --localization-backend capon_1src \
      --no-localization-vad-enabled \
      --fd-noise-covariance-mode estimated_target_subtractive_frozen \
      --fd-diag-load 0.01 \
      --fd-trace-diagonal-loading-factor "$LOAD" \
      --target-activity-rnn-update-mode estimated_target_activity \
      --target-activity-detector-mode target_blocker_calibrated \
      --target-activity-detector-backend silero_fused \
      --target-activity-update-every-n-fast-frames 2 \
      --postfilter-method off \
      --own-voice-suppression-mode off \
      > "$LOG_PATH" 2>&1 &
  echo "PID: $!"
done

echo "All sweeps launched."

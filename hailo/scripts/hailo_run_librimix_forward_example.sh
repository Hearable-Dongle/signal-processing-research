#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

RUN_TS="${HAILO_RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="hailo/module_runs/${RUN_TS}_librimix_forward"
mkdir -p "$RUN_DIR"

PY="hailo/to-onnx-env/bin/python"

"$PY" -m hailo.librimix_hailo_forward_example \
  --output_dir "$RUN_DIR" \
  ${MIX_WAV:+--mix_wav "$MIX_WAV"} \
  ${LIBRIMIX_ROOT:+--librimix_root "$LIBRIMIX_ROOT"} \
  ${SPLIT:+--split "$SPLIT"} \
  ${MODEL_ID:+--model_id "$MODEL_ID"} | tee "${RUN_DIR}/run.log"

echo "[DONE] outputs in ${RUN_DIR}"
echo "Metrics: ${RUN_DIR}/librimix_hailo_forward_metrics.json"

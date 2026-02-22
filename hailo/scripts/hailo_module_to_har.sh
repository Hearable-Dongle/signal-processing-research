#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <module> <run_tag> [export args ...]"
  exit 2
fi

MODULE="$1"
RUN_TAG="$2"
shift 2

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

ONNX_PY="hailo/to-onnx-env/bin/python"
HAILO_PY="hailo/to-hailo-env/bin/python"

TS="${HAILO_RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="hailo/module_runs/${TS}"
SUMMARY="${HAILO_SUMMARY_PATH:-${RUN_DIR}/summary.tsv}"
mkdir -p "$RUN_DIR"

RAW_ONNX="${RUN_DIR}/${RUN_TAG}_raw.onnx"
PATCHED_ONNX="${RUN_DIR}/${RUN_TAG}_patched.onnx"
HAR_PATH="${RUN_DIR}/${RUN_TAG}.har"
LOG_PATH="${RUN_DIR}/${RUN_TAG}.log"

if [[ ! -f "$SUMMARY" ]]; then
  echo -e "stage\tmodule\trun_tag\tonnx_path\thar_path\thar_success\texception_type\texception_head\tlog_path" > "$SUMMARY"
fi

set +e
echo "[START] module=${MODULE} run_tag=${RUN_TAG} ts=$(date -Iseconds)" > "$LOG_PATH"
"$ONNX_PY" -m hailo.export_hailo_module_to_onnx --module "$MODULE" --output "$RAW_ONNX" "$@" >> "$LOG_PATH" 2>&1
RC=$?
if [[ $RC -eq 0 ]]; then
  "$ONNX_PY" -m hailo.patch_onnx "$RAW_ONNX" "$PATCHED_ONNX" >> "$LOG_PATH" 2>&1
  RC=$?
fi
if [[ $RC -eq 0 ]]; then
  "$HAILO_PY" -m hailo.onnx_to_hailo "$PATCHED_ONNX" "$HAR_PATH" --model_name "$RUN_TAG" --hw_arch hailo8 >> "$LOG_PATH" 2>&1
  RC=$?
fi
echo "[END] module=${MODULE} run_tag=${RUN_TAG} rc=${RC} ts=$(date -Iseconds)" >> "$LOG_PATH"
set -e

HAR_SUCCESS="false"
if [[ $RC -eq 0 && -f "$HAR_PATH" ]]; then
  HAR_SUCCESS="true"
fi

EXC_TYPE=""
EXC_HEAD=""
if [[ "$HAR_SUCCESS" != "true" ]]; then
  EXC_TYPE="$(rg -o "[A-Za-z]+Exception" "$LOG_PATH" -N | tail -n1 || true)"
  EXC_HEAD="$(rg -n "Translation failed|Step 3 Failed|error|Error|Exception" "$LOG_PATH" -S | tail -n1 | sed 's/\t/ /g' || true)"
fi

ROW_TMP="$(mktemp "${RUN_DIR}/.row.XXXXXX")"
echo -e "${HAILO_STAGE:-unknown}\t${MODULE}\t${RUN_TAG}\t${PATCHED_ONNX}\t${HAR_PATH}\t${HAR_SUCCESS}\t${EXC_TYPE}\t${EXC_HEAD}\t${LOG_PATH}" > "$ROW_TMP"
cat "$ROW_TMP" >> "$SUMMARY"
rm -f "$ROW_TMP"

if [[ "$HAR_SUCCESS" == "true" ]]; then
  echo "[PASS] ${MODULE} (${RUN_TAG}) -> ${HAR_PATH}"
else
  echo "[FAIL] ${MODULE} (${RUN_TAG}) rc=${RC} log=${LOG_PATH}"
  exit 1
fi

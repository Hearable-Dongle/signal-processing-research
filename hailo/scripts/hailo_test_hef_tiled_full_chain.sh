#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

RUN_TS="${HAILO_RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="hailo/module_runs/${RUN_TS}"
mkdir -p "$RUN_DIR"

PY="hailo/to-onnx-env/bin/python"
BACKEND="${BACKEND:-torch_proxy}"
DECODER_SUMMARY_TSV="${DECODER_SUMMARY_TSV:-hailo/module_runs/20260223_052929/allocator_mapping_fixes_summary.tsv}"
if [[ -z "${MASKER_SUMMARY_TSV:-}" ]]; then
  MASKER_SUMMARY_TSV="$(ls -1t hailo/module_runs/*/masker_allocator_fixes_summary.tsv 2>/dev/null | head -n1 || true)"
fi
if [[ -z "${MASKER_SUMMARY_TSV:-}" || ! -f "$MASKER_SUMMARY_TSV" ]]; then
  echo "[error] MASKER_SUMMARY_TSV is not set and no masker_allocator_fixes_summary.tsv was found."
  exit 2
fi
SUMMARY="${RUN_DIR}/hef_tiled_full_chain_summary.tsv"
echo -e "run_tag\tbackend\twav_t\ttile_w\tsuccess\toutput_json" > "$SUMMARY"

run_case() {
  local tag="$1"
  local wav_t="$2"
  local tile_w="$3"
  local out_json="${RUN_DIR}/${tag}.json"
  local ok="false"
  set +e
  "$PY" -m hailo.hef_tiled_full_chain \
    --output_json "$out_json" \
    --backend "$BACKEND" \
    --decoder_summary_tsv "$DECODER_SUMMARY_TSV" \
    --masker_summary_tsv "$MASKER_SUMMARY_TSV" \
    --wav_t "$wav_t" \
    --tile_w "$tile_w" > "${RUN_DIR}/${tag}.log" 2>&1
  local rc=$?
  set -e
  if [[ $rc -eq 0 ]]; then
    ok="true"
  fi
  echo -e "${tag}\t${BACKEND}\t${wav_t}\t${tile_w}\t${ok}\t${out_json}" >> "$SUMMARY"
  if [[ "$ok" == "true" ]]; then
    echo "[PASS] ${tag}"
  else
    echo "[FAIL] ${tag} (see ${RUN_DIR}/${tag}.log)"
  fi
}

run_case "hef_full_chain_wav16000_tile256" 16000 256
run_case "hef_full_chain_wav16000_tile512" 16000 512
run_case "hef_full_chain_wav8192_tile256" 8192 256

echo "[DONE] run_ts=${RUN_TS}"
echo "Summary: ${SUMMARY}"

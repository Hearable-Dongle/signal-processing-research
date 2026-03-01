#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

RUN_TS="${HAILO_RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="hailo/module_runs/${RUN_TS}"
mkdir -p "$RUN_DIR"

PY="hailo/to-onnx-env/bin/python"
SUMMARY_TSV="${SUMMARY_TSV:-hailo/module_runs/20260223_052929/allocator_mapping_fixes_summary.tsv}"
BACKEND="${BACKEND:-torch_proxy}"

SUMMARY="${RUN_DIR}/hef_tiled_decoder_path_summary.tsv"
echo -e "run_tag\tbackend\tlatent_w\ttile_w\tsuccess\toutput_json" > "$SUMMARY"

run_case() {
  local tag="$1"
  local latent_w="$2"
  local tile_w="$3"
  local out_json="${RUN_DIR}/${tag}.json"
  local ok="false"
  set +e
  "$PY" -m hailo.hef_tiled_decoder_path \
    --output_json "$out_json" \
    --summary_tsv "$SUMMARY_TSV" \
    --backend "$BACKEND" \
    --latent_w "$latent_w" \
    --tile_w "$tile_w" > "${RUN_DIR}/${tag}.log" 2>&1
  local rc=$?
  set -e
  if [[ $rc -eq 0 ]]; then
    ok="true"
  fi
  echo -e "${tag}\t${BACKEND}\t${latent_w}\t${tile_w}\t${ok}\t${out_json}" >> "$SUMMARY"
  if [[ "$ok" == "true" ]]; then
    echo "[PASS] ${tag}"
  else
    echo "[FAIL] ${tag} (see ${RUN_DIR}/${tag}.log)"
  fi
}

run_case "hef_tile_lat2000_w256" 2000 256
run_case "hef_tile_lat1024_w256" 1024 256
run_case "hef_tile_lat2000_w512" 2000 512

echo "[DONE] run_ts=${RUN_TS}"
echo "Summary: ${SUMMARY}"

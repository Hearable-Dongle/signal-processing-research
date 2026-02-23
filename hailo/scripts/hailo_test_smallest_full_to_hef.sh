#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}" )/../.." && pwd)"
cd "$ROOT_DIR"

RUN_TS="${HAILO_RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
export HAILO_RUN_TS="$RUN_TS"
RUN_DIR="hailo/module_runs/${RUN_TS}"
mkdir -p "$RUN_DIR"

CALIB_NPZ="${CALIB_NPZ:-hailo/calibration_1000ms_16k_64.npz}"
HW_ARCH="${HW_ARCH:-hailo8}"
USE_QUICK_OPT="${USE_QUICK_OPT:-1}"
STOP_ON_FIRST_SUCCESS="${STOP_ON_FIRST_SUCCESS:-1}"
CLEANUP_INTERMEDIATES="${CLEANUP_INTERMEDIATES:-1}"
CLEANUP_REMOVE_HAR_ON_PASS="${CLEANUP_REMOVE_HAR_ON_PASS:-0}"

SUMMARY="${RUN_DIR}/smallest_full_hef_summary.tsv"
if [[ ! -f "$SUMMARY" ]]; then
  echo -e "run_tag\tconfig\thar_success\thef_success\texception_type\tfailure_head\thar_path\thef_path\thar_log\thef_log\tfailure_log" > "$SUMMARY"
fi

run_case() {
  local tag="$1"
  local n_filters="$2"
  local bn_chan="$3"
  local hid_chan="$4"
  local skip_chan="$5"

  local har_path="${RUN_DIR}/${tag}.har"
  local hef_path="${RUN_DIR}/${tag}.hef"
  local har_log="${RUN_DIR}/${tag}.log"
  local hef_log="${RUN_DIR}/${tag}_hef.log"
  local fail_log="${RUN_DIR}/${tag}_compile_failure.txt"
  local config="n_src=2,n_filters=${n_filters},bn_chan=${bn_chan},hid_chan=${hid_chan},skip_chan=${skip_chan},n_blocks=1,n_repeats=1,time_len=16000"

  echo "[CASE] ${tag} (${config})"

  local har_success="false"
  set +e
  HAILO_STAGE="stage_smallest_full" HAILO_SUMMARY_PATH="${RUN_DIR}/summary.tsv" ./hailo/scripts/hailo_module_to_har.sh hailo_convtasnet "$tag" \
    --n_src 2 --n_filters "$n_filters" --bn_chan "$bn_chan" --hid_chan "$hid_chan" --skip_chan "$skip_chan" \
    --n_blocks 1 --n_repeats 1 --truncate_k_blocks 0 \
    --mask_mul_mode normal --skip_topology_mode project --decoder_mode conv1x1_head --time_len 16000
  local har_rc=$?
  set -e

  if [[ $har_rc -eq 0 && -f "$har_path" ]]; then
    har_success="true"
  fi

  local hef_success="false"
  local exc_type=""
  local failure_head=""

  if [[ "$har_success" == "true" ]]; then
    set +e
    if [[ "$USE_QUICK_OPT" == "1" ]]; then
      hailo/to-hailo-env/bin/python -m hailo.har_to_hef \
        "$har_path" "$hef_path" \
        --model_name "$tag" \
        --hw_arch "$HW_ARCH" \
        --calib_npz "$CALIB_NPZ" \
        --quick_opt \
        --log_failed_layers_path "$fail_log" > "$hef_log" 2>&1
    else
      hailo/to-hailo-env/bin/python -m hailo.har_to_hef \
        "$har_path" "$hef_path" \
        --model_name "$tag" \
        --hw_arch "$HW_ARCH" \
        --calib_npz "$CALIB_NPZ" \
        --log_failed_layers_path "$fail_log" > "$hef_log" 2>&1
    fi
    local hef_rc=$?
    set -e

    if [[ $hef_rc -eq 0 && -f "$hef_path" ]]; then
      hef_success="true"
    else
      exc_type="$(rg -o "[A-Za-z]+Exception" "$hef_log" -N | tail -n1 || true)"
      failure_head="$(rg -n "Agent infeasible|Mapping Failed|BackendAllocatorException|Failed|Exception" "$hef_log" -S | tail -n1 | sed 's/\t/ /g' || true)"
    fi
  else
    exc_type="HARBuildFailed"
    failure_head="har_to_hef skipped because HAR build failed"
  fi

  echo -e "${tag}\t${config}\t${har_success}\t${hef_success}\t${exc_type}\t${failure_head}\t${har_path}\t${hef_path}\t${har_log}\t${hef_log}\t${fail_log}" >> "$SUMMARY"

  if [[ "$har_success" == "true" && "$hef_success" == "true" ]]; then
    echo "[PASS] ${tag}"
    if [[ "$CLEANUP_INTERMEDIATES" == "1" ]]; then
      rm -f "${RUN_DIR}/${tag}_raw.onnx" "${RUN_DIR}/${tag}_patched.onnx" "${RUN_DIR}/${tag}_raw.onnx.data"
      if [[ "$CLEANUP_REMOVE_HAR_ON_PASS" == "1" ]]; then
        rm -f "$har_path"
      fi
    fi
    return 10
  fi

  if [[ "$har_success" == "true" ]]; then
    echo "[HEF FAIL] ${tag}"
  else
    echo "[HAR FAIL] ${tag}"
  fi
  if [[ "$CLEANUP_INTERMEDIATES" == "1" ]]; then
    rm -f "${RUN_DIR}/${tag}_raw.onnx" "${RUN_DIR}/${tag}_patched.onnx" "${RUN_DIR}/${tag}_raw.onnx.data"
  fi
  return 0
}

# Keep architecture shape realistic (time_len=16000, n_blocks=1, n_repeats=1),
# scale channels down only as needed.
cases=(
  "smallest_full_f256_bn64_hid128_skip128 256 64 128 128"
  "smallest_full_f256_bn48_hid96_skip96 256 48 96 96"
  "smallest_full_f256_bn32_hid64_skip64 256 32 64 64"
)

for row in "${cases[@]}"; do
  # shellcheck disable=SC2086
  run_case $row
  rc=$?
  if [[ "$STOP_ON_FIRST_SUCCESS" == "1" && $rc -eq 10 ]]; then
    echo "[STOP] first HEF success achieved"
    break
  fi
done

echo "[DONE] Smallest full HEF run_ts=${RUN_TS}"
echo "Summary: ${SUMMARY}"

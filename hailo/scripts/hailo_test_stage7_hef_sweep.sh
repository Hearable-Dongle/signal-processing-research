#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

RUN_TS="${HAILO_RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
export HAILO_RUN_TS="$RUN_TS"

RUN_DIR="hailo/module_runs/${RUN_TS}"
SUMMARY="${RUN_DIR}/hef_summary.tsv"
mkdir -p "$RUN_DIR"

if [[ ! -f "$SUMMARY" ]]; then
  echo -e "run_tag\thar_path\thef_path\thef_success\texception_type\tfailure_head\tcompile_log\tfail_log" > "$SUMMARY"
fi

CALIB_NPZ="${CALIB_NPZ:-hailo/calibration_1000ms_16k_64.npz}"
HW_ARCH="${HW_ARCH:-hailo8}"

run_case() {
  local run_tag="$1"
  local n_blocks="$2"
  local n_repeats="$3"
  local bn_chan="$4"
  local hid_chan="$5"

  local har_path="${RUN_DIR}/${run_tag}.har"
  local hef_path="${RUN_DIR}/${run_tag}.hef"
  local compile_log="${RUN_DIR}/${run_tag}_hef.log"
  local fail_log="${RUN_DIR}/${run_tag}_compile_failure.txt"

  echo "[CASE] ${run_tag} (blocks=${n_blocks}, repeats=${n_repeats}, bn=${bn_chan}, hid=${hid_chan})"

  ./hailo/scripts/hailo_module_to_har.sh hailo_convtasnet "$run_tag" \
    --n_src 2 --n_filters 256 --bn_chan "$bn_chan" --hid_chan "$hid_chan" --skip_chan 128 \
    --n_blocks "$n_blocks" --n_repeats "$n_repeats" --truncate_k_blocks 0 \
    --mask_mul_mode normal --skip_topology_mode project --decoder_mode conv1x1_head --time_len 16000

  set +e
  hailo/to-hailo-env/bin/python -m hailo.har_to_hef \
    "$har_path" "$hef_path" \
    --model_name "$run_tag" \
    --hw_arch "$HW_ARCH" \
    --calib_npz "$CALIB_NPZ" \
    --quick_opt \
    --log_failed_layers_path "$fail_log" > "$compile_log" 2>&1
  local rc=$?
  set -e

  local hef_success="false"
  if [[ $rc -eq 0 && -f "$hef_path" ]]; then
    hef_success="true"
  fi

  local exc_type=""
  local fail_head=""
  if [[ "$hef_success" != "true" ]]; then
    exc_type="$(rg -o "[A-Za-z]+Exception" "$compile_log" -N | tail -n1 || true)"
    fail_head="$(rg -n "Agent infeasible|Mapping Failed|BackendAllocatorException|Failed|Exception" "$compile_log" -S | tail -n1 | sed 's/\t/ /g' || true)"
  fi

  echo -e "${run_tag}\t${har_path}\t${hef_path}\t${hef_success}\t${exc_type}\t${fail_head}\t${compile_log}\t${fail_log}" >> "$SUMMARY"

  if [[ "$hef_success" == "true" ]]; then
    echo "[HEF PASS] ${run_tag} -> ${hef_path}"
  else
    echo "[HEF FAIL] ${run_tag} rc=${rc} log=${compile_log}"
  fi
}

# Baseline target config first.
run_case "s7_hef_full_b8_r3_bn128_hid256" 8 3 128 256

# Channel reductions at fixed depth.
run_case "s7_hef_full_b8_r3_bn96_hid192" 8 3 96 192
run_case "s7_hef_full_b8_r3_bn64_hid128" 8 3 64 128

# Depth reductions.
run_case "s7_hef_mid_b6_r3_bn96_hid192" 6 3 96 192
run_case "s7_hef_small_b4_r2_bn64_hid128" 4 2 64 128

echo "[DONE] Stage7 HEF sweep run_ts=${RUN_TS}"
echo "Summary: ${SUMMARY}"

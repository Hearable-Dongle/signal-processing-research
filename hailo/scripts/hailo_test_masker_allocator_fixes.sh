#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

RUN_TS="${HAILO_RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
export HAILO_RUN_TS="$RUN_TS"
RUN_DIR="hailo/module_runs/${RUN_TS}"
mkdir -p "$RUN_DIR"

HW_ARCH="${HW_ARCH:-hailo8}"
USE_QUICK_OPT="${USE_QUICK_OPT:-1}"
CLEANUP_INTERMEDIATES="${CLEANUP_INTERMEDIATES:-1}"
CLEANUP_REMOVE_HAR_ON_PASS="${CLEANUP_REMOVE_HAR_ON_PASS:-0}"
CASE_FILTER="${CASE_FILTER:-}"
PROFILE="${PROFILE:-quick}" # quick|full
MODEL_STATE_DICT="${MODEL_STATE_DICT:-}"

SUMMARY="${RUN_DIR}/masker_allocator_fixes_summary.tsv"
if [[ ! -f "$SUMMARY" ]]; then
  echo -e "stage\tmodule\trun_tag\tinput_len\tinput_chan\tcalib_npz\thar_path\thef_path\thar_success\thef_success\texception_type\tfailure_head\thar_log\thef_log\tfailure_log" > "$SUMMARY"
fi

mk_calib_npz() {
  local path="$1"
  local n_samples="$2"
  local t_len="$3"
  local c_chan="$4"
  python - <<PY
import numpy as np
p = ${path@Q}
n = int(${n_samples@Q})
t = int(${t_len@Q})
c = int(${c_chan@Q})
rng = np.random.default_rng(1337)
calib = rng.uniform(-1.0, 1.0, size=(n, 1, t, c)).astype(np.float32)
np.savez_compressed(p, calib_data=calib)
print(p)
PY
}

in_filter() {
  local tag="$1"
  if [[ -z "$CASE_FILTER" ]]; then
    return 0
  fi
  IFS=',' read -r -a arr <<< "$CASE_FILTER"
  local x
  for x in "${arr[@]}"; do
    if [[ "$x" == "$tag" ]]; then
      return 0
    fi
  done
  return 1
}

run_case() {
  local stage="$1"
  local module="$2"
  local tag="$3"
  local input_len="$4"
  local input_chan="$5"
  shift 5

  if ! in_filter "$tag"; then
    echo "[SKIP] ${tag}"
    return 0
  fi

  local calib_npz="${RUN_DIR}/calib_${input_len}_c${input_chan}.npz"
  mk_calib_npz "$calib_npz" 16 "$input_len" "$input_chan" >/dev/null

  local har_path="${RUN_DIR}/${tag}.har"
  local hef_path="${RUN_DIR}/${tag}.hef"
  local har_log="${RUN_DIR}/${tag}.log"
  local hef_log="${RUN_DIR}/${tag}_hef.log"
  local fail_log="${RUN_DIR}/${tag}_compile_failure.txt"

  echo "[CASE] ${tag} (module=${module}, input_len=${input_len}, input_chan=${input_chan})"

  local har_success="false"
  set +e
  HAILO_STAGE="$stage" HAILO_SUMMARY_PATH="${RUN_DIR}/summary.tsv" ./hailo/scripts/hailo_module_to_har.sh "$module" "$tag" "$@"
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
        --calib_npz "$calib_npz" \
        --input_length_policy error \
        --quick_opt \
        --log_failed_layers_path "$fail_log" > "$hef_log" 2>&1
    else
      hailo/to-hailo-env/bin/python -m hailo.har_to_hef \
        "$har_path" "$hef_path" \
        --model_name "$tag" \
        --hw_arch "$HW_ARCH" \
        --calib_npz "$calib_npz" \
        --input_length_policy error \
        --log_failed_layers_path "$fail_log" > "$hef_log" 2>&1
    fi
    local hef_rc=$?
    set -e

    if [[ $hef_rc -eq 0 && -f "$hef_path" ]]; then
      hef_success="true"
    else
      exc_type="$(rg -o "[A-Za-z]+(Exception|Error)" "$hef_log" -N | tail -n1 || true)"
      failure_head="$(rg -n "BadInputsShape|Unexpected model input|Agent infeasible|Mapping Failed|BackendAllocatorException|Failed|Exception|Error" "$hef_log" -S | tail -n1 | sed 's/\t/ /g' || true)"
    fi
  else
    exc_type="HARBuildFailed"
    failure_head="har_to_hef skipped because HAR build failed"
  fi

  echo -e "${stage}\t${module}\t${tag}\t${input_len}\t${input_chan}\t${calib_npz}\t${har_path}\t${hef_path}\t${har_success}\t${hef_success}\t${exc_type}\t${failure_head}\t${har_log}\t${hef_log}\t${fail_log}" >> "$SUMMARY"

  if [[ "$har_success" == "true" && "$hef_success" == "true" ]]; then
    echo "[PASS] ${tag}"
  elif [[ "$har_success" == "true" ]]; then
    echo "[HEF FAIL] ${tag}"
  else
    echo "[HAR FAIL] ${tag}"
  fi

  if [[ "$CLEANUP_INTERMEDIATES" == "1" ]]; then
    rm -f "${RUN_DIR}/${tag}_raw.onnx" "${RUN_DIR}/${tag}_patched.onnx" "${RUN_DIR}/${tag}_raw.onnx.data"
    if [[ "$CLEANUP_REMOVE_HAR_ON_PASS" == "1" && "$hef_success" == "true" ]]; then
      rm -f "$har_path"
    fi
  fi
}

COMMON_ARGS=(--n_src 2 --n_filters 256 --bn_chan 128 --hid_chan 256 --skip_chan 128 --n_blocks 1 --n_repeats 1 --truncate_k_blocks 1 --mask_mul_mode normal --skip_topology_mode project --decoder_mode conv1x1_head --time_len 16000)
if [[ -n "$MODEL_STATE_DICT" ]]; then
  COMMON_ARGS+=(--state_dict_path "$MODEL_STATE_DICT" --state_dict_strict 0)
fi

# 1) temporal sweeps for non-block masker paths
for L in 2000 1024 512 256 128; do
  run_case "stage1_temporal" "convtas_masker_only" "allocfix_temporal_masker_only_w${L}" "$L" 256 \
    "${COMMON_ARGS[@]}" --time_len $((L * 8))
  run_case "stage1_temporal" "convtas_masker_tcn_block0_only" "allocfix_temporal_masker_tcn0_w${L}" "$L" 128 \
    "${COMMON_ARGS[@]}" --time_len $((L * 8))
done

# 2) blockwise smoke
run_case "stage2_blocks" "convtas_masker_bottleneck_block" "allocfix_masker_bneck_o0_i0_w256" 256 64 \
  "${COMMON_ARGS[@]}" --block_chan 64 --out_block_idx 0 --in_block_idx 0 --time_len 2048
run_case "stage2_blocks" "convtas_masker_tcn0_inconv_block" "allocfix_masker_tcn0_in_o0_i0_w256" 256 64 \
  "${COMMON_ARGS[@]}" --block_chan 64 --out_block_idx 0 --in_block_idx 0 --time_len 2048
run_case "stage2_blocks" "convtas_masker_tcn0_depth_block" "allocfix_masker_tcn0_depth_b0_w256" 256 64 \
  "${COMMON_ARGS[@]}" --block_chan 64 --depth_block_idx 0 --time_len 2048
run_case "stage2_blocks" "convtas_masker_tcn0_res_block" "allocfix_masker_tcn0_res_o0_i0_w256" 256 64 \
  "${COMMON_ARGS[@]}" --block_chan 64 --out_block_idx 0 --in_block_idx 0 --time_len 2048
run_case "stage2_blocks" "convtas_masker_tcn0_skip_block" "allocfix_masker_tcn0_skip_o0_i0_w256" 256 64 \
  "${COMMON_ARGS[@]}" --block_chan 64 --out_block_idx 0 --in_block_idx 0 --time_len 2048
run_case "stage2_blocks" "convtas_masker_head_block" "allocfix_masker_head_o0_i0_w256" 256 64 \
  "${COMMON_ARGS[@]}" --block_chan 64 --out_block_idx 0 --in_block_idx 0 --time_len 2048

if [[ "$PROFILE" == "full" ]]; then
  for ob in 0 1; do
    for ib in 0 1 2 3; do
      run_case "stage2_blocks_full" "convtas_masker_bottleneck_block" "allocfix_masker_bneck_o${ob}_i${ib}_w256" 256 64 \
        "${COMMON_ARGS[@]}" --block_chan 64 --out_block_idx "$ob" --in_block_idx "$ib" --time_len 2048
    done
  done

  for ob in 0 1 2 3; do
    for ib in 0 1; do
      run_case "stage2_blocks_full" "convtas_masker_tcn0_inconv_block" "allocfix_masker_tcn0_in_o${ob}_i${ib}_w256" 256 64 \
        "${COMMON_ARGS[@]}" --block_chan 64 --out_block_idx "$ob" --in_block_idx "$ib" --time_len 2048
    done
  done

  for db in 0 1 2 3; do
    run_case "stage2_blocks_full" "convtas_masker_tcn0_depth_block" "allocfix_masker_tcn0_depth_b${db}_w256" 256 64 \
      "${COMMON_ARGS[@]}" --block_chan 64 --depth_block_idx "$db" --time_len 2048
  done

  for ob in 0 1; do
    for ib in 0 1 2 3; do
      run_case "stage2_blocks_full" "convtas_masker_tcn0_res_block" "allocfix_masker_tcn0_res_o${ob}_i${ib}_w256" 256 64 \
        "${COMMON_ARGS[@]}" --block_chan 64 --out_block_idx "$ob" --in_block_idx "$ib" --time_len 2048
      run_case "stage2_blocks_full" "convtas_masker_tcn0_skip_block" "allocfix_masker_tcn0_skip_o${ob}_i${ib}_w256" 256 64 \
        "${COMMON_ARGS[@]}" --block_chan 64 --out_block_idx "$ob" --in_block_idx "$ib" --time_len 2048
    done
  done

  for ob in 0 1 2 3 4 5 6 7; do
    for ib in 0 1; do
      run_case "stage2_blocks_full" "convtas_masker_head_block" "allocfix_masker_head_o${ob}_i${ib}_w256" 256 64 \
        "${COMMON_ARGS[@]}" --block_chan 64 --out_block_idx "$ob" --in_block_idx "$ib" --time_len 2048
    done
  done
fi

echo "[DONE] Masker allocator fixes run_ts=${RUN_TS}"
echo "Summary: ${SUMMARY}"

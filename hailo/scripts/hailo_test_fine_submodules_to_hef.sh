#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

RUN_TS="${HAILO_RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
export HAILO_RUN_TS="$RUN_TS"
RUN_DIR="hailo/module_runs/${RUN_TS}"
mkdir -p "$RUN_DIR"

CALIB_NPZ="${CALIB_NPZ:-hailo/calibration_1000ms_16k_64.npz}"
HW_ARCH="${HW_ARCH:-hailo8}"
USE_QUICK_OPT="${USE_QUICK_OPT:-1}"
CLEANUP_INTERMEDIATES="${CLEANUP_INTERMEDIATES:-1}"
CLEANUP_REMOVE_HAR_ON_PASS="${CLEANUP_REMOVE_HAR_ON_PASS:-0}"

# Optional comma-separated case filter.
CASE_FILTER="${CASE_FILTER:-}"

SUMMARY="${RUN_DIR}/fine_submodule_hef_summary.tsv"
if [[ ! -f "$SUMMARY" ]]; then
  echo -e "stage\tmodule\trun_tag\thar_path\thef_path\thar_success\thef_success\texception_type\tfailure_head\thar_log\thef_log\tfailure_log" > "$SUMMARY"
fi

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
  shift 3

  if ! in_filter "$tag"; then
    echo "[SKIP] ${tag}"
    return 0
  fi

  local har_path="${RUN_DIR}/${tag}.har"
  local hef_path="${RUN_DIR}/${tag}.hef"
  local har_log="${RUN_DIR}/${tag}.log"
  local hef_log="${RUN_DIR}/${tag}_hef.log"
  local fail_log="${RUN_DIR}/${tag}_compile_failure.txt"

  echo "[CASE] stage=${stage} module=${module} tag=${tag}"

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

  echo -e "${stage}\t${module}\t${tag}\t${har_path}\t${hef_path}\t${har_success}\t${hef_success}\t${exc_type}\t${failure_head}\t${har_log}\t${hef_log}\t${fail_log}" >> "$SUMMARY"

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

# Encoder finer split
run_case "stage_fine_encoder" "encoder_conv_only" "fine_encoder_conv_only" \
  --n_filters 256 --encdec_kernel_size 16 --encdec_stride 8 --time_len 16000

# Full-model finer split from same architectural family as hailo_convtasnet_k1
run_case "stage_fine_full" "convtas_encoder_only" "fine_full_encoder_only" \
  --n_src 2 --n_filters 256 --bn_chan 128 --hid_chan 256 --skip_chan 128 --n_blocks 1 --n_repeats 1 \
  --truncate_k_blocks 1 --mask_mul_mode bypass --skip_topology_mode project --decoder_mode conv1x1_head --time_len 16000

run_case "stage_fine_full" "convtas_masker_only" "fine_full_masker_only" \
  --n_src 2 --n_filters 256 --bn_chan 128 --hid_chan 256 --skip_chan 128 --n_blocks 1 --n_repeats 1 \
  --truncate_k_blocks 1 --mask_mul_mode bypass --skip_topology_mode project --decoder_mode conv1x1_head --time_len 16000

run_case "stage_fine_masker" "convtas_masker_bottleneck_only" "fine_full_masker_bottleneck_only" \
  --n_src 2 --n_filters 256 --bn_chan 128 --hid_chan 256 --skip_chan 128 --n_blocks 1 --n_repeats 1 \
  --truncate_k_blocks 1 --mask_mul_mode bypass --skip_topology_mode project --decoder_mode conv1x1_head --time_len 16000

run_case "stage_fine_masker" "convtas_masker_tcn_block0_only" "fine_full_masker_tcn_block0_only" \
  --n_src 2 --n_filters 256 --bn_chan 128 --hid_chan 256 --skip_chan 128 --n_blocks 1 --n_repeats 1 \
  --truncate_k_blocks 1 --mask_mul_mode bypass --skip_topology_mode project --decoder_mode conv1x1_head --time_len 16000

run_case "stage_fine_masker" "convtas_masker_mask_head_only" "fine_full_masker_mask_head_only" \
  --n_src 2 --n_filters 256 --bn_chan 128 --hid_chan 256 --skip_chan 128 --n_blocks 1 --n_repeats 1 \
  --truncate_k_blocks 1 --mask_mul_mode bypass --skip_topology_mode project --decoder_mode conv1x1_head --time_len 16000

run_case "stage_fine_full" "convtas_source_projector_only" "fine_full_source_projector_only" \
  --n_src 2 --n_filters 256 --bn_chan 128 --hid_chan 256 --skip_chan 128 --n_blocks 1 --n_repeats 1 \
  --truncate_k_blocks 1 --mask_mul_mode bypass --skip_topology_mode project --decoder_mode conv1x1_head --time_len 16000

run_case "stage_fine_full" "convtas_decoder_pre_only" "fine_full_decoder_pre_only" \
  --n_src 2 --n_filters 256 --bn_chan 128 --hid_chan 256 --skip_chan 128 --n_blocks 1 --n_repeats 1 \
  --truncate_k_blocks 1 --mask_mul_mode bypass --skip_topology_mode project --decoder_mode conv1x1_head --time_len 16000

run_case "stage_fine_full" "convtas_decoder_only" "fine_full_decoder_only" \
  --n_src 2 --n_filters 256 --bn_chan 128 --hid_chan 256 --skip_chan 128 --n_blocks 1 --n_repeats 1 \
  --truncate_k_blocks 1 --mask_mul_mode bypass --skip_topology_mode project --decoder_mode conv1x1_head --time_len 16000

echo "[DONE] Fine submodule HEF test run_ts=${RUN_TS}"
echo "Summary: ${SUMMARY}"

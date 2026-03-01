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

SUMMARY="${RUN_DIR}/module_hef_calib_match_summary.tsv"
if [[ ! -f "$SUMMARY" ]]; then
  echo -e "stage\tmodule\trun_tag\tinput_len\tcalib_npz\thar_path\thef_path\thar_success\thef_success\texception_type\tfailure_head\thar_log\thef_log\tfailure_log" > "$SUMMARY"
fi

mk_calib_npz() {
  local path="$1"
  local n_samples="$2"
  local t_len="$3"
  python - <<PY
import numpy as np
p = ${path@Q}
n = int(${n_samples@Q})
t = int(${t_len@Q})
rng = np.random.default_rng(1337)
calib = rng.uniform(-1.0, 1.0, size=(n, 1, t, 1)).astype(np.float32)
np.savez_compressed(p, calib_data=calib)
print(p)
PY
}

CALIB_1024="${RUN_DIR}/calib_1024.npz"
CALIB_2000="${RUN_DIR}/calib_2000.npz"
CALIB_16000="${RUN_DIR}/calib_16000.npz"
mk_calib_npz "$CALIB_1024" 16 1024 >/dev/null
mk_calib_npz "$CALIB_2000" 16 2000 >/dev/null
mk_calib_npz "$CALIB_16000" 16 16000 >/dev/null

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
  local calib_npz="$5"
  shift 5

  if ! in_filter "$tag"; then
    echo "[SKIP] ${tag}"
    return 0
  fi

  local har_path="${RUN_DIR}/${tag}.har"
  local hef_path="${RUN_DIR}/${tag}.hef"
  local har_log="${RUN_DIR}/${tag}.log"
  local hef_log="${RUN_DIR}/${tag}_hef.log"
  local fail_log="${RUN_DIR}/${tag}_compile_failure.txt"

  echo "[CASE] ${tag} (input_len=${input_len}, calib=$(basename "$calib_npz"))"

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
      exc_type="$(rg -o "[A-Za-z]+Exception" "$hef_log" -N | tail -n1 || true)"
      failure_head="$(rg -n "BadInputsShape|Agent infeasible|Mapping Failed|BackendAllocatorException|Failed|Exception" "$hef_log" -S | tail -n1 | sed 's/\t/ /g' || true)"
    fi
  else
    exc_type="HARBuildFailed"
    failure_head="har_to_hef skipped because HAR build failed"
  fi

  echo -e "${stage}\t${module}\t${tag}\t${input_len}\t${calib_npz}\t${har_path}\t${hef_path}\t${har_success}\t${hef_success}\t${exc_type}\t${failure_head}\t${har_log}\t${hef_log}\t${fail_log}" >> "$SUMMARY"

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

run_case "stage1_primitives" "norm" "calib_match_norm_affine" 1024 "$CALIB_1024" --in_chan 128 --norm_mode affine --time_len 1024
run_case "stage1_primitives" "norm" "calib_match_norm_identity" 1024 "$CALIB_1024" --in_chan 128 --norm_mode identity --time_len 1024
run_case "stage1_primitives" "activation" "calib_match_activation_sigmoid" 1024 "$CALIB_1024" --in_chan 128 --mask_act sigmoid --time_len 1024
run_case "stage2_block" "conv1d_block" "calib_match_conv1d_block_skip0" 1024 "$CALIB_1024" --in_chan 128 --hid_chan 256 --skip_chan 0 --kernel_size 3 --dilation 1 --time_len 1024
run_case "stage3_tdconvnet" "tdconvnet" "calib_match_tdconvnet_k1" 1024 "$CALIB_1024" --in_chan 128 --out_chan 128 --n_src 2 --bn_chan 128 --hid_chan 256 --skip_chan 128 --n_blocks 1 --n_repeats 1 --truncate_k_blocks 1 --time_len 1024 --mask_act sigmoid
run_case "stage4_encdec" "encoder" "calib_match_encoder" 16000 "$CALIB_16000" --n_filters 256 --encdec_kernel_size 16 --encdec_stride 8 --time_len 16000
run_case "stage4_encdec" "decoder" "calib_match_decoder_conv1x1" 2000 "$CALIB_2000" --in_chan 256 --n_src 2 --time_len 2000
run_case "stage5_full_k1" "hailo_convtasnet" "calib_match_hailo_convtasnet_k1" 16000 "$CALIB_16000" --n_src 2 --n_filters 256 --bn_chan 128 --hid_chan 256 --skip_chan 128 --n_blocks 1 --n_repeats 1 --truncate_k_blocks 1 --mask_mul_mode bypass --skip_topology_mode project --decoder_mode conv1x1_head --time_len 16000

echo "[DONE] Module HEF calib-match run_ts=${RUN_TS}"
echo "Summary: ${SUMMARY}"

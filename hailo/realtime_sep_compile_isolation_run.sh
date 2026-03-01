#!/usr/bin/env bash
set -euo pipefail

# Goal:
# Run a focused compile-isolation experiment sequence for Hailo export of
# real-time edge speaker separation. This script avoids broad sweeps and
# isolates the dominant compile blockers (skip/concat/mask-mul and TCN depth)
# to identify a compileable profile quickly.

cd /home/mkeller/mkeller/signal-processing-research

OUT_DIR="hailo/night_runs2"
TS=$(date +%Y%m%d_%H%M%S)
RUN_DIR="$OUT_DIR/$TS"
SUMMARY="$OUT_DIR/summary.tsv"
CALIB_NPZ=${CALIB_NPZ:-"hailo/calibration_1000ms_16k_64.npz"}
CASE_TIMEOUT_SEC=${CASE_TIMEOUT_SEC:-3600}
HEARTBEAT_SEC=${HEARTBEAT_SEC:-30}
RESUME_FROM_RUN_TAG=${RESUME_FROM_RUN_TAG:-""}
FULL_SWEEP=${FULL_SWEEP:-"false"}
FORCE_NEW_SUMMARY=${FORCE_NEW_SUMMARY:-"false"}

mkdir -p "$RUN_DIR"
mkdir -p "$OUT_DIR"

if [[ "$FORCE_NEW_SUMMARY" == "true" || ! -f "$SUMMARY" ]]; then
  echo -e "run_tag\texport_flags\tonnx_size_mb\tfailure_stage\texception_type\texception_head\treshape_layer_count\troot_cause_family\tfirst_10_failed_layers\tcontains_shortcut1\tcontains_concat1\tcontains_ew_mult1\thar_success\thef_success\tcompile_success\tcase_status\tcommand_status\ttimed_out\tduration_sec\tkey_trace\tlog_path\tfailure_json\thar_path\thef_path" > "$SUMMARY"
fi

RESUME_GUARD_REACHED="false"

trap 'echo "[TRAP_ERR] ts=$(date -Iseconds) line=${LINENO} cmd=${BASH_COMMAND}" >&2' ERR

json_get() {
  local json_path="$1"
  local expr="$2"
  python3 - "$json_path" "$expr" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
expr = sys.argv[2]
if not path.exists():
    print("")
    raise SystemExit(0)

data = json.loads(path.read_text())
if expr == "exception_type":
    print(data.get("exception_type", ""))
elif expr == "first10":
    layers = data.get("reshape_layers") or []
    if layers:
        print(",".join(layers[:10]))
    else:
        neg = data.get("negexp") or {}
        print(neg.get("layer", ""))
elif expr == "reshape_count":
    print(len(data.get("reshape_layers") or []))
elif expr == "exception_head":
    msg = (data.get("exception_message") or "").replace("\\t", " ").replace("\\n", " ")
    print(msg[:180])
elif expr == "root_cause_family":
    layers = set(data.get("reshape_layers") or [])
    msg = (data.get("exception_message") or "").lower()
    etype = (data.get("exception_type") or "").lower()
    has_shortcut = "shortcut1" in layers
    has_concat = "concat1" in layers
    has_ew_mult = "ew_mult1" in layers
    has_conv = any(x.startswith("conv") for x in layers)
    if "unsupported dimensions" in msg:
        print("unsupported_dimensions")
    elif "negativeslopeexponentnonfixable" in etype or "unsupported required slope" in msg:
        print("quantization_negative_exponent")
    elif has_shortcut and has_concat and has_ew_mult:
        print("skip_concat_maskmul")
    elif has_shortcut and has_concat:
        print("skip_concat")
    elif has_ew_mult and has_conv:
        print("maskmul_plus_conv_chain")
    elif has_ew_mult:
        print("maskmul")
    elif has_conv:
        print("conv_chain_layout")
    elif layers:
        print("reshape_allocator")
    else:
        print("unknown")
else:
    print("")
PY
}

infer_failure_stage() {
  local log_path="$1"
  if rg -q "Step 3 Failed|Translation failed" "$log_path"; then
    echo "translate"
    return
  fi
  if rg -q "Step 4 Failed" "$log_path"; then
    if rg -q "Starting Compilation\.\.\." "$log_path"; then
      echo "compile"
    elif rg -q "Starting Optimization" "$log_path"; then
      echo "optimize"
    else
      echo "optimize"
    fi
    return
  fi
  echo ""
}

already_completed() {
  local run_tag="$1"
  if [[ ! -f "$SUMMARY" ]]; then
    echo "false"
    return
  fi
  if awk -F '\t' -v tag="$run_tag" 'NR>1 && $1==tag {found=1} END{exit(found?0:1)}' "$SUMMARY"; then
    echo "true"
  else
    echo "false"
  fi
}

should_run_tag() {
  local run_tag="$1"

  if [[ "$(already_completed "$run_tag")" == "true" ]]; then
    echo "false"
    return
  fi

  if [[ -z "$RESUME_FROM_RUN_TAG" ]]; then
    echo "true"
    return
  fi

  if [[ "$RESUME_GUARD_REACHED" == "true" ]]; then
    echo "true"
    return
  fi

  if [[ "$run_tag" == "$RESUME_FROM_RUN_TAG" ]]; then
    RESUME_GUARD_REACHED="true"
    echo "true"
    return
  fi

  echo "false"
}

contains_token() {
  local token="$1"
  local log_path="$2"
  local fail_txt="$3"
  local fail_json="$4"
  if rg -q "$token" "$log_path" "$fail_txt" "$fail_json" 2>/dev/null; then
    echo "true"
  else
    echo "false"
  fi
}

extract_key_trace() {
  local log_path="$1"
  local trace
  trace=$(rg -n "Failed to add spatial reshapes|Reshape is needed for layers|Unsupported kernel type|Super deconv|Memory units capacity exceeded|BackendAllocatorException|Failed to produce compiled graph|Translation failed with error|NegativeSlopeExponentNonFixable" "$log_path" -S | tail -n 3 | sed 's/\t/ /g' | tr '\n' '|' || true)
  echo "$trace"
}

run_case() {
  local run_tag="$1"
  local disable_skip="$2"
  local mask_mul_mode="$3"
  local force_n_src_1="$4"
  local bypass_concat="$5"
  local skip_topology_mode="$6"
  local deconv_mode="$7"
  local truncate_k="$8"

  local base="convtas_${run_tag//-/_}"
  local log_path="$RUN_DIR/${run_tag}.log"
  local fail_txt="$RUN_DIR/${run_tag}.failure.txt"
  local fail_json="${fail_txt}.json"
  local hef_path="hailo/${base}.hef"
  local har_path="hailo/${base}.har"
  local patched_onnx="hailo/${base}_patched.onnx"

  rm -f "$fail_txt" "$fail_json"
  rm -f "$hef_path"

  if [[ "$(should_run_tag "$run_tag")" != "true" ]]; then
    echo "[SKIP] ${run_tag} already completed or before resume gate"
    return
  fi

  local start_epoch end_epoch duration_sec
  local start_ts end_ts
  start_epoch=$(date +%s)
  start_ts=$(date -Iseconds)
  echo "[START] ${run_tag} ts=${start_ts}"

  set +e
  local cmd_status=0
  local timed_out="false"

  local -a case_cmd=(
    env
    EXPORT_PROFILE=baseline
    ACT_REPLACE=relu
    NORM_REPLACE=identity
    NORM_MODE=channel
    NORM_EPS=1e-8
    CALIB_NPZ="$CALIB_NPZ"
    DISABLE_SKIP="$disable_skip"
    MASK_MUL_MODE="$mask_mul_mode"
    FORCE_N_SRC_1="$force_n_src_1"
    BYPASS_CONCAT="$bypass_concat"
    SKIP_TOPOLOGY_MODE="$skip_topology_mode"
    DECONV_MODE="$deconv_mode"
    TRUNCATE_K_BLOCKS="$truncate_k"
    LOG_FAILED_LAYERS_PATH="$fail_txt"
    ./hailo/run_conversion_flow.sh "$base" "${base}.har"
  )

  if [[ "$CASE_TIMEOUT_SEC" -gt 0 ]]; then
    timeout --signal=TERM --kill-after=30 "${CASE_TIMEOUT_SEC}s" "${case_cmd[@]}" > "$log_path" 2>&1 &
  else
    "${case_cmd[@]}" > "$log_path" 2>&1 &
  fi
  local run_pid=$!

  while kill -0 "$run_pid" 2>/dev/null; do
    sleep "$HEARTBEAT_SEC"
    if kill -0 "$run_pid" 2>/dev/null; then
      echo "[HEARTBEAT] ${run_tag} ts=$(date -Iseconds) pid=${run_pid}"
    fi
  done

  wait "$run_pid"
  cmd_status=$?
  if [[ "$cmd_status" -eq 124 ]]; then
    timed_out="true"
    echo "[TIMEOUT] ${run_tag} exceeded ${CASE_TIMEOUT_SEC}s" >> "$log_path"
  fi
  set -e

  local har_success="false"
  if [[ -f "$har_path" ]] || rg -q "Step 3 Success" "$log_path"; then
    har_success="true"
  fi

  local hef_success="false"
  if [[ $cmd_status -eq 0 && -f "$hef_path" ]] || rg -q "Step 4 Success" "$log_path"; then
    hef_success="true"
  fi
  local compile_success="$hef_success"

  local onnx_size_mb=""
  if [[ -f "$patched_onnx" ]]; then
    onnx_size_mb=$(python3 - <<PY
from pathlib import Path
p=Path("$patched_onnx")
print(f"{p.stat().st_size/(1024*1024):.3f}")
PY
)
  fi

  local failure_stage
  failure_stage=$(infer_failure_stage "$log_path")
  if [[ "$timed_out" == "true" ]]; then
    failure_stage="timeout"
  fi

  local exception_type
  exception_type=$(json_get "$fail_json" "exception_type")

  local first10
  first10=$(json_get "$fail_json" "first10")
  local exception_head
  exception_head=$(json_get "$fail_json" "exception_head")
  local reshape_layer_count
  reshape_layer_count=$(json_get "$fail_json" "reshape_count")
  local root_cause_family
  root_cause_family=$(json_get "$fail_json" "root_cause_family")

  local contains_shortcut1
  contains_shortcut1=$(contains_token "shortcut1" "$log_path" "$fail_txt" "$fail_json")
  local contains_concat1
  contains_concat1=$(contains_token "concat1" "$log_path" "$fail_txt" "$fail_json")
  local contains_ew_mult1
  contains_ew_mult1=$(contains_token "ew_mult1" "$log_path" "$fail_txt" "$fail_json")
  local key_trace
  key_trace=$(extract_key_trace "$log_path")

  local export_flags="disable_skip=${disable_skip},mask_mul_mode=${mask_mul_mode},force_n_src_1=${force_n_src_1},bypass_concat=${bypass_concat},skip_topology_mode=${skip_topology_mode},deconv_mode=${deconv_mode},truncate_k_blocks=${truncate_k}"
  local case_status="failed"
  if [[ "$compile_success" == "true" ]]; then
    case_status="compile_success"
  elif [[ "$har_success" == "true" ]]; then
    case_status="har_only_success"
  fi

  end_epoch=$(date +%s)
  end_ts=$(date -Iseconds)
  duration_sec=$((end_epoch - start_epoch))
  echo "[END] ${run_tag} rc=${cmd_status} case_status=${case_status} har_success=${har_success} hef_success=${hef_success} duration_sec=${duration_sec} ts=${end_ts}"

  local row_tmp
  row_tmp=$(mktemp "${OUT_DIR}/.summary_row.XXXXXX")
  echo -e "${run_tag}\t${export_flags}\t${onnx_size_mb}\t${failure_stage}\t${exception_type}\t${exception_head}\t${reshape_layer_count}\t${root_cause_family}\t${first10}\t${contains_shortcut1}\t${contains_concat1}\t${contains_ew_mult1}\t${har_success}\t${hef_success}\t${compile_success}\t${case_status}\t${cmd_status}\t${timed_out}\t${duration_sec}\t${key_trace}\t${log_path}\t${fail_json}\t${har_path}\t${hef_path}" > "$row_tmp"
  cat "$row_tmp" >> "$SUMMARY"
  rm -f "$row_tmp"

  echo "[DONE] ${run_tag} status=${cmd_status} har_success=${har_success} compile_success=${compile_success} log=${log_path}"
}

run_case "k1-project-mask-bypass-deconv-reduced64" "false" "bypass" "false" "false" "project" "reduced_deconv_64" "1"
run_case "k1-project-mask-bypass-conv1x1-head" "false" "bypass" "false" "false" "project" "conv1x1_head" "1"
run_case "k1-project-mask-bypass-deconv-reduced64-force-n-src-1" "false" "bypass" "true" "false" "project" "reduced_deconv_64" "1"
run_case "k1-project-mask-bypass-conv1x1-head-force-n-src-1" "false" "bypass" "true" "false" "project" "conv1x1_head" "1"

if [[ "$FULL_SWEEP" == "true" ]]; then
  run_case "baseline" "false" "normal" "false" "false" "concat" "grouped" "0"
  run_case "disable-skip" "true" "normal" "false" "false" "concat" "grouped" "0"
  run_case "mask-bypass" "false" "bypass" "false" "false" "concat" "grouped" "0"
  run_case "concat-bypass" "false" "normal" "false" "true" "concat" "grouped" "0"
  run_case "mask-and-concat-bypass" "false" "bypass" "false" "true" "concat" "grouped" "0"
  run_case "force-n-src-1" "false" "normal" "true" "false" "concat" "grouped" "0"
  run_case "truncate-k-1-project" "false" "normal" "false" "false" "project" "grouped" "1"
  run_case "truncate-k-1-project-mask-bypass" "false" "bypass" "false" "false" "project" "grouped" "1"
  run_case "truncate-k-1-project-mask-bypass-deconv-fallback" "false" "bypass" "false" "false" "project" "ungrouped_blockdiag" "1"
  run_case "truncate-k-1-project-mask-bypass-deconv-reduced128" "false" "bypass" "false" "false" "project" "reduced_deconv_128" "1"
  run_case "truncate-k-1-project-mask-bypass-deconv-reduced64" "false" "bypass" "false" "false" "project" "reduced_deconv_64" "1"
  run_case "truncate-k-1-project-mask-bypass-conv1x1-head" "false" "bypass" "false" "false" "project" "conv1x1_head" "1"

  for k in 1 2 4 8 12 16 24; do
    run_case "truncate-k-${k}" "false" "normal" "false" "false" "concat" "grouped" "$k"
  done
fi

echo "[RESULT] summary=${SUMMARY}"
echo "[RESULT] run_logs=${RUN_DIR}"

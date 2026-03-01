#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

BASE_TAG="${BASE_TAG:-end-to-end-hybrid-run}"
MODEL_ID="${MODEL_ID:-JorisCos/ConvTasNet_Libri2Mix_sepnoisy_16k}"
DEC_RUN_TS="${DEC_RUN_TS:-${BASE_TAG}-decoder}"
MASK_RUN_TS="${MASK_RUN_TS:-${BASE_TAG}-masker}"
STATE_DICT_PATH="${STATE_DICT_PATH:-/tmp/${BASE_TAG}_hailo_state_dict.pt}"
OUTPUT_TARBALL="${OUTPUT_TARBALL:-/tmp/${BASE_TAG}_hailo_true_hybrid_artifacts.tgz}"
RUN_PARALLEL="${RUN_PARALLEL:-1}"
DEC_PROFILE="${DEC_PROFILE:-full}"
MASK_PROFILE="${MASK_PROFILE:-quick}"

DEC_SUMMARY="hailo/module_runs/${DEC_RUN_TS}/allocator_mapping_fixes_summary.tsv"
MASK_SUMMARY="hailo/module_runs/${MASK_RUN_TS}/masker_allocator_fixes_summary.tsv"
AGG_SUMMARY="hailo/module_runs/${BASE_TAG}_summary.tsv"

mkdir -p "hailo/module_runs"

echo "[1/5] Building Hailo state dict from pretrained model"
hailo/to-onnx-env/bin/python -m hailo.build_hailo_state_dict_from_pretrained \
  --model_name "$MODEL_ID" \
  --output "$STATE_DICT_PATH"

run_decoder() {
  echo "[2/5] Producing decoder HEFs (PROFILE=${DEC_PROFILE}, RUN_TS=${DEC_RUN_TS})"
  MODEL_STATE_DICT="$STATE_DICT_PATH" \
  PROFILE="$DEC_PROFILE" \
  HAILO_RUN_TS="$DEC_RUN_TS" \
  ./hailo/scripts/hailo_test_allocator_mapping_fixes.sh
}

run_masker() {
  echo "[2/5] Producing masker HEFs (PROFILE=${MASK_PROFILE}, RUN_TS=${MASK_RUN_TS})"
  MODEL_STATE_DICT="$STATE_DICT_PATH" \
  PROFILE="$MASK_PROFILE" \
  HAILO_RUN_TS="$MASK_RUN_TS" \
  ./hailo/scripts/hailo_test_masker_allocator_fixes.sh
}

if [[ "$RUN_PARALLEL" == "1" ]]; then
  run_decoder &
  dec_pid=$!
  run_masker &
  mask_pid=$!
  wait "$dec_pid"
  wait "$mask_pid"
else
  run_decoder
  run_masker
fi

if [[ ! -f "$DEC_SUMMARY" ]]; then
  echo "[error] Missing decoder summary: $DEC_SUMMARY"
  exit 2
fi
if [[ ! -f "$MASK_SUMMARY" ]]; then
  echo "[error] Missing masker summary: $MASK_SUMMARY"
  exit 2
fi

echo "[3/5] Validating required decoder/masker rows"

dec_bad="$(awk -F'\t' 'NR>1 && $3 ~ /^allocfix_full_source_/ && $10 != "true" {print $3}' "$DEC_SUMMARY")"
if [[ -n "$dec_bad" ]]; then
  echo "[error] Some required decoder full-source rows failed:"
  echo "$dec_bad"
  exit 3
fi

for tag in \
  allocfix_block_decpre_h0_o0_i0_w256 \
  allocfix_block_decpre_h0_o0_i1_w256 \
  allocfix_block_dechead_s0_i0_w256 \
  allocfix_block_dechead_s0_i1_w256
do
  ok="$(awk -F'\t' -v tag="$tag" 'NR>1 && $3==tag {print $10}' "$DEC_SUMMARY" | tail -n1)"
  if [[ "$ok" != "true" ]]; then
    echo "[error] Required decoder tag missing or failed: $tag"
    exit 3
  fi
done

for tag in \
  allocfix_masker_bneck_o0_i0_w256 \
  allocfix_masker_tcn0_in_o0_i0_w256 \
  allocfix_masker_tcn0_depth_b0_w256 \
  allocfix_masker_tcn0_res_o0_i0_w256 \
  allocfix_masker_tcn0_skip_o0_i0_w256 \
  allocfix_masker_head_o0_i0_w256
do
  ok="$(awk -F'\t' -v tag="$tag" 'NR>1 && $3==tag {print $10}' "$MASK_SUMMARY" | tail -n1)"
  if [[ "$ok" != "true" ]]; then
    echo "[error] Required masker tag missing or failed: $tag"
    exit 3
  fi
done

dec_hef_count="$(awk -F'\t' 'NR>1 && $10=="true" {n++} END{print n+0}' "$DEC_SUMMARY")"
mask_hef_count="$(awk -F'\t' 'NR>1 && $10=="true" {n++} END{print n+0}' "$MASK_SUMMARY")"

{
  echo -e "role\trun_ts\tsummary_tsv\thef_success_count\tmodel_id\tstate_dict"
  echo -e "decoder\t${DEC_RUN_TS}\t${DEC_SUMMARY}\t${dec_hef_count}\t${MODEL_ID}\t${STATE_DICT_PATH}"
  echo -e "masker\t${MASK_RUN_TS}\t${MASK_SUMMARY}\t${mask_hef_count}\t${MODEL_ID}\t${STATE_DICT_PATH}"
} > "$AGG_SUMMARY"

echo "[4/5] Packaging artifacts: $OUTPUT_TARBALL"
tmp_list="/tmp/${BASE_TAG}_hef_files.txt"
awk -F'\t' 'NR>1 && $9=="true" {print $8}' "$DEC_SUMMARY" > "$tmp_list"
awk -F'\t' 'NR>1 && $9=="true" {print $8}' "$MASK_SUMMARY" >> "$tmp_list"
echo "$DEC_SUMMARY" >> "$tmp_list"
echo "$MASK_SUMMARY" >> "$tmp_list"
echo "$AGG_SUMMARY" >> "$tmp_list"
tar -czf "$OUTPUT_TARBALL" -T "$tmp_list"

echo "[5/5] Done"
echo "Aggregate summary: $AGG_SUMMARY"
echo "Decoder summary: $DEC_SUMMARY"
echo "Masker summary: $MASK_SUMMARY"
echo "Tarball: $OUTPUT_TARBALL"
echo
echo "RPi env exports:"
echo "export DECODER_SUMMARY_TSV=\"$DEC_SUMMARY\""
echo "export MASKER_SUMMARY_TSV=\"$MASK_SUMMARY\""

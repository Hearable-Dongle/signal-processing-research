#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

RUN_TS="${HAILO_RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="hailo/module_runs/${RUN_TS}_hybrid_librimix"
mkdir -p "$RUN_DIR"

BACKEND="${BACKEND:-hailo_runtime}"
DECODER_SUMMARY_TSV="${DECODER_SUMMARY_TSV:-hailo/module_runs/20260223_052929/allocator_mapping_fixes_summary.tsv}"
MASKER_SUMMARY_TSV="${MASKER_SUMMARY_TSV:-hailo/module_runs/20260223_082226/masker_allocator_fixes_summary.tsv}"
MIX_WAV="${MIX_WAV:-hailo/sanity_librimix3/sanity_mix.wav}"
MODEL_ID="${MODEL_ID:-JorisCos/ConvTasNet_Libri3Mix_sepclean_8k}"
SAMPLE_RATE="${SAMPLE_RATE:-8000}"
N_SRC="${N_SRC:-2}"
MAX_SECONDS="${MAX_SECONDS:-4.0}"
TILE_W="${TILE_W:-256}"
BLOCK_CHAN="${BLOCK_CHAN:-64}"

hailo/to-onnx-env/bin/python -m hailo.hybrid_hailo_librimix_validation \
  --backend "$BACKEND" \
  --decoder_summary_tsv "$DECODER_SUMMARY_TSV" \
  --masker_summary_tsv "$MASKER_SUMMARY_TSV" \
  --mix_wav "$MIX_WAV" \
  --model_id "$MODEL_ID" \
  --out_dir "$RUN_DIR" \
  --sample_rate "$SAMPLE_RATE" \
  --n_src "$N_SRC" \
  --max_seconds "$MAX_SECONDS" \
  --tile_w "$TILE_W" \
  --block_chan "$BLOCK_CHAN"

echo "[DONE] Hybrid validation run: $RUN_DIR"
echo "Metrics: $RUN_DIR/hybrid_validation_metrics.json"

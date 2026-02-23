#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

if [[ -z "${HEF_PATH:-}" ]]; then
  echo "HEF_PATH is required" >&2
  exit 2
fi

RUN_TS="${HAILO_RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-hailo/module_runs/${RUN_TS}/monolithic_runtime}"
MIX_WAV="${MIX_WAV:-}"
LIBRIMIX_ROOT="${LIBRIMIX_ROOT:-}"

if [[ -z "$MIX_WAV" ]]; then
  if [[ -n "$LIBRIMIX_ROOT" ]]; then
    MIX_WAV="$(find "$LIBRIMIX_ROOT" -type f -name '*.wav' | head -n1 || true)"
  else
    MIX_WAV="$(python - <<'PY'
from pathlib import Path
try:
    from general_utils.constants import LIBRIMIX_PATH
except Exception:
    print("")
    raise SystemExit(0)
root = Path(LIBRIMIX_PATH)
if not root.exists():
    print("")
    raise SystemExit(0)
files = sorted(root.rglob("*.wav"))
print(str(files[0]) if files else "")
PY
)"
  fi
fi

if [[ -z "$MIX_WAV" || ! -f "$MIX_WAV" ]]; then
  echo "Could not resolve MIX_WAV. Set MIX_WAV or LIBRIMIX_ROOT." >&2
  exit 2
fi

hailo/to-hailo-env/bin/python -m hailo.monolithic_hef_runtime_infer \
  --hef "$HEF_PATH" \
  --mix_wav "$MIX_WAV" \
  --out_dir "$OUT_DIR"

echo "[DONE] runtime outputs: $OUT_DIR"

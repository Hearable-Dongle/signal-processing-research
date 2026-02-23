#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

OUT_DIR="${OUT_DIR:-hailo/sanity_librimix3}"
MIX_WAV="${MIX_WAV:-}"

OUT_DIR="$OUT_DIR" MIX_WAV="$MIX_WAV" python - <<'PY'
from pathlib import Path
import json
import os
import shutil

from general_utils.constants import LIBRIMIX_PATH

root = Path(LIBRIMIX_PATH)
out = Path(os.environ["OUT_DIR"])
out.mkdir(parents=True, exist_ok=True)

mix_arg = os.environ.get("MIX_WAV", "").strip()
if mix_arg:
    mix = Path(mix_arg)
else:
    libri3 = root / "Libri3Mix"
    mixes = sorted(libri3.glob("**/mix_clean/*.wav"))
    mix = None
    for m in mixes:
        s1 = Path(str(m).replace("/mix_clean/", "/s1/"))
        s2 = Path(str(m).replace("/mix_clean/", "/s2/"))
        s3 = Path(str(m).replace("/mix_clean/", "/s3/"))
        if s1.exists() and s2.exists() and s3.exists():
            mix = m
            break
    if mix is None:
        raise FileNotFoundError(f"No Libri3Mix sample with s1/s2/s3 found under {libri3}")

if not mix.exists():
    raise FileNotFoundError(f"mix wav does not exist: {mix}")
s1 = Path(str(mix).replace("/mix_clean/", "/s1/"))
s2 = Path(str(mix).replace("/mix_clean/", "/s2/"))
s3 = Path(str(mix).replace("/mix_clean/", "/s3/"))
for p in (s1, s2, s3):
    if not p.exists():
        raise FileNotFoundError(f"missing source wav: {p}")

copy_map = {
    "sanity_mix.wav": mix,
    "sanity_voice_1.wav": s1,
    "sanity_voice_2.wav": s2,
    "sanity_voice_3.wav": s3,
}
for name, src in copy_map.items():
    shutil.copy2(src, out / name)

meta = {
    "librimix_root": str(root),
    "selected_mix": str(mix),
    "selected_s1": str(s1),
    "selected_s2": str(s2),
    "selected_s3": str(s3),
    "copied_files": {k: str(out / k) for k in copy_map.keys()},
}
(out / "sanity_manifest.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
print(json.dumps(meta, indent=2))
PY

echo "[DONE] sanity assets copied to ${OUT_DIR}"

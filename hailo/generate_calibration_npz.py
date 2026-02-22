import argparse
import math
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from general_utils.constants import LIBRIMIX_PATH


def _find_candidate_wavs(librimix_root: Path, include_clean_only: bool) -> List[Path]:
    if include_clean_only:
        preferred_dirs = [
            librimix_root / "Libri2Mix/wav16k/min/train-100/mix_clean",
            librimix_root / "Libri2Mix/wav16k/min/train-360/mix_clean",
            librimix_root / "Libri2Mix/wav16k/max/train-100/mix_clean",
            librimix_root / "Libri2Mix/wav16k/max/train-360/mix_clean",
            librimix_root / "Libri2Mix/wav16k/min/test/mix_clean",
            librimix_root / "Libri2Mix/wav16k/max/test/mix_clean",
        ]
        files: List[Path] = []
        for d in preferred_dirs:
            if d.exists():
                files.extend(d.glob("*.wav"))
        if files:
            return sorted(set(files))
        # Fallback to recursive search if expected Libri2Mix structure is absent.
        patterns = ["**/mix_clean/*.wav", "**/mix_both/*.wav", "**/mix_single/*.wav"]
    else:
        patterns = ["**/*.wav"]

    files = []
    for pattern in patterns:
        files.extend(librimix_root.glob(pattern))

    # De-duplicate while keeping deterministic ordering.
    unique = sorted(set(files))
    return unique


def _to_float_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    if np.issubdtype(audio.dtype, np.integer):
        max_val = np.iinfo(audio.dtype).max
        audio = audio.astype(np.float32) / max_val
    else:
        audio = audio.astype(np.float32)

    return np.clip(audio, -1.0, 1.0)


def _resample_if_needed(audio: np.ndarray, source_sr: int, target_sr: int) -> np.ndarray:
    if source_sr == target_sr:
        return audio
    g = math.gcd(source_sr, target_sr)
    up = target_sr // g
    down = source_sr // g
    return resample_poly(audio, up=up, down=down).astype(np.float32)


def _extract_fixed_chunk(audio: np.ndarray, chunk_len: int, rng: np.random.Generator) -> np.ndarray:
    if len(audio) >= chunk_len:
        start = int(rng.integers(0, len(audio) - chunk_len + 1))
        return audio[start : start + chunk_len]

    padded = np.zeros((chunk_len,), dtype=np.float32)
    padded[: len(audio)] = audio
    return padded


def _format_layout(batch_nt: np.ndarray, layout: str) -> np.ndarray:
    if layout == "nt":
        return batch_nt
    if layout == "nct":
        return batch_nt[:, np.newaxis, :]
    if layout == "nhwc":
        return batch_nt[:, np.newaxis, :, np.newaxis]
    if layout == "nchw":
        return batch_nt[:, np.newaxis, np.newaxis, :]
    raise ValueError(f"Unsupported layout: {layout}")


def _build_dataset(
    wav_paths: List[Path],
    num_samples: int,
    target_sr: int,
    clip_ms: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, List[str]]:
    chunk_len = int(target_sr * clip_ms / 1000.0)
    if chunk_len <= 0:
        raise ValueError(f"Invalid chunk length for clip_ms={clip_ms}, target_sr={target_sr}")

    selected_paths: List[str] = []
    chunks: List[np.ndarray] = []

    sampled_indices = rng.integers(0, len(wav_paths), size=num_samples)
    for idx in sampled_indices:
        wav_path = wav_paths[int(idx)]
        source_sr, audio = wavfile.read(str(wav_path))
        audio = _to_float_mono(audio)
        audio = _resample_if_needed(audio, source_sr=source_sr, target_sr=target_sr)
        chunk = _extract_fixed_chunk(audio, chunk_len=chunk_len, rng=rng)
        chunks.append(chunk)
        selected_paths.append(str(wav_path))

    batch_nt = np.stack(chunks, axis=0).astype(np.float32)
    return batch_nt, selected_paths


def main():
    parser = argparse.ArgumentParser(description="Generate calibration NPZ from LibriMix WAVs")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("hailo/calibration_200ms_16k.npz"),
        help="Output NPZ path",
    )
    parser.add_argument("--num_samples", type=int, default=64, help="Number of calibration examples")
    parser.add_argument("--clip_ms", type=int, default=200, help="Clip duration in milliseconds")
    parser.add_argument("--sample_rate", type=int, default=16000, help="Target sample rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--layout",
        choices=["nt", "nct", "nhwc", "nchw"],
        default="nt",
        help="Output tensor layout in NPZ key 'calib_data'",
    )
    parser.add_argument(
        "--librimix_root",
        type=Path,
        default=LIBRIMIX_PATH,
        help="Root path for LibriMix dataset",
    )
    parser.add_argument(
        "--include_any_wav",
        action="store_true",
        help="If set, scan all WAVs. Default restricts to common mixture folders first.",
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    wav_paths = _find_candidate_wavs(args.librimix_root, include_clean_only=not args.include_any_wav)
    if not wav_paths:
        raise FileNotFoundError(f"No WAV files found under {args.librimix_root}")

    batch_nt, selected_paths = _build_dataset(
        wav_paths=wav_paths,
        num_samples=args.num_samples,
        target_sr=args.sample_rate,
        clip_ms=args.clip_ms,
        rng=rng,
    )
    calib_data = _format_layout(batch_nt, layout=args.layout)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        calib_data=calib_data,
        selected_paths=np.asarray(selected_paths, dtype=object),
        sample_rate=np.asarray(args.sample_rate, dtype=np.int32),
        clip_ms=np.asarray(args.clip_ms, dtype=np.int32),
        layout=np.asarray(args.layout),
    )

    print(f"Saved: {args.output}")
    print(f"calib_data shape: {calib_data.shape}, dtype: {calib_data.dtype}, layout: {args.layout}")
    print(f"source wav count scanned: {len(wav_paths)}")


if __name__ == "__main__":
    main()

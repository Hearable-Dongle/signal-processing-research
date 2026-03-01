import argparse
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf


def _load_runtime_api():
    try:
        from hailo_platform import HEF, VDevice  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("install pyhailort/hailo_platform to run HEF inference") from exc
    return HEF, VDevice


def _pick_stream_name(available: list[str], explicit: str | None) -> str:
    if explicit:
        if explicit not in available:
            raise ValueError(
                f"Requested stream '{explicit}' not found. Available: {available}"
            )
        return explicit
    if len(available) == 1:
        return available[0]
    raise ValueError(
        "HEF stream selection is ambiguous. Pass --input-stream/--output-stream explicitly. "
        f"Available streams: {available}"
    )


def _read_mono_16k(path: Path, max_samples: int) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1).astype(np.float32)
    if sr != 16000:
        raise ValueError(f"Expected 16kHz WAV for milestone runtime validation, got {sr}")
    if max_samples > 0 and audio.shape[0] > max_samples:
        audio = audio[:max_samples]
    return audio.astype(np.float32), sr


def _reshape_input(audio: np.ndarray) -> np.ndarray:
    # NHWC with shape (N, H, W, C) == (1, 1, T, 1)
    return audio[np.newaxis, np.newaxis, :, np.newaxis].astype(np.float32)


def _split_outputs(output: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    out = np.asarray(output)
    out = np.squeeze(out)
    if out.ndim == 1:
        return out, out
    if out.ndim == 2:
        if out.shape[0] >= 2:
            return out[0], out[1]
        if out.shape[1] >= 2:
            return out[:, 0], out[:, 1]
    raise ValueError(f"Unsupported output shape for source split: {output.shape}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run monolithic HEF on one WAV and save separated outputs")
    parser.add_argument("--hef", required=True)
    parser.add_argument("--mix_wav", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--input-stream", default=None)
    parser.add_argument("--output-stream", default=None)
    parser.add_argument("--max-seconds", type=float, default=2.0)
    args = parser.parse_args()

    hef_path = Path(args.hef)
    mix_path = Path(args.mix_wav)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    HEF, VDevice = _load_runtime_api()

    audio, sr = _read_mono_16k(mix_path, max_samples=max(1, int(args.max_seconds * 16000)))
    model_input = _reshape_input(audio)

    start = time.perf_counter()

    hef = HEF(str(hef_path))
    with VDevice() as target:
        config = target.create_infer_model(hef)
        input_names = list(config.input_names)
        output_names = list(config.output_names)
        input_name = _pick_stream_name(input_names, args.input_stream)
        output_name = _pick_stream_name(output_names, args.output_stream)

        bindings = config.create_bindings()
        bindings.input(input_name).set_buffer(model_input)
        output_buffer = np.empty(bindings.output(output_name).shape, dtype=np.float32)
        bindings.output(output_name).set_buffer(output_buffer)

        configured = config.configure()
        job = configured.run_async([bindings])
        job.wait(10_000)

    elapsed = time.perf_counter() - start

    src1, src2 = _split_outputs(output_buffer)
    src1 = np.asarray(src1, dtype=np.float32)
    src2 = np.asarray(src2, dtype=np.float32)

    mix_out = out_dir / "mix.wav"
    src1_out = out_dir / "sep_src1.wav"
    src2_out = out_dir / "sep_src2.wav"
    metrics_out = out_dir / "runtime_metrics.json"

    sf.write(mix_out, audio, sr)
    sf.write(src1_out, src1, sr)
    sf.write(src2_out, src2, sr)

    metrics = {
        "ok": True,
        "hef": str(hef_path),
        "mix_wav": str(mix_path),
        "input_samples": int(audio.shape[0]),
        "output_samples_src1": int(src1.shape[0]),
        "output_samples_src2": int(src2.shape[0]),
        "sample_rate": int(sr),
        "latency_ms": float(elapsed * 1000.0),
        "rtf": float(elapsed / max(1e-9, audio.shape[0] / sr)),
        "input_stream": input_name,
        "output_stream": output_name,
        "device_info": "VDevice",
    }
    metrics_out.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    print(f"Saved outputs to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

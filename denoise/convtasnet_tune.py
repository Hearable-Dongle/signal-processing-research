import argparse
from pathlib import Path
import sys
import time

import torch
import torchaudio
import matplotlib.pyplot as plt

# Add project root to allow sibling imports
sys.path.append(str(Path(__file__).resolve().parents[1]))
from own_voice_suppression.audio_utils import prep_audio

from asteroid.models import ConvTasNet


# -----------------------------
# Built-in latency presets sweep
# -----------------------------
LATENCY_PRESETS = [
    {
        "name": "baseline",
        "window_sec": 2.0,
        "stride_sec": 0.5,
        "pad_sec": 0.5,
        "fade_sec": 0.05,
        "stitching": "crossfade",
    },
    {
        "name": "lowlat_500ms",
        "window_sec": 0.5,
        "stride_sec": 0.25,
        "pad_sec": 0.10,
        "fade_sec": 0.02,
        "stitching": "crossfade",
    },
    {
        "name": "lowlat_250ms",
        "window_sec": 0.25,
        "stride_sec": 0.125,
        "pad_sec": 0.05,
        "fade_sec": 0.01,
        "stitching": "crossfade",
    },
]


class ConvTasNetWrapper:
    """
    ConvTasNet single-speaker enhancement wrapper with tunable inference parameters.
    """
    MODEL_ID = "JorisCos/ConvTasNet_Libri1Mix_enhsingle_16k"
    NATIVE_SR = 16000

    def __init__(self, device: torch.device, pad_sec: float = 0.5, rms_norm: bool = True, rms_eps: float = 1e-8):
        self.device = device
        self.pad_sec = float(pad_sec)
        self.rms_norm = bool(rms_norm)
        self.rms_eps = float(rms_eps)

        print(f"[Model] Loading {self.MODEL_ID} on {device} ...")
        self.model = ConvTasNet.from_pretrained(self.MODEL_ID).to(device)
        self.model.eval()

    def process(self, noisy_chunk: torch.Tensor) -> torch.Tensor:
        """
        noisy_chunk: (1, T) tensor at 16 kHz
        returns: (1, T) enhanced tensor at 16 kHz
        """
        with torch.no_grad():
            pad_samples = int(self.pad_sec * self.NATIVE_SR)

            if pad_samples > 0:
                padded = torch.nn.functional.pad(
                    noisy_chunk.unsqueeze(1),  # (B,1,T)
                    (pad_samples, pad_samples),
                    mode="reflect",
                ).squeeze(1)  # (B,T)
            else:
                padded = noisy_chunk

            # Asteroid ConvTasNet returns shape (B, nsrc, T)
            est_padded = self.model(padded)  # (B,1,T_padded)
            est_padded = est_padded[:, 0, :]  # (B,T_padded)

            if pad_samples > 0:
                est = est_padded[:, pad_samples:-pad_samples]
            else:
                est = est_padded

            if self.rms_norm:
                in_rms = torch.sqrt(torch.mean(noisy_chunk ** 2, dim=-1, keepdim=True))
                out_rms = torch.sqrt(torch.mean(est ** 2, dim=-1, keepdim=True)) + self.rms_eps
                est = est * (in_rms / out_rms)

        return est


def denoise_long_audio(
    enhancer: ConvTasNetWrapper,
    noisy_wav: torch.Tensor,
    window_sec: float = 2.0,
    stride_sec: float = 0.5,
    stitching: str = "crossfade",
    fade_sec: float = 0.05,
):
    """
    Sliding-window denoising with either overwrite stitching or crossfade stitching.

    noisy_wav: (1, N) at enhancer.NATIVE_SR
    returns: (output_wav, avg_chunk_latency_s, rtf)
    """
    sr = enhancer.NATIVE_SR
    window_samples = int(window_sec * sr)
    stride_samples = int(stride_sec * sr)
    fade_samples = int(fade_sec * sr)

    if stride_samples <= 0 or window_samples <= 0:
        raise ValueError("window_sec and stride_sec must be > 0")
    if stride_samples > window_samples:
        raise ValueError("stride_sec must be <= window_sec")
    if stitching not in ("overwrite", "crossfade"):
        raise ValueError("stitching must be 'overwrite' or 'crossfade'")
    if stitching == "crossfade" and fade_samples <= 0:
        raise ValueError("fade_sec must be > 0 for crossfade")

    num_samples = noisy_wav.shape[1]
    out = torch.zeros_like(noisy_wav)

    total_infer = 0.0
    chunks = 0

    device = noisy_wav.device
    pos = 0

    while pos < num_samples:
        end = pos + window_samples
        chunk = noisy_wav[:, pos:end]

        # Pad last chunk to full window
        if chunk.shape[1] < window_samples:
            pad = window_samples - chunk.shape[1]
            chunk = torch.nn.functional.pad(chunk, (0, pad))

        t0 = time.monotonic()
        enhanced = enhancer.process(chunk)
        t1 = time.monotonic()
        total_infer += (t1 - t0)
        chunks += 1

        # Determine how much of this chunk is "real" (no pad)
        real_len = min(window_samples, num_samples - pos)
        enhanced = enhanced[:, :real_len]

        if pos == 0 or stitching == "overwrite":
            # Overwrite strategy (simple but can click at boundaries)
            out[:, pos:pos + real_len] = enhanced
        else:
            # Crossfade overlap region
            overlap = min(fade_samples, pos, real_len)
            if overlap > 0:
                fade_in = torch.linspace(0, 1, overlap, device=device).unsqueeze(0)
                fade_out = 1 - fade_in

                out[:, pos:pos + overlap] = (
                    out[:, pos:pos + overlap] * fade_out + enhanced[:, :overlap] * fade_in
                )
                out[:, pos + overlap:pos + real_len] = enhanced[:, overlap:]
            else:
                out[:, pos:pos + real_len] = enhanced

        pos += stride_samples
        if pos >= num_samples:
            break

    avg_latency = total_infer / chunks if chunks else 0.0
    rtf = avg_latency / window_sec if window_sec > 0 else 0.0
    return out, avg_latency, rtf


def _save_spectrogram_pair(noisy_16k: torch.Tensor, denoised_16k: torch.Tensor, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    plt.title("Original (Noisy) - 16k")
    plt.specgram(noisy_16k.detach().cpu().numpy()[0], Fs=16000, NFFT=1024, noverlap=512)

    plt.subplot(2, 1, 2)
    plt.title("Denoised (ConvTasNet)")
    plt.specgram(denoised_16k.detach().cpu().numpy()[0], Fs=16000, NFFT=1024, noverlap=512)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def run_single_config(
    noisy_wav_16k: torch.Tensor,
    input_stem: str,
    device: torch.device,
    base_output_dir: Path,
    cfg: dict,
    rms_norm: bool,
    rms_eps: float,
):
    """
    Runs one config (either from CLI or a preset) and saves outputs into its own folder.
    Returns a dict of metrics for printing/comparison.
    """
    enhancer = ConvTasNetWrapper(
        device=device,
        pad_sec=cfg["pad_sec"],
        rms_norm=rms_norm,
        rms_eps=rms_eps,
    )

    denoised_16k, avg_lat_s, rtf = denoise_long_audio(
        enhancer,
        noisy_wav_16k,
        window_sec=cfg["window_sec"],
        stride_sec=cfg["stride_sec"],
        stitching=cfg["stitching"],
        fade_sec=cfg["fade_sec"],
    )

    run_name = (
        f"{cfg['name']}_pad{cfg['pad_sec']}_win{cfg['window_sec']}_str{cfg['stride_sec']}_"
        f"{cfg['stitching']}_fade{cfg['fade_sec']}_{input_stem}"
    )
    run_dir = base_output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save audio
    out_wav_path = run_dir / "denoised.wav"
    torchaudio.save(out_wav_path, denoised_16k.detach().cpu(), 16000)

    # Save plot
    plot_path = run_dir / "spectrogram.png"
    _save_spectrogram_pair(noisy_wav_16k, denoised_16k, plot_path)

    # Approx algorithmic delay estimate (very rough):
    # - buffering ~ window_sec
    # - lookahead ~ pad_sec (because reflect padding includes "future")
    approx_algo_delay_ms = (cfg["window_sec"] + cfg["pad_sec"]) * 1000.0

    return {
        "name": cfg["name"],
        "window_sec": cfg["window_sec"],
        "stride_sec": cfg["stride_sec"],
        "pad_sec": cfg["pad_sec"],
        "fade_sec": cfg["fade_sec"],
        "stitching": cfg["stitching"],
        "avg_latency_ms": avg_lat_s * 1000.0,
        "rtf": rtf,
        "approx_algo_delay_ms": approx_algo_delay_ms,
        "out_dir": str(run_dir),
    }


def main():
    parser = argparse.ArgumentParser(description="ConvTasNet tuning runner (isolated from main denoise.py)")
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("denoise/outputs/convtasnet_tuning"))

    # ConvTasNet params
    parser.add_argument("--pad-sec", type=float, default=0.5, help="Reflect padding before/after chunk (future context)")
    parser.add_argument("--rms-norm", action="store_true", help="Match output RMS to input RMS per chunk")
    parser.add_argument("--no-rms-norm", dest="rms_norm", action="store_false")
    parser.set_defaults(rms_norm=True)
    parser.add_argument("--rms-eps", type=float, default=1e-8)

    # Chunking params
    parser.add_argument("--window-sec", type=float, default=2.0)
    parser.add_argument("--stride-sec", type=float, default=0.5)

    # Stitching params
    parser.add_argument("--stitching", type=str, choices=["overwrite", "crossfade"], default="crossfade")
    parser.add_argument("--fade-sec", type=float, default=0.05)

    # New: built-in sweep mode
    parser.add_argument("--sweep-latency", action="store_true", help="Run built-in latency sweep presets")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Run] Device: {device}")

    print("[Audio] Loading input...")
    noisy_wav, sr = torchaudio.load(args.input_path)

    # Convert to mono if needed
    if noisy_wav.shape[0] > 1:
        noisy_wav = torch.mean(noisy_wav, dim=0, keepdim=True)

    # Resample to ConvTasNet native SR
    noisy_wav_16k = prep_audio(noisy_wav, sr, 16000).to(device)
    input_stem = args.input_path.stem

    base_output_dir = Path(args.output_dir)

    if args.sweep_latency:
        print("\n[SWEEP] Running built-in latency presets...\n")
        sweep_dir = base_output_dir / "latency_sweep"
        sweep_dir.mkdir(parents=True, exist_ok=True)

        results = []
        for cfg in LATENCY_PRESETS:
            cfg_run = dict(cfg)  # copy
            res = run_single_config(
                noisy_wav_16k=noisy_wav_16k,
                input_stem=input_stem,
                device=device,
                base_output_dir=sweep_dir,
                cfg=cfg_run,
                rms_norm=args.rms_norm,
                rms_eps=args.rms_eps,
            )
            results.append(res)
            print(f"  ✓ {res['name']} -> {res['out_dir']}")

        print("\n================ LATENCY SWEEP RESULTS ================")
        print(f"{'Preset':<14} | {'win':<5} | {'str':<5} | {'pad':<5} | {'fade':<5} | {'RTF':<6} | {'chunk_ms':<9} | {'~algo_ms':<9}")
        print("-" * 95)
        for r in results:
            print(
                f"{r['name']:<14} | {r['window_sec']:<5.2f} | {r['stride_sec']:<5.2f} | {r['pad_sec']:<5.2f} | "
                f"{r['fade_sec']:<5.2f} | {r['rtf']:<6.3f} | {r['avg_latency_ms']:<9.1f} | {r['approx_algo_delay_ms']:<9.1f}"
            )
        print("-" * 95)
        print("[Done] Sweep outputs saved under:", sweep_dir)
        return

    # Normal single-run behavior (CLI params)
    cfg = {
        "name": "single",
        "window_sec": args.window_sec,
        "stride_sec": args.stride_sec,
        "pad_sec": args.pad_sec,
        "fade_sec": args.fade_sec,
        "stitching": args.stitching,
    }

    res = run_single_config(
        noisy_wav_16k=noisy_wav_16k,
        input_stem=input_stem,
        device=device,
        base_output_dir=base_output_dir,
        cfg=cfg,
        rms_norm=args.rms_norm,
        rms_eps=args.rms_eps,
    )

    print(f"[Perf] Avg chunk latency: {res['avg_latency_ms']:.2f} ms")
    print(f"[Perf] RTF: {res['rtf']:.3f}")
    print(f"[Perf] Approx algorithmic delay (rough): {res['approx_algo_delay_ms']:.1f} ms")
    print(f"[Done] Outputs in: {res['out_dir']}")


if __name__ == "__main__":
    main()

import argparse
from pathlib import Path
import sys
import time
import math
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import torch
import torchaudio

# Add project root to allow sibling imports
sys.path.append(str(Path(__file__).resolve().parents[1]))
from own_voice_suppression.audio_utils import prep_audio

# Try to use your project's SII/SI-SDR helpers if present
try:
    from own_voice_suppression.validate_source_separation import calculate_sii_from_audio, align_volume
except Exception:
    calculate_sii_from_audio = None
    align_volume = None

try:
    from own_voice_suppression.validate_voice_detection import compute_si_sdr
except Exception:
    compute_si_sdr = None


@dataclass
class SweepResult:
    name: str
    frame_ms: float
    hop_ms: float
    lookahead_ms: float
    est_e2e_latency_ms: float
    mean_hop_compute_ms: float
    p95_hop_compute_ms: float
    rtf: float
    sii: Optional[float]
    si_sdr: Optional[float]
    out_path: str


def percentile(x: List[float], p: float) -> float:
    if not x:
        return 0.0
    xs = np.array(x, dtype=np.float64)
    return float(np.percentile(xs, p))


def estimate_e2e_latency_ms(frame_ms: float, lookahead_ms: float, io_buffer_ms: float = 20.0, sched_ms: float = 10.0) -> float:
    """
    Rough end-to-end budget:
      - algorithmic: ~ frame length + lookahead
      - I/O buffering: ~ 10-30 ms typical on Linux audio stacks
      - scheduling jitter: ~ 5-20 ms
    Adjust io_buffer_ms/sched_ms once you measure on CM5.
    """
    return frame_ms + lookahead_ms + io_buffer_ms + sched_ms


class StreamingSpectralGate:
    """
    Low-latency streaming spectral gate.
    - STFT with small frame/hop
    - noise tracker (min / EMA)
    - soft mask applied per-frame
    """

    def __init__(
        self,
        sr: int,
        n_fft: int,
        hop: int,
        win: int,
        lookahead_frames: int = 0,
        noise_smooth: float = 0.98,
        noise_floor: float = 1e-6,
        thresh_mul: float = 1.5,
        mask_floor: float = 0.05,
    ):
        self.sr = sr
        self.n_fft = n_fft
        self.hop = hop
        self.win = win
        self.lookahead_frames = int(lookahead_frames)

        self.noise_smooth = float(noise_smooth)
        self.noise_floor = float(noise_floor)
        self.thresh_mul = float(thresh_mul)
        self.mask_floor = float(mask_floor)

        self.window = torch.hann_window(win)
        self.noise_psd = None  # (freq_bins,)

        # Ring buffers for streaming
        self.in_buf = torch.zeros(0)
        self.out_buf = torch.zeros(0)

        # Frame queue for optional lookahead
        self.mag_queue: List[torch.Tensor] = []
        self.phase_queue: List[torch.Tensor] = []

    def _stft_frame(self, frame: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        frame: (win,) real tensor
        returns: mag, phase, each (freq_bins,)
        """
        X = torch.fft.rfft(frame * self.window, n=self.n_fft)
        mag = torch.abs(X)
        phase = torch.angle(X)
        return mag, phase

    def _istft_frame(self, mag: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        """
        mag/phase: (freq_bins,)
        returns time frame length win (after iFFT and window)
        """
        X = mag * torch.exp(1j * phase)
        x = torch.fft.irfft(X, n=self.n_fft)
        # Take first win samples, window for overlap-add
        xw = x[: self.win] * self.window
        return xw

    def _update_noise(self, mag: torch.Tensor):
        psd = mag ** 2
        if self.noise_psd is None:
            self.noise_psd = psd.clone()
        else:
            # EMA noise tracking; tends to keep "background" model
            self.noise_psd = self.noise_smooth * self.noise_psd + (1.0 - self.noise_smooth) * psd
        self.noise_psd = torch.clamp(self.noise_psd, min=self.noise_floor)

    def _apply_gate(self, mag: torch.Tensor) -> torch.Tensor:
        """
        Soft mask based on noise PSD estimate.
        """
        self._update_noise(mag)
        noise_mag = torch.sqrt(self.noise_psd)
        thresh = self.thresh_mul * noise_mag

        # soft mask: if mag >> thresh -> ~1, else -> mask_floor
        # mask = clamp((mag - thresh) / (mag + eps), mask_floor, 1)
        eps = 1e-8
        mask = (mag - thresh) / (mag + eps)
        mask = torch.clamp(mask, min=self.mask_floor, max=1.0)
        return mag * mask

    def process_stream(self, x: torch.Tensor) -> torch.Tensor:
        """
        Streaming processing.
        x: (N,) float32
        returns: y: denoised samples for whatever can be produced now.
        """
        if x.ndim != 1:
            raise ValueError("x must be 1D mono tensor")

        # append to input buffer
        self.in_buf = torch.cat([self.in_buf, x.cpu()])

        # output buffer if empty
        if self.out_buf.numel() == 0:
            self.out_buf = torch.zeros(0)

        # While enough for one hop frame extraction:
        # We'll use hop-sized progression, but each analysis frame is win samples.
        y_out_chunks = []

        # We keep an overlap-add accumulator for streaming synthesis:
        # Here we do simple OLA with hop.
        while self.in_buf.numel() >= self.win:
            frame = self.in_buf[: self.win]
            self.in_buf = self.in_buf[self.hop :]  # advance by hop

            mag, phase = self._stft_frame(frame)

            # queue for lookahead
            self.mag_queue.append(mag)
            self.phase_queue.append(phase)

            # only process when we have enough lookahead to decide current frame
            if len(self.mag_queue) > self.lookahead_frames:
                mag0 = self.mag_queue.pop(0)
                ph0 = self.phase_queue.pop(0)

                mag_d = self._apply_gate(mag0)
                y_frame = self._istft_frame(mag_d, ph0)

                # Overlap-add: we emit hop samples each step.
                # Keep an internal synthesis buffer of length win, shift by hop.
                if self.out_buf.numel() < self.win:
                    self.out_buf = torch.cat([self.out_buf, torch.zeros(self.win - self.out_buf.numel())])

                self.out_buf[: self.win] += y_frame

                # emit hop samples
                y_emit = self.out_buf[: self.hop].clone()
                y_out_chunks.append(y_emit)

                # shift synthesis buffer
                self.out_buf = torch.cat([self.out_buf[self.hop :], torch.zeros(self.hop)])

        if y_out_chunks:
            return torch.cat(y_out_chunks)
        return torch.zeros(0)


def run_one_config(
    noisy_16k: torch.Tensor,
    clean_16k: Optional[torch.Tensor],
    sr: int,
    out_dir: Path,
    name: str,
    frame_ms: float,
    hop_ms: float,
    lookahead_ms: float,
    params: Dict[str, Any],
) -> SweepResult:
    """
    Simulates streaming by feeding hop-sized blocks and timing per hop.
    """
    n_fft = int(round(sr * frame_ms / 1000.0))
    # keep FFT size power-of-2-ish for speed (optional)
    # but don't exceed too much
    def next_pow2(n: int) -> int:
        return 1 << (n - 1).bit_length()

    n_fft = max(128, next_pow2(n_fft))
    win = int(round(sr * frame_ms / 1000.0))
    hop = int(round(sr * hop_ms / 1000.0))
    lookahead_frames = int(round(lookahead_ms / hop_ms)) if hop_ms > 0 else 0

    denoiser = StreamingSpectralGate(
        sr=sr,
        n_fft=n_fft,
        hop=hop,
        win=win,
        lookahead_frames=lookahead_frames,
        noise_smooth=params["noise_smooth"],
        thresh_mul=params["thresh_mul"],
        mask_floor=params["mask_floor"],
    )

    # stream loop
    x = noisy_16k.squeeze(0).cpu()
    hop_times_ms: List[float] = []

    y_chunks = []
    t_start = time.monotonic()

    # Feed in hop-sized packets (like realtime audio callback)
    for i in range(0, x.numel(), hop):
        block = x[i : i + hop]
        if block.numel() < hop:
            # pad last block
            block = torch.nn.functional.pad(block, (0, hop - block.numel()))

        t0 = time.monotonic()
        y_block = denoiser.process_stream(block)
        t1 = time.monotonic()
        hop_times_ms.append((t1 - t0) * 1000.0)

        if y_block.numel() > 0:
            y_chunks.append(y_block)

    # flush remaining queued frames (lookahead)
    # keep feeding zeros until queue empties
    for _ in range(lookahead_frames + 4):
        t0 = time.monotonic()
        y_block = denoiser.process_stream(torch.zeros(hop))
        t1 = time.monotonic()
        hop_times_ms.append((t1 - t0) * 1000.0)
        if y_block.numel() > 0:
            y_chunks.append(y_block)

    t_total = time.monotonic() - t_start

    y = torch.cat(y_chunks) if y_chunks else torch.zeros(0)
    # trim/pad to input length
    y = y[: x.numel()]
    if y.numel() < x.numel():
        y = torch.nn.functional.pad(y, (0, x.numel() - y.numel()))

    y = y.unsqueeze(0)  # (1,N)

    # Save
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}.wav"
    torchaudio.save(out_path, y, sr)

    # Metrics
    sii_val = None
    si_sdr_val = None
    if clean_16k is not None:
        y_m = y
        c_m = clean_16k
        if align_volume is not None:
            y_m = align_volume(y_m, c_m)
        if compute_si_sdr is not None:
            si_sdr_val = float(compute_si_sdr(y_m, c_m))
        if calculate_sii_from_audio is not None:
            # noise estimate = (y - clean) treated as noise
            noise_est = (y_m - c_m)
            sii_val = float(calculate_sii_from_audio(c_m, noise_est, sr))

    mean_hop_ms = float(np.mean(hop_times_ms)) if hop_times_ms else 0.0
    p95_hop_ms = percentile(hop_times_ms, 95.0)

    # RTF for streaming hop processing
    audio_sec = x.numel() / sr
    rtf = (t_total / audio_sec) if audio_sec > 0 else 0.0

    est_e2e = estimate_e2e_latency_ms(frame_ms=frame_ms, lookahead_ms=lookahead_ms)

    return SweepResult(
        name=name,
        frame_ms=frame_ms,
        hop_ms=hop_ms,
        lookahead_ms=lookahead_ms,
        est_e2e_latency_ms=est_e2e,
        mean_hop_compute_ms=mean_hop_ms,
        p95_hop_compute_ms=p95_hop_ms,
        rtf=rtf,
        sii=sii_val,
        si_sdr=si_sdr_val,
        out_path=str(out_path),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--noisy", type=Path, required=True, help="Noisy input wav")
    p.add_argument("--clean", type=Path, default=None, help="Optional clean reference wav for SII/SI-SDR")
    p.add_argument("--out-dir", type=Path, default=Path("denoise/outputs/streaming_sweep"))
    p.add_argument("--sr", type=int, default=16000)

    # Sweep sets
    p.add_argument("--frames-ms", type=str, default="20,25,32", help="Comma-separated frame sizes in ms")
    p.add_argument("--hops-ms", type=str, default="10,12.5,16", help="Comma-separated hop sizes in ms")
    p.add_argument("--lookahead-ms", type=str, default="0,10,20", help="Comma-separated lookahead in ms")

    # Gate params (we sweep these too)
    p.add_argument("--thresh-mul", type=str, default="1.2,1.5,1.8", help="Comma-separated threshold multipliers")
    p.add_argument("--mask-floor", type=str, default="0.02,0.05,0.1", help="Comma-separated minimum mask")
    p.add_argument("--noise-smooth", type=str, default="0.97,0.985,0.995", help="Comma-separated EMA smoothing")

    # Latency constraint
    p.add_argument("--max-e2e-ms", type=float, default=100.0, help="Only report configs with estimated E2E <= this")

    args = p.parse_args()

    def parse_list(s: str) -> List[float]:
        return [float(x.strip()) for x in s.split(",") if x.strip()]

    frames = parse_list(args.frames_ms)
    hops = parse_list(args.hops_ms)
    lookaheads = parse_list(args.lookahead_ms)
    thresh_muls = parse_list(args.thresh_mul)
    mask_floors = parse_list(args.mask_floor)
    noise_smooths = parse_list(args.noise_smooth)

    device = torch.device("cpu")  # streaming gate runs on CPU; later you swap in Hailo NN
    sr = int(args.sr)

    # Load audio
    noisy_wav, noisy_sr = torchaudio.load(args.noisy)
    noisy_16k = prep_audio(noisy_wav, noisy_sr, sr).to(device)
    if noisy_16k.shape[0] > 1:
        noisy_16k = torch.mean(noisy_16k, dim=0, keepdim=True)

    clean_16k = None
    if args.clean is not None:
        clean_wav, clean_sr = torchaudio.load(args.clean)
        clean_16k = prep_audio(clean_wav, clean_sr, sr).to(device)
        if clean_16k.shape[0] > 1:
            clean_16k = torch.mean(clean_16k, dim=0, keepdim=True)

        # trim to match
        n = min(clean_16k.shape[1], noisy_16k.shape[1])
        clean_16k = clean_16k[:, :n]
        noisy_16k = noisy_16k[:, :n]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: List[SweepResult] = []

    # Sweep
    idx = 0
    for frame_ms in frames:
        for hop_ms in hops:
            if hop_ms <= 0 or frame_ms <= 0:
                continue
            if hop_ms > frame_ms:
                continue

            for look_ms in lookaheads:
                # quick latency filter (rough): frame + lookahead + 20 + 10 <= max
                est = estimate_e2e_latency_ms(frame_ms, look_ms)
                if est > args.max_e2e_ms:
                    continue

                for tm in thresh_muls:
                    for mf in mask_floors:
                        for ns in noise_smooths:
                            idx += 1
                            name = f"cfg{idx:03d}_F{frame_ms:g}_H{hop_ms:g}_LA{look_ms:g}_TM{tm:g}_MF{mf:g}_NS{ns:g}"
                            params = {"thresh_mul": tm, "mask_floor": mf, "noise_smooth": ns}

                            print(f"[{idx:03d}] Running {name} (est_e2e~{est:.1f}ms)")
                            res = run_one_config(
                                noisy_16k=noisy_16k,
                                clean_16k=clean_16k,
                                sr=sr,
                                out_dir=out_dir,
                                name=name,
                                frame_ms=frame_ms,
                                hop_ms=hop_ms,
                                lookahead_ms=look_ms,
                                params=params,
                            )
                            results.append(res)
                            print(
                                f"      hop_mean={res.mean_hop_compute_ms:.2f}ms p95={res.p95_hop_compute_ms:.2f}ms "
                                f"RTF={res.rtf:.3f} SII={res.sii if res.sii is not None else 'n/a'}"
                            )

    if not results:
        print("No configs met the latency constraint. Increase --max-e2e-ms or reduce IO/sched assumptions in code.")
        return

    # Rank: maximize SII first (if available), else minimize compute then
    def key(r: SweepResult):
        sii = r.sii if r.sii is not None else -1.0
        return (sii, -r.rtf, -r.p95_hop_compute_ms)

    results_sorted = sorted(results, key=key, reverse=True)

    print("\n================== TOP RESULTS (<= target latency) ==================")
    print(f"{'name':<45} | {'e2e_ms':>7} | {'hop_ms':>7} | {'p95_ms':>7} | {'RTF':>6} | {'SII':>6} | {'SI-SDR':>7}")
    print("-" * 110)
    for r in results_sorted[:10]:
        print(
            f"{r.name:<45} | {r.est_e2e_latency_ms:7.1f} | {r.mean_hop_compute_ms:7.2f} | {r.p95_hop_compute_ms:7.2f} | "
            f"{r.rtf:6.3f} | {('%.3f' % r.sii) if r.sii is not None else '  n/a'} | {('%.2f' % r.si_sdr) if r.si_sdr is not None else '  n/a'}"
        )

    print("\nOutputs saved to:", out_dir)
    print("Tip: Listen to the top configs' WAV files and verify they match the metric ranking.")


if __name__ == "__main__":
    main()
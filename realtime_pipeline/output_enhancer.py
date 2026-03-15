from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass

import numpy as np
from scipy.signal import resample_poly

from .contracts import PipelineConfig


class PostFilterState:
    def __init__(self, frame_samples: int, cfg: PipelineConfig):
        self.n = int(frame_samples)
        self.fft_n = max(4, 2 * self.n)
        self.cfg = cfg
        self._win = np.sqrt(np.hanning(self.fft_n)).astype(np.float64)
        self._prev_in = np.zeros(self.n, dtype=np.float64)
        self._ola_tail = np.zeros(self.n, dtype=np.float64)
        self.noise_psd: np.ndarray | None = None
        self.speech_psd: np.ndarray | None = None
        self.gain_prev: np.ndarray | None = None
        self.post_snr_prev: np.ndarray | None = None

    def _smooth_gain_frequency(self, gain: np.ndarray) -> np.ndarray:
        radius = max(0, int(self.cfg.postfilter_freq_smoothing_bins))
        if radius <= 0:
            return gain
        width = (2 * radius) + 1
        positions = np.arange(width, dtype=np.float64) - float(radius)
        kernel = np.exp(-0.5 * (positions / max(float(radius), 1.0)) ** 2)
        kernel /= np.sum(kernel)
        return np.convolve(gain, kernel, mode="same")

    def process(self, frame: np.ndarray, speech_activity: float) -> np.ndarray:
        x = np.asarray(frame, dtype=np.float64).reshape(-1)
        block = np.concatenate([self._prev_in, x], axis=0)
        self._prev_in = x.copy()
        xw = block * self._win
        x_fft = np.fft.rfft(xw, n=self.fft_n)
        psd = np.abs(x_fft) ** 2

        if self.noise_psd is None:
            self.noise_psd = psd.copy()
        if self.speech_psd is None:
            self.speech_psd = np.maximum(psd * 0.25, 1e-12)
        if self.gain_prev is None:
            self.gain_prev = np.ones_like(psd, dtype=np.float64)
        if self.post_snr_prev is None:
            self.post_snr_prev = np.maximum(psd / np.maximum(self.noise_psd, 1e-12) - 1.0, 0.0)

        n_alpha = float(np.clip(self.cfg.postfilter_noise_ema_alpha, 0.0, 1.0))
        s_alpha = float(np.clip(self.cfg.postfilter_speech_ema_alpha, 0.0, 1.0))
        g_alpha = float(np.clip(self.cfg.postfilter_gain_ema_alpha, 0.0, 1.0))
        dd_alpha = float(np.clip(self.cfg.postfilter_dd_alpha, 0.0, 0.999))
        floor = float(np.clip(self.cfg.postfilter_gain_floor, 0.05, 1.0))
        speech_update_scale = float(np.clip(self.cfg.postfilter_noise_update_speech_scale, 0.0, 1.0))
        gain_max_step_db = float(max(0.0, self.cfg.postfilter_gain_max_step_db))

        output_rms = float(np.sqrt(np.mean(x**2) + 1e-12))
        output_activity = float(np.clip((output_rms - 0.003) / 0.02, 0.0, 1.0))
        posterior_snr_hint = float(np.clip(np.mean(psd / np.maximum(self.noise_psd, 1e-12) - 1.0) / 6.0, 0.0, 1.0))
        speech_presence = float(np.clip(max(float(speech_activity), output_activity, posterior_snr_hint), 0.0, 1.0))
        noise_alpha_eff = n_alpha * ((1.0 - speech_presence) + (speech_presence * speech_update_scale))
        noise_observation = np.minimum(psd, self.noise_psd * (1.0 + (2.5 * noise_alpha_eff)) + 1e-12)
        self.noise_psd = (1.0 - noise_alpha_eff) * self.noise_psd + noise_alpha_eff * noise_observation

        post_snr = np.maximum(psd / np.maximum(self.noise_psd, 1e-12) - 1.0, 0.0)
        dd_prior = dd_alpha * (self.gain_prev**2) * self.post_snr_prev + (1.0 - dd_alpha) * post_snr
        speech_observation = np.maximum(psd - self.noise_psd, 0.0)
        self.speech_psd = (1.0 - s_alpha) * self.speech_psd + s_alpha * speech_observation
        prior_snr = np.maximum(dd_prior, self.speech_psd / np.maximum(self.noise_psd, 1e-12))
        gain = prior_snr / (1.0 + prior_snr)
        gain = np.maximum(gain, floor)
        gain = self._smooth_gain_frequency(gain)
        if gain_max_step_db > 0.0:
            max_ratio = float(10.0 ** (gain_max_step_db / 20.0))
            gain = np.clip(gain, self.gain_prev / max_ratio, self.gain_prev * max_ratio)
        gain = (1.0 - g_alpha) * self.gain_prev + g_alpha * gain
        self.gain_prev = gain
        self.post_snr_prev = post_snr

        y_fft = x_fft * gain
        y_block = np.fft.irfft(y_fft, n=self.fft_n).real
        yw = y_block * self._win
        out = self._ola_tail + yw[: self.n]
        self._ola_tail = yw[self.n :].copy()
        return out.astype(np.float32, copy=False)


class RNNoiseBackend:
    def __init__(self) -> None:
        self._lib = None
        self._state = None
        self._py_backend = None
        self.backend_name = ""
        self.error = ""
        self._load()

    def _load(self) -> None:
        candidates = [
            os.environ.get("RNNOISE_LIB"),
            "librnnoise.so",
            "librnnoise.dylib",
            "rnnoise.dll",
        ]
        for candidate in candidates:
            if not candidate:
                continue
            try:
                lib = ctypes.cdll.LoadLibrary(candidate)
                lib.rnnoise_create.restype = ctypes.c_void_p
                lib.rnnoise_destroy.argtypes = [ctypes.c_void_p]
                lib.rnnoise_process_frame.argtypes = [
                    ctypes.c_void_p,
                    ctypes.POINTER(ctypes.c_float),
                    ctypes.POINTER(ctypes.c_float),
                ]
                state = lib.rnnoise_create(None)
                if not state:
                    continue
                self._lib = lib
                self._state = state
                self.backend_name = str(candidate)
                self.error = ""
                return
            except OSError:
                continue
            except Exception as exc:  # pragma: no cover
                self.error = str(exc)
        try:
            from pyrnnoise import RNNoise as PyRNNoise

            self._py_backend = PyRNNoise(48000)
            self.backend_name = "pyrnnoise"
            self.error = ""
        except Exception as exc:  # pragma: no cover
            self.error = str(exc)

    @property
    def available(self) -> bool:
        return (self._lib is not None and self._state is not None) or (self._py_backend is not None)

    def process_48k(self, audio_48k: np.ndarray) -> np.ndarray:
        if not self.available:
            raise RuntimeError(self.error or "RNNoise unavailable")
        x_48k = np.asarray(audio_48k, dtype=np.float32).reshape(-1)
        frame_len = 480
        if x_48k.shape[0] % frame_len:
            x_48k = np.pad(x_48k, (0, frame_len - (x_48k.shape[0] % frame_len)))
        if self._py_backend is not None:
            parts = []
            for _vad, den in self._py_backend.denoise_chunk(x_48k):
                den_arr = np.asarray(den, dtype=np.float32).reshape(-1)
                if den_arr.dtype.kind in {"i", "u"} or np.max(np.abs(den_arr)) > 1.5:
                    den_arr = den_arr / 32768.0
                parts.append(den_arr)
            return np.concatenate(parts).astype(np.float32, copy=False)
        out = np.zeros_like(x_48k, dtype=np.float32)
        for start in range(0, x_48k.shape[0], frame_len):
            frame = np.ascontiguousarray(x_48k[start : start + frame_len], dtype=np.float32)
            out_frame = np.zeros(frame_len, dtype=np.float32)
            self._lib.rnnoise_process_frame(
                ctypes.c_void_p(self._state),
                out_frame.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                frame.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            )
            out[start : start + frame_len] = out_frame
        return out

    def close(self) -> None:
        if self._lib is not None and self._state is not None:
            try:
                self._lib.rnnoise_destroy(ctypes.c_void_p(self._state))
            except Exception:
                pass
            self._state = None


class RNNoiseEnhancer:
    TARGET_SR = 48000

    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.backend = RNNoiseBackend()

    @property
    def available(self) -> bool:
        return self.backend.available

    def process(self, frame: np.ndarray) -> np.ndarray:
        if not self.available:
            return np.asarray(frame, dtype=np.float32)
        x = np.asarray(frame, dtype=np.float32).reshape(-1)
        input_gain = float(10.0 ** (float(self.cfg.rnnoise_input_gain_db) / 20.0))
        wet_mix = float(np.clip(self.cfg.rnnoise_wet_mix, 0.0, 1.0))
        boosted = x * input_gain
        expected_48k = int(round(boosted.shape[0] * self.TARGET_SR / float(self.cfg.sample_rate_hz)))
        if int(self.cfg.sample_rate_hz) != self.TARGET_SR:
            x_48k = resample_poly(boosted, self.TARGET_SR, int(self.cfg.sample_rate_hz)).astype(np.float32, copy=False)
        else:
            x_48k = boosted.astype(np.float32, copy=False)
        den_48k = self.backend.process_48k(x_48k)[:expected_48k]
        if int(self.cfg.sample_rate_hz) != self.TARGET_SR:
            den = resample_poly(den_48k, int(self.cfg.sample_rate_hz), self.TARGET_SR).astype(np.float32, copy=False)
        else:
            den = den_48k.astype(np.float32, copy=False)
        if den.shape[0] < x.shape[0]:
            den = np.pad(den, (0, x.shape[0] - den.shape[0]))
        den = den[: x.shape[0]]
        mixed = (wet_mix * den) + ((1.0 - wet_mix) * x)
        return np.asarray(mixed, dtype=np.float32)

    def close(self) -> None:
        self.backend.close()


@dataclass(frozen=True)
class OutputEnhancerMetadata:
    output_enhancer_mode: str
    rnnoise_backend: str
    rnnoise_available: bool
    rnnoise_error: str


class OutputEnhancerChain:
    def __init__(self, frame_samples: int, cfg: PipelineConfig):
        self.cfg = cfg
        self.mode = _resolve_output_enhancer_mode(cfg)
        self.outer_frame_samples = int(frame_samples)
        self.inner_frame_samples = _resolve_output_enhancer_frame_samples(cfg)
        self.postfilter = PostFilterState(frame_samples=self.inner_frame_samples, cfg=cfg) if self.mode in {"shared_wiener", "shared_wiener_rnnoise"} else None
        self.rnnoise = RNNoiseEnhancer(cfg) if self.mode in {"rnnoise", "shared_wiener_rnnoise"} else None

    def process(self, frame: np.ndarray, speech_activity: float) -> np.ndarray:
        out = np.asarray(frame, dtype=np.float32).reshape(-1)
        if self.inner_frame_samples <= 0 or self.inner_frame_samples >= out.shape[0]:
            if self.postfilter is not None:
                out = self.postfilter.process(out, speech_activity=speech_activity)
            if self.rnnoise is not None:
                out = self.rnnoise.process(out)
            return np.asarray(out, dtype=np.float32)
        chunks: list[np.ndarray] = []
        for start in range(0, out.shape[0], self.inner_frame_samples):
            piece = out[start : start + self.inner_frame_samples]
            if piece.shape[0] < self.inner_frame_samples:
                piece = np.pad(piece, (0, self.inner_frame_samples - piece.shape[0]))
            if self.postfilter is not None:
                piece = self.postfilter.process(piece, speech_activity=speech_activity)
            if self.rnnoise is not None:
                piece = self.rnnoise.process(piece)
            chunks.append(np.asarray(piece, dtype=np.float32))
        return np.concatenate(chunks)[: out.shape[0]].astype(np.float32, copy=False)

    def metadata(self) -> OutputEnhancerMetadata:
        rnnoise_backend = ""
        rnnoise_available = False
        rnnoise_error = ""
        if self.rnnoise is not None:
            rnnoise_backend = str(self.rnnoise.backend.backend_name)
            rnnoise_available = bool(self.rnnoise.available)
            rnnoise_error = str(self.rnnoise.backend.error)
        return OutputEnhancerMetadata(
            output_enhancer_mode=self.mode,
            rnnoise_backend=rnnoise_backend,
            rnnoise_available=rnnoise_available,
            rnnoise_error=rnnoise_error,
        )

    def close(self) -> None:
        if self.rnnoise is not None:
            self.rnnoise.close()


def _resolve_output_enhancer_mode(cfg: PipelineConfig) -> str:
    mode = str(getattr(cfg, "output_enhancer_mode", "auto")).strip().lower()
    if mode == "auto":
        return "shared_wiener" if bool(cfg.postfilter_enabled) else "off"
    if mode in {"off", "shared_wiener", "rnnoise", "shared_wiener_rnnoise"}:
        return mode
    return "off"


def _resolve_output_enhancer_frame_samples(cfg: PipelineConfig) -> int:
    frame_ms = float(getattr(cfg, "output_enhancer_frame_ms", 0.0) or 0.0)
    if frame_ms <= 0.0:
        frame_ms = float(cfg.fast_frame_ms)
    samples = int(round(float(cfg.sample_rate_hz) * (frame_ms / 1000.0)))
    return max(1, samples)


def apply_output_enhancer_audio(
    audio: np.ndarray,
    *,
    cfg: PipelineConfig,
    frame_samples: int,
    speech_activity: float | np.ndarray = 0.0,
) -> tuple[np.ndarray, OutputEnhancerMetadata]:
    chain = OutputEnhancerChain(frame_samples=frame_samples, cfg=cfg)
    try:
        x = np.asarray(audio, dtype=np.float32).reshape(-1)
        outputs: list[np.ndarray] = []
        if np.isscalar(speech_activity):
            activity_source = None
            activity_default = float(speech_activity)
        else:
            activity_source = np.asarray(speech_activity, dtype=np.float64).reshape(-1)
            activity_default = 0.0
        for start in range(0, x.shape[0], frame_samples):
            frame = x[start : start + frame_samples]
            if frame.shape[0] < frame_samples:
                frame = np.pad(frame, (0, frame_samples - frame.shape[0]))
            if activity_source is None:
                activity = activity_default
            else:
                end = min(activity_source.shape[0], start + frame_samples)
                activity = float(np.clip(np.mean(activity_source[start:end]) if end > start else 0.0, 0.0, 1.0))
            outputs.append(chain.process(frame, speech_activity=activity))
        out = np.concatenate(outputs)[: x.shape[0]] if outputs else np.zeros_like(x)
        return np.asarray(out, dtype=np.float32), chain.metadata()
    finally:
        chain.close()

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import webrtcvad  # type: ignore
except ImportError:  # pragma: no cover - exercised via fallback behavior
    webrtcvad = None

try:  # pragma: no cover - optional dependency
    import torch
    from scipy.signal import resample_poly
    from silero_vad import get_speech_timestamps, load_silero_vad  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    torch = None
    resample_poly = None
    get_speech_timestamps = None
    load_silero_vad = None


DEFAULT_WEBRTC_VAD_MODE = 2
DEFAULT_WEBRTC_FRAME_MS = 20
DEFAULT_WEBRTC_HANGOVER_FRAMES = 4
MIN_PCM_VALUE = -32768
MAX_PCM_VALUE = 32767


@dataclass(frozen=True, slots=True)
class VADDecision:
    speech_active: bool
    speech_score: float
    raw_active: bool
    hangover_remaining: int
    frame_ms: int
    source: str


class WebRTCVADGate:
    def __init__(
        self,
        *,
        sample_rate_hz: int,
        mode: int = DEFAULT_WEBRTC_VAD_MODE,
        frame_ms: int = DEFAULT_WEBRTC_FRAME_MS,
        hangover_frames: int = DEFAULT_WEBRTC_HANGOVER_FRAMES,
    ) -> None:
        self.sample_rate_hz = int(sample_rate_hz)
        self.mode = int(mode)
        self.frame_ms = int(frame_ms)
        self.hangover_frames = int(hangover_frames)
        self._hangover_remaining = 0
        self._vad = None if webrtcvad is None else webrtcvad.Vad(self.mode)

    def process(self, mono_audio: np.ndarray) -> VADDecision:
        mono = np.asarray(mono_audio, dtype=np.float32).reshape(-1)
        if mono.size == 0:
            return VADDecision(
                speech_active=False,
                speech_score=0.0,
                raw_active=False,
                hangover_remaining=int(self._hangover_remaining),
                frame_ms=self.frame_ms,
                source="empty",
            )

        if self._vad is None:
            return self._fallback_energy_gate(mono)

        frame_samples = int(round((self.sample_rate_hz * self.frame_ms) / 1000.0))
        if frame_samples <= 0 or mono.size < frame_samples:
            return self._fallback_energy_gate(mono)

        usable = (mono.size // frame_samples) * frame_samples
        if usable <= 0:
            return self._fallback_energy_gate(mono)
        pcm = np.clip(mono[:usable], -1.0, 1.0)
        pcm = np.round(pcm * MAX_PCM_VALUE).astype(np.int16, copy=False)
        frames = pcm.reshape(-1, frame_samples)
        flags = [bool(self._vad.is_speech(frame.tobytes(), self.sample_rate_hz)) for frame in frames]
        score = float(np.mean(flags)) if flags else 0.0
        raw_active = bool(any(flags))
        if raw_active:
            self._hangover_remaining = int(self.hangover_frames)
        else:
            self._hangover_remaining = max(0, int(self._hangover_remaining) - 1)
        speech_active = bool(raw_active or self._hangover_remaining > 0)
        return VADDecision(
            speech_active=speech_active,
            speech_score=score,
            raw_active=raw_active,
            hangover_remaining=int(self._hangover_remaining),
            frame_ms=self.frame_ms,
            source="webrtcvad",
        )

    def _fallback_energy_gate(self, mono: np.ndarray) -> VADDecision:
        rms = float(np.sqrt(np.mean(np.square(mono))) if mono.size else 0.0)
        raw_active = bool(rms >= 0.015)
        if raw_active:
            self._hangover_remaining = int(self.hangover_frames)
        else:
            self._hangover_remaining = max(0, int(self._hangover_remaining) - 1)
        speech_active = bool(raw_active or self._hangover_remaining > 0)
        return VADDecision(
            speech_active=speech_active,
            speech_score=float(np.clip(rms / 0.05, 0.0, 1.0)),
            raw_active=raw_active,
            hangover_remaining=int(self._hangover_remaining),
            frame_ms=self.frame_ms,
            source="energy_fallback",
        )


class SileroVADGate:
    def __init__(
        self,
        *,
        sample_rate_hz: int,
        frame_ms: int = DEFAULT_WEBRTC_FRAME_MS,
        hangover_frames: int = DEFAULT_WEBRTC_HANGOVER_FRAMES,
    ) -> None:
        if load_silero_vad is None or get_speech_timestamps is None or torch is None or resample_poly is None:
            raise RuntimeError("silero_fused target activity backend requested but silero_vad/torch/scipy is unavailable.")
        self.sample_rate_hz = int(sample_rate_hz)
        self.frame_ms = int(frame_ms)
        self.hangover_frames = int(hangover_frames)
        self._hangover_remaining = 0
        self._model = load_silero_vad()

    def process(self, mono_audio: np.ndarray) -> VADDecision:
        mono = np.asarray(mono_audio, dtype=np.float32).reshape(-1)
        if mono.size == 0:
            return VADDecision(
                speech_active=False,
                speech_score=0.0,
                raw_active=False,
                hangover_remaining=int(self._hangover_remaining),
                frame_ms=self.frame_ms,
                source="silero_empty",
            )
        target_sr = 16000
        if int(self.sample_rate_hz) != target_sr:
            mono = resample_poly(mono, target_sr, int(self.sample_rate_hz)).astype(np.float32, copy=False)
        audio = torch.from_numpy(np.clip(mono, -1.0, 1.0))
        speech_segments = get_speech_timestamps(audio, self._model, sampling_rate=target_sr)
        voiced = 0
        for seg in speech_segments:
            voiced += max(0, int(seg.get("end", 0)) - int(seg.get("start", 0)))
        score = float(np.clip(voiced / max(len(mono), 1), 0.0, 1.0))
        raw_active = bool(speech_segments)
        if raw_active:
            self._hangover_remaining = int(self.hangover_frames)
        else:
            self._hangover_remaining = max(0, int(self._hangover_remaining) - 1)
        speech_active = bool(raw_active or self._hangover_remaining > 0)
        return VADDecision(
            speech_active=speech_active,
            speech_score=score,
            raw_active=raw_active,
            hangover_remaining=int(self._hangover_remaining),
            frame_ms=self.frame_ms,
            source="silero_vad",
        )

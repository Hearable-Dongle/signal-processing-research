from __future__ import annotations

from typing import Protocol

import numpy as np


class SpeakerEmbeddingBackend(Protocol):
    def embed(self, audio: np.ndarray, sample_rate_hz: int) -> np.ndarray | None:
        ...


def _normalize(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-12:
        return arr.astype(np.float32, copy=False)
    return (arr / norm).astype(np.float32, copy=False)


class SpeechbrainECAPABackend:
    def __init__(self, device: str = "cpu") -> None:
        self._device = str(device)
        self._classifier = None
        self._torch = None

    def _load(self) -> None:
        if self._classifier is not None:
            return
        import torch
        import torchaudio
        import huggingface_hub

        if not hasattr(torchaudio, "list_audio_backends"):
            torchaudio.list_audio_backends = lambda: ["soundfile"]  # type: ignore[attr-defined]
        if not hasattr(torchaudio, "set_audio_backend"):
            torchaudio.set_audio_backend = lambda _backend: None  # type: ignore[attr-defined]
        orig_hf_download = huggingface_hub.hf_hub_download

        def _hf_hub_download_compat(*args, **kwargs):
            if "use_auth_token" in kwargs and "token" not in kwargs:
                kwargs["token"] = kwargs.pop("use_auth_token")
            else:
                kwargs.pop("use_auth_token", None)
            return orig_hf_download(*args, **kwargs)

        huggingface_hub.hf_hub_download = _hf_hub_download_compat
        from speechbrain.inference.speaker import EncoderClassifier

        self._torch = torch
        try:
            self._classifier = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir="pretrained_models/spkrec-ecapa-voxceleb",
            ).to(torch.device(self._device))
        finally:
            huggingface_hub.hf_hub_download = orig_hf_download

    def embed(self, audio: np.ndarray, sample_rate_hz: int) -> np.ndarray | None:
        if int(sample_rate_hz) != 16000:
            raise ValueError(f"SpeechbrainECAPABackend expects 16 kHz audio, got {sample_rate_hz}")
        x = np.asarray(audio, dtype=np.float32).reshape(-1)
        if x.size == 0:
            return None
        self._load()
        with self._torch.no_grad():
            emb = self._classifier.encode_batch(self._torch.tensor(x, dtype=self._torch.float32).unsqueeze(0))
        arr = emb.squeeze().detach().cpu().numpy()
        return _normalize(arr)


class WavLMXVectorBackend:
    def __init__(self, device: str = "cpu", model_id: str = "microsoft/wavlm-base-plus-sv") -> None:
        self._device = str(device)
        self._model_id = str(model_id)
        self._torch = None
        self._processor = None
        self._model = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector

        self._torch = torch
        self._processor = Wav2Vec2FeatureExtractor.from_pretrained(self._model_id)
        self._model = WavLMForXVector.from_pretrained(self._model_id).to(torch.device(self._device)).eval()

    def embed(self, audio: np.ndarray, sample_rate_hz: int) -> np.ndarray | None:
        if int(sample_rate_hz) != 16000:
            raise ValueError(f"WavLMXVectorBackend expects 16 kHz audio, got {sample_rate_hz}")
        x = np.asarray(audio, dtype=np.float32).reshape(-1)
        if x.size == 0:
            return None
        self._load()
        inputs = self._processor(
            x,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with self._torch.no_grad():
            out = self._model(**inputs)
        return _normalize(out.embeddings.squeeze().detach().cpu().numpy())


def build_session_embedding_backend(model_name: str, device: str = "cpu") -> SpeakerEmbeddingBackend:
    key = str(model_name).strip().lower()
    if key in {"ecapa_voxceleb", "ecapa-voxceleb", "speechbrain_ecapa"}:
        return SpeechbrainECAPABackend(device=device)
    if key in {"wavlm_xvector", "wavlm-large", "wavlm", "wavlm_base_plus_sv", "wavlm-base-plus-sv"}:
        return WavLMXVectorBackend(device=device, model_id="microsoft/wavlm-base-plus-sv")
    if key in {"wavlm_base_sv", "wavlm-base-sv"}:
        return WavLMXVectorBackend(device=device, model_id="microsoft/wavlm-base-sv")
    raise ValueError(f"Unsupported speaker embedding model: {model_name}")

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pyroomacoustics as pra
import torch
import torch.nn as nn
from scipy import signal

from simulation.mic_array_profiles import mic_positions_xyz
from realtime_pipeline.localization_backends import LocalizationBackendBase, LocalizationBackendResult, _pair_indices


FREQ_RANGE_HZ = (300, 2500)
HIDDEN_UNITS = 64
LEARNING_RATE = 1e-3
EPOCHS = 12
BATCH_SIZE = 128
TRAIN_SAMPLES = 4096
VAL_SAMPLES = 512
CHECKPOINT_DIR = Path("pretrained_models/localization/ipd_regressor")
CHECKPOINT_PATH = CHECKPOINT_DIR / "ipd_regressor.pt"
MANIFEST_PATH = CHECKPOINT_DIR / "training_manifest.json"


class _IPDMLP(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, HIDDEN_UNITS),
            nn.ReLU(),
            nn.Linear(HIDDEN_UNITS, HIDDEN_UNITS),
            nn.ReLU(),
            nn.Linear(HIDDEN_UNITS, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _target_xy(angle_deg: np.ndarray) -> np.ndarray:
    rad = np.deg2rad(angle_deg)
    return np.stack([np.cos(rad), np.sin(rad)], axis=1).astype(np.float32)


def extract_ipd_features(audio: np.ndarray, fs: int, nfft: int, overlap: float) -> np.ndarray:
    f_vec, _t_vec, zxx = signal.stft(
        audio,
        fs=fs,
        nperseg=nfft,
        noverlap=int(nfft * overlap),
        boundary=None,
        padded=False,
    )
    mask = (f_vec >= FREQ_RANGE_HZ[0]) & (f_vec <= FREQ_RANGE_HZ[1])
    z = zxx[:, mask, :]
    if z.shape[1] == 0 or z.shape[2] == 0:
        return np.zeros(len(_pair_indices(audio.shape[0])) * 4, dtype=np.float32)
    features: list[np.ndarray] = []
    for i, j in _pair_indices(audio.shape[0]):
        ipd = np.unwrap(np.angle(z[i] * np.conj(z[j])), axis=0)
        feat = np.mean(ipd, axis=1)
        features.append(feat.astype(np.float32))
    return np.concatenate(features, axis=0).astype(np.float32)


def _simulate_training_example(rng: np.random.Generator, fs: int, nfft: int, overlap: float) -> tuple[np.ndarray, float]:
    mic_pos = mic_positions_xyz("respeaker_v3_0457")
    e_abs = None
    max_order = None
    room_dim = None
    rt60 = None
    for _ in range(32):
        candidate_room_dim = [float(rng.uniform(2.8, 5.5)), float(rng.uniform(2.8, 5.5)), 2.8]
        candidate_rt60 = float(rng.uniform(0.08, 0.28))
        try:
            e_abs, max_order = pra.inverse_sabine(candidate_rt60, candidate_room_dim)
            room_dim = candidate_room_dim
            rt60 = candidate_rt60
            break
        except ValueError:
            continue
    if e_abs is None or max_order is None or room_dim is None or rt60 is None:
        room_dim = [4.0, 4.0, 2.8]
        rt60 = 0.18
        e_abs, max_order = 0.35, 8
    room = pra.ShoeBox(room_dim, fs=fs, materials=pra.Material(e_abs), max_order=max_order)
    center = np.asarray([room_dim[0] / 2.0, room_dim[1] / 2.0, 1.2], dtype=np.float64)
    room.add_microphone_array(pra.MicrophoneArray((mic_pos + center).T, fs=fs))
    angle_deg = float(rng.uniform(0.0, 360.0))
    max_distance = max(0.6, min(center[0], center[1], room_dim[0] - center[0], room_dim[1] - center[1]) - 0.35)
    distance = float(rng.uniform(0.5, min(1.8, max_distance)))
    src = center + np.asarray(
        [distance * math.cos(math.radians(angle_deg)), distance * math.sin(math.radians(angle_deg)), 0.0],
        dtype=np.float64,
    )
    t = np.arange(int(0.2 * fs), dtype=np.float64) / fs
    carrier = (
        0.6 * np.sin(2.0 * np.pi * rng.uniform(220.0, 480.0) * t)
        + 0.3 * np.sin(2.0 * np.pi * rng.uniform(700.0, 1500.0) * t + rng.uniform(-1.0, 1.0))
    )
    envelope = 0.5 + 0.5 * np.sin(2.0 * np.pi * rng.uniform(1.0, 4.0) * t + rng.uniform(-1.0, 1.0))
    sig = (carrier * envelope).astype(np.float32)
    room.add_source(src, signal=sig)
    room.simulate()
    audio = np.asarray(room.mic_array.signals, dtype=np.float32)
    audio += rng.normal(0.0, 0.003, size=audio.shape).astype(np.float32)
    return extract_ipd_features(audio, fs=fs, nfft=nfft, overlap=overlap), angle_deg


def ensure_trained(*, fs: int, nfft: int, overlap: float, force: bool = False) -> Path:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    if CHECKPOINT_PATH.exists() and MANIFEST_PATH.exists() and not force:
        return CHECKPOINT_PATH

    rng = np.random.default_rng(0)
    train_x = []
    train_y = []
    for _ in range(TRAIN_SAMPLES):
        feat, angle = _simulate_training_example(rng, fs, nfft, overlap)
        train_x.append(feat)
        train_y.append(angle)
    val_x = []
    val_y = []
    for _ in range(VAL_SAMPLES):
        feat, angle = _simulate_training_example(rng, fs, nfft, overlap)
        val_x.append(feat)
        val_y.append(angle)

    x_train = torch.from_numpy(np.stack(train_x, axis=0))
    y_train = torch.from_numpy(_target_xy(np.asarray(train_y, dtype=np.float32)))
    x_val = torch.from_numpy(np.stack(val_x, axis=0))
    y_val = torch.from_numpy(_target_xy(np.asarray(val_y, dtype=np.float32)))
    model = _IPDMLP(int(x_train.shape[1]))
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    device = _device()
    model = model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    for _epoch in range(EPOCHS):
        order = torch.randperm(x_train.shape[0])
        model.train()
        for start in range(0, x_train.shape[0], BATCH_SIZE):
            idx = order[start : start + BATCH_SIZE]
            xb = x_train[idx].to(device)
            yb = y_train[idx].to(device)
            optim.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optim.step()
        model.eval()
        with torch.no_grad():
            val_pred = model(x_val.to(device))
            val_loss = float(loss_fn(val_pred, y_val.to(device)).item())
        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "state_dict": model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
                    "input_dim": int(x_train.shape[1]),
                    "fs": int(fs),
                    "nfft": int(nfft),
                    "overlap": float(overlap),
                },
                CHECKPOINT_PATH,
            )
    MANIFEST_PATH.write_text(
        json.dumps(
            {
                "train_samples": TRAIN_SAMPLES,
                "val_samples": VAL_SAMPLES,
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "best_val_loss": best_val,
                "cuda_available": torch.cuda.is_available(),
                "cuda_device_count": torch.cuda.device_count(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return CHECKPOINT_PATH


class IPDRegressorBackend(LocalizationBackendBase):
    def __init__(self, **kwargs):
        kwargs["freq_range"] = tuple(FREQ_RANGE_HZ)
        super().__init__(**kwargs)
        self._model: _IPDMLP | None = None
        self._device = _device()
        self._checkpoint_path = ensure_trained(fs=self.fs, nfft=self.nfft, overlap=self.overlap)

    def _load_model(self, input_dim: int) -> _IPDMLP:
        if self._model is not None:
            return self._model
        payload = torch.load(self._checkpoint_path, map_location=self._device)
        model = _IPDMLP(int(payload["input_dim"]))
        model.load_state_dict(payload["state_dict"])
        model.to(self._device)
        model.eval()
        self._model = model
        return model

    def process(self, audio: np.ndarray) -> LocalizationBackendResult:
        feat = extract_ipd_features(np.asarray(audio, dtype=np.float32), fs=self.fs, nfft=self.nfft, overlap=self.overlap)
        model = self._load_model(int(feat.shape[0]))
        with torch.no_grad():
            pred = model(torch.from_numpy(feat[None, :]).to(self._device)).detach().cpu().numpy()[0]
        angle = float(np.degrees(np.arctan2(pred[1], pred[0])) % 360.0)
        confidence = float(np.linalg.norm(pred))
        return LocalizationBackendResult(
            peaks_deg=[angle],
            peak_scores=[confidence],
            score_spectrum=None,
            debug={
                "backend": "ipd_regressor",
                "checkpoint_path": str(self._checkpoint_path.resolve()),
                "feature_dim": int(feat.shape[0]),
                "device": str(self._device),
            },
        )


def main() -> None:
    from realtime_pipeline.localization_strategies.cli import run_backend_cli

    run_backend_cli("ipd_regressor")


if __name__ == "__main__":
    main()

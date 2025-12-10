import argparse
from pathlib import Path
import torch
import torchaudio
import matplotlib.pyplot as plt
from typing import Literal

from asteroid.models import ConvTasNet
from speechbrain.inference.enhancement import SpectralMaskEnhancement
import noisereduce as nr
from df.enhance import enhance, init_df

from own_voice_suppression.audio_utils import prep_audio

WINDOW_SEC = 2.0  
STRIDE_SEC = 0.5

MODEL_OPTIONS = ["convtasnet", "metricgan", "deepfilternet", "spectral-gating"]
ModelOption = Literal["convtasnet", "metricgan", "deepfilternet", "spectral-gating"]

class ConvTasNetWrapper:
    """ Wraps Asteroid's ConvTasNet for single-speaker enhancement. """
    MODEL_ID = "JorisCos/ConvTasNet_Libri1Mix_enhsingle_16k"
    NATIVE_SR = 16000
    
    def __init__(self, device):
        self.device = device
        print(f"[Model] Loading {self.MODEL_ID}...")
        self.model = ConvTasNet.from_pretrained(self.MODEL_ID).to(device)
        self.model.eval()
        
    def process(self, noisy_chunk: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            estimate = self.model(noisy_chunk)
        return estimate[:, 0, :]

class MetricGANWrapper:
    """ Wraps SpeechBrain's MetricGAN+ """
    MODEL_ID = "speechbrain/metricgan-plus-voicebank"
    NATIVE_SR = 16000
    
    def __init__(self, device):
        self.device = device
        print(f"[Model] Loading {self.MODEL_ID}...")
        self.model = SpectralMaskEnhancement.from_hparams(
            source=self.MODEL_ID, 
            savedir=f"pretrained_models/{self.MODEL_ID}",
            run_opts={"device": str(device)}
        )
        
    def process(self, noisy_chunk: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            lengths = torch.ones(noisy_chunk.shape[0], device=self.device)
            return self.model.enhance_batch(noisy_chunk, lengths=lengths)

class DeepFilterNetWrapper:
    """ Wraps DeepFilterNet3 """
    NATIVE_SR = 48000

    def __init__(self, device):
        if device.type == "cuda":
            print("[Warning] DeepFilterNet3 does not support GPU inference. Using CPU instead.")

        self.device = torch.device("cpu")
        print("[Model] Loading DeepFilterNet3...")
        self.model, self.df_state, _ = init_df(config_allow_defaults=True)

    def process(self, noisy_chunk: torch.Tensor) -> torch.Tensor:
        noisy_chunk = noisy_chunk.to(self.device)
        with torch.no_grad():
            return enhance(self.model, self.df_state, noisy_chunk)

class SpectralGatingWrapper:
    """ Wraps the noisereduce library for traditional spectral gating. """
    NATIVE_SR = 16000 # Standardized to 16k but not necessary

    def __init__(self, device):
        self.device = device
        print("[Model] Initializing Spectral Gating...")
        
    def process(self, noisy_chunk: torch.Tensor) -> torch.Tensor:
        noisy_np = noisy_chunk.squeeze(0).cpu().numpy()
        
        reduced_np = nr.reduce_noise(y=noisy_np, sr=self.NATIVE_SR)
        
        return torch.from_numpy(reduced_np).unsqueeze(0).to(self.device)


def main(input_path, output_directory, model_type: ModelOption, window_sec=WINDOW_SEC, stride_sec=STRIDE_SEC):
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / "denoised.wav"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")

    if model_type == "convtasnet":
        enhancer = ConvTasNetWrapper(device)
    elif model_type == "metricgan":
        enhancer = MetricGANWrapper(device)
    elif model_type == "deepfilternet":
        enhancer = DeepFilterNetWrapper(device)
    elif model_type == "spectral-gating":
        enhancer = SpectralGatingWrapper(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    working_sr = enhancer.NATIVE_SR
    print(f"Working Sample Rate: {working_sr} Hz")

    print("Loading audio...")
    noisy_wav, sr = torchaudio.load(input_path)
    noisy_wav = prep_audio(noisy_wav, sr, working_sr).to(device)

    window_samples = int(window_sec * working_sr)
    stride_samples = int(stride_sec * working_sr)
    num_samples = noisy_wav.shape[1]

    output_buffer = torch.zeros_like(noisy_wav)
    
    print(f"Denoising {num_samples/working_sr:.2f}s of audio...")

    current_start = 0
    while current_start + window_samples <= num_samples:
        chunk = noisy_wav[:, current_start : current_start + window_samples]
        enhanced_chunk = enhancer.process(chunk)
        
        if current_start == 0:
            output_buffer[:, 0 : window_samples] = enhanced_chunk
        else:
            stride_idx = window_samples - stride_samples
            new_content = enhanced_chunk[:, stride_idx:]
            update_start = current_start + stride_idx
            update_end = current_start + window_samples
            output_buffer[:, update_start : update_end] = new_content
            
        current_start += stride_samples

    output_buffer = prep_audio(output_buffer, working_sr, 16000)
    torchaudio.save(output_path, output_buffer.cpu(), 16000)
    print(f"Saved denoised audio to {output_path}")
    
    print("Generating comparison plot...")
    plt.figure(figsize=(12, 6))
    
    noisy_16k = prep_audio(noisy_wav, working_sr, 16000)

    plt.subplot(2, 1, 1)
    plt.title("Original (Noisy)")
    plt.specgram(noisy_16k.cpu().numpy()[0], Fs=16000, NFFT=1024, noverlap=512)
    
    plt.subplot(2, 1, 2)
    plt.title(f"Enhanced (Clean) - {model_type}")
    plt.specgram(output_buffer.cpu().numpy()[0], Fs=16000, NFFT=1024, noverlap=512)
    
    plt.tight_layout()
    plt.savefig(output_directory / "denoising_spectrogram.png")
    print("Spectrogram saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=Path, required=True, help="Path to noisy audio")
    parser.add_argument("--model-type", type=str, choices=MODEL_OPTIONS, default="convtasnet")
    parser.add_argument("--output-directory", type=Path)
    
    args = parser.parse_args()

    if not args.output_directory:
        args.output_directory = Path("denoise/outputs") / f"{args.model_type}_{args.input_path.stem}"

    main(args.input_path, args.output_directory, args.model_type)
import argparse
from pathlib import Path
import sys
import torch
import torchaudio
import matplotlib.pyplot as plt
from typing import Literal
import numpy as np
from scipy import signal
import pywt

from asteroid.models import ConvTasNet
from speechbrain.inference.enhancement import SpectralMaskEnhancement
import noisereduce as nr
from df.enhance import enhance, init_df

# Add project root to allow sibling imports
sys.path.append(str(Path(__file__).resolve().parents[1]))
from own_voice_suppression.audio_utils import prep_audio

WINDOW_SEC = 2.0  
STRIDE_SEC = 0.5

MODEL_OPTIONS = ["convtasnet", "metricgan", "deepfilternet", "spectral-gating", "spectral-subtraction", "wiener", "wavelet", "high-pass", "notch"]
ModelOption = Literal["convtasnet", "metricgan", "deepfilternet", "spectral-gating", "spectral-subtraction", "wiener", "wavelet", "high-pass", "notch"]

class ConvTasNetWrapper:
    """ 
    Wraps Asteroid's ConvTasNet for single-speaker enhancement. 
    Includes internal padding and volume normalization to fix stitching artifacts.
    """
    MODEL_ID = "JorisCos/ConvTasNet_Libri1Mix_enhsingle_16k"
    NATIVE_SR = 16000
    
    def __init__(self, device):
        self.device = device
        print(f"[Model] Loading {self.MODEL_ID}...")
        self.model = ConvTasNet.from_pretrained(self.MODEL_ID).to(device)
        self.model.eval()
        
    def process(self, noisy_chunk: torch.Tensor) -> torch.Tensor:
        """
        Process the chunk with internal padding and energy normalization 
        to ensure smooth stitching in the main loop.
        """
        with torch.no_grad():
            # Pad the input with 0.5s of reflected audio so the model has future contxt
            pad_samples = int(0.5 * self.NATIVE_SR)
            padded_input = torch.nn.functional.pad(
                noisy_chunk.unsqueeze(1),  
                (pad_samples, pad_samples), 
                mode='reflect'
            ).squeeze(1)

            estimate_padded = self.model(padded_input)

            # Crop the padding back off to get the original size
            estimate = estimate_padded[:, 0, pad_samples:-pad_samples]

            # ConvTasNet changes the volume arbitrarily, so we normalize output energy back to their input RMS
            input_energy = torch.sqrt(torch.mean(noisy_chunk ** 2, dim=-1, keepdim=True))
            output_energy = torch.sqrt(torch.mean(estimate ** 2, dim=-1, keepdim=True)) + 1e-8
            estimate = estimate * (input_energy / output_energy)

        return estimate

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

class WienerFilterWrapper:
    """ Wraps scipy.signal.wiener for denoising. """
    NATIVE_SR = 16000 # Not strictly necessary, but good for consistency

    def __init__(self, device):
        self.device = device
        print("[Model] Initializing Wiener Filter...")

    def process(self, noisy_chunk: torch.Tensor) -> torch.Tensor:
        noisy_np = noisy_chunk.squeeze(0).cpu().numpy()
        denoised_np = signal.wiener(noisy_np, mysize=5)
        return torch.from_numpy(denoised_np).unsqueeze(0).to(self.device)

class WaveletDenoisingWrapper:
    """ Denoising using Wavelet Transform. """
    NATIVE_SR = 16000

    def __init__(self, device):
        self.device = device
        print("[Model] Initializing Wavelet Denoising...")

    def process(self, noisy_chunk: torch.Tensor) -> torch.Tensor:
        noisy_np = noisy_chunk.squeeze(0).cpu().numpy()
        
        # Decompose to get wavelet coefficients
        coeffs = pywt.wavedec(noisy_np, 'db8', level=6)
        
        # Calculate threshold
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        threshold = sigma * np.sqrt(2 * np.log(len(noisy_np)))
        
        # Threshold coefficients
        coeffs_thresh = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]
        
        # Reconstruct signal
        denoised_np = pywt.waverec(coeffs_thresh, 'db8')
        
        # Ensure same length
        if len(denoised_np) > len(noisy_np):
            denoised_np = denoised_np[:len(noisy_np)]
        elif len(denoised_np) < len(noisy_np):
            denoised_np = np.pad(denoised_np, (0, len(noisy_np) - len(denoised_np)), 'constant')

        return torch.from_numpy(denoised_np).unsqueeze(0).to(self.device)

class HighPassFilterWrapper:
    """ Denoising using a High-Pass Filter. """
    NATIVE_SR = 16000
    CUTOFF_FREQ = 80 # Hz

    def __init__(self, device):
        self.device = device
        print(f"[Model] Initializing High-Pass Filter (cutoff: {self.CUTOFF_FREQ} Hz)...")
        # Design filter
        nyquist = 0.5 * self.NATIVE_SR
        norm_cutoff = self.CUTOFF_FREQ / nyquist
        self.b, self.a = signal.butter(5, norm_cutoff, btype='high', analog=False)

    def process(self, noisy_chunk: torch.Tensor) -> torch.Tensor:
        noisy_np = noisy_chunk.squeeze(0).cpu().numpy()
        denoised_np = signal.filtfilt(self.b, self.a, noisy_np)
        return torch.from_numpy(denoised_np.copy()).unsqueeze(0).to(self.device)

class NotchFilterWrapper:
    """ Denoising using a Notch Filter. """
    NATIVE_SR = 16000
    NOTCH_FREQ = 60 # Hz
    QUALITY_FACTOR = 30.0

    def __init__(self, device):
        self.device = device
        print(f"[Model] Initializing Notch Filter (freq: {self.NOTCH_FREQ} Hz)...")
        # Design filter
        self.b, self.a = signal.iirnotch(self.NOTCH_FREQ, self.QUALITY_FACTOR, self.NATIVE_SR)

    def process(self, noisy_chunk: torch.Tensor) -> torch.Tensor:
        noisy_np = noisy_chunk.squeeze(0).cpu().numpy()
        denoised_np = signal.lfilter(self.b, self.a, noisy_np)
        return torch.from_numpy(denoised_np.copy()).unsqueeze(0).to(self.device)


def load_enhancer(model_type: ModelOption, device):
    """Loads the specified enhancement model."""
    print(f"[Model] Loading {model_type}...")
    if model_type == "convtasnet":
        enhancer = ConvTasNetWrapper(device)
    elif model_type == "metricgan":
        enhancer = MetricGANWrapper(device)
    elif model_type == "deepfilternet":
        enhancer = DeepFilterNetWrapper(device)
    elif model_type == "spectral-gating" or model_type == "spectral-subtraction":
        if model_type == "spectral-subtraction":
            print("[Info] 'spectral-subtraction' is an alias for 'spectral-gating'.")
        enhancer = SpectralGatingWrapper(device)
    elif model_type == "wiener":
        enhancer = WienerFilterWrapper(device)
    elif model_type == "wavelet":
        enhancer = WaveletDenoisingWrapper(device)
    elif model_type == "high-pass":
        enhancer = HighPassFilterWrapper(device)
    elif model_type == "notch":
        enhancer = NotchFilterWrapper(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    return enhancer

def denoise_long_audio(enhancer, noisy_wav: torch.Tensor):
    """ Denoises a long audio file using a sliding window. """
    working_sr = enhancer.NATIVE_SR
    window_samples = int(WINDOW_SEC * working_sr)
    stride_samples = int(STRIDE_SEC * working_sr)
    num_samples_audio = noisy_wav.shape[1]

    output_buffer = torch.zeros_like(noisy_wav)
    
    current_start = 0
    with torch.no_grad():
        while current_start + window_samples <= num_samples_audio:
            chunk = noisy_wav[:, current_start : current_start + window_samples]
            enhanced_chunk = enhancer.process(chunk)
            
            if current_start == 0:
                output_buffer[:, 0:window_samples] = enhanced_chunk
            else:
                stride_idx = window_samples - stride_samples
                new_content = enhanced_chunk[:, stride_idx:]
                update_start = current_start + stride_idx
                update_end = current_start + window_samples
                output_buffer[:, update_start:update_end] = new_content
                
            current_start += stride_samples
    
        if current_start < num_samples_audio:
            last_chunk = noisy_wav[:, current_start:]
            if last_chunk.shape[1] < window_samples:
                padding = window_samples - last_chunk.shape[1]
                last_chunk = torch.nn.functional.pad(last_chunk, (0, padding))
            
            enhanced_chunk = enhancer.process(last_chunk)

            remaining_len = num_samples_audio - current_start
            stride_idx = window_samples - stride_samples
            
            if current_start > 0:
                update_start = current_start + stride_idx
                if update_start < num_samples_audio:
                    len_to_copy = num_samples_audio - update_start
                    output_buffer[:, update_start:] = enhanced_chunk[:, stride_idx:stride_idx + len_to_copy]
            else:
                output_buffer[:, current_start:] = enhanced_chunk[:, :remaining_len]

    return output_buffer

def main(input_path, output_directory, model_type: ModelOption, window_sec=WINDOW_SEC, stride_sec=STRIDE_SEC):
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / "denoised.wav"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")

    enhancer = load_enhancer(model_type, device)
    working_sr = enhancer.NATIVE_SR
    print(f"Working Sample Rate: {working_sr} Hz")

    print("Loading audio...")
    noisy_wav, sr = torchaudio.load(input_path)
    noisy_wav = prep_audio(noisy_wav, sr, working_sr).to(device)
    
    print(f"Denoising {noisy_wav.shape[1]/working_sr:.2f}s of audio...")
    output_buffer = denoise_long_audio(enhancer, noisy_wav)

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
import argparse
from collections import deque
from typing import Literal, Tuple
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import torchaudio
from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector

from asteroid.models import ConvTasNet

from speechbrain.inference.separation import SepformerSeparation as SepFormer

import diart.functional as diart_func
from diart.models import SegmentationModel


from own_voice_suppression.audio_utils import prep_audio, resample, torch_trusted_load

WAVLM_REQUIRED_SR = 16_000  
SMOOTHING_WINDOW = 5
DETECTION_THRESHOLD = 0.55
WINDOW_SEC = 2
STRIDE_SEC = 0.100

MODEL_OPTIONS = ["convtasnet", "sepformer", "diart"]
ModelOption = Literal["convtasnet", "sepformer", "diart"]

def resample(audio: torch.Tensor, orig_sr: int, new_sr: int) -> torch.Tensor:
    if orig_sr == new_sr:
        return audio
    resampler = torchaudio.transforms.Resample(orig_sr, new_sr).to(audio.device)
    return resampler(audio)


class WavLMVerifier:
    MODEL_ID = "microsoft/wavlm-base-plus-sv"
    
    def __init__(self, device):
        self.device = device
        print(f"[Verifier] Loading {self.MODEL_ID}...")
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(self.MODEL_ID)
        self.model = WavLMForXVector.from_pretrained(self.MODEL_ID).to(device)
        self.model.eval()

    def get_embedding(self, wav_tensor: torch.Tensor, input_sr: int) -> torch.Tensor:
        """
        Extracts embedding. 
        Auto-upsamples to 16k if the input (from separation model) is 8k.
        """
        if wav_tensor.shape[1] < 100: 
            raise ValueError("Input audio too short for embedding extraction, must have at least 100 samples.")

        if input_sr != WAVLM_REQUIRED_SR:
            wav_tensor = resample(wav_tensor, input_sr, WAVLM_REQUIRED_SR)

        wav_np = wav_tensor.squeeze(0).cpu().numpy()
        inputs = self.processor(wav_np, sampling_rate=WAVLM_REQUIRED_SR, return_tensors="pt", padding=True).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            emb = F.normalize(outputs.embeddings, p=2, dim=1)
        return emb

    def score(self, chunk: torch.Tensor, target_emb: torch.Tensor, input_sr: int) -> float:
        chunk_emb = self.get_embedding(chunk, input_sr)
        return F.cosine_similarity(chunk_emb, target_emb).item()


class AsteroidConvTasNetWrapper:
    """ Native Rate: 16000 Hz """
    NATIVE_SR = 16000
    
    def __init__(self, device):
        print("[Model] Loading ConvTasNet (16k causal)...")
        torch.load = torch_trusted_load
        self.model = ConvTasNet.from_pretrained("JorisCos/ConvTasNet_Libri2Mix_sepnoisy_16k")
        self.model.to(device)
    
    def process(self, mix_8k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            est_sources = self.model(mix_8k) # Returns (Batch, Sources, Time) at 8k
        return est_sources[:, 0, :], est_sources[:, 1, :]


class SpeechBrainSepFormerWrapper:
    """ Native Rate: 8000 Hz """
    NATIVE_SR = 8000

    def __init__(self, device):
        print("[Model] Loading SepFormer (wsj02mix)...")
        self.model = SepFormer.from_hparams(
            source="speechbrain/sepformer-wsj02mix", 
            savedir="pretrained_models/sepformer",
            run_opts={"device": str(device)}
        )

    def process(self, mix_8k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            est_sources = self.model.separate_batch(mix_8k) 
        return est_sources[:, :, 0], est_sources[:, :, 1]


class DiartWrapper:
    """ Native Rate: 16000 Hz (Standard for Diarization) """
    NATIVE_SR = 16000

    def __init__(self, device):
        if diart_func is None: raise ImportError("Diart not installed.")
        print("[Model] Loading Diart Streaming Pipeline...")
        torch.load = torch_trusted_load
        self.segmentation = SegmentationModel.from_pretrained("pyannote/segmentation-3.0")
        self.segmentation.to(device)
        
    def process(self, mix_16k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            segmentation = self.segmentation(mix_16k.unsqueeze(1)) 
        
        probs = segmentation[0] 
        num_speakers = probs.shape[1]
        
        # Interpolate masks to match audio length
        probs_t = probs.T.unsqueeze(0)
        probs_interp = F.interpolate(probs_t, size=mix_16k.shape[1], mode='linear', align_corners=False)
        
        s1_mask = (probs_interp[0, 0, :] > 0.5).float()
        s2_mask = (probs_interp[0, 1, :] > 0.5).float() if num_speakers > 1 else torch.zeros_like(s1_mask)

        return mix_16k * s1_mask, mix_16k * s2_mask


def main(enrolment_path, mixed_path, output_directory, model_type: ModelOption, suppress: bool, window_sec=WINDOW_SEC, stride_sec=STRIDE_SEC):
    
    output_directory.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")

    if model_type == "convtasnet":
        extractor = AsteroidConvTasNetWrapper(device)
    elif model_type == "sepformer":
        extractor = SpeechBrainSepFormerWrapper(device)
    elif model_type == "diart":
        extractor = DiartWrapper(device)
    else:
        raise ValueError("Unknown model type")
    
    working_sr = extractor.NATIVE_SR
    print(f"Working Sample Rate: {working_sr} Hz")

    verifier = WavLMVerifier(device)

    print("Prepping audio...")
    enroll_wav, sr = torchaudio.load(enrolment_path)

    # Enrollment needs to be 16k for WavLM
    enroll_wav = prep_audio(enroll_wav, sr, WAVLM_REQUIRED_SR).to(device)
    
    mixed_wav, sr = torchaudio.load(mixed_path)
    mixed_wav = prep_audio(mixed_wav, sr, working_sr).to(device)

    target_emb = verifier.get_embedding(enroll_wav, input_sr=WAVLM_REQUIRED_SR)

    window_samples = int(window_sec * working_sr)
    stride_samples = int(stride_sec * working_sr)
    num_samples = mixed_wav.shape[1]

    if suppress:
        output_buffer = torch.zeros_like(mixed_wav)
    else:
        output_buffer_s1 = torch.zeros_like(mixed_wav)
        output_buffer_s2 = torch.zeros_like(mixed_wav)
    
    log = []
    
    # Avoid permutation flips in separation mode
    anchor_emb_s1 = None
    anchor_emb_s2 = None
    ALPHA = 0.95 # Smoothing factor for anchor embedding updates
    
    print(f"Processing {num_samples/working_sr:.2f}s...")

    current_start = 0
    while current_start + window_samples <= num_samples:
        
        chunk = mixed_wav[:, current_start : current_start + window_samples]
        
        s1, s2 = extractor.process(chunk)
        
        if suppress: # Suppression mode
            score_s1 = verifier.score(s1, target_emb, input_sr=working_sr)
            score_s2 = verifier.score(s2, target_emb, input_sr=working_sr)
            
            target_detected = False
            kept_audio = chunk 
            
            if score_s1 > DETECTION_THRESHOLD and score_s1 > score_s2:
                kept_audio = s2 
                target_detected = True
            elif score_s2 > DETECTION_THRESHOLD and score_s2 > score_s1:
                kept_audio = s1 
                target_detected = True
            
            log.append({"time": current_start / working_sr, "detected": int(target_detected)})
            
            if current_start == 0:
                output_buffer[:, 0 : window_samples] = kept_audio
            else:
                stride_start_idx = window_samples - stride_samples
                update_start = current_start + stride_start_idx
                update_end = current_start + window_samples
                
                new_content = kept_audio[:, stride_start_idx:]
                output_buffer[:, update_start : update_end] = new_content
        
        else: # Separation mode (with speaker tracking)
            if anchor_emb_s1 is None:
                anchor_emb_s1 = verifier.get_embedding(s1, input_sr=working_sr)
                anchor_emb_s2 = verifier.get_embedding(s2, input_sr=working_sr)
                final_s1, final_s2 = s1, s2
            else:
                emb1 = verifier.get_embedding(s1, input_sr=working_sr)
                emb2 = verifier.get_embedding(s2, input_sr=working_sr)
                
                # Check permutation against anchors
                score_perm_orig = F.cosine_similarity(emb1, anchor_emb_s1) + F.cosine_similarity(emb2, anchor_emb_s2)
                score_perm_flipped = F.cosine_similarity(emb1, anchor_emb_s2) + F.cosine_similarity(emb2, anchor_emb_s1)

                if score_perm_orig >= score_perm_flipped:
                    final_s1, final_s2 = s1, s2
                    anchor_emb_s1 = ALPHA * anchor_emb_s1 + (1 - ALPHA) * emb1
                    anchor_emb_s2 = ALPHA * anchor_emb_s2 + (1 - ALPHA) * emb2
                else:
                    final_s1, final_s2 = s2, s1
                    anchor_emb_s1 = ALPHA * anchor_emb_s1 + (1 - ALPHA) * emb2
                    anchor_emb_s2 = ALPHA * anchor_emb_s2 + (1 - ALPHA) * emb1

            s1_rms = torch.sqrt(torch.mean(final_s1**2)).item()
            s2_rms = torch.sqrt(torch.mean(final_s2**2)).item()
            log.append({"time": current_start / working_sr, "s1_rms": s1_rms, "s2_rms": s2_rms})
            
            # Overlap-add for the consistently ordered sources
            if current_start == 0:
                output_buffer_s1[:, 0 : window_samples] = final_s1
                output_buffer_s2[:, 0 : window_samples] = final_s2
            else:
                stride_start_idx = window_samples - stride_samples
                update_start = current_start + stride_start_idx
                update_end = current_start + window_samples
                
                output_buffer_s1[:, update_start : update_end] = final_s1[:, stride_start_idx:]
                output_buffer_s2[:, update_start : update_end] = final_s2[:, stride_start_idx:]

        current_start += stride_samples

    if suppress:
        output_path = output_directory / f"suppressed_{model_type}.wav"
        torchaudio.save(output_path, output_buffer.cpu(), working_sr)
        print(f"Saved suppressed audio to {output_path} @ {working_sr}Hz")
        
        times = [d["time"] for d in log]
        dets = [d["detected"] for d in log]
        
        plt.figure(figsize=(10, 4))
        plt.fill_between(times, dets, step="pre", alpha=0.4, color='red', label="Target Detected")
        plt.plot(times, dets, drawstyle="steps", color='red')
        plt.ylim(-0.1, 1.1)
        plt.title(f"Target Presence ({model_type})")
        plt.xlabel("Time (s)")
        plt.yticks([0, 1], ["Absent", "Present"])
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_directory / f"presence_{model_type}.png")

    else: # Separation mode
        output_path_s1 = output_directory / f"separated_{model_type}_source1.wav"
        output_path_s2 = output_directory / f"separated_{model_type}_source2.wav"
        torchaudio.save(output_path_s1, output_buffer_s1.cpu(), working_sr)
        torchaudio.save(output_path_s2, output_buffer_s2.cpu(), working_sr)
        print(f"Saved separated sources to {output_path_s1} and {output_path_s2} @ {working_sr}Hz")
        
        times = [d["time"] for d in log]
        s1_rms = [d["s1_rms"] for d in log]
        s2_rms = [d["s2_rms"] for d in log]

        plt.figure(figsize=(10, 4))
        plt.plot(times, s1_rms, label="Source 1 Amplitude")
        plt.plot(times, s2_rms, label="Source 2 Amplitude")
        plt.title(f"Source Amplitudes ({model_type})")
        plt.xlabel("Time (s)")
        plt.ylabel("RMS Amplitude")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(output_directory / f"amplitudes_{model_type}.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--enrolment-path", type=Path, required=True)
    parser.add_argument("--mixed-path", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--model-type", type=str, choices=MODEL_OPTIONS, default="convtasnet")
    parser.add_argument("--suppress", action="store_true", help="Enable suppression of the target speaker. If not set, separates all sources.")

    args = parser.parse_args()

    if not args.output_directory:
        args.output_directory = Path("own_voice_suppression/outputs") / \
            Path("suppression" if args.suppress else "separation") / \
                (f"{args.model_type}_enrol_{args.enrolment_path.stem}_mix_{args.mixed_path.stem}")

    main(args.enrolment_path, args.mixed_path, args.output_directory, args.model_type, args.suppress)
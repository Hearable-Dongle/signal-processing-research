import argparse
from typing import Literal, Tuple
from pathlib import Path
from collections import deque

import torch
import torch.nn.functional as F
import torchaudio
from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector

from asteroid.models import ConvTasNet
from speechbrain.inference.separation import SepformerSeparation as SepFormer
import diart.functional as diart_func
from diart.models import SegmentationModel

from own_voice_suppression.plot_utils import plot_source_amplitudes, plot_target_presence
from own_voice_suppression.audio_utils import prep_audio, resample, torch_trusted_load

WAVLM_REQUIRED_SR = 16_000  
DETECTION_THRESHOLD = 0.55
WINDOW_SEC = 2.0
STRIDE_SEC = 0.5

MODEL_OPTIONS = ["convtasnet", "sepformer", "diart"]
ModelOption = Literal["convtasnet", "sepformer", "diart"]


class WavLMVerifier:
    MODEL_ID = "microsoft/wavlm-base-plus-sv"
    
    def __init__(self, device):
        self.device = device
        print(f"[Verifier] Loading {self.MODEL_ID}...")
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(self.MODEL_ID)
        self.model = WavLMForXVector.from_pretrained(self.MODEL_ID).to(device)
        self.model.eval()

    def get_embedding(self, wav_tensor: torch.Tensor, input_sr: int) -> torch.Tensor:
        if wav_tensor.shape[1] < 400:
             return torch.zeros(1, 512).to(self.device)
        if input_sr != WAVLM_REQUIRED_SR:
            wav_tensor = resample(wav_tensor, input_sr, WAVLM_REQUIRED_SR)
        wav_np = wav_tensor.squeeze(0).cpu().numpy()
        inputs = self.processor(wav_np, sampling_rate=WAVLM_REQUIRED_SR, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return F.normalize(outputs.embeddings, p=2, dim=1)

    def score(self, chunk: torch.Tensor, target_emb: torch.Tensor, input_sr: int) -> float:
        chunk_emb = self.get_embedding(chunk, input_sr)
        return F.cosine_similarity(chunk_emb, target_emb).item()

class AsteroidConvTasNetWrapper:
    NATIVE_SR = 16000
    def __init__(self, device):
        print("[Model] Loading ConvTasNet (16k causal)...")
        torch.load = torch_trusted_load
        self.model = ConvTasNet.from_pretrained("JorisCos/ConvTasNet_Libri2Mix_sepnoisy_16k").to(device)
        self.model.eval()
    
    def process(self, mix_chunk: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            pad_samples = int(0.5 * self.NATIVE_SR)
            padded_input = F.pad(mix_chunk.unsqueeze(1), (pad_samples, pad_samples), mode='reflect').squeeze(1)
            est_sources_padded = self.model(padded_input)
            est_sources = est_sources_padded[:, :, pad_samples:-pad_samples]
            s1, s2 = est_sources[:, 0, :], est_sources[:, 1, :]
            separated_mix = s1 + s2
            input_rms = torch.sqrt(torch.mean(mix_chunk ** 2, dim=-1, keepdim=True))
            separated_mix_rms = torch.sqrt(torch.mean(separated_mix ** 2, dim=-1, keepdim=True)) + 1e-8
            scaling_factor = input_rms / separated_mix_rms
        return s1 * scaling_factor, s2 * scaling_factor

class SpeechBrainSepFormerWrapper:
    NATIVE_SR = 8000
    def __init__(self, device):
        print("[Model] Loading SepFormer (wsj02mix)...")
        self.model = SepFormer.from_hparams("speechbrain/sepformer-wsj02mix", savedir="pretrained_models/sepformer", run_opts={"device": str(device)})

    def process(self, mix_chunk: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            pad_samples = int(0.5 * self.NATIVE_SR)
            padded_input = F.pad(mix_chunk.unsqueeze(1), (pad_samples, pad_samples), mode='reflect').squeeze(1)
            est_sources_padded = self.model.separate_batch(padded_input)
            est_sources = est_sources_padded[:, pad_samples:-pad_samples, :]
            s1, s2 = est_sources[:, :, 0], est_sources[:, :, 1]
            separated_mix = s1 + s2
            input_rms = torch.sqrt(torch.mean(mix_chunk ** 2, dim=-1, keepdim=True))
            separated_mix_rms = torch.sqrt(torch.mean(separated_mix ** 2, dim=-1, keepdim=True)) + 1e-8
            scaling_factor = input_rms / separated_mix_rms
        return s1 * scaling_factor, s2 * scaling_factor

class DiartWrapper:
    NATIVE_SR = 16000
    def __init__(self, device):
        if diart_func is None: raise ImportError("Diart not installed.")
        print("[Model] Loading Diart Streaming Pipeline...")
        torch.load = torch_trusted_load
        self.segmentation = SegmentationModel.from_pretrained("pyannote/segmentation-3.0").to(device)
        
    def process(self, mix_16k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            segmentation = self.segmentation(mix_16k.unsqueeze(1))
        probs = segmentation[0]
        num_speakers = probs.shape[1]
        probs_t = probs.T.unsqueeze(0)
        probs_interp = F.interpolate(probs_t, size=mix_16k.shape[1], mode='linear', align_corners=False)
        s1_mask = (probs_interp[0, 0, :] > 0.5).float()
        s2_mask = (probs_interp[0, 1, :] > 0.5).float() if num_speakers > 1 else torch.zeros_like(s1_mask)
        return mix_16k * s1_mask, mix_16k * s2_mask


def run_separation_pipeline(
    mixed_audio: torch.Tensor,
    orig_sr_mix: int,
    enrolment_audio: torch.Tensor,
    orig_sr_enrol: int,
    model_type: ModelOption,
    device: torch.device,
    suppress: bool,
    detection_threshold: float = DETECTION_THRESHOLD,
    window_sec: float = WINDOW_SEC,
    stride_sec: float = STRIDE_SEC,
    smoothing_window: int = 10
) -> Tuple[Tuple[torch.Tensor, ...], list, int]:
    """
    Runs the full source separation and suppression/tracking pipeline on audio tensors.
    """

    if model_type == "convtasnet":
        extractor = AsteroidConvTasNetWrapper(device)

    elif model_type == "sepformer":
        extractor = SpeechBrainSepFormerWrapper(device)

    elif model_type == "diart":
        extractor = DiartWrapper(device)

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    verifier = WavLMVerifier(device)
    working_sr = extractor.NATIVE_SR

    enrolment_for_verifier = prep_audio(enrolment_audio, orig_sr_enrol, WAVLM_REQUIRED_SR).to(device)
    mixed_audio = prep_audio(mixed_audio, orig_sr_mix, working_sr).to(device)

    target_emb = verifier.get_embedding(enrolment_for_verifier, input_sr=WAVLM_REQUIRED_SR)

    window_samples = int(window_sec * working_sr)
    stride_samples = int(stride_sec * working_sr)

    num_samples = mixed_audio.shape[1]

    output_buffer = torch.zeros_like(mixed_audio)
    output_buffer_s1 = torch.zeros_like(mixed_audio) if not suppress else None
    output_buffer_s2 = torch.zeros_like(mixed_audio) if not suppress else None

    log = []

    anchor_emb_s1, anchor_emb_s2 = None, None
    score_buffer = deque(maxlen=smoothing_window)
    ALPHA = 0.95

    current_start = 0

    while current_start + window_samples <= num_samples:
        chunk = mixed_audio[:, current_start : current_start + window_samples]
        s1, s2 = extractor.process(chunk)

        if suppress:
            score_s1 = verifier.score(s1, target_emb, input_sr=working_sr)
            score_s2 = verifier.score(s2, target_emb, input_sr=working_sr)
            
            raw_score = max(score_s1, score_s2)
            score_buffer.append(raw_score)
            smoothed_score = sum(score_buffer) / len(score_buffer)

            target_detected = False
            kept_audio = chunk 

            if smoothed_score >= detection_threshold:
                if score_s1 > score_s2:
                    kept_audio = s2 
                    target_detected = True
                else:
                    kept_audio = s1
                    target_detected = True

            log.append({
                "time": current_start / working_sr, 
                "score": raw_score,
                "smoothed_score": smoothed_score,
                "detected": int(target_detected)
            })

            if current_start == 0:
                output_buffer[:, 0:window_samples] = kept_audio

            else:
                fade_len = window_samples - stride_samples
                fade_in = torch.linspace(0, 1, fade_len).to(device)
                fade_out = torch.linspace(1, 0, fade_len).to(device)
                output_buffer[:, current_start : current_start + fade_len] *= fade_out
                output_buffer[:, current_start : current_start + fade_len] += kept_audio[:, :fade_len] * fade_in
                output_buffer[:, current_start + fade_len : current_start + window_samples] = kept_audio[:, fade_len:]

        else: # Separation mode
            if anchor_emb_s1 is None:
                anchor_emb_s1 = verifier.get_embedding(s1, input_sr=working_sr)
                anchor_emb_s2 = verifier.get_embedding(s2, input_sr=working_sr)
            else:
                emb1 = verifier.get_embedding(s1, input_sr=working_sr)
                emb2 = verifier.get_embedding(s2, input_sr=working_sr)

                score_perm_orig = F.cosine_similarity(emb1, anchor_emb_s1) + F.cosine_similarity(emb2, anchor_emb_s2)
                score_perm_flipped = F.cosine_similarity(emb1, anchor_emb_s2) + F.cosine_similarity(emb2, anchor_emb_s1)

                if score_perm_orig < score_perm_flipped:
                    s1, s2 = s2, s1
                    emb1, emb2 = emb2, emb1

                anchor_emb_s1 = ALPHA * anchor_emb_s1 + (1 - ALPHA) * emb1
                anchor_emb_s2 = ALPHA * anchor_emb_s2 + (1 - ALPHA) * emb2

            log.append({
                "time": current_start / working_sr,
                "s1_rms": torch.sqrt(torch.mean(s1**2)).item(),
                "s2_rms": torch.sqrt(torch.mean(s2**2)).item()
            })

            if current_start == 0:
                output_buffer_s1[:, :window_samples] = s1
                output_buffer_s2[:, :window_samples] = s2

            else:
                # Assuming simple overlap-add for separation mode for now
                stride_start_idx = window_samples - stride_samples
                output_buffer_s1[:, current_start + stride_start_idx : current_start + window_samples] = s1[:, stride_start_idx:]
                output_buffer_s2[:, current_start + stride_start_idx : current_start + window_samples] = s2[:, stride_start_idx:]

        current_start += stride_samples
    
    if suppress:
        return (output_buffer,), log, working_sr

    else:
        return (output_buffer_s1, output_buffer_s2), log, working_sr

def main():
    parser = argparse.ArgumentParser(description="""

        Performs speaker separation or suppression on an audio file.

        In 'separation' mode (default), it separates the mixed audio into two tracks.

        In 'suppression' mode (`--suppress`), it removes the enrolled target speaker.

    """)
    parser.add_argument("--enrolment-path", type=Path, required=True, help="Path to the enrolment audio file for the target speaker.")
    parser.add_argument("--mixed-path", type=Path, required=True, help="Path to the mixed audio file to be processed.")
    parser.add_argument("--output-directory", type=Path, help="Directory to save the output audio files and plots.")
    parser.add_argument("--model-type", type=str, choices=MODEL_OPTIONS, default="convtasnet", help="Which separation model to use.")
    parser.add_argument("--suppress", action="store_true", help="Enable suppression of the target speaker. If not set, separates all sources.")

    args = parser.parse_args()

    if not args.output_directory:
        mode = "suppression" if args.suppress else "separation"
        args.output_directory = Path(f"own_voice_suppression/outputs/{mode}/{args.model_type}_enrol_{args.enrolment_path.stem}_mix_{args.mixed_path.stem}")

    args.output_directory.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")

    enrolment_audio, sr_enrol = torchaudio.load(args.enrolment_path)
    mixed_audio, sr_mix = torchaudio.load(args.mixed_path)
    print(f"Processing {mixed_audio.shape[1] / WAVLM_REQUIRED_SR:.2f}s of audio...")

    output_audios, log, working_sr = run_separation_pipeline(
        mixed_audio=mixed_audio,
        orig_sr_mix=sr_mix,
        enrolment_audio=enrolment_audio,
        orig_sr_enrol=sr_enrol,
        model_type=args.model_type,
        device=device,
        suppress=args.suppress,
    )

    if args.suppress:
        output_path = args.output_directory / f"suppressed_{args.model_type}.wav"
        torchaudio.save(output_path, output_audios[0].cpu(), working_sr)
        print(f"\nSaved suppressed audio to {output_path} @ {working_sr}Hz")

        if log:
            plot_path = args.output_directory / f"presence_{args.model_type}.png"
            plot_target_presence(log, plot_path, args.model_type)

    else:
        output_path_s1 = args.output_directory / f"separated_{args.model_type}_source1.wav"
        output_path_s2 = args.output_directory / f"separated_{args.model_type}_source2.wav"
        torchaudio.save(output_path_s1, output_audios[0].cpu(), working_sr)
        torchaudio.save(output_path_s2, output_audios[1].cpu(), working_sr)
        print(f"\nSaved separated sources to {output_path_s1} and {output_path_s2} @ {working_sr}Hz")

        if log:
            plot_path = args.output_directory / f"amplitudes_{args.model_type}.png"
            plot_source_amplitudes(log, plot_path, args.model_type)


if __name__ == "__main__":
    main()

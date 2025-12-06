import argparse
from collections import deque
import os
from typing import Literal
from os import PathLike
from pathlib import Path

import numpy as np
from pyannote.audio import Model, Pipeline
from speechbrain.inference.speaker import EncoderClassifier
from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector

from torch import Tensor
import torch
import torch.nn.functional as F
import torchaudio
import matplotlib.pyplot as plt

from general_utils.resample_audio import resample

TARGET_SR = 16_000
SMOOTHING_WINDOW = 10
SPEAKER_DETECTION_THRESHOLD = 0.65

DEFAULT_CLASSIFIER = "wavlm-large"
CLASSIFIER_OPTIONS = ["ecapa-voxceleb", "wavlm-large", "xvect-voxceleb", "pyannote-diarization"]

ClassifierOption = Literal["ecapa-voxceleb", "wavlm-large", "xvect-voxceleb", "pyannote-diarization"]


class PyannoteDiarizationWrapper:
    """
    A wrapper around the pyannote.audio speaker embedding model.
    This uses the same embedding model (`pyannote/speaker-embedding`) as the
    `pyannote/speaker-diarization-3.1` pipeline.
    Requires a Hugging Face token that has accepted the user agreements for the models.
    """
    MODEL_ID = "pyannote/embedding"
    # MODEL_ID = "pyannote/speaker-diarization-3.1" # Use with pipeline
    
    def __init__(self, device=torch.device("cpu")):
        self.device = device
        print(f"Loading {self.MODEL_ID} on {device}...")
        
        # Pyannote models use a HF token for gated models.
        hf_token = os.environ.get("HUGGING_FACE_TOKEN")
        if hf_token is None:
            print("Warning: Hugging Face token not found. This might fail if the model is gated.")
        
        # Hack: Override torch.load to disable weights_only loading
        # See https://github.com/m-bain/whisperX/issues/1304
        _original_torch_load = torch.load

        def _trusted_load(*args, **kwargs):
            kwargs['weights_only'] = False
            return _original_torch_load(*args, **kwargs)

        torch.load = _trusted_load
        
        # self.embedding_model = Pipeline.from_pretrained(
        #     self.MODEL_ID,
        # ).to(device)
        self.embedding_model = Model.from_pretrained(
            self.MODEL_ID,
        ).to(device)
        print("SELF> EMBEDDING_MODEL", self.embedding_model)
        self.embedding_model.eval()

    def encode_batch(self, wavs: torch.Tensor) -> torch.Tensor:
        """
        Mimics SpeechBrain's encode_batch.
        Input: (Batch, Time)
        Output: (Batch, 1, Embedding_Dim)
        """
        if wavs.dim() == 1: # Add back batch dimension
            wavs = wavs.unsqueeze(0) 
        
        all_embeddings = []
        with torch.no_grad():
            for i in range(wavs.shape[0]):
                waveform_slice = wavs[i:i+1] # Keep 2D for pyannote
                
                file = {"waveform": waveform_slice.cpu(), "sample_rate": TARGET_SR}
                
                embedding_np = self.embedding_model(file)
                
                # Average the embeddings if multiple chunks are returned 
                if embedding_np.ndim > 1 and embedding_np.shape[0] > 1:
                    embedding_np = np.mean(embedding_np, axis=0, keepdims=True)
                
                embedding_torch = torch.from_numpy(embedding_np).to(self.device)
                all_embeddings.append(embedding_torch)

        embeddings = torch.cat(all_embeddings, dim=0) # (B, emb_dim)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings.unsqueeze(1)


class WavLMClassifierWrapper:
    MODEL_ID = "microsoft/wavlm-base-plus-sv"
    
    def __init__(self, device=torch.device("cpu")):
        self.device = device
        print(f"Loading {self.MODEL_ID} on {device}...")
        
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(self.MODEL_ID)
        
        self.model = WavLMForXVector.from_pretrained(self.MODEL_ID).to(self.device)
        self.model.eval()

    def encode_batch(self, wavs: torch.Tensor) -> torch.Tensor:
        """
        Mimics SpeechBrain's encode_batch.
        Input: (Batch, Time)
        Output: (Batch, 1, Embedding_Dim)
        """
        if wavs.dim() > 1:
            wavs_np = wavs.squeeze(0).cpu().numpy() if wavs.shape[0] == 1 else wavs.cpu().numpy()
            if wavs_np.ndim > 1: wavs_np = wavs_np[0] # Handle (1, T) case
        else:
            wavs_np = wavs.cpu().numpy()

        inputs = self.processor(
            wavs_np, 
            sampling_rate=TARGET_SR, 
            return_tensors="pt", 
            padding=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            # Returns embeddings directly
            embeddings = outputs.embeddings
            embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings.unsqueeze(1)

def prep_audio(audio: Tensor, orig_sr: int, target_sr: int = TARGET_SR) -> Tensor:
    """
    Ensures mono channel and resamples to target sample rate.
    """
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)
    if orig_sr != target_sr:
        audio = resample(
            audio, 
            orig_sr=orig_sr, 
            new_sr=target_sr
        )
    return audio

def get_target_speaker_confidence(live_chunk: Tensor, target_embedding: Tensor, classifier: EncoderClassifier):
    """
    live_chunk: 100ms audio tensor (1, 1600)
    target_embedding: Pre-computed embedding of the user (1, 1, 192)
    """
    
    live_embedding = classifier.encode_batch(live_chunk)
    
    score = F.cosine_similarity(live_embedding.squeeze(1), target_embedding.squeeze(1))
    
    return score.item()


def main(enrolment_path: PathLike, 
         mixed_path: PathLike, 
         output_directory: PathLike, 
         classifier_type: ClassifierOption = DEFAULT_CLASSIFIER, 
         smoothing_window: int = SMOOTHING_WINDOW,
         speaker_detection_threshold: int = SPEAKER_DETECTION_THRESHOLD,
         ):
    
    output_path = Path(output_directory) / f"suppressed.wav"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier_device = device
    
    # Workaround for xvect-voxceleb: it seems to have a bug causing tensors to be on CPU
    # during GPU execution. We force it to CPU.
    if classifier_type == "xvect-voxceleb":
        print("INFO: Using CPU for xvect-voxceleb due to a device compatibility issue.")
        classifier_device = torch.device("cpu")

    if classifier_type == "ecapa-voxceleb":
        classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb", 
            savedir="pretrained_models/spkrec-ecapa-voxceleb"
        ).to(classifier_device)
    elif classifier_type == "xvect-voxceleb":
        classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-xvect-voxceleb",
            savedir="pretrained_models/spkrec-xvect-voxceleb"
        ).to(classifier_device)
    elif classifier_type == "wavlm-large":
        classifier = WavLMClassifierWrapper(device=classifier_device)
    elif classifier_type == "pyannote-diarization":
        classifier = PyannoteDiarizationWrapper(device=classifier_device)
    else:
        raise ValueError(f"Unsupported classifier type: {classifier_type}")
        
    enrolment_audio, sr_enroll = torchaudio.load(enrolment_path)
    enrolment_audio = prep_audio(enrolment_audio, sr_enroll, TARGET_SR).to(classifier_device)

    mixed_audio, sr_mix = torchaudio.load(mixed_path)
    mixed_audio = prep_audio(mixed_audio, sr_mix, TARGET_SR).to(classifier_device)
    
    
    with torch.no_grad():
        target_embedding = classifier.encode_batch(enrolment_audio).to(classifier_device)

    window_sec = 0.600
    stride_sec = 0.100
    
    window_samples = int(window_sec * TARGET_SR)
    stride_samples = int(stride_sec * TARGET_SR)
    num_samples = mixed_audio.shape[1]

    # Create a mask to track detections (1.0 = Speaker Detected, 0.0 = Not Detected)
    detection_mask = torch.zeros_like(mixed_audio)
    confidence_logs = []

    print(f"Processing {num_samples / TARGET_SR:.2f}s of audio with {window_sec}s context...")
    
    current_start = 0
    score_buffer = deque(maxlen=smoothing_window)
    
    while current_start + window_samples <= num_samples:
        
        chunk = mixed_audio[:, current_start : current_start + window_samples]
        
        with torch.no_grad():
            score = get_target_speaker_confidence(
                live_chunk=chunk, 
                target_embedding=target_embedding, 
                classifier=classifier, 
                )
        
        score_buffer.append(score)
        smoothed_score = sum(score_buffer) / len(score_buffer)
        
        # Use smoothed score for detection decision
        is_present = smoothed_score >= speaker_detection_threshold

        current_time_sec = current_start / TARGET_SR
        confidence_logs.append({
            "time": current_time_sec, 
            "score": score, 
            "smoothed_score": smoothed_score,
            "detected": int(is_present)}
            )
        
        if is_present:
            if current_start == 0:
                detection_mask[:, 0 : window_samples] = 1.0
            else:
                update_start = current_start + window_samples - stride_samples
                update_end = current_start + window_samples
                detection_mask[:, update_start : update_end] = 1.0

        current_start += stride_samples

    output_audio = (mixed_audio * (1 - detection_mask)).cpu()

    time_steps = [log["time"] for log in confidence_logs]
    scores = [log["score"] for log in confidence_logs]
    smoothed_scores = [log["smoothed_score"] for log in confidence_logs]

    plt.plot(time_steps, scores, label="Raw Confidence Score")
    plt.plot(time_steps, smoothed_scores, label=f"Smoothed Confidence Score (window={smoothing_window})")
    plt.axhline(y=speaker_detection_threshold, color='r', linestyle='--', label="Detection Threshold")
    plt.xlabel("Time (s)")
    plt.ylabel("Confidence Score")
    plt.title("Speaker Detection Confidence Over Time")
    plt.legend(loc="lower right")
    plt.savefig(output_directory / f"confidence_plot.png")

    print(f"Saving to {output_path}")
    torchaudio.save(output_path, output_audio, TARGET_SR)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--enrolment-path",
        type=Path,
        required=True,
        help="Path to the enrolment audio file.",
    )
    parser.add_argument(
        "--mixed-path",
        type=Path,
        required=True,
        help="Path to the mixed audio file.",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        help="Directory to save the output audio file.",
    )
    parser.add_argument(
        "--classifier-type",
        type=str,
        choices=CLASSIFIER_OPTIONS,
        default=DEFAULT_CLASSIFIER,
        help="Type of speaker classifier to use.",
    )
    
    args = parser.parse_args()
    
    if not args.output_directory:
        args.output_directory = Path("own_voice_suppression") / "outputs" / f"suppressed_using_{args.classifier_type}_{args.mixed_path.stem}_from_{args.enrolment_path.stem}"
    
    args.output_directory.mkdir(parents=True, exist_ok=True)

    main(
        enrolment_path=args.enrolment_path, 
        mixed_path=args.mixed_path, 
        output_directory=args.output_directory,
        classifier_type=args.classifier_type,
    )
    
    

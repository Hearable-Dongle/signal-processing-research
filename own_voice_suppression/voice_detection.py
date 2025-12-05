import argparse
from os import PathLike
from pathlib import Path
from speechbrain.inference.speaker import EncoderClassifier
from torch import Tensor
import torch
import torch.nn.functional as F
import torchaudio

from general_utils.resample_audio import resample

TARGET_SR = 16_000


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

def is_target_speaker(live_chunk: Tensor, target_embedding: Tensor, classifier: EncoderClassifier, threshold=0.25):
    """
    live_chunk: 100ms audio tensor (1, 1600)
    target_embedding: Pre-computed embedding of the user (1, 1, 192)
    """
    
    live_embedding = classifier.encode_batch(live_chunk)
    
    score = F.cosine_similarity(live_embedding.squeeze(1), target_embedding.squeeze(1))
    
    return score.item() > threshold, score.item()


def main(enrolment_path: PathLike, mixed_path: PathLike, output_path: PathLike):
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb", 
        savedir="pretrained_models/spkrec-ecapa-voxceleb"
    )
    enrolment_audio, sr_enroll = torchaudio.load(enrolment_path)
    enrolment_audio = prep_audio(enrolment_audio, sr_enroll, TARGET_SR)

    mixed_audio, sr_mix = torchaudio.load(mixed_path)
    mixed_audio = prep_audio(mixed_audio, sr_mix, TARGET_SR)
    
    print("Works so far")
    
    with torch.no_grad():
        target_embedding = classifier.encode_batch(enrolment_audio)

    window_sec = 0.600
    stride_sec = 0.100
    
    window_samples = int(window_sec * TARGET_SR)
    stride_samples = int(stride_sec * TARGET_SR)
    num_samples = mixed_audio.shape[1]

    # Create a mask to track detections (1.0 = Speaker Detected, 0.0 = Not Detected)
    detection_mask = torch.zeros_like(mixed_audio)

    print(f"Processing {num_samples / TARGET_SR:.2f}s of audio with {window_sec}s context...")
    
    current_start = 0
    
    while current_start + window_samples <= num_samples:
        
        # 600ms context buffer
        chunk = mixed_audio[:, current_start : current_start + window_samples]
        
        with torch.no_grad():
            is_present, score = is_target_speaker(chunk, target_embedding, classifier, threshold=0.25)
        
        if is_present:
            if current_start == 0:
                detection_mask[:, 0 : window_samples] = 1.0
            else:
                update_start = current_start + window_samples - stride_samples
                update_end = current_start + window_samples
                detection_mask[:, update_start : update_end] = 1.0

        current_start += stride_samples

    output_audio = mixed_audio * (1 - detection_mask)

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
        "--output-path",
        type=Path,
        help="Path to save the output audio file.",
    )
    
    
    
    args = parser.parse_args()
    
    if not args.output_path:
        args.output_path = args.mixed_path.parent / f"suppressed_{args.mixed_path.name}_from_{args.enrolment_path.stem}.wav"

    main(
        enrolment_path=args.enrolment_path, 
        mixed_path=args.mixed_path, 
        output_path=args.output_path
    )
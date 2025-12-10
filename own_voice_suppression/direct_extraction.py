import pathlib
from os import PathLike
import argparse
import time
import torch
import torchaudio
import torch.nn.functional as F

from general_utils.resample_audio import resample
from speakerbeam.src.models.td_speakerbeam import TimeDomainSpeakerBeam

MODEL_PATH = "speakerbeam/example/model.pth"



def direct_own_voice_suppression(
    enrolment_audio: torch.Tensor, enrolment_sr: int, mixed_audio: torch.Tensor, mixed_sr: int
    ):
    """
    Extracts the enrolment speaker from the mix and subtracts it.
    Note: Both inputs are expected to be 1D (Mono) or (1, Time).
    """
    MODEL_SAMPLE_RATE = 8_000
    resampled_enrolment = resample(
        enrolment_audio, 
        orig_sr=enrolment_sr, 
        new_sr=MODEL_SAMPLE_RATE
    )
    
    resampled_mixed = resample(
        mixed_audio,
        orig_sr=mixed_sr,
        new_sr=MODEL_SAMPLE_RATE
    )
    model = TimeDomainSpeakerBeam.from_pretrained(MODEL_PATH)
    model.eval()
    with torch.no_grad():
        start = time.time()
        est_source = model(resampled_mixed, resampled_enrolment)
        total_time = time.time() - start
        return mixed_audio, total_time, MODEL_SAMPLE_RATE
        # return mixed_audio - est_source.squeeze(0), total_time, MODEL_SAMPLE_RATE
        # return mixed_audio - est_source.squeeze(0), total_time
        # return est_source.squeeze(0) - mixed_audio, total_time

def main(enrolment_path: PathLike, mixed_path: PathLike, output_path: PathLike):
    """
    Performs own voice suppression on the mixed audio file using the enrolment audio file path
    """
    
    enrolment_audio, sr_enroll = torchaudio.load(enrolment_path)
    mixed_audio, sr_mix = torchaudio.load(mixed_path)
    
    # Ensure mono channel
    if enrolment_audio.shape[0] > 1:
        enrolment_audio = torch.mean(enrolment_audio, dim=0, keepdim=True)
    if mixed_audio.shape[0] > 1:
        mixed_audio = torch.mean(mixed_audio, dim=0, keepdim=True)

    suppressed_audio, inference_time, output_sample_rate = direct_own_voice_suppression(
        enrolment_audio, sr_enroll, mixed_audio, sr_mix
        )

    print(f"Inference time: {inference_time:.2f} seconds")
    print(f"Length of output audio: {mixed_audio.shape[-1] / sr_mix:.2f} seconds")
    torchaudio.save(output_path, suppressed_audio / suppressed_audio.abs().max(), output_sample_rate)
    print(f"Saved suppressed audio to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mixed-path",
        type=pathlib.Path,
        required=True,
        help="Path to the mixed audio file.",
    )
    parser.add_argument(
        "--enrolment-path",
        type=pathlib.Path,
        required=True,
        help="Path to the enrolment audio file.",
    )
    parser.add_argument(
        "--output-path",
        type=pathlib.Path,
        help="Path to save the output audio file.",
    )
    
    args = parser.parse_args()
    
    if not args.output_path:
        args.output_path = args.mixed_path.parent / f"suppressed_{args.mixed_path.name}_direct_extraction.wav"

    main(
        enrolment_path=args.enrolment_path,
        mixed_path=args.mixed_path, 
        output_path=args.output_path
    )
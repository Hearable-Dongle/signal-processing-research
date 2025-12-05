import pathlib
from os import PathLike
import argparse
import torch
import torchaudio
import torch.nn.functional as F
from asteroid.models import ConvTasNet
from speechbrain.inference.speaker import EncoderClassifier
import time



MODEL_PATH = "asteroid/egs/librimix/ConvTasNet/exp/tmp/best_model.pth"

def blind_own_voice_suppression(enrolment_audio: torch.Tensor, mixed_audio: torch.Tensor):
    if mixed_audio.dim() == 1:
        model_input = mixed_audio.unsqueeze(0).unsqueeze(0)
    else:
        model_input = mixed_audio.unsqueeze(0)
    
    sep_model = ConvTasNet(
        n_src=2,
        n_repeats=3,
        n_blocks=8,
        norm_type="cLN", # Must use 'cLN' (Cumulative Layer Norm) for causal systems. 'gLN' is non-causal.
        causal=True, # Enforces causal padding (past-only) in convolutions
        mask_act="relu"
    )
    
    sep_model.load_state_dict(torch.load(MODEL_PATH, weights_only=False)["state_dict"])
    
    sep_model.eval() # Set to evaluation mode

    start = time.time()


    # Model to score d-vector
    # From: https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb
    spk_classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb", 
        savedir="tmp_model"
    )    

    with torch.no_grad():
        estimated_sources = sep_model(model_input)

    
    # ECAPA expects (Batch, Time), assumes audio is already 16k
    emb_enroll = spk_classifier.encode_batch(enrolment_audio)
    
    source_0 = estimated_sources[0, 0, :]
    # source_0 /= source_0.max()
    source_0 /= torch.norm(source_0, p=2)
    emb_src0 = spk_classifier.encode_batch(source_0)
    
    source_1 = estimated_sources[0, 1, :]
    # source_1 /= source_1.max()
    source_1 /= torch.norm(source_1, p=2)
    emb_src1 = spk_classifier.encode_batch(source_1)

    score_0 = F.cosine_similarity(emb_enroll, emb_src0, dim=-1)
    score_1 = F.cosine_similarity(emb_enroll, emb_src1, dim=-1)
    total_time = time.time() - start

    print(f"Similarity Score Source 0: {score_0.item()}")
    print(f"Similarity Score Source 1: {score_1.item()}")
    print(f"Total processing time: {total_time:.2f} seconds")
    print(f"Length of output audio: {mixed_audio.shape[-1] / 16000:.2f} seconds")

    return source_0

    if score_0 > score_1:
        print("User detected in Stream 0. Playing Stream 1.")
        return source_1
    else:
        print("User detected in Stream 1. Playing Stream 0.")
        return source_0
    
    

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

    suppressed_audio = blind_own_voice_suppression(enrolment_audio, mixed_audio)
    
    # Ensure dimensions are (Channels, Time)
    if suppressed_audio.dim() == 1:
        suppressed_audio = suppressed_audio.unsqueeze(0)
        
    torchaudio.save(output_path, suppressed_audio, sr_mix)
    print(f"Saved suppressed audio to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--enrolment-path",
        type=pathlib.Path,
        required=True,
        help="Path to the enrolment audio file.",
    )
    parser.add_argument(
        "--mixed-path",
        type=pathlib.Path,
        required=True,
        help="Path to the mixed audio file.",
    )
    parser.add_argument(
        "--output-path",
        type=pathlib.Path,
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
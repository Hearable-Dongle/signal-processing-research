import torch
from torch import Tensor 
from general_utils.resample_audio import resample


def prep_audio(audio: Tensor, orig_sr: int, target_sr: int) -> Tensor:
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

    
# Hack: Override torch.load to disable weights_only loading
# See https://github.com/m-bain/whisperX/issues/1304
_original_torch_load = torch.load

def torch_trusted_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)

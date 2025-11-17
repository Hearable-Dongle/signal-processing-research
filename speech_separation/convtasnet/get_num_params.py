import argparse

import soundfile as sf
import torch
from asteroid.models import ConvTasNet

def main():
    model_name = "JorisCos/ConvTasNet_Libri2Mix_sepnoisy_16k"
    print(f"Loading pre-trained model: {model_name}...")

    model = ConvTasNet.from_pretrained(model_name)
    
    total_params = sum(param.numel() for param in model.parameters())

    print("Total params", total_params)


if __name__ == "__main__":
    main()


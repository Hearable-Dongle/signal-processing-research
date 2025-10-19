import argparse
import os
import torch
import soundfile as sf
from asteroid.models import ConvTasNet

def main():
    model_name = "JorisCos/ConvTasNet_Libri2Mix_sepnoisy_16k"
    print(f"Loading pre-trained model: {model_name}...")
    model = ConvTasNet.from_pretrained(model_name)
    torch.save(model.state_dict(), 'conv_tasnet_model.pth')

if __name__ == "__main__":
    main()

import torch
import torchaudio
import argparse
from .model import SpeakerCountCRNN

# Constants (Must match training)
SAMPLE_RATE = 16000
N_MELS = 64
N_FFT = 400
HOP_LENGTH = 160

class SpeakerCounter:
    def __init__(self, model_path, device='cpu'):
        self.device = device
        self.model = SpeakerCountCRNN(num_classes=5)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(device)
        self.model.eval()
        
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS
        ).to(device)

    def predict(self, audio_path):
        """
        Predicts number of speakers from an audio file.
        """
        waveform, sr = torchaudio.load(audio_path)
        
        # Resample if needed
        if sr != SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE).to(waveform.device)
            waveform = resampler(waveform)
            
        # Mix down to mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        # Pad or Crop to 1 second?
        # For this simplified model, we handle whatever length, 
        # but fixed size is better for edge.
        # Let's crop/pad to 1 second (16000 samples)
        target_len = 16000
        if waveform.shape[1] < target_len:
            waveform = torch.nn.functional.pad(waveform, (0, target_len - waveform.shape[1]))
        else:
            waveform = waveform[:, :target_len]
            
        waveform = waveform.to(self.device)
        
        # Spectrogram
        spec = self.mel_transform(waveform)
        spec = torch.log(spec + 1e-9)
        
        # Add batch dim: (1, 1, F, T)
        spec = spec.unsqueeze(0)
        
        with torch.no_grad():
            logits = self.model(spec)
            probs = torch.softmax(logits, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            
        return pred, probs[0].cpu().numpy()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("audio_path", help="Path to audio file")
    parser.add_argument("--model", default="speaker_count_crnn.pth", help="Path to model checkpoint")
    args = parser.parse_args()
    
    counter = SpeakerCounter(args.model)
    count, probs = counter.predict(args.audio_path)
    print(f"Predicted Speaker Count: {count}")
    print(f"Probabilities: {probs}")

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
from torch.utils.data import Dataset, DataLoader
from .model import SpeakerCountCRNN

# --- Configuration ---
SAMPLE_RATE = 16000
DURATION = 1.0 # seconds
N_MELS = 64
N_FFT = 400
HOP_LENGTH = 160
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-3
MODEL_SAVE_PATH = "speaker_count_crnn.pth"

class AudioDataset(Dataset):
    """
    Dataset wrapper.
    For real usage, implement file loading logic here.
    """
    def __init__(self, num_samples=1000):
        self.num_samples = num_samples
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS
        )

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate dummy data: Random noise + sine waves to simulate speakers?
        # Real data should load wav files mixed with 0-4 speakers.
        
        # Placeholder: Random tensor
        # 1 second of audio
        waveform = torch.randn(1, int(SAMPLE_RATE * DURATION)) 
        
        # Random label 0-4
        label = torch.randint(0, 5, (1,)).item()
        
        # Compute Mel Spectrogram
        # Shape: (1, n_mels, time)
        spec = self.mel_transform(waveform)
        
        # Normalize (log mel)
        spec = torch.log(spec + 1e-9)
        
        return spec, label

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data
    dataset = AudioDataset(num_samples=1000)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Model
    model = SpeakerCountCRNN(num_classes=5).to(device)
    
    # Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Loop
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        correct = 0
        total = 0
        
        for specs, labels in dataloader:
            specs = specs.to(device)
            labels = labels.to(device)
            
            # Forward
            outputs = model(specs)
            loss = criterion(outputs, labels)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {total_loss/len(dataloader):.4f}, Accuracy: {100*correct/total:.2f}%")

    # Save
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")
    
    # Export ONNX (Ready for Hailo)
    dummy_input = torch.randn(1, 1, N_MELS, 100).to(device) # Approx shape
    onnx_path = MODEL_SAVE_PATH.replace(".pth", ".onnx")
    torch.onnx.export(
        model, 
        dummy_input, 
        onnx_path, 
        input_names=['input'], 
        output_names=['output'],
        opset_version=11 
    )
    print(f"Model exported to {onnx_path}")

if __name__ == "__main__":
    train()

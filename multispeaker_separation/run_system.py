import os
import torch
import torch.nn as nn
from multispeaker_separation.inference import SpeakerSeparationSystem
from multispeaker_separation.count_speakers.inference import SpeakerCounter
from multispeaker_separation.speaker_counter import PyannoteCounter

# Mock Asteroid ConvTasNet for testing
class MockConvTasNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))
    
    def forward(self, x):
        # Return same shape as input
        return x

    @classmethod
    def from_pretrained(cls, path):
        return cls()

# Patch ConvTasNet in inference module to use Mock
import multispeaker_separation.inference
multispeaker_separation.inference.ConvTasNet = MockConvTasNet

def create_dummy_models(model_dir):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    for k in range(1, 6):
        path = os.path.join(model_dir, f"convtasnet_{k}spk.pth")
        if not os.path.exists(path):
            # Just create an empty file or a dummy torch file
            # Since we patched from_pretrained, the file content doesn't matter much 
            # as long as it exists. But let's save a valid state dict just in case.
            torch.save({}, path)
            print(f"Created dummy model: {path}")

def main():
    print("--- Setting up Speaker Separation System ---")
    
    # 1. Setup Models
    model_dir = "multispeaker_separation/models"
    create_dummy_models(model_dir)
    
    # 2. Initialize Separation System
    # Using PyTorch backend for this test
    sep_system = SpeakerSeparationSystem(model_dir, backend='pytorch')
    print("Separation System Initialized.")
    
    # 3. Initialize Counters
    # CRNN Counter
    crnn_path = os.path.join(model_dir, "speaker_count_crnn.pth")
    if os.path.exists(crnn_path):
        crnn_counter = SpeakerCounter(crnn_path, device='cpu')
        print("CRNN Counter Initialized.")
    else:
        print("CRNN Model not found.")
        crnn_counter = None

    # Pyannote Counter (might fail without token, so wrapping in try)
    try:
        # Pass a dummy token or None. It will likely fail to load the pipeline if not authenticated.
        # But we check initialization.
        pyannote_counter = PyannoteCounter(auth_token="HF_TOKEN_PLACEHOLDER")
        print("Pyannote Counter Initialized (Warning: Pipeline might fail without valid token).")
    except Exception as e:
        print(f"Pyannote Counter failed to init: {e}")
        pyannote_counter = None

    # 4. Run Inference on Dummy Audio
    print("\n--- Running Inference Test ---")
    
    # Create 1 second dummy audio (16kHz)
    dummy_audio = torch.randn(1, 16000)
    # Save to file for counters
    torchaudio.save("dummy_test_audio.wav", dummy_audio, 16000)
    
    # Test CRNN
    if crnn_counter:
        count, probs = crnn_counter.predict("dummy_test_audio.wav")
        print(f"[CRNN] Predicted Speakers: {count} (Probs: {probs})")
        
        # Use this count for separation
        try:
            # We treat 0 as 1 or handle it? The CRNN has class 0.
            # Models are 1-5.
            if count == 0: count = 1
            if count > 5: count = 5
            
            output = sep_system.separate(dummy_audio, count)
            print(f"[Separation] Output shape for {count} speakers: {output.shape}")
        except Exception as e:
            print(f"[Separation] Failed: {e}")

    # Test Pyannote
    if pyannote_counter and pyannote_counter.pipeline:
        try:
            count = pyannote_counter.count("dummy_test_audio.wav")
            print(f"[Pyannote] Predicted Speakers: {count}")
        except Exception as e:
            print(f"[Pyannote] Failed (Expected if no token): {e}")

    print("\n--- Test Complete ---")
    os.remove("dummy_test_audio.wav")

if __name__ == "__main__":
    import torchaudio
    main()
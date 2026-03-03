import torch.nn as nn

class SpeakerCountCRNN(nn.Module):
    def __init__(self, num_classes=5, input_channels=1):
        """
        Lightweight CRNN for speaker counting.
        Classes: 0, 1, 2, 3, 4+ (5 classes)
        """
        super(SpeakerCountCRNN, self).__init__()
        
        # Convolutional feature extractor
        # Input expected: (Batch, 1, n_mels, time_frames)
        # e.g., (B, 1, 64, 100) for 1s audio @ 16kHz, hop=160
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)) # (B, 16, 32, 50)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)) # (B, 32, 16, 25)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # Pool frequency only, keep time resolution high? 
            # Or pool both. 25 frames is enough for counting.
            nn.MaxPool2d(kernel_size=(2, 1)) # (B, 64, 8, 25)
        )
        
        # Recurrent layer
        # Input features for RNN: 64 * 8 = 512
        self.rnn_input_dim = 64 * 8
        self.hidden_dim = 128
        self.gru = nn.GRU(
            input_size=self.rnn_input_dim, 
            hidden_size=self.hidden_dim, 
            num_layers=1, 
            batch_first=True, 
            bidirectional=False # Simpler for edge deployment
        )
        
        # Classifier
        self.fc = nn.Linear(self.hidden_dim, num_classes)

    def forward(self, x):
        # x: (B, 1, F, T)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        # Prepare for RNN: (B, C, F, T) -> (B, T, C*F)
        b, c, f, t = x.size()
        x = x.permute(0, 3, 1, 2).contiguous() # (B, T, C, F)
        x = x.view(b, t, -1) # (B, T, C*F)
        
        # RNN
        # out: (B, T, H)
        # h_n: (1, B, H)
        out, h_n = self.gru(x)
        
        # Use last hidden state
        # h_n is (1, B, H) -> (B, H)
        final_state = h_n.squeeze(0)
        
        logits = self.fc(final_state)
        return logits

def get_model():
    return SpeakerCountCRNN()

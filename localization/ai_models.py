import torch
import torch.nn as nn

class HailoDOANet(nn.Module):
    def __init__(self, num_pairs=4, gcc_width=128, output_classes=360):
        super(HailoDOANet, self).__init__()
        # 1D CNN to extract features from GCC-PHAT lags
        self.conv_block = nn.Sequential(
            nn.Conv1d(num_pairs, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(16) 
        )
        
        # Temporal tracking
        self.gru = nn.GRU(input_size=128 * 16, hidden_size=128, batch_first=True)
        
        # Spatial Likelihood Map
        self.fc = nn.Linear(128, output_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (Batch, TimeSteps, Pairs, GCC_Width)
        # Note: The input is expected to be (Batch, TimeSteps, Pairs, GCC_Width)
        b, t, p, w = x.shape
        x = x.view(b * t, p, w)
        x = self.conv_block(x)
        x = x.view(b, t, -1)
        x, _ = self.gru(x)
        x = self.fc(x[:, -1, :]) # Take last timestep
        return self.sigmoid(x)

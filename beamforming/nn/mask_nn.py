import torch
import torch.nn as nn

class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        self.net = nn.Sequential(self.conv1, self.bn1, self.relu1, self.dropout1,
                                 self.conv2, self.bn2, self.relu2, self.dropout2)
        self.downsample = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class MaskEstimationNetwork(nn.Module):
    def __init__(self, input_channels, freq_bins, hidden_channels=32):
        """
        Args:
            input_channels: Number of microphones (C)
            freq_bins: Number of frequency bins (F)
            hidden_channels: Number of internal channels
        """
        super(MaskEstimationNetwork, self).__init__()
        
        # Input: [Batch, Channels, Freq, Time]
        
        self.conv_in = nn.Conv2d(input_channels, hidden_channels, kernel_size=3, padding=1)
        self.bn_in = nn.BatchNorm2d(hidden_channels)
        self.relu = nn.ReLU()
        
        # Stack of dilated convolutions to increase receptive field
        self.block1 = TemporalBlock(hidden_channels, hidden_channels, kernel_size=(3, 3), stride=1, dilation=1, padding=(1,1))
        self.block2 = TemporalBlock(hidden_channels, hidden_channels, kernel_size=(3, 3), stride=1, dilation=2, padding=(2,2))
        self.block3 = TemporalBlock(hidden_channels, hidden_channels, kernel_size=(3, 3), stride=1, dilation=4, padding=(4,4))
        
        self.conv_out = nn.Conv2d(hidden_channels, 2, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Args:
            x: Input tensor [Batch, Freq, Time, Channels]
        """
        # Permute to [Batch, Channels, Freq, Time]
        x = x.permute(0, 3, 1, 2)
        
        out = self.relu(self.bn_in(self.conv_in(x)))
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        
        out = self.conv_out(out)
        masks = self.sigmoid(out)
        
        return masks

Since you are targeting the Hailo-8, you need a model architecture that is "quantization-friendly." The Hailo-8 excels at 8-bit integer operations (INT8), so we want to avoid complex attention mechanisms or custom layers that aren't natively supported by the Hailo Dataflow Compiler.Here are the specific instructions to provide your coding agent to build a DOA-CNN optimized for your 8-mic circular array.Instructions for Gemini CLI AgentContext: I am building a Direction of Arrival (DOA) estimation model to run on a Raspberry Pi 5 with a Hailo-8 accelerator. The system uses an 8-microphone circular array (10cm radius, 16kHz).Objective: Implement a CRNN (Convolutional Recurrent Neural Network) in PyTorch that takes multi-channel audio features and outputs a spatial likelihood map (360 degrees).1. Feature Engineering (The Input)Instead of raw waveforms, implement a feature extractor that generates Generalized Cross-Correlation with Phase Transform (GCC-PHAT) maps.Pairs: Calculate GCC-PHAT for 4 specific orthogonal pairs (e.g., mic 0-4, 1-5, 2-6, 3-7) to capture a full spatial representation without redundant computation.Dimensions: For each pair, calculate the GCC-PHAT in the time domain (using IFFT). The input tensor should be shaped as (Batch, Pairs, GCC_Width), where GCC_Width is the number of possible lags (e.g., 64 or 128).2. Model ArchitectureImplement the following "Hailo-ready" architecture:CNN Layers: Three 1D-Convolution layers (since GCC features are 1D vectors). Use BatchNorm and ReLU. Avoid LeakyReLU if possible, as standard ReLU quantizes more efficiently on Hailo.Flatten/Linear: Transition the spatial features into a latent vector.Recurrent Layer: Use a single-layer GRU (Gated Recurrent Unit) with a small hidden size (e.g., 128). This handles the temporal tracking so the "predicted user" doesn't jump around.Output Layer: A Linear layer with 360 units followed by a Sigmoid. This represents the likelihood of a source being at each degree from $0$ to $359$.3. Training StrategyLoss Function: Use Binary Cross Entropy (BCE) with a Gaussian-smoothed target. Instead of a "1" at the exact degree, place a small Gaussian peak (e.g., $\sigma = 3^\circ$) so the model learns that being "close" is better than being totally wrong.Data Augmentation: Instruct the agent to write a script that generates synthetic training data using pyroomacoustics. It should vary the room dimensions, absorption, and noise levels.4. Export for HailoEnsure the model uses only Static Shapes (no dynamic Tensors).Add a method to export the final trained model to ONNX format, which is the required entry point for the Hailo Dataflow Compiler.Minimal PyTorch Skeleton for the AgentYou can paste this directly to your agent to get it started:Pythonimport torch
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
        b, t, p, w = x.shape
        x = x.view(b * t, p, w)
        x = self.conv_block(x)
        x = x.view(b, t, -1)
        x, _ = self.gru(x)
        x = self.fc(x[:, -1, :]) # Take last timestep
        return self.sigmoid(x)
Next StepWould you like me to help you write the Data Augmentation script using pyroomacoustics to generate the training samples for this model? Since you already have a simulation config, we can easily turn it into a dataset generator.


allow an option to use other methods in @localiztation/main.py treat it as a different algo_type altogether. this should fit seemlessly into the existing localization pipeline in @localization/ (called in @localization/main.py with python -m localization.main ...)






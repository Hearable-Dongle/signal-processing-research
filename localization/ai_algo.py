import numpy as np
import scipy.signal as signal
import torch
import warnings
from .ai_models import HailoDOANet

class AILocalization:
    def __init__(self, mic_pos, fs=16000, nfft=512, overlap=0.5, 
                 max_sources=2, model_path=None, gcc_width=128, **kwargs):
        """
        AI-based Localization using HailoDOANet.
        
        Args:
            mic_pos: (3, M) numpy array of microphone positions.
            fs: Sampling frequency.
            nfft: FFT size for STFT (used for GCC-PHAT).
            overlap: Overlap fraction.
            max_sources: Max sources to detect.
            model_path: Path to trained model weights.
            gcc_width: Number of lags to keep for GCC-PHAT features.
        """
        self.mic_pos = mic_pos
        self.fs = fs
        self.nfft = nfft
        self.overlap = overlap
        self.max_sources = max_sources
        self.gcc_width = gcc_width
        
        # Initialize model
        # Assuming 4 pairs for 8 mics
        self.num_pairs = 4
        self.model = HailoDOANet(num_pairs=self.num_pairs, gcc_width=gcc_width)
        
        if model_path:
            try:
                state_dict = torch.load(model_path, map_location='cpu')
                self.model.load_state_dict(state_dict)
                print(f"Loaded AI model from {model_path}")
            except Exception as e:
                print(f"Error loading model from {model_path}: {e}")
                print("Using random weights.")
        else:
            print("Warning: No model path provided. Using initialized random weights.")
            
        self.model.eval()

    def process(self, audio):
        """
        Process multichannel audio to find sources.
        
        Args:
            audio: (M, N) numpy array of multichannel audio.
            
        Returns:
            estimated_doas: List of estimated angles (radians).
            histogram: The angular likelihood map (averaged).
            history: List of (time, angle) tuples.
        """
        M_mics, N_samples = audio.shape
        
        # 1. Feature Extraction: GCC-PHAT
        # We need to compute STFT first
        f_vec, t_vec, Zxx = signal.stft(audio, fs=self.fs, nperseg=self.nfft, 
                                       noverlap=int(self.nfft * self.overlap))
        # Zxx: (M, F, T)
        M, F, T = Zxx.shape
        
        # Defined Pairs (mic 0-4, 1-5, 2-6, 3-7 for 8 mics)
        # If not 8 mics, we might need logic. Assuming 8 mics based on instructions.
        if M_mics == 8:
            pairs = [(0, 4), (1, 5), (2, 6), (3, 7)]
        else:
            # Fallback or error? Let's just try to form pairs 0-M/2 etc
            half = M_mics // 2
            pairs = [(i, i + half) for i in range(half)]
            if len(pairs) != self.num_pairs:
                warnings.warn(f"Number of pairs {len(pairs)} does not match model expectation {self.num_pairs}.")
        
        # Compute GCC-PHAT for each pair and time step
        # Output: (T, Pairs, GCC_Width)
        
        features = np.zeros((T, len(pairs), self.gcc_width), dtype=np.float32)
        
        for p_idx, (i, j) in enumerate(pairs):
            Xi = Zxx[i] # (F, T)
            Xj = Zxx[j]
            
            # Generalized Cross Correlation
            R_12 = Xi * np.conj(Xj)
            
            # Phase Transform (PHAT) weighting
            denom = np.abs(R_12)
            denom[denom < 1e-10] = 1e-10
            G_phat = R_12 / denom
            
            # Inverse FFT to get time domain GCC
            # axis=0 is Frequency. We want IFFT over frequency.
            # n=nfft ensures we get the full time correlation
            gcc_time = np.fft.irfft(G_phat, n=self.nfft, axis=0) # (nfft, T)
            
            # fftshift to center 0 lag? irfft result: lag 0 is at index 0. 
            # lags: 0, 1, ... n/2, -n/2, ... -1
            # We usually want to shift it so 0 is in center if we take a window.
            # But irfft returns [0, 1, ..., N-1].
            # Let's perform fftshift along axis 0
            gcc_time = np.fft.fftshift(gcc_time, axes=0)
            
            # Crop to gcc_width around center
            center = self.nfft // 2
            half_width = self.gcc_width // 2
            
            start = center - half_width
            end = center + half_width
            
            # Handle odd width if necessary, but assume even
            gcc_crop = gcc_time[start:end, :] # (Width, T)
            
            # Transpose to (T, Width) and store
            features[:, p_idx, :] = gcc_crop.T
            
        # 2. Model Inference
        # Model expects (Batch, TimeSteps, Pairs, GCC_Width)
        # We can process the whole sequence as one batch of 1 sample with T timesteps, 
        # but the GRU might be designed for shorter sequences or streaming.
        # The forward method takes x, does conv, then view(b, t, -1) -> gru -> takes LAST timestep.
        # This implies the model predicts the DOA at the END of the sequence provided.
        # So we should probably slide a window or just process frame by frame.
        # If we process frame by frame (T=1), the GRU state is not maintained unless we modify the model to accept hidden state.
        # The current model implementation RE-INITIALIZES GRU hidden state in every forward call (no hidden state arg).
        # So providing T=1 means GRU has no history.
        # Providing T=All means it gives one prediction for the very end.
        
        # To get a history of predictions, we should probably modify the model to return sequence or accept hidden state.
        # BUT I cannot change the model architecture given in instructions easily (or I should have?).
        # The instructions say: "Recurrent Layer: Use a single-layer GRU... Output Layer: ... likelihood ... "
        # The provided code `x = self.fc(x[:, -1, :])` explicitly takes the last time step.
        
        # To get a history, we can feed a sliding window of context.
        # e.g. context window of 16 frames.
        
        context_size = 16
        history = []
        
        # Prepare batch of sliding windows
        # Creating a batch of (BatchSize, Context, Pairs, Width)
        # This might be heavy memory-wise if T is large.
        
        input_tensor = torch.from_numpy(features) # (T, Pairs, Width)
        
        # Let's just iterate and construct batch on the fly
        batch_size = 32
        windows = []
        time_indices = []
        
        for t in range(context_size, T):
            window = input_tensor[t-context_size : t]
            windows.append(window)
            time_indices.append(t)
            
            if len(windows) >= batch_size or t == T - 1:
                batch_x = torch.stack(windows) # (B, Context, Pairs, Width)
                with torch.no_grad():
                    # (B, 360)
                    output = self.model(batch_x)
                
                # Process outputs
                probs = output.numpy() # (B, 360)
                
                for k in range(len(probs)):
                    # Find peak
                    p = probs[k]
                    best_idx = np.argmax(p)
                    angle_rad = np.deg2rad(best_idx) # Index 0..359 corresponds to degrees
                    
                    t_abs = t_vec[time_indices[k]]
                    history.append((t_abs, angle_rad))
                
                windows = []
                time_indices = []
        
        if not history:
            return [], np.zeros(360), []

        # 3. Aggregate Histogram
        # Extract angles from history
        angles = [h[1] for h in history]
        
        # Simple histogram of estimates
        hist, edges = np.histogram(angles, bins=360, range=(0, 2*np.pi))
        
        # Smooth histogram (optional, but good for peak finding)
        hist = np.convolve(hist, np.ones(5)/5, mode='same')
        
        # 4. Find Peaks (Final DOAs)
        final_doas = []
        peaks = []
        for i in range(len(hist)):
            prev = hist[(i-1)%len(hist)]
            curr = hist[i]
            next_val = hist[(i+1)%len(hist)]
            if curr > prev and curr >= next_val:
                peaks.append((curr, i))
        
        peaks.sort(key=lambda x: x[0], reverse=True)
        
        # Convert bin index to radians
        # bins are 0..359 mapping to 0..2pi
        bin_width = 2*np.pi / 360
        
        for p_val, p_idx in peaks[:self.max_sources]:
             angle = p_idx * bin_width + bin_width/2
             final_doas.append(angle)
             
        return final_doas, hist, history


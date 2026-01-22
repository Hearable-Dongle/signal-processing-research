import numpy as np
import scipy.signal as signal
from scipy.linalg import eigh

class LocalizationSystem:
    def __init__(self, mic_pos, fs=16000, nfft=512, overlap=0.5, epsilon=0.2, d_freq=2, freq_range=(200, 3000), max_sources=4):
        """
        Args:
            mic_pos: (3, M) numpy array of microphone positions.
            fs: Sampling frequency.
            nfft: FFT size.
            overlap: Overlap fraction (0 to 1).
            epsilon: SSZ detection threshold (correlation >= 1 - epsilon).
            d_freq: Number of frequency components to group/average per zone.
            freq_range: Tuple (min_freq, max_freq) in Hz to process.
            max_sources: Maximum number of sources to detect in Matching Pursuit.
        """
        self.mic_pos = mic_pos
        self.fs = fs
        self.nfft = nfft
        self.overlap = overlap
        self.epsilon = epsilon
        self.d_freq = d_freq
        self.freq_range = freq_range
        self.max_sources = max_sources
        self.c = 343.0  # Speed of sound

    def process(self, audio):
        """
        Process multichannel audio to find sources.
        
        Args:
            audio: (M, N) numpy array of multichannel audio.
            
        Returns:
            estimated_doas: List of estimated angles (radians).
            histogram: The angular spectrum/histogram before MP.
        """
        # 1. STFT
        # audio is (M, N). scipy.signal.stft expects (..., time)
        # Returns f_vec, t_vec, Zxx. Zxx shape: (M, F, T)
        f_vec, t_vec, Zxx = signal.stft(audio, fs=self.fs, nperseg=self.nfft, 
                                       noverlap=int(self.nfft * self.overlap))
        
        M, F, T = Zxx.shape
        
        ssz_doas = []
        
        # Grid for DOA search
        search_angles = np.linspace(0, 2*np.pi, 360, endpoint=False)
        
        # Determine frequency bins to process
        f_min, f_max = self.freq_range
        f_idx_start = np.searchsorted(f_vec, f_min)
        f_idx_end = np.searchsorted(f_vec, f_max)
        
        # Iterate over Time-Frequency Bins (or Zones)
        # Group frequency bins by d_freq
        for t_idx in range(T):
            for f_idx in range(f_idx_start, f_idx_end - self.d_freq + 1, self.d_freq):
                # Extract block: (M, d_freq)
                block = Zxx[:, f_idx : f_idx + self.d_freq, t_idx] # Shape (M, d)
                
                # --- New SSZ Detection: Magnitude Correlation Coefficient ---
                # r' = sum(|Xi||Xj|) / sqrt(sum|Xi|^2 sum|Xj|^2)
                # Averaged over adjacent pairs.
                
                mags = np.abs(block) # (M, d)
                
                # Compute energies for denominator
                energies = np.sum(mags**2, axis=1) # (M,)
                
                # Check for silence to avoid div by zero
                if np.any(energies < 1e-10):
                    continue
                
                avg_corr = 0.0
                
                for i in range(M):
                    j = (i + 1) % M
                    
                    # Numerator: sum over freq of |Xi||Xj|
                    num = np.sum(mags[i] * mags[j])
                    
                    # Denominator
                    den = np.sqrt(energies[i] * energies[j])
                    
                    corr = num / den
                    avg_corr += corr
                
                avg_corr /= M
                
                if avg_corr >= (1.0 - self.epsilon):
                    # It is an SSZ. Estimate DOA.
                    
                    # For DOA estimation, we still need the Spatial Correlation Matrix R
                    # R = (1/d) * sum(x * x^H)
                    R = np.zeros((M, M), dtype=complex)
                    for k in range(self.d_freq):
                        vec = block[:, k][:, np.newaxis] # (M, 1)
                        R += vec @ vec.conj().T
                    R /= self.d_freq
                    
                    # Frequencies in this zone
                    freqs = f_vec[f_idx : f_idx + self.d_freq]
                    
                    best_angle = self._cics_doa(R, freqs, search_angles)
                    ssz_doas.append(best_angle)

        # 2. Histogram Construction
        # Simple histogram
        hist, edges = np.histogram(ssz_doas, bins=360, range=(0, 2*np.pi))
        
        # 3. Matching Pursuit (Updated)
        final_doas = self._matching_pursuit(hist, edges)
        
        return final_doas, hist

    def _cics_doa(self, R, freqs, angles):
        """
        Find angle phi that maximizes coherent sum.
        """
        M = self.mic_pos.shape[1]
        num_angles = len(angles)
        
        # Mic positions (2D)
        pos_x = self.mic_pos[0, :]
        pos_y = self.mic_pos[1, :]
        
        # Direction vectors for all search angles
        u_x = np.cos(angles)
        u_y = np.sin(angles)
        
        # Use center frequency of the zone
        center_freq = np.mean(freqs)
        omega = 2 * np.pi * center_freq
        k_wave = omega / self.c
        
        # Calculate exponents for all mics and angles: (M, NumAngles)
        proj = pos_x[:, np.newaxis] * u_x + pos_y[:, np.newaxis] * u_y
        phases = 1j * k_wave * proj # Corrected sign
        steering_vecs = np.exp(phases) # (M, A)
        
        # Beamformer Output Power: P(phi) = v^H R v
        temp = R @ steering_vecs
        P = np.sum(np.conj(steering_vecs) * temp, axis=0)
        
        best_idx = np.argmax(np.abs(P))
        return angles[best_idx]

    def _matching_pursuit(self, hist, edges):
        """
        Iterative matching pursuit on the histogram.
        Uses Dual-Width Atoms and Smoothing.
        """
        # 1. SMOOTH the histogram first
        # Use an averaging filter (approx 5 degrees -> 5 bins if 360 bins)
        kernel_size = 5
        smoothing_kernel = np.ones(kernel_size) / kernel_size
        
        # Circular convolution for smoothing
        smoothed_hist = np.convolve(np.pad(hist, (kernel_size, kernel_size), mode='wrap'), smoothing_kernel, mode='same')
        # Crop back to original size (handling the padding)
        smoothed_hist = smoothed_hist[kernel_size:-kernel_size]
        
        # Ensure we didn't drift size
        if len(smoothed_hist) != len(hist):
             # Fallback to simple same mode without wrap if logic fails, but let's try to be precise
             smoothed_hist = np.convolve(hist, smoothing_kernel, mode='same')

        # 2. Define Dual Atoms
        # Assuming 360 bins (1 degree per bin)
        # Narrow: 40 deg -> 41 bins (must be odd usually for symmetry)
        # Wide: 80 deg -> 81 bins
        # Paper suggests 81 and 161 for L=720 (0.5 deg resolution).
        # For L=360, we scale down by 2.
        # Narrow ~ 41, Wide ~ 81.
        atom_narrow_width = 41
        atom_wide_width = 81
        
        atom_narrow = np.blackman(atom_narrow_width)
        atom_wide = np.blackman(atom_wide_width)
        
        current_hist = smoothed_hist.copy().astype(float)
        detected_sources = []
        
        # Threshold: Dynamic or static?
        # Paper suggests dynamic based on history, but here we process batch.
        # Let's use a relative threshold of the max peak found initially
        initial_max = np.max(current_hist)
        threshold = initial_max * 0.15 # Stop if peak is < 15% of max
        
        for j in range(self.max_sources):
            # Find peak using Narrow logic conceptually (peak of smoothed hist)
            peak_idx = np.argmax(current_hist)
            peak_val = current_hist[peak_idx]
            
            if peak_val < threshold:
                break
            
            # Convert bin index to angle
            angle = (edges[peak_idx] + edges[peak_idx+1]) / 2
            detected_sources.append(angle)
            
            # 3. MASKING: Subtract using the WIDE atom
            # We subtract the WIDE atom scaled to the peak value
            self._subtract_circular_atom(current_hist, peak_idx, atom_wide, peak_val)
            
        return detected_sources

    def _subtract_circular_atom(self, hist, center_idx, atom, scale):
        """
        Subtracts a scaled atom from the histogram handling circular wrapping.
        """
        N = len(hist)
        L = len(atom)
        start = center_idx - L // 2
        
        indices = np.arange(start, start + L)
        wrap_indices = indices % N
        
        hist[wrap_indices] = np.maximum(0, hist[wrap_indices] - atom * scale)
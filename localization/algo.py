import numpy as np
import scipy.signal as signal
from scipy.linalg import eigh

class LocalizationSystem:
    def __init__(self, mic_pos, fs=16000, nfft=512, overlap=0.5, epsilon=0.2, d_freq=2, freq_range=(200, 3000)):
        """
        Args:
            mic_pos: (3, M) numpy array of microphone positions.
            fs: Sampling frequency.
            nfft: FFT size.
            overlap: Overlap fraction (0 to 1).
            epsilon: SSZ detection threshold (lambda2 / lambda1 < epsilon).
            d_freq: Number of frequency components to group/average per zone.
            freq_range: Tuple (min_freq, max_freq) in Hz to process.
        """
        self.mic_pos = mic_pos
        self.fs = fs
        self.nfft = nfft
        self.overlap = overlap
        self.epsilon = epsilon
        self.d_freq = d_freq
        self.freq_range = freq_range
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
                
                # Compute Spatial Correlation Matrix R (averaged over the d freq bins)
                # R = (1/d) * sum(x * x^H)
                R = np.zeros((M, M), dtype=complex)
                for k in range(self.d_freq):
                    vec = block[:, k][:, np.newaxis] # (M, 1)
                    R += vec @ vec.conj().T
                R /= self.d_freq
                
                # SSZ Detection
                # Eigen decomposition
                # eigh returns eigenvalues in ascending order
                eigvals = eigh(R, eigvals_only=True)
                # eigvals are sorted ascending: lambda_0 <= lambda_1 ... <= lambda_{M-1}
                # We want lambda_{M-2} / lambda_{M-1} < epsilon
                # (Ratio of second largest to largest)
                
                if eigvals[-1] > 1e-10: # Avoid division by zero/noise
                    ratio = eigvals[-2] / eigvals[-1]
                    if ratio < self.epsilon:
                        # It is an SSZ. Estimate DOA.
                        # CICS: maximize coherent sum of phase rotated cross-power spectra.
                        # Actually for a single zone, this is equivalent to finding the peak 
                        # of the beamformer spectrum or SRP.
                        
                        # Frequencies in this zone
                        freqs = f_vec[f_idx : f_idx + self.d_freq]
                        
                        best_angle = self._cics_doa(R, freqs, search_angles)
                        ssz_doas.append(best_angle)

        # 2. Histogram Construction
        # Simple histogram
        hist, edges = np.histogram(ssz_doas, bins=360, range=(0, 2*np.pi))
        
        # Smooth histogram? The prompt mentions "matching pursuit loop w blackman window atoms".
        # This implies we use the MP on the histogram itself.
        
        # 3. Matching Pursuit
        final_doas = self._matching_pursuit(hist, edges)
        
        return final_doas, hist

    def _cics_doa(self, R, freqs, angles):
        """
        Find angle phi that maximizes coherent sum.
        """
        M = self.mic_pos.shape[1]
        num_angles = len(angles)
        
        # CICS function: P(phi) = Sum_{i,j} | R_ij | * cos( phase(R_ij) - omega * tau_ij(phi) )
        # OR simply Beamformer Output Power: w^H R w
        # The prompt says: "maximize coherent sum of phase rotated cross-power spectra"
        # This usually means: Sum_{i!=j} R_ij * exp(-j * omega * tau_ij)
        # We can sum this over the 'd' frequencies.
        
        # Let's vectorize over angles.
        
        # Mic positions (2D)
        pos_x = self.mic_pos[0, :]
        pos_y = self.mic_pos[1, :]
        
        # Direction vectors for all search angles
        # u = [cos(phi), sin(phi)]
        u_x = np.cos(angles)
        u_y = np.sin(angles)
        
        cost = np.zeros(num_angles)
        
        # Precompute TDOAs for all pairs is expensive? 
        # Let's optimize: sum over pairs.
        
        # We can iterate over frequencies in the zone
        for k, f_hz in enumerate(freqs):
            omega = 2 * np.pi * f_hz
            k_wave = omega / self.c
            
            # Extract R for this freq (Wait, R passed in is Averaged R)
            # If R is averaged, we can't strictly apply different phase shifts for different freqs 
            # effectively unless we assume narrowband or center freq.
            # "d=2" is small, so we can use the center frequency of the zone.
            pass
            
        # Use center frequency of the zone for the phase shift approximation
        center_freq = np.mean(freqs)
        omega = 2 * np.pi * center_freq
        k_wave = omega / self.c
        
        # SRP-PHAT style or just SRP on R
        # P(phi) = sum_{i, j} R_ij * exp(-j * k * (r_i - r_j) . u)
        
        # Vectorized implementation:
        # P(phi) = v(phi)^H * R * v(phi)  where v is steering vector
        # v_i(phi) = exp(-j * k * (x_i cos phi + y_i sin phi))
        
        # Calculate exponents for all mics and angles: (M, NumAngles)
        # pos (2, M) . u (2, NumAngles) -> (M, NumAngles)
        proj = pos_x[:, np.newaxis] * u_x + pos_y[:, np.newaxis] * u_y
        phases = 1j * k_wave * proj
        steering_vecs = np.exp(phases) # (M, A)
        
        # P(phi) = sum_phi | v_phi^H * R * v_phi |
        #        = sum_phi real(sum_i sum_j v_i* R_ij v_j)
        
        # Optimize: P = sum( (v^H R) .* v^T, axis=0 ) ?
        # R @ steering_vecs -> (M, A)
        temp = R @ steering_vecs
        # Element-wise multiply by conj(steering_vecs) and sum over M
        P = np.sum(np.conj(steering_vecs) * temp, axis=0)
        
        best_idx = np.argmax(np.abs(P))
        return angles[best_idx]

    def _matching_pursuit(self, hist, edges):
        """
        Iterative matching pursuit on the histogram.
        Atoms: Blackman window.
        """
        # Create a Blackman window atom
        # We need to define the width of the atom. 
        # Pavlidi 2013: "The width of the window is determined by the array aperture"
        # Let's pick a reasonable width, e.g., 30 degrees or similar.
        # Or better, define it in bins.
        
        current_hist = hist.astype(float)
        detected_sources = []
        
        # Atom definition
        # Approx width of main lobe. For circular array, maybe 20-40 degrees?
        # Let's say width is ~40 degrees -> ~40 bins if 360 bins.
        atom_width = 40
        atom = np.blackman(atom_width)
        
        # Iterate
        max_iter = 10 # Safety break
        for _ in range(max_iter):
            peak_idx = np.argmax(current_hist)
            peak_val = current_hist[peak_idx]
            
            if peak_val < 5: # Threshold to stop (noise floor)
                break
                
            # Convert bin index to angle
            # edges has 361 points for 360 bins
            angle = (edges[peak_idx] + edges[peak_idx+1]) / 2
            detected_sources.append(angle)
            
            # Subtract atom
            # We need to handle circular wrap-around
            indices = np.arange(peak_idx - atom_width//2, peak_idx - atom_width//2 + atom_width)
            wrap_indices = indices % len(current_hist)
            
            # Scale atom
            scaled_atom = atom * peak_val
            
            # Subtract (relu to keep positive?)
            current_hist[wrap_indices] = np.maximum(0, current_hist[wrap_indices] - scaled_atom)
            
        return detected_sources

import numpy as np
import scipy.signal as signal
from scipy.linalg import eigh

class SSZLocalization:
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
        ssz_history = [] # List of (time, angle)

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
                    ssz_history.append((t_vec[t_idx], best_angle))

        # 2. Histogram Construction
        # Simple histogram
        hist, edges = np.histogram(ssz_doas, bins=360, range=(0, 2*np.pi))
        
        # 3. Matching Pursuit (Updated)
        final_doas = self._matching_pursuit(hist, edges)
        
        return final_doas, hist, ssz_history

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


class GMDALaplace:
    def __init__(self, mic_pos, fs=16000, nfft=512, overlap=0.5, 
                 freq_range=(200, 3000), max_sources=4,
                 power_thresh_percentile=90, mdl_beta=3.0,
                 mic_type_is_circular=True):
        """
        Implementation of Zhang & Rao (2010) GMDA-Laplace localization.
        
        Args:
            mic_pos: (3, M) numpy array of microphone positions.
            fs: Sampling frequency.
            nfft: FFT size.
            overlap: Overlap fraction (0 to 1).
            freq_range: Tuple (min_freq, max_freq) in Hz.
            max_sources: Max sources to consider for MDL.
            power_thresh_percentile: Percentile for power thresholding (e.g. 90 = top 10%).
            mdl_beta: Penalty factor for MDL (> 0.5).
        """
        self.mic_pos = mic_pos
        self.fs = fs
        self.nfft = nfft
        self.overlap = overlap
        self.freq_range = freq_range
        self.max_sources = max_sources
        self.power_thresh_percentile = power_thresh_percentile
        self.mdl_beta = mdl_beta
        self.c = 343.0
        self.mic_type_is_circular = mic_type_is_circular
        
    def process(self, audio):
        """
        Process multichannel audio to find sources using GMDA-Laplace.
        
        Returns:
            estimated_doas: List of estimated angles (radians).
            histogram: (Dummy) Angular histogram for viz compatibility.
            history: (Dummy) History for viz compatibility.
        """
        # 1. STFT
        f_vec, t_vec, Zxx = signal.stft(audio, fs=self.fs, nperseg=self.nfft, 
                                       noverlap=int(self.nfft * self.overlap))
        # Zxx: (M, F, T)
        M_mics, F, T = Zxx.shape
        
        # Frequency Mask
        f_min, f_max = self.freq_range
        f_mask = (f_vec >= f_min) & (f_vec <= f_max)
        f_indices = np.where(f_mask)[0]
        
        # 2. Data Selection (High SNR)
        # Power of ref mic (0)
        power = np.abs(Zxx[0, f_mask, :])**2 # (F_sub, T)
        if power.size == 0:
            return [], np.zeros(360), []

        threshold = np.percentile(power, self.power_thresh_percentile)
        mask_snr = power >= threshold # (F_sub, T) boolean
        
        indices_f_sub, indices_t = np.where(mask_snr) 
        
        obs_Y = []
        obs_Omega = []
        
        for i in range(len(indices_f_sub)):
            f_idx = f_indices[indices_f_sub[i]]
            t_idx = indices_t[i]
            
            omega = 2 * np.pi * f_vec[f_idx]
            if omega == 0: continue

            # Phase diffs relative to mic 0
            ref_spec = Zxx[0, f_idx, t_idx]
            y_i = []
            for m in range(1, M_mics):
                mic_spec = Zxx[m, f_idx, t_idx]
                phase_diff = np.angle(ref_spec * np.conjugate(mic_spec))
                y_i.append(phase_diff)
            
            obs_Y.append(y_i)
            obs_Omega.append(omega)
            
        obs_Y = np.array(obs_Y) # (N_obs, M-1)
        obs_Omega = np.array(obs_Omega) # (N_obs,)
        N_obs = len(obs_Y)
        
        if N_obs == 0:
            return [], np.zeros(360), []
            
        print(f"GMDA: Selected {N_obs} TF points.")

        # 3. Initialization using ITD Histogram (via SRP/Grid Search for robustness)
        # We need to propose initial slopes for candidate sources.
        # Paper says "Convert IPDs to ITDs, bin them".
        # We'll generate a histogram of DOAs using a simple SRP-like method on the selected points.
        
        search_angles = np.linspace(0, 2*np.pi, 72, endpoint=False) # 5 deg resolution for init
        init_hist = np.zeros(len(search_angles))
        
        # Precompute delays for grid: (M-1, A)
        grid_delays = np.zeros((M_mics-1, len(search_angles)))
        for m in range(1, M_mics):
            diff = self.mic_pos[:, m] - self.mic_pos[:, 0] # Vector from 0 to m
            for ai, ang in enumerate(search_angles):
                u = np.array([np.cos(ang), np.sin(ang), 0])
                # Delay tau_1m = - (r_m - r_1) . u / c
                grid_delays[m-1, ai] = -np.dot(diff, u) / self.c
        
        # Populate histogram (Accumulate consistency)
        # For each obs, find closest angle
        for i in range(N_obs):
            # Cost: sum | wrap(psi - omega * delay) |
            # Vectorized over angles
            psi = obs_Y[i][:, np.newaxis] # (M-1, 1)
            omega = obs_Omega[i]
            pred_phase = omega * grid_delays # (M-1, A)
            
            # Wrap error to [-pi, pi]
            err = np.angle(np.exp(1j * (psi - pred_phase)))
            cost = np.sum(np.abs(err), axis=0) # (A,)
            
            best_ang_idx = np.argmin(cost)
            init_hist[best_ang_idx] += 1
            
        # Smooth histogram
        init_hist = np.convolve(np.pad(init_hist, 2, 'wrap'), np.ones(3)/3, 'same')[2:-2]

        # Find peaks
        # Simple peak finding
        peaks = []
        for i in range(len(init_hist)):
            prev = init_hist[(i-1)%len(init_hist)]
            curr = init_hist[i]
            next_val = init_hist[(i+1)%len(init_hist)]
            if curr > prev and curr > next_val:
                peaks.append((curr, i))
        
        peaks.sort(key=lambda x: x[0], reverse=True)
        candidate_indices = [p[1] for p in peaks[:self.max_sources]]
        
        # 4. Model Selection (MDL) loop
        best_model = None
        best_mdl_score = -np.inf
        
        # Test m from 1 to max_sources (or len(peaks))
        limit_sources = min(len(peaks), self.max_sources)
        if limit_sources == 0: limit_sources = 1
        
        # Pre-convert candidate angles to slopes for initialization
        # Slopes alpha_{k} correspond to delay tau_{1,k}
        # We need a set of slopes for each source.
        
        all_candidate_slopes = [] # List of (M-1,) arrays
        for idx in candidate_indices:
             all_candidate_slopes.append(grid_delays[:, idx])
        
        # Fallback if no peaks
        while len(all_candidate_slopes) < self.max_sources:
             all_candidate_slopes.append(np.zeros(M_mics-1))

        for m_sources in range(1, limit_sources + 1):
            # Init Parameters
            # alpha: (m_sources, M-1)
            alpha = np.array(all_candidate_slopes[:m_sources])
            
            # b: (m_sources,) variance
            # Init b somewhat large? Paper doesn't specify.
            # E[|x|] = b. Residuals roughly uniform in [-pi, pi] -> avg |x| ~ pi/2?
            # Start with 1.0
            b = np.ones(m_sources)
            
            # priors: (m_sources,)
            pi_mix = np.ones(m_sources) / m_sources
            
            # EM Loop
            max_iter = 20
            for iter_num in range(max_iter):
                # E-step
                # P(j|i) propto pi_j * prod_k p(y_ik | j)
                # log P_unnorm(j, i) = log pi_j - (M-1) log(2b_j) - sum_k |err|/b_j
                
                log_probs = np.zeros((N_obs, m_sources))
                
                for j in range(m_sources):
                    # Calc error for source j
                    # err: (N_obs, M-1)
                    # wrap(psi - alpha * omega)
                    
                    pred = np.outer(obs_Omega, alpha[j]) # (N, M-1)
                    diff = obs_Y - pred
                    # Wrap
                    err = np.angle(np.exp(1j * diff))
                    abs_err = np.abs(err)
                    sum_abs_err = np.sum(abs_err, axis=1) # (N,)
                    
                    log_probs[:, j] = np.log(pi_mix[j] + 1e-10) \
                                      - (M_mics - 1) * np.log(2 * b[j] + 1e-10) \
                                      - sum_abs_err / (b[j] + 1e-10)
                
                # Normalize via log-sum-exp
                max_log = np.max(log_probs, axis=1, keepdims=True)
                log_probs -= max_log
                probs = np.exp(log_probs)
                probs /= np.sum(probs, axis=1, keepdims=True) # (N, m_src)
                
                # M-step
                
                # Update priors
                N_j = np.sum(probs, axis=0) # (m_src,)
                pi_mix = N_j / N_obs
                
                # Update b
                # b_j = (1 / (N * (M-1) * pi_j)) * sum_i P(j|i) sum_k |err|
                for j in range(m_sources):
                    pred = np.outer(obs_Omega, alpha[j])
                    diff = obs_Y - pred
                    err = np.angle(np.exp(1j * diff))
                    sum_abs_err = np.sum(np.abs(err), axis=1)
                    
                    weighted_err = np.sum(probs[:, j] * sum_abs_err)
                    denom = N_obs * (M_mics - 1) * pi_mix[j]
                    b[j] = weighted_err / (denom + 1e-10)
                
                # Update alpha using Newton (IRLS)
                for j in range(m_sources):
                    # For each mic pair k, update alpha_{j,k}
                    # We can update them independently as cross-terms don't exist in likelihood (diagonal cov)
                    
                    for k in range(M_mics - 1):
                        # Data for this regression: obs_Y[:, k], obs_Omega[:]
                        # Weights: probs[:, j]
                        
                        self._newton_update(alpha, j, k, obs_Y[:, k], obs_Omega, probs[:, j])

            # Calculate MDL
            # Log Likelihood
            # L = sum_i log( sum_j pi_j p(y_i|j) )
            # We can recompute this or approx
            
            final_log_probs = np.zeros((N_obs, m_sources))
            for j in range(m_sources):
                 pred = np.outer(obs_Omega, alpha[j])
                 diff = obs_Y - pred
                 err = np.angle(np.exp(1j * diff))
                 sum_abs_err = np.sum(np.abs(err), axis=1)
                 
                 term = np.log(pi_mix[j] + 1e-10) - (M_mics - 1)*np.log(2*b[j] + 1e-10) - sum_abs_err/(b[j]+1e-10)
                 final_log_probs[:, j] = term
            
            # Sum over sources in log domain
            # log(sum exp(x))
            m_max = np.max(final_log_probs, axis=1)
            ll_per_point = m_max + np.log(np.sum(np.exp(final_log_probs - m_max[:, np.newaxis]), axis=1))
            total_ll = np.sum(ll_per_point)
            
            # Penalty
            # params = m*(M_mics + 1) - 1
            n_params = m_sources * ( (M_mics - 1) + 2 ) - 1
            # Paper says beta * k * ln N
            penalty = self.mdl_beta * n_params * np.log(N_obs)
            
            mdl = total_ll - penalty
            
            if mdl > best_mdl_score:
                best_mdl_score = mdl
                best_model = {
                    'alpha': alpha.copy(),
                    'b': b.copy(),
                    'pi': pi_mix.copy(),
                    'm': m_sources
                }
        
        # 5. Extract DOAs from best model
        final_doas = []
        if best_model is not None:
            alphas = best_model['alpha'] # (m, M-1)
            
            # Use finer grid for final readout
            fine_angles = np.linspace(0, 2*np.pi, 720, endpoint=False)
            fine_delays = np.zeros((M_mics-1, len(fine_angles)))
            for m in range(1, M_mics):
                diff = self.mic_pos[:, m] - self.mic_pos[:, 0]
                for ai, ang in enumerate(fine_angles):
                    u = np.array([np.cos(ang), np.sin(ang), 0])
                    fine_delays[m-1, ai] = -np.dot(diff, u) / self.c
            
            # Convert slopes back to DOAs
            # Minimize error between alpha and grid_delays
            for j in range(best_model['m']):
                if best_model['pi'][j] < 0.1: 
                    # Ignore this detection as it doesn't represent enough data
                    continue    
                src_alpha = alphas[j] # (M-1,)
                
                # Check against grid
                # fine_delays: (M-1, A)
                dist = np.sum((fine_delays - src_alpha[:, np.newaxis])**2, axis=0)
                best_ang_idx = np.argmin(dist)
                final_doas.append(fine_angles[best_ang_idx])
        
        # Viz data (just return init histogram for debug)
        viz_hist = init_hist if len(init_hist) == 360 else np.interp(np.linspace(0, 72, 360), np.arange(72), init_hist)
        
        return final_doas, viz_hist, []

    def _newton_update(self, alpha, j, k, y_vec, omega_vec, weights):
        """
        Update alpha_{j,k} using Newton/IRLS for L1 regression.
        Minimize sum_i w_i |wrap(y_i - alpha * omega_i)|
        """
        # Current estimate
        curr_alpha = alpha[j, k]
        
        # IRLS Loop (just 1 or 2 steps is usually enough per EM iter)
        for _ in range(2):
            # Calculate residual
            # Unwrapped target estimate:
            # We want y_i ~ alpha * omega.
            # Residual r = wrap(y - alpha*omega)
            # Linearized target Y_lin = alpha*omega + r
            
            pred = curr_alpha * omega_vec
            diff = y_vec - pred
            r = np.angle(np.exp(1j * diff))
            
            # Weight for L1: W' = W / |r|
            # Regularize |r| to avoid div by zero
            denom = np.abs(r)
            denom[denom < 1e-6] = 1e-6
            
            W_prime = weights / denom
            
            # Weighted Least Squares
            # Minimize sum W' (r)^2  -> sum W' (Y_lin - a*w)^2 ?
            # Wait, r is the error.
            # We want to find shift da such that r_new ~ 0.
            # r_new = r_old - da * omega
            # Minimize sum W' (r_old - da * omega)^2
            
            # da = sum(W' * r_old * omega) / sum(W' * omega^2)
            
            num = np.sum(W_prime * r * omega_vec)
            den = np.sum(W_prime * omega_vec**2)
            
            if den < 1e-10:
                break
                
            d_alpha = num / den
            curr_alpha += d_alpha
            
        alpha[j, k] = curr_alpha


class SRPPHATLocalization:
    def __init__(self, mic_pos, fs=16000, nfft=512, overlap=0.5, 
                 freq_range=(200, 3000), max_sources=4,
                 **kwargs):
        """
        Implementation of SRP-PHAT with SNR-based weighting.
        
        Args:
            mic_pos: (3, M) numpy array of microphone positions.
            fs: Sampling frequency.
            nfft: FFT size.
            overlap: Overlap fraction (0 to 1).
            freq_range: Tuple (min_freq, max_freq) in Hz.
            max_sources: Number of peaks to find.
        """
        self.mic_pos = mic_pos
        self.fs = fs
        self.nfft = nfft
        self.overlap = overlap
        self.freq_range = freq_range
        self.max_sources = max_sources
        self.c = 343.0
        
    def process(self, audio):
        """
        Process multichannel audio to find sources using SRP-PHAT.
        
        Args:
            audio: (M, N) numpy array of multichannel audio.
            
        Returns:
            estimated_doas: List of estimated angles (radians).
            histogram: Angular power spectrum P(theta).
            history: Dummy history.
        """
        M_mics, N_samples = audio.shape
        
        # 1. STFT
        f_vec, t_vec, Zxx = signal.stft(audio, fs=self.fs, nperseg=self.nfft, 
                                       noverlap=int(self.nfft * self.overlap))
        # Zxx: (M, F, T)
        
        # 2. Frequency Masking & Selection
        f_min, f_max = self.freq_range
        f_mask = (f_vec >= f_min) & (f_vec <= f_max)
        relevant_freqs = f_vec[f_mask]
        Zxx_roi = Zxx[:, f_mask, :] # (M, F_roi, T)
        
        if Zxx_roi.shape[1] == 0:
            return [], np.zeros(360), []

        # 3. Calculate Weighting W(f) based on SNR
        # Estimate Noise Floor: Average power of quietest 10% of frames
        # Compute frame energy
        frame_energy = np.sum(np.sum(np.abs(Zxx_roi)**2, axis=0), axis=0) # (T,)
        threshold_energy = np.percentile(frame_energy, 10)
        noise_frames_mask = frame_energy <= threshold_energy
        
        # If we have no noise frames (unlikely unless constant sound), use bottom 1%
        if np.sum(noise_frames_mask) == 0:
            threshold_energy = np.percentile(frame_energy, 1)
            noise_frames_mask = frame_energy <= threshold_energy
            
        # Noise Spectrum N(f): Average over noise frames, average over mics
        # (F_roi,)
        if np.sum(noise_frames_mask) > 0:
            noise_spec = np.mean(np.mean(np.abs(Zxx_roi[:, :, noise_frames_mask])**2, axis=2), axis=0)
        else:
            # Fallback: estimate from min across time
            noise_spec = np.min(np.mean(np.abs(Zxx_roi)**2, axis=0), axis=1)

        # Signal Spectrum S(f): Average over active frames
        # Active frames: e.g. top 50% energy
        active_thresh = np.percentile(frame_energy, 50)
        active_mask = frame_energy >= active_thresh
        
        if np.sum(active_mask) == 0:
            # Fallback to all frames
            active_mask = np.ones_like(frame_energy, dtype=bool)

        signal_spec = np.mean(np.mean(np.abs(Zxx_roi[:, :, active_mask])**2, axis=2), axis=0)
        
        # SNR Weighting W(f)
        # Simple Wiener-like or binary: if Signal > 2 * Noise -> 1, else 0
        snr_ratio = signal_spec / (noise_spec + 1e-10)
        W_f = np.zeros_like(relevant_freqs)
        W_f[snr_ratio > 2.0] = 1.0 # Hard threshold or soft? Instructions: "zeroed out" -> implies hard.
        
        # Frequency weighting to emphasize higher frequencies (better resolution)
        W_f *= (relevant_freqs**2)
        W_f /= (np.max(W_f) + 1e-10) # Normalize


        
        # 4. GCC-PHAT Calculation (Averaged over active frames)
        # We need the Cross-Spectrum Matrix averaged over time
        # R_ij(f) = sum_t (X_i X_j* / |X_i X_j*|)
        
        n_pairs = M_mics * (M_mics - 1) // 2
        pairs = []
        for i in range(M_mics):
            for j in range(i + 1, M_mics):
                pairs.append((i, j))
        
        # Pre-allocate averaged GCC: (n_pairs, F_roi)
        avg_GCC = np.zeros((n_pairs, len(relevant_freqs)), dtype=complex)
        
        active_indices = np.where(active_mask)[0]
        
        # We can vectorize over active frames?
        # Memory check: M=4, F=200, T=500 -> Small.
        X_active = Zxx_roi[:, :, active_indices] # (M, F, T_active)
        
        for p_idx, (i, j) in enumerate(pairs):
            Xi = X_active[i]
            Xj = X_active[j]
            prod = Xi * np.conj(Xj)
            denom = np.abs(prod)
            denom[denom < 1e-10] = 1e-10
            
            # PHAT
            R_inst = prod / denom # (F, T)
            
            # Average over time
            avg_GCC[p_idx, :] = np.mean(R_inst, axis=1)

        # 5. SRP Summation
        # P(theta) = sum_pair sum_f W(f) * Real(GCC(f) * exp(j * 2pi * f * tau))
        
        search_angles = np.linspace(0, 2*np.pi, 360, endpoint=False)
        
        # Calculate Tau for all pairs and angles
        # tau_ij(theta) = (p_i - p_j) . u(theta) / c
        # Delays in seconds
        
        delays = np.zeros((n_pairs, len(search_angles)))
        for p_idx, (i, j) in enumerate(pairs):
            # Vector pointing from i to j: r_j - r_i
            # Delay tau_ij = t_i - t_j = (r_j - r_i) . u / c
            diff = self.mic_pos[:, j] - self.mic_pos[:, i]
            
            for a_idx, ang in enumerate(search_angles):
                u = np.array([np.cos(ang), np.sin(ang), 0])
                delays[p_idx, a_idx] = np.dot(diff, u) / self.c

        # Computation
        # We need sum_f W(f) * Real( GCC_pair(f) * exp(j 2pi f tau) )
        
        # Vectorized:
        # avg_GCC: (P, F)
        # W_f: (F,)
        # delays: (P, A)
        # relevant_freqs: (F,)
        
        # Construct exponent: (P, F, A)
        omega = 2 * np.pi * relevant_freqs # (F,)
        
        # Phase terms: omega * tau
        phase = omega[np.newaxis, :, np.newaxis] * delays[:, np.newaxis, :]
        steer_vec = np.exp(1j * phase)
        
        # Apply SNR weights to GCC
        # weighted_GCC: (P, F, 1)
        weighted_GCC = (avg_GCC * W_f[np.newaxis, :])[:, :, np.newaxis]
        
        # Element-wise multiply and sum
        # term: (P, F, A)
        term = weighted_GCC * steer_vec
        
        # Sum over F and P
        # Real part
        P_theta = np.sum(np.sum(np.real(term), axis=1), axis=0) # (A,)
        
        # Normalize P_theta for "histogram" look
        # Clip negative values (sidelobes) to zero instead of lifting everything
        P_theta[P_theta < 0] = 0
        
        if np.max(P_theta) > 0:
            P_theta = P_theta / np.max(P_theta)
            
        # 6. Peak Finding (Simple greedy with Ghost Suppression)
        # Find local maxima
        
        peaks = []
        # Circular check
        for i in range(len(P_theta)):
            prev = P_theta[(i-1)%len(P_theta)]
            curr = P_theta[i]
            next_val = P_theta[(i+1)%len(P_theta)]
            if curr > prev and curr >= next_val:
                peaks.append((curr, i))
                
        peaks.sort(key=lambda x: x[0], reverse=True)
        
        final_doas = []
        accepted_sources = [] # List of (angle_deg, amplitude)
        
        # Ghost suppression parameters
        # Threshold: if peak < 0.85 * parent_peak and at 180 deg, ignore it.
        # This works because we improved weighting to reduce ghost ratio to ~0.6.
        suppression_threshold = 0.85 
        ghost_angle_tol = 25.0 # degrees tolerance
        
        for p_val, p_idx in peaks:
            if len(final_doas) >= self.max_sources:
                break
                
            angle = search_angles[p_idx]
            angle_deg = np.degrees(angle)
            
            is_ghost = False
            for acc_ang, acc_amp in accepted_sources:
                # Check if this peak is a ghost of an already accepted source
                # Ghost location is 180 degrees from source
                ghost_loc = (acc_ang + 180) % 360
                
                # Circular distance
                diff = abs(angle_deg - ghost_loc)
                if diff > 180: diff = 360 - diff
                
                if diff < ghost_angle_tol:
                    # It is spatially close to a ghost location.
                    # Check amplitude ratio.
                    if p_val < suppression_threshold * acc_amp:
                        is_ghost = True
                        break
            
            if not is_ghost:
                final_doas.append(angle)
                accepted_sources.append((angle_deg, p_val))

        # 7. Generate Time-History for Visualization
        # Re-run SRP per frame for the history plot
        history = []
        
        # We use a coarser grid or fewer frames if T is very large, but T ~ 500 is fine.
        # Construct exponent for all search angles
        omega = 2 * np.pi * relevant_freqs
        phase = omega[np.newaxis, :, np.newaxis] * delays[:, np.newaxis, :]
        steer_vec = np.exp(1j * phase)
        
        # Process each time frame
        for t_idx in range(len(t_vec)):
            # Check if frame is active
            if not active_mask[t_idx]:
                continue
            
            # Instantaneous GCC-PHAT
            X_inst = Zxx_roi[:, :, t_idx] # (M, F)
            frame_GCC = np.zeros((n_pairs, len(relevant_freqs)), dtype=complex)
            for p_idx, (i, j) in enumerate(pairs):
                prod = X_inst[i] * np.conj(X_inst[j])
                denom = np.abs(prod)
                denom[denom < 1e-10] = 1e-10
                frame_GCC[p_idx, :] = prod / denom
            
            # Apply frequency weighting
            weighted_GCC = (frame_GCC * W_f[np.newaxis, :])[:, :, np.newaxis]
            
            # SRP for this frame
            P_frame = np.sum(np.sum(np.real(weighted_GCC * steer_vec), axis=1), axis=0)
            
            if np.max(P_frame) > 0:
                best_idx = np.argmax(P_frame)
                history.append((t_vec[t_idx], search_angles[best_idx]))
            
        return final_doas, P_theta, history

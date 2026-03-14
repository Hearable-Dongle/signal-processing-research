import numpy as np
import pyroomacoustics as pra
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
        Implementation of base SRP-PHAT.
        
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
        self.accumulation_sec = float(kwargs.get("accumulation_sec", 0.5))
        self.min_active_frames = int(kwargs.get("min_active_frames", 3))
        self.rms_floor = float(kwargs.get("rms_floor", 5e-4))
        self.speech_ratio_threshold = float(kwargs.get("speech_ratio_threshold", 0.62))
        self.rms_ratio_threshold = float(kwargs.get("rms_ratio_threshold", 1.2))
        self.flux_threshold = float(kwargs.get("flux_threshold", 0.12))
        self.noise_floor_alpha = float(kwargs.get("noise_floor_alpha", 0.95))
        self.pair_selection_mode = str(kwargs.get("pair_selection_mode", "all"))

    def _selected_pairs(self, m_mics: int) -> list[tuple[int, int]]:
        pairs = [(i, j) for i in range(m_mics) for j in range(i + 1, m_mics)]
        if self.pair_selection_mode != "adjacent_only" or len(pairs) <= 1:
            return pairs
        pair_distances = []
        for i, j in pairs:
            diff = np.asarray(self.mic_pos[:, i] - self.mic_pos[:, j], dtype=np.float64)
            pair_distances.append((float(np.linalg.norm(diff)), (i, j)))
        min_distance = min(distance for distance, _pair in pair_distances)
        tol = max(1e-6, min_distance * 0.05)
        return [pair for distance, pair in pair_distances if distance <= min_distance + tol]

    def _speech_features(self, chunk: np.ndarray, prev_mag: np.ndarray | None, noise_floor: float):
        mono = np.mean(chunk, axis=0)
        rms = float(np.sqrt(np.mean(np.square(mono)) + 1e-12))
        spec = np.abs(np.fft.rfft(mono * np.hanning(mono.size)))
        freqs = np.fft.rfftfreq(mono.size, d=1.0 / self.fs)
        speech_mask = (freqs >= 250.0) & (freqs <= 3500.0)
        total_energy = float(np.sum(spec**2) + 1e-12)
        speech_energy = float(np.sum(spec[speech_mask] ** 2))
        speech_ratio = speech_energy / total_energy
        if prev_mag is None:
            flux = 0.0
        else:
            flux = float(np.mean(np.maximum(spec - prev_mag, 0.0)) / (np.mean(prev_mag) + 1e-12))
        rms_ratio = rms / max(noise_floor, 1e-8)
        speech_active = bool(
            rms > self.rms_floor
            and speech_ratio >= self.speech_ratio_threshold
            and (rms_ratio >= self.rms_ratio_threshold or flux >= self.flux_threshold)
        )
        return speech_active, rms, spec

    def _update_noise_floor(self, noise_floor: float, rms: float, speech_active: bool) -> float:
        if not np.isfinite(noise_floor) or noise_floor <= 0.0:
            return rms
        if speech_active:
            return float(noise_floor)
        return float(self.noise_floor_alpha * noise_floor + (1.0 - self.noise_floor_alpha) * rms)
        
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
        
        # NOTE: This intentionally gives up parity with the previous scene-level
        # SRP-PHAT implementation in favor of matching the validated
        # debug_localization winner: MSC-weighted SRP-PHAT with speech-active
        # accumulation. That path was materially better under directional noise.
        noverlap = min(int(round(self.nfft * self.overlap)), self.nfft - 1)
        f_vec, t_vec, Zxx = signal.stft(
            audio,
            fs=self.fs,
            nperseg=self.nfft,
            noverlap=noverlap,
            boundary=None,
            padded=False,
        )

        f_min, f_max = self.freq_range
        f_mask = (f_vec >= f_min) & (f_vec <= f_max)
        relevant_freqs = f_vec[f_mask].astype(float)
        Zxx_roi = Zxx[:, f_mask, :]

        if Zxx_roi.shape[1] == 0:
            return [], np.zeros(360), []
        pairs = self._selected_pairs(M_mics)
        pair_freq_masks: dict[tuple[int, int], np.ndarray] = {}
        for i, j in pairs:
            diff = np.asarray(self.mic_pos[:, i] - self.mic_pos[:, j], dtype=np.float64)
            pair_distance_m = float(np.linalg.norm(diff))
            if pair_distance_m <= 1e-9:
                pair_freq_masks[(i, j)] = np.ones_like(relevant_freqs, dtype=bool)
                continue
            pair_alias_hz = float(self.c / (2.0 * pair_distance_m))
            pair_freq_masks[(i, j)] = relevant_freqs <= pair_alias_hz
        search_angles = np.linspace(0, 2 * np.pi, 360, endpoint=False)
        dirs = np.stack([np.cos(search_angles), np.sin(search_angles), np.zeros_like(search_angles)], axis=1)
        frame_specs = []
        frame_times = []
        frame_active = []
        prev_mag = None
        noise_floor = 0.0
        accum_frames = max(1, int(round(self.accumulation_sec / max((1.0 - self.overlap) * self.nfft / self.fs, 1e-6))))

        for t_idx in range(Zxx_roi.shape[2]):
            start = int(round(t_idx * (self.nfft - noverlap)))
            stop = min(start + self.nfft, N_samples)
            if stop - start <= 8:
                continue
            chunk = audio[:, start:stop]
            speech_active, rms, prev_mag = self._speech_features(chunk, prev_mag, noise_floor)
            noise_floor = self._update_noise_floor(noise_floor, rms, speech_active)
            frame_spec = np.zeros(search_angles.shape[0], dtype=float)
            for i, j in pairs:
                pair_mask = pair_freq_masks[(i, j)]
                if not np.any(pair_mask):
                    continue
                cross = Zxx_roi[i, :, t_idx] * np.conj(Zxx_roi[j, :, t_idx])
                phat = cross / np.maximum(np.abs(cross), 1e-10)
                auto_i = np.abs(Zxx_roi[i, :, t_idx]) ** 2
                auto_j = np.abs(Zxx_roi[j, :, t_idx]) ** 2
                msc = (np.abs(cross) ** 2) / np.maximum(auto_i * auto_j, 1e-10)
                msc = np.clip(msc, 0.0, 1.0)
                if np.max(msc) > 1e-10:
                    msc = msc / np.max(msc)
                phat = np.where(pair_mask, phat, 0.0)
                msc = np.where(pair_mask, msc, 0.0)
                diff = self.mic_pos[:, i] - self.mic_pos[:, j]
                tau = (dirs @ diff) / self.c  # (A,)
                phase = 2 * np.pi * relevant_freqs[:, np.newaxis] * tau[np.newaxis, :]
                steered = np.real(phat[:, np.newaxis] * np.exp(-1j * phase))
                frame_spec += np.sum(steered * msc[:, np.newaxis], axis=0)
            frame_specs.append(frame_spec)
            frame_times.append(float(t_vec[t_idx]) if t_idx < len(t_vec) else float(start / self.fs))
            frame_active.append(bool(speech_active))

        if not frame_specs:
            return [], np.zeros(360), []

        active_specs = [spec for spec, active in zip(frame_specs, frame_active) if active]
        if len(active_specs) >= self.min_active_frames:
            P_theta = np.mean(np.stack(active_specs, axis=0), axis=0)
        else:
            P_theta = np.mean(np.stack(frame_specs, axis=0), axis=0)
        
        # Normalize P_theta for "histogram" look
        # Clip negative values (sidelobes) to zero instead of lifting everything
        P_theta[P_theta < 0] = 0
        
        if np.max(P_theta) > 0:
            P_theta = P_theta / np.max(P_theta)
            
        peaks = []
        for i in range(len(P_theta)):
            prev = P_theta[(i-1) % len(P_theta)]
            curr = P_theta[i]
            next_val = P_theta[(i+1) % len(P_theta)]
            if curr > prev and curr >= next_val:
                peaks.append((curr, i))
        peaks.sort(key=lambda x: x[0], reverse=True)

        final_doas = []
        for p_val, p_idx in peaks:
            if len(final_doas) >= self.max_sources:
                break
            angle = search_angles[p_idx]
            final_doas.append(angle)

        history = []
        for idx in range(len(frame_specs)):
            start_idx = max(0, idx - accum_frames + 1)
            window_specs = [spec for spec, active in zip(frame_specs[start_idx:idx + 1], frame_active[start_idx:idx + 1]) if active]
            if len(window_specs) < self.min_active_frames:
                continue
            hist_spec = np.mean(np.stack(window_specs, axis=0), axis=0)
            if np.max(hist_spec) > 0.0:
                history.append((frame_times[idx], search_angles[int(np.argmax(hist_spec))]))

        return final_doas, P_theta, history


class CaponLocalization:
    def __init__(
        self,
        mic_pos,
        fs=16000,
        nfft=512,
        overlap=0.5,
        freq_range=(200, 3000),
        max_sources=1,
        **kwargs,
    ):
        self.mic_pos = np.asarray(mic_pos, dtype=np.float64)
        self.fs = int(fs)
        self.nfft = int(nfft)
        self.overlap = float(overlap)
        self.freq_range = tuple(int(v) for v in freq_range)
        self.max_sources = int(max_sources)
        self.c = 343.0
        self.grid_size = int(kwargs.get("grid_size", 360))
        self.diagonal_loading = float(kwargs.get("diagonal_loading", 1e-3))

    def process(self, audio):
        m_mics, _n_samples = audio.shape
        noverlap = min(int(round(self.nfft * self.overlap)), self.nfft - 1)
        f_vec, _t_vec, Zxx = signal.stft(
            audio,
            fs=self.fs,
            nperseg=self.nfft,
            noverlap=noverlap,
            boundary=None,
            padded=False,
        )

        f_min, f_max = self.freq_range
        f_mask = (f_vec >= f_min) & (f_vec <= f_max)
        relevant_freqs = f_vec[f_mask].astype(float)
        Zxx_roi = Zxx[:, f_mask, :]
        if Zxx_roi.shape[1] == 0 or Zxx_roi.shape[2] == 0:
            return [], np.zeros(self.grid_size), []

        search_angles = np.linspace(0, 2 * np.pi, self.grid_size, endpoint=False)
        dirs = np.stack([np.cos(search_angles), np.sin(search_angles), np.zeros_like(search_angles)], axis=1)
        spectrum = np.zeros(self.grid_size, dtype=np.float64)
        eye = np.eye(m_mics, dtype=np.complex128)

        for f_idx, freq_hz in enumerate(relevant_freqs):
            snapshots = np.asarray(Zxx_roi[:, f_idx, :], dtype=np.complex128)
            if snapshots.ndim != 2 or snapshots.shape[1] == 0:
                continue
            cov = (snapshots @ snapshots.conj().T) / max(1, snapshots.shape[1])
            trace_scale = float(np.real(np.trace(cov))) / max(1, m_mics)
            load = max(self.diagonal_loading * max(trace_scale, 1e-8), 1e-8)
            cov_loaded = cov + load * eye
            try:
                cov_inv = np.linalg.pinv(cov_loaded, hermitian=True)
            except TypeError:
                cov_inv = np.linalg.pinv(cov_loaded)

            mic_projections = self.mic_pos.T @ dirs.T  # (M, A)
            tau = mic_projections / float(self.c)
            tau = tau - np.mean(tau, axis=0, keepdims=True)
            steering = np.exp(-1j * 2.0 * np.pi * float(freq_hz) * tau)  # (M, A)
            numerators = np.einsum("ma,mn,na->a", steering.conj(), cov_inv, steering, optimize=True)
            denom = np.maximum(np.real(numerators), 1e-10)
            spectrum += 1.0 / denom

        spectrum = np.asarray(np.real(spectrum), dtype=np.float64)
        spectrum[~np.isfinite(spectrum)] = 0.0
        spectrum[spectrum < 0.0] = 0.0
        vmax = float(np.max(spectrum))
        if vmax > 0.0:
            spectrum /= vmax
        best_idx = int(np.argmax(spectrum)) if spectrum.size else 0
        return [float(search_angles[best_idx])], spectrum, []


class PyroomacousticsDOABase:
    def __init__(
        self,
        mic_pos,
        fs=16000,
        nfft=512,
        overlap=0.5,
        freq_range=(200, 3000),
        max_sources=4,
        grid_size=360,
        num_iter=5,
        method_name="MUSIC",
        frequency_normalization=None,
    ):
        self.mic_pos = mic_pos
        self.fs = fs
        self.nfft = nfft
        self.overlap = overlap
        self.freq_range = freq_range
        self.max_sources = max_sources
        self.grid_size = grid_size
        self.num_iter = num_iter
        self.method_name = method_name
        self.frequency_normalization = frequency_normalization

    def _build_estimator(self):
        doa_cls = getattr(pra.doa, self.method_name)
        search_angles = np.linspace(0, 2 * np.pi, self.grid_size, endpoint=False)
        kwargs = {
            "L": self.mic_pos,
            "fs": self.fs,
            "nfft": self.nfft,
            "num_src": self.max_sources,
            "mode": "far",
            "azimuth": search_angles,
        }

        if self.method_name in {"CSSM", "WAVES"}:
            kwargs["num_iter"] = self.num_iter

        if self.frequency_normalization is not None:
            kwargs["frequency_normalization"] = self.frequency_normalization

        return doa_cls(**kwargs), search_angles

    def process(self, audio):
        _, _, Zxx = signal.stft(
            audio,
            fs=self.fs,
            nperseg=self.nfft,
            noverlap=int(self.nfft * self.overlap),
        )
        # Zxx shape expected by pyroomacoustics DOA: (M, F, snapshots)
        if Zxx.shape[1] == 0 or Zxx.shape[2] == 0:
            return [], np.zeros(self.grid_size), []

        estimator, _ = self._build_estimator()
        try:
            estimator.locate_sources(
                Zxx,
                num_src=self.max_sources,
                freq_range=list(self.freq_range),
            )
        except Exception:
            # Subspace methods can fail on degenerate bins/snapshots (e.g., singular matrix).
            # Keep pipeline robust by returning no detections for this chunk.
            return [], np.zeros(self.grid_size), []

        doa_est = getattr(estimator, "azimuth_recon", None)
        if doa_est is None:
            estimated_doas = []
        else:
            estimated_doas = [float(a % (2 * np.pi)) for a in np.asarray(doa_est).tolist()]

        spectrum = np.asarray(estimator.grid.values, dtype=float).reshape(-1)
        if spectrum.size != self.grid_size:
            x_old = np.linspace(0, 1, spectrum.size, endpoint=False)
            x_new = np.linspace(0, 1, self.grid_size, endpoint=False)
            spectrum = np.interp(x_new, x_old, spectrum)
        if np.max(spectrum) > 0:
            spectrum = spectrum / np.max(spectrum)
        spectrum[spectrum < 0] = 0

        return estimated_doas, spectrum, []


class MUSICLocalization(PyroomacousticsDOABase):
    def __init__(self, **kwargs):
        super().__init__(method_name="MUSIC", frequency_normalization=False, **kwargs)


class NormMUSICLocalization(PyroomacousticsDOABase):
    def __init__(self, **kwargs):
        super().__init__(method_name="NormMUSIC", frequency_normalization=True, **kwargs)


class CSSMLocalization(PyroomacousticsDOABase):
    def __init__(self, **kwargs):
        super().__init__(method_name="CSSM", **kwargs)


class WAVESLocalization(PyroomacousticsDOABase):
    def __init__(self, **kwargs):
        super().__init__(method_name="WAVES", **kwargs)

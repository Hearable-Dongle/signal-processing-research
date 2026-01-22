import numpy as np
from numpy.typing import NDArray


def wng_mvdr_steepest(
    Rnn: NDArray[np.float64],
    steering_vecs: list[NDArray[np.complex128]],
    gamma: float,
    mu: float,
    iteration_count: int,
) -> tuple[NDArray[np.complex128], NDArray[np.float64]]:
    # TODO: Update to handle multiple speakers
    # Create constraint matrix from steering vectors
    C = np.hstack(steering_vecs)

    # Create distortionless beamformer weights in steering vector direction
    weight_vec = C @ np.linalg.pinv(C.conj().T @ C) @ np.ones((len(steering_vecs), 1))

    # Initialize list for storing noise power history
    power_history: list[np.float64] = list()

    # Execute specified iteration count of steepest descent
    for _ in range(iteration_count):
        # Compute noise power, the objective function
        power = np.float64(np.real(weight_vec.conj().T @ (Rnn @ weight_vec)))

        # Append noise power to history
        power_history.append(power)

        # Determine gradient direction of increased noise power
        grad = 2 * (Rnn @ weight_vec)

        # Update gradient
        w_tilde = weight_vec - mu * grad

        # Enforce distortionless constraint for all steering vectors
        w1 = w_tilde - C @ np.linalg.pinv(C.conj().T @ C) @ (C.conj().T @ w_tilde - np.ones((len(steering_vecs), 1)))

        # Enforce WNG constraint
        norm2 = np.real(w1.conj().T @ w1)
        max_norm2 = 1 / gamma
        if norm2 > max_norm2:
            w1 = w1 * np.sqrt(max_norm2 / norm2)

        # Set weight vector for next iteration
        weight_vec = w1

    # Return optimzed weight vector and power history
    return weight_vec, np.asarray(power_history)


def wng_mvdr_newton(
    Rnn: NDArray[np.float64],
    steering_vecs: list[NDArray[np.complex128]],
    gamma: float,
    mu: float,
    iteration_count: int,
) -> tuple[NDArray[np.complex128], NDArray[np.float64]]:
    """
    WNG-MVDR beamformer using Newton's method optimization.

    Uses Newton's method with Hessian (second derivative) for potentially faster
    convergence than steepest descent. The mu parameter acts as a step size damping
    factor for stability.

    Parameters:
    -----------
    Rnn : NDArray[np.float64]
        Noise covariance matrix (M x M)
    steering_vecs : list[NDArray[np.complex128]]
        Steering vectors in target directions (M x 1)
    gamma : float
        WNG constraint parameter (gamma = 10^(gamma_dB/10))
    mu : float
        Step size damping factor (0 < mu <= 1, typically 0.01-0.5)
    iteration_count : int
        Number of Newton iterations to perform

    Returns:
    --------
    NDArray[np.complex128]
        Optimized beamformer weight vector (M x 1)
    """
    # TODO: Update to handle multiple speakers
    # Create constraint matrix from steering vectors
    C = np.hstack(steering_vecs)

    # Create distortionless beamformer weights in steering vector direction
    weight_vec = C @ np.linalg.pinv(C.conj().T @ C) @ np.ones((len(steering_vecs), 1))

    # Initialize list for storing noise power history
    power_history: list[np.float64] = list()

    # Execute specified iteration count of Newton's method
    for _ in range(iteration_count):
        # Compute noise power, the objective function
        power = np.float64(np.real(weight_vec.conj().T @ (Rnn @ weight_vec)))

        # Append noise power to history
        power_history.append(power)

        # Gradient: grad f(w) = 2 * Rnn * w (same as steepest descent)
        grad = 2 * (Rnn @ weight_vec)

        # Hessian: H = 2 * Rnn (constant for quadratic objective)
        # For Newton's method: w_new = w - H^(-1) * grad
        # This simplifies to: w_new = w - inv(Rnn) * (Rnn * w) = w - w = 0
        # So we need to be more careful and use damped Newton: w_new = w - mu * H^(-1) * grad

        delta = np.linalg.pinv(Rnn) @ grad

        # Damped Newton update (mu acts as step size)
        w_tilde = weight_vec - mu * delta

        # Enforce distortionless constraint for all steering vectors
        w1 = w_tilde - C @ np.linalg.pinv(C.conj().T @ C) @ (C.conj().T @ w_tilde - np.ones((len(steering_vecs), 1)))

        # Enforce WNG constraint
        norm2 = np.real(w1.conj().T @ w1)
        max_norm2 = 1 / gamma
        if norm2 > max_norm2:
            w1 = w1 * np.sqrt(max_norm2 / norm2)

        weight_vec = w1

    # Return optimzed weight vector and power history
    return weight_vec, np.asarray(power_history)


def compute_steering_vector(
    mic_pos: NDArray[np.float64],
    mic_loc: NDArray[np.float64],
    fvec: NDArray[np.float64],
    signal_loc: NDArray[np.float64],
    sound_speed: float,
) -> NDArray[np.complex128]:
    # Extract dimensional information
    mic_count = mic_pos.shape[1]
    freq_bin_count = fvec.size

    # Initialize steering vector list
    steering_vecs = []

    # Iterate over signal sources
    for source_loc in signal_loc:
        # Compute distance between microphones and signal source
        dist = np.linalg.norm(mic_pos.T - (source_loc - mic_loc), axis=1)

        # Determine time delays per microphone from signal source
        tau = dist / sound_speed

        # Initialize steering matrix
        steering_vec = np.zeros((freq_bin_count, mic_count), dtype=complex)

        # Verify frequency vector is defined
        if fvec.size == 0:
            msg = "No STFT was computed"
            raise ValueError(msg)

        # Iterate over frequency vectors per microphone
        for freq_idx, freq in enumerate(fvec):
            # Determine phase delays
            steering_vec[freq_idx, :] = np.exp(-1j * 2 * np.pi * freq * tau)

        # Add steering vector to list
        steering_vecs.append(steering_vec)

    # Return the steering vectors
    return steering_vecs


def apply_beamformer_stft(
    data: NDArray[np.complex128], weights: NDArray[np.complex128]
) -> NDArray[np.complex128]:
    # Extract dimensional information
    freq_bin_count, time_frame_count, _ = data.shape

    # Initialize filtered data array
    filtered_data = np.zeros((freq_bin_count, time_frame_count), dtype=complex)

    # Iterate through frequency bins
    for freq_bin_idx in range(freq_bin_count):
        # Extract weights for frequency bin
        freq_bin_weights = weights[freq_bin_idx, :].reshape(-1, 1).conj()

        # Iterate through time frames
        for time_frame_idx in range(time_frame_count):
            # Extract data for frequency bin and time frame
            freq_bin_data = data[freq_bin_idx, time_frame_idx, :].reshape(-1, 1)

            # Apply associated weights to frequency bin data
            filtered_data[freq_bin_idx, time_frame_idx] = (freq_bin_weights.T @ freq_bin_data)[0, 0]

    # Return filtered data
    return filtered_data


def lcmv_solver(
    Rnn: NDArray[np.float64],
    steering_vecs: list[NDArray[np.complex128]],
    response_vec: NDArray[np.complex128] | None = None
) -> NDArray[np.complex128]:
    """
    Standard closed-form LCMV solver.
    """
    # Create constraint matrix from steering vectors
    C = np.hstack(steering_vecs)
    
    if response_vec is None:
        # Default to distortionless response (1.0) for all constraints
        response_vec = np.ones((C.shape[1], 1), dtype=np.complex128)
        
    R_inv = np.linalg.pinv(Rnn)
    
    # w = R^-1 C (C^H R^-1 C)^-1 g
    # Compute C^H R^-1 C
    inner_matrix = C.conj().T @ R_inv @ C
    
    # Compute inverse of inner matrix
    inner_inv = np.linalg.pinv(inner_matrix)
    
    # Compute weights
    w = R_inv @ C @ inner_inv @ response_vec
    
    return w


def gsc_solver(
    Rnn: NDArray[np.float64],
    steering_vecs: list[NDArray[np.complex128]],
    response_vec: NDArray[np.complex128] | None = None
) -> NDArray[np.complex128]:
    """
    Generalized Sidelobe Canceler (GSC) solver.
    """
    # Create constraint matrix from steering vectors
    C = np.hstack(steering_vecs)
    M, K = C.shape
    
    if response_vec is None:
        response_vec = np.ones((K, 1), dtype=np.complex128)
        
    # 1. Quiescent vector w_q = C (C^H C)^-1 g
    w_q = C @ np.linalg.pinv(C.conj().T @ C) @ response_vec
    
    # 2. Blocking Matrix B
    # Use QR decomposition to find null space of C^H
    # Q matrix from QR of C contains basis for range(C) and null(C^H)
    # C is M x K. Q is M x M.
    Q, _ = np.linalg.qr(C, mode='complete')
    B = Q[:, K:] # M x (M-K)
    
    if B.shape[1] == 0:
        # No degrees of freedom left
        return w_q
        
    # 3. Adaptive weights w_a
    # w_a = (B^H R B)^-1 B^H R w_q
    
    denom = B.conj().T @ Rnn @ B
    num = B.conj().T @ Rnn @ w_q
    
    w_a = np.linalg.pinv(denom) @ num
    
    # Total weights
    w = w_q - B @ w_a
    
    return w
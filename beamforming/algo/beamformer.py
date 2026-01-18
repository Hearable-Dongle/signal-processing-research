import numpy as np
from numpy.typing import NDArray


def wng_mvdr_steepest(
    Rnn: NDArray[np.float64],
    steering_vec: NDArray[np.complex128],
    gamma: float,
    mu: float,
    iteration_count: int,
) -> tuple[NDArray[np.complex128], NDArray[np.float64]]:
    # Create distortionless beamformer weights in steering vector direction
    weight_vec = steering_vec / (steering_vec.conj().T @ steering_vec)

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

        # Enforce distortionless constraint
        alpha = (steering_vec.conj().T @ w_tilde - 1) / (steering_vec.conj().T @ steering_vec)
        w1 = w_tilde - steering_vec * alpha

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
    steering_vec: NDArray[np.complex128],
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
    steering_vec : NDArray[np.complex128]
        Steering vector in target direction (M x 1)
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

    # Create distortionless beamformer weights in steering vector direction
    weight_vec = steering_vec / (steering_vec.conj().T @ steering_vec)

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

        # Enforce distortionless constraint (19)
        # Equation (19) = 1  to satisfy integral (22-23)
        alpha = (steering_vec.conj().T @ w_tilde - 1) / (steering_vec.conj().T @ steering_vec)
        w1 = w_tilde - steering_vec * alpha

        # Enforce WNG constraint
        norm2 = np.real(w1.conj().T @ w1)
        max_norm2 = 1 / gamma
        if norm2 > max_norm2:
            w1 = w1 * np.sqrt(max_norm2 / norm2)

        weight_vec = w1

    # Return optimzed weight vector and power history
    return weight_vec, np.asarray(power_history)


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

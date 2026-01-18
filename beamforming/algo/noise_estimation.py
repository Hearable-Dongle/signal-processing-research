import numpy as np
from numpy.typing import NDArray


def estimate_Rnn(noise_seg: NDArray[np.float64]):
    # Compute sample covariance matrix of noise across microphones
    Rnn = noise_seg.T @ noise_seg / noise_seg.shape[0]

    # Symmetrize covariance matrix and return matrix
    return 0.5 * (Rnn + Rnn.T)


def reduce_Rnn(Rnn: NDArray[np.float64], component_count: int):
    # Compute eigen-decomposition of symmetric (eigh instead of eig) noise covariance matrix
    eigvals, eigvecs = np.linalg.eigh(Rnn)

    # Sort indices for eigenvalues in descending order
    idxes = np.argsort(eigvals)[::-1]

    # Reorder matrix of eigenvectors and eigenvalues according to descending order of eigenvalues
    eigvecs_desc = eigvecs[:, idxes]
    eigvals_desc = eigvals[idxes]

    # Select specified number of directions with largest noise variance
    eigvecs_k = eigvecs_desc[:, :component_count]
    eigvals_k = np.diag(eigvals_desc[:component_count])

    # Reconstruct approximate covariance matrix using top specified number of components
    Rnn_reduced = eigvecs_k @ eigvals_k @ eigvecs_k.T

    # Symmetrize approximate covariance matrix and return matrix
    return 0.5 * (Rnn_reduced + Rnn_reduced.T)


def regularize_Rnn(Rnn: NDArray[np.float64], reg_factor: float):
    # Regularize covariance matrix of noise
    Rnn_reg = Rnn + reg_factor * np.eye(Rnn.shape[0], dtype=Rnn.dtype)

    # Symmetrize covariance matrix and return matrix
    return 0.5 * (Rnn_reg + Rnn_reg.T)

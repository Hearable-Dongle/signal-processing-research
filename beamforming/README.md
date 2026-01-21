# Beamforming Simulation Project

This directory contains a beamforming simulation project designed to process audio signals in a simulated acoustic environment.

## Project Gist

The project implements a Frequency-Domain Beamforming pipeline:

1.  **Simulation**: A virtual room is created to simulate sound propagation. This is done twice:
    *   Once for the mixed audio (speech + noise) captured by a microphone array.
    *   Once for pure noise (ground truth) captured by the same microphone array.
2.  **Transformation**: Both the mixed microphone audio and the pure noise audio are converted into the frequency domain using Short-Time Fourier Transform (STFT).
3.  **Statistics (The "Brain")**:
    *   The Spatial Covariance Matrix ($R_{nn}$) is computed using the pure noise simulation, characterizing the spatial properties of the noise.
    *   Steering Vectors are computed based on the known locations of the target speech sources.
4.  **Solver (The "Muscle")**: Iterative algorithms (Steepest Descent and Newton's Method) are used to calculate complex weights for each frequency bin. These weights aim to minimize noise power while maintaining a constant gain towards the target speech source.
5.  **Reconstruction**: The processed frequency-domain signals are converted back into a waveform using Inverse STFT.
6.  **Evaluation**: The reconstructed audio is compared against the original dry source files to assess the beamforming performance.

## How to Run

To set up the environment and run the simulation:

1.  **Install Dependencies**: Ensure you have Python installed. Then, create and activate a virtual environment (recommended) and install the required packages:
    ```bash
    python -m venv beamforming-env
    source beamforming-env/bin/activate
    pip install -r beamforming/requirements.txt
    ```

2.  **Run the Simulation**: From the parent directory (e.g., `signal-processing-research/`), execute the main script as a Python module:
    ```bash
    python -m beamforming.main
    ```
    The simulation will read its configuration from `beamforming/config/config.json` and output results (processed audio files, plots) into the `beamforming/output/` directory.

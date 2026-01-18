# Beamforming Simulation

This project simulates a microphone array and applies beamforming techniques to enhance speech signals from target speakers in a multi-speaker environment.

## Quickstart

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the simulation:**
   ```bash
   python3 interface.py
   ```

## Configuration

The simulation is configured through the `config/config.json` file. This file allows you to define the room acoustics, microphone array, audio sources, and beamforming parameters.

### Room Configuration

- `room_settings`: Defines the physical dimensions of the room and the number of reflections to simulate.

### Microphone Array Configuration

- `mic_settings`: Defines the number of microphones, their spacing, location, and the array type (e.g., `circular`).

### Audio Sources

The `sources` array allows you to define multiple audio sources within the room. Each source has the following properties:

- `input`: The path to the audio file for the source.
- `loc`: The 3D coordinates of the source in the room.
- `classification`: The type of source. This can be either `signal` for a target speaker or `noise` for background noise.

### Beamforming Parameters

- `noise_pc_count`: The number of principal components to use for noise reduction.
- `noise_reg_factor`: The regularization factor for the noise covariance matrix.
- `frame_duration`: The duration of the STFT frames in milliseconds.

## Running Simulations

To run a simulation, modify the `config/config.json` file to define your desired scenario and then run the `interface.py` script.

### Command-Line Arguments

You can also specify the configuration file using a command-line argument:

- `--config`: Path to the configuration file. Defaults to `config/config.json`.

Example:
```bash
python3 interface.py --config my_custom_config.json
```

You can also specify a custom output directory:
- `--output`: Path to the output directory.

Example:
```bash
python3 interface.py --config my_custom_config.json --output my_output_directory
```

### Example: Two Target Speakers

To simulate a scenario with two target speakers, you would configure the `sources` array in `config/config.json` as follows:

```json
"sources": [
    {
        "input": "50.flac",
        "loc": [2.5, 1.0, 1.5],
        "classification": "signal"
    },
    {
        "input": "02.flac",
        "loc": [2.5, 3.0, 1.5],
        "classification": "signal"
    },
    {
        "input": "noise.wav",
        "loc": [4.0, 2.0, 1.5],
        "classification": "noise"
    }
],
```

### Example: Three Target Speakers

To simulate a scenario with three target speakers, you would configure the `sources` array as follows:

```json
"sources": [
    {
        "input": "50.flac",
        "loc": [2.5, 1.0, 1.5],
        "classification": "signal"
    },
    {
        "input": "02.flac",
        "loc": [2.5, 3.0, 1.5],
        "classification": "signal"
    },
    {
        "input": "another_speaker.wav",
        "loc": [1.0, 2.0, 1.5],
        "classification": "signal"
    },
    {
        "input": "noise.wav",
        "loc": [4.0, 2.0, 1.5],
        "classification": "noise"
    }
],
```

## Output

By default, the simulation output is saved to a directory in the `output` folder named after the configuration file (e.g., `output/my_config`). You can override this by using the `--output` command-line argument.

The output directory will contain:

- A copy of the configuration file used for the simulation.
- `audio/`: The raw microphone audio and the filtered audio from each beamforming method.
- `plots/`: Plots of the microphone and room layout, beam patterns, and optimization history.
- `log.txt`: A log file containing the quantitative evaluation metrics (RMSE, MSE, SNR, and SI-SDR).

You can analyze the log file to quantitatively assess the performance of the beamformer in different scenarios.

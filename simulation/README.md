# Simulation System

System for generating simulated multi-channel audio datasets using `pyroomacoustics`.

## Setup
Create a venv from `beamforming/requirements.txt`.
```sh
python -m venv sim-env
source sim-env/bin/activate
pip install -r beamforming/requirements.txt
```

## Files
- `simulation_config.py`: Configuration loader/saver (JSON).
- `simulator.py`: Core simulation engine using `pyroomacoustics`.
- `create_restaurant_scene.py`: Dataset generator for restaurant environments.
- `create_library_scene.py`: Dataset generator for library environments.

## Execution
Run scripts from the project root:
```bash
# Generate restaurant configs
python -m simulation.create_restaurant_scene

# Generate library configs
python -m simulation.create_library_scene
```
Configs are output to `simulation/simulations/configs/`.

For conversation-style scenes with frame-level activity truth, overlap intervals, noise metadata, and preset environments, use [sim/realistic_conversations/README.md](/home/mkeller/mkeller/signal-processing-research/sim/realistic_conversations/README.md).

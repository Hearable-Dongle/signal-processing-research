import random
import glob
from pathlib import Path
from typing import List
import numpy as np

from general_utils.constants import LIBRIMIX_PATH
from simulation.simulation_config import SimulationConfig, Room, MicrophoneArray, SimulationAudio, SimulationSource

def get_audio_files(base_path: Path, pattern: str) -> List[Path]:
    return list(base_path.glob(pattern))

def generate_random_position(room_dims: List[float], z_height: float = 1.5, margin: float = 0.5) -> List[float]:
    x = random.uniform(margin, room_dims[0] - margin)
    y = random.uniform(margin, room_dims[1] - margin)
    z = z_height 
    return [x, y, z]

def main():
    random.seed(42)
    np.random.seed(42)

    output_dir = Path("simulation/simulations/configs/restaurant_scene")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Scanning for audio files in {LIBRIMIX_PATH}...")
    
    speech_files = get_audio_files(LIBRIMIX_PATH / "LibriSpeech/train-clean-100", "**/*.flac")
    if not speech_files:
        print("Warning: No speech files found. Checking alternate extensions/paths...")
        speech_files = get_audio_files(LIBRIMIX_PATH / "LibriSpeech/train-clean-100", "**/*.wav")
        
    noise_files = get_audio_files(LIBRIMIX_PATH / "wham_noise/tr", "**/*.wav")

    print(f"Found {len(speech_files)} speech files and {len(noise_files)} noise files.")

    if not speech_files or not noise_files:
        print("Error: Could not find necessary audio files.")
        return

    speakers_counts = [1, 2, 3, 4, 5]
    scenes_per_count = 40

    for k in speakers_counts:
        for i in range(scenes_per_count):
            # Room Definition (Small Restaurant)
            # Width/Length: 5m - 10m
            # Height: 3m - 4m
            room_dims = [
                round(random.uniform(5, 10), 2),
                round(random.uniform(5, 10), 2),
                round(random.uniform(3, 4), 2)
            ]
            
            room = Room(dimensions=room_dims, absorption=0.2)

            # 2. Microphone Array
            # Center of room with slight jitter
            center_x = room_dims[0] / 2 + random.uniform(-0.5, 0.5)
            center_y = room_dims[1] / 2 + random.uniform(-0.5, 0.5)
            center_z = 1.5 # Fixed height for mic
            
            mic_array = MicrophoneArray(
                mic_center=[round(center_x, 2), round(center_y, 2), center_z],
                mic_radius=0.1,
                mic_count=4
            )

            # Audio Sources
            sources = []
            
            # Select k distinct speech files
            selected_speech_files = random.sample(speech_files, k)
            
            for speech_file in selected_speech_files:
                pos = generate_random_position(room_dims, z_height=1.5)
                try:
                    rel_path = speech_file.relative_to(LIBRIMIX_PATH)
                except ValueError:
                    rel_path = speech_file
                
                sources.append(SimulationSource(
                    loc=[round(p, 2) for p in pos],
                    audio_path=str(rel_path),
                    gain=1.0 # Normal speech
                ))
            
            # Add Background Noise
            noise_file = random.choice(noise_files)
            try:
                rel_noise_path = noise_file.relative_to(LIBRIMIX_PATH)
            except ValueError:
                rel_noise_path = noise_file
                
            # Random position for noise (maybe distinct from speech? or just random)
            noise_pos = generate_random_position(room_dims, z_height=random.uniform(0.5, 2.5))
            
            # Vary intensity
            noise_gain = round(random.uniform(0.1, 0.8), 2)
            
            sources.append(SimulationSource(
                loc=[round(p, 2) for p in noise_pos],
                audio_path=str(rel_noise_path),
                gain=noise_gain
            ))

            simulation_audio = SimulationAudio(
                sources=sources,
                duration=10.0,
                fs=16000
            )

            config = SimulationConfig(
                room=room,
                microphone_array=mic_array,
                audio=simulation_audio
            )

            filename = f"restaurant_k{k}_scene{i:02d}.json"
            config.to_file(output_dir / filename)
            
    print(f"Generated {len(speakers_counts) * scenes_per_count} configuration files in {output_dir}")

if __name__ == "__main__":
    main()

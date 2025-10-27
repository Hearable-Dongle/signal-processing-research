import argparse
import logging
import sounddevice as sd
from scipy.io.wavfile import write

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def record_audio(duration, sample_rate):
    """
    Records audio from the default input device.

    Args:
        duration (int): The duration of the recording in seconds.
        sample_rate (int): The sample rate of the recording.

    Returns:
        numpy.ndarray: The recorded audio data.
    """
    logging.info(f"Recording for {duration} seconds at {sample_rate} Hz...")
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
    sd.wait()  # Wait until recording is finished
    logging.info("Recording finished.")
    return audio_data

def save_audio(file_path, audio_data, sample_rate):
    """
    Saves audio data to a WAV file.

    Args:
        file_path (str): The path to save the WAV file.
        audio_data (numpy.ndarray): The audio data to save.
        sample_rate (int): The sample rate of the audio.
    """
    logging.info(f"Saving audio to {file_path}...")
    write(file_path, sample_rate, audio_data)
    logging.info("Audio saved successfully.")

def main():
    parser = argparse.ArgumentParser(description="Record voice and save to a file.")
    parser.add_argument("--output-file", type=str, help="The path to save the output WAV file.")
    parser.add_argument("--duration", type=int, default=5, help="The duration of the recording in seconds.")
    parser.add_argument("--sample-rate", type=int, default=44100, help="The sample rate of the recording.")
    args = parser.parse_args()

    audio_data = record_audio(args.duration, args.sample_rate)
    save_audio(args.output_file, audio_data, args.sample_rate)

if __name__ == "__main__":
    main()


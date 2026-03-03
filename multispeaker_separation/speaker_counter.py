import threading
from pyannote.audio import Pipeline


class PyannoteCounter:
    def __init__(self, auth_token=None):
        """
        Args:
            auth_token (str): HuggingFace auth token for pyannote.audio (required for some pipelines).
        """
        self.pipeline = None
        if Pipeline:
            # TODO: clean this up!!
            try:
                # 'pyannote/speaker-diarization' is the standard pipeline
                # It requires an auth token usually.
                # Newer versions use 'use_auth_token' (deprecated) or 'token' or implicit login.
                # We'll try passing it as a kwarg if provided, otherwise rely on env/login.
                if auth_token:
                     self.pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=auth_token)
                else:
                     self.pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")
            except TypeError:
                # Fallback for versions that don't accept use_auth_token or renamed it
                try:
                     self.pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")
                except Exception as e:
                     print(f"Failed to initialize pyannote pipeline (fallback): {e}")
            except Exception as e:
                print(f"Failed to initialize pyannote pipeline: {e}")
        else:
            print("pyannote.audio not installed.")

    def count(self, audio_path):
        """
        Counts speakers in the audio file.
        
        Args:
            audio_path (str): Path to audio file.
            
        Returns:
            int: Number of unique speakers detected.
        """
        if not self.pipeline:
            raise RuntimeError("Pipeline not initialized.")
        
        diarization = self.pipeline(audio_path)
        
        # diarization is an Annotation object. 
        # distinct labels gives the set of speakers.
        unique_speakers = set(diarization.labels())
        return len(unique_speakers)

    def count_async(self, audio_path, callback):
        """
        Runs counting in a separate thread and calls callback with result.
        
        Args:
            audio_path (str): Path to audio file.
            callback (callable): Function accepting (int) count.
        """
        def task():
            try:
                count = self.count(audio_path)
                callback(count)
            except Exception as e:
                print(f"Error in async counting: {e}")
                callback(None)
                
        thread = threading.Thread(target=task)
        thread.start()
        return thread

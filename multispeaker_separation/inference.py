import os
import torch
import numpy as np
from asteroid.models import ConvTasNet

# Try importing HailoRT, or define a dummy for development if not present
try:
    from hailo_platform import HEF, VDevice, HailoStreamInterface, InferVStreams, ConfigureParams, InputVStreamParams, OutputVStreamParams, FormatType
    HAILO_AVAILABLE = True
except ImportError:
    HAILO_AVAILABLE = False


class ConvTasNetInferencer:
    """
    Handles inference for ConvTasNet using either PyTorch or HailoRT backend.
    """
    def __init__(self, model_path, backend='pytorch', **kwargs):
        """
        Args:
            model_path (str): Path to the model file (.pth for pytorch, .hef for hailo).
            backend (str): 'pytorch' or 'hailo'.
        """
        self.model_path = model_path
        self.backend = backend.lower()
        self.model = None
        
        if self.backend == 'pytorch' and ConvTasNet is None:
             raise ImportError("Asteroid library is required for PyTorch backend.")
        if self.backend == 'hailo' and not HAILO_AVAILABLE:
             print("Warning: HailoRT not detected. Hailo inference will fail.")

        self.load_model()

    def load_model(self):
        if self.backend == 'pytorch':
            self.model = ConvTasNet.from_pretrained(self.model_path)
            self.model.eval()
        elif self.backend == 'hailo':
            # TODO: handle the device context properly
            self.hef = HEF(self.model_path)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def infer(self, audio_input):
        """
        Runs inference on the audio input.
        
        Args:
            audio_input (np.ndarray or torch.Tensor): Input audio waveform. 
                                                      Shape: (batch, channels, time) or (channels, time)
        
        Returns:
            np.ndarray: Separated audio sources.
        """
        if self.backend == 'pytorch':
            return self._infer_pytorch(audio_input)
        elif self.backend == 'hailo':
            return self._infer_hailo(audio_input)
        
    def _infer_pytorch(self, audio):
        if not isinstance(audio, torch.Tensor):
            audio = torch.from_numpy(audio)
        
        if audio.dim() == 1:
            audio = audio.unsqueeze(0).unsqueeze(0)
        elif audio.dim() == 2:
            audio = audio.unsqueeze(1) # Assuming (batch, time) -> (batch, 1, time)

        with torch.no_grad():
            output = self.model(audio)
        
        return output.cpu().numpy()

    def _infer_hailo(self, audio):
        if not HAILO_AVAILABLE:
            raise RuntimeError("HailoRT is not available.")
        
        if isinstance(audio, torch.Tensor):
            audio = audio.cpu().numpy()
            
        # Hailo expects input shape typically as (Batch, Height, Width, Channels) or similar depending on model
        audio = audio.astype(np.float32)

        # Might be better to keep newtork_group and streams open for multiple inferences
        results = None

        with VDevice() as target:
            configure_params = ConfigureParams.create_from_hef(hef=self.hef, interface=HailoStreamInterface.PCIe)
            network_groups = target.configure(self.hef, configure_params)
            network_group = network_groups[0]
            
            input_vstream_params = InputVStreamParams.make(network_group, format_type=FormatType.FLOAT32)
            output_vstream_params = OutputVStreamParams.make(network_group, format_type=FormatType.FLOAT32)

            with InferVStreams(network_group, input_vstream_params, output_vstream_params) as infer_pipeline:
                # Expecting a dictionary or list depending on inputs
                # Using the first input layer name if possible or passing as list
                input_data = {self.hef.get_input_vstream_infos()[0].name: audio}
                
                output_data = infer_pipeline.infer(input_data)
                
                # output_data is a dictionary {output_name: np.array}
                # We assume single output for separation (or combined mask)
                results = list(output_data.values())[0]

        return results

    def post_process(self, output):
        return output


class SpeakerSeparationSystem:
    """
    Main inference engine wrapper that manages multiple models for different speaker counts.
    """
    def __init__(self, model_dir, backend='pytorch', max_speakers=5):
        """
        Args:
            model_dir (str): Directory containing model files named like 'convtasnet_2spk.pth'
            backend (str): 'pytorch' or 'hailo'
            max_speakers (int): Maximum number of speakers to support (default 5).
        """
        self.backend = backend
        self.models = {}
        self.max_speakers = max_speakers
        self.model_dir = model_dir
        
        self._load_all_models()

    def _load_all_models(self):
        extension = 'hef' if self.backend == 'hailo' else 'pth'
        
        for k in range(1, self.max_speakers + 1):
            # Naming convention assumption: convtasnet_{k}spk.{ext}
            # Or similar. We'll try a standard pattern.
            filename = f"convtasnet_{k}spk.{extension}"
            path = os.path.join(self.model_dir, filename)
            
            if os.path.exists(path):
                print(f"Loading model for {k} speakers from {path}...")
                self.models[k] = ConvTasNetInferencer(path, backend=self.backend)
            else:
                print(f"Warning: Model file {path} not found. Skipping {k} speakers.")

    def separate(self, audio, num_speakers):
        """
        Separate audio based on the number of speakers.
        """
        if num_speakers not in self.models:
            raise ValueError(f"No model loaded for {num_speakers} speakers.")
        
        inferencer = self.models[num_speakers]
        raw_output = inferencer.infer(audio)
        return inferencer.post_process(raw_output)

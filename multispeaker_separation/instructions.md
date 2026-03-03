I am trying to make a system for speaker separation then synthesis using ConvTasNet network. I want to develop this in Pytorch but then have a system to run on my RPi where I am making calls to a .hef binary using the HailoRT library and SDK. 

1. I'd like for you to set up a class which handles calling the model on either Pytorch or HailoRT depending on a flag passed to the constructor. The class should have methods for loading the model, running inference, and post-processing the output to get separated audio signals. 


2. Make another class which is essentially a wrapper around this bigger class, which is treated like our main inference engine, but when we call it, we explicitly state how many speakers are in the mixture. This class should load ConvTasNet models for 1, 2, 3, 4, and 5 speakers separately (set this max speaker count in constructor but default to 5 initially).

3. I'll need a way to determine how many speakers are in an audio clip. Use pyannote.audio to create a speaker counting system which can be run in the background or on another thread and the output can be passed to the main speaker separation inference engine

4. I don't know if this Pyannote audio can be run on Hailo-8, so I need an alternative lightweight method for speaker counting that can be deployed on edge devices. In @multispeaker_separation/count_speakers/ directory, I need you to develop and train a small CRNN (using Pytorch but with supported operations from Hailo shown in https://hailo.ai/developer-zone/documentation/dataflow-compiler-v3-33-0/?sp_referrer=sdk/supported_layers.html) to count the number of speakers (0 to 4+) in a 1-second audio clip. The model should take a Mel-Spectrogram as input and output a classification of the number of speakers. Please provide training code, model definition, and inference code that can be converted to Hailo .hef format later.


All code should be written in the multispeaker_separation/ directory, but can call code from other modules. I want all files to be run with `python -m multispeaker_separation.<path_to_file>` from the root directory, so make sure you set up the __init__.py files and imports correctly.



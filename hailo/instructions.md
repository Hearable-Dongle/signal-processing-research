You are an agent to create a system to convert a Convtasnet model from Pytorch to .hef format using ONNX as an intermediate step. The user has provided you with the necessary code snippets and context to complete this task.


The current flow is:
1. Define convtasnet model in Pytorch. This is done in "@asteroid/models/conv_tasnet.py"
2. Write a wrapper to remove the non-hailo 8 compatible parts. This is done in "@hailo/convtasnet_to_onnx.py"
3. Convert to onnx with the wrapper.
4. Patch the onnx to fix any issues. This is done in "@hailo/patch_onnx.py"
5. Convert the patched onnx to .hef using hailo sdk. This is done in "@hailo/onnx_to_hef.py." This is where its failing. 

I want to avoid doing too many details with changing the Onnx file and adding in tooo many monkey patches (although this is fine). Ideally, I want to update the wrapper to make the onnx more compatible with hailo sdk.

Docs for supported operations available at: https://hailo.ai/developer-zone/documentation/dataflow-compiler-v3-33-0/?sp_referrer=sdk/supported_layers.html 

More implementation details of convtasnet available at https://asteroid.readthedocs.io/en/v0.3.3/apidoc/asteroid.models.conv_tasnet.html and https://github.com/asteroid-team/asteroid/blob/master/asteroid/models/conv_tasnet.py


ALL model Pytorch --> onnx should use python executable in "@hailo/to-onnx-env/bin/python", and all onnx --> hef should use python executable in "@hailo/to-hailo-env/bin/python". The to-hailo-env is the current venv

ALL produced code should be in the hailo/ directory.


I already asked an agent to do some work and this is what it came up with:


1. Diagnosis
* Issue: The original ConvTasNet is a 1D Temporal Convolutional Network (3D tensors: Batch, Channels, Time). The
    Hailo SDK failed to convert this, throwing Unsupported dimensions and later UnsupportedConv3DError because it
    interpreted the 1D convolutions with specific layouts as unsupported 3D operations or mismatched shapes.
* Solution: Convert the entire model to use 2D Convolutions with a 4D tensor layout [Batch, Channels, Time, 1],
    essentially treating the time dimension as "Height" and adding a dummy "Width" of 1.

2. Code Changes (hailo/convtasnet_to_onnx.py)
* Model Transformation (`convert_model_to_4d`):
    * Wrote a function to recursively traverse the PyTorch model.
    * Replaced all nn.Conv1d layers with nn.Conv2d (kernel size (K, 1), stride (S, 1), etc.).
    * Replaced all nn.ConvTranspose1d with nn.ConvTranspose2d.
    * Asteroid Specifics: Explicitly handled Asteroid's Encoder and Decoder classes, extracting their filterbank
        weights and reshaping them to 4D [Out, In, K, 1].
* Monkey Patching:
    * `norms._glob_norm`: Patched the Global Layer Norm function to calculate mean/variance correctly over the
        Channel (1) and Time (2) dimensions of the new 4D tensors.
    * `TDConvNet.forward`: Patched the masker's forward pass to handle 4D inputs and correctly reshape/view the
        tensors (flattening Batch and Source dimensions) during the masking operation.
* Export Wrapper (`HailoExportWrapper`):
    * Updated to accept 3D input, unsqueeze it to 4D [1, 1, 16000, 1].
    * Handled the broadcasting of masks by manually repeating the encoded features (repeat_interleave) to ensure
        explicit shape matching for the ONNX exporter.
    * Reshaped the final 4D output back to the original 3D [Batch, n_src, Time] format.

3. Pipeline Updates
* `hailo/onnx_to_hef.py`: Removed hardcoded end_node_names to allow the compiler to attempt converting the full
    valid subgraph of the new 4D model.
* `hailo/patch_onnx.py`: Kept in the pipeline to ensure explicit kernel_shape attributes are present, which aids
    the Hailo parser.

Next Steps for Instructions: Convert 1D audio models to 'Vertical' 4D (NxCxHx1) structures using Conv2d layers. This forces the model into a standard image-like format that the Hailo compiler
handles natively.

I gave this to another agent and it entered a loop:

> The assistant is in a repetitive loop. It has executed the same `run_shell_command` pipeline four times in the last several turns. Each time, the pipeline fails at the `runner.optimize` step with the same `Unsupported dimensions` error, specifically related to a dimension of size 262144. The assistant's attempts to fix this by changing the calibration data shape and decomposing a normalization function were unsuccessful, indicating a cognitive loop where it is unable to diagnose the root cause of the recurring error. While the latest action involves a major strategy shift to a "Horizontal" orientation, it follows the same unproductive pattern of making a change and re-running the entire failing process.

What might be the root cause of this? Output all issues and suggestions in a concise manner in "issues.md"

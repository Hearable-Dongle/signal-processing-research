"""
Analyzes and prints the GFLOPS, size, and number of parameters for a given model.
This script is designed to work with the models used in source_separation.py.
"""
import argparse
import torch
import tempfile
import os
from pathlib import Path
import sys

from fvcore.nn import FlopCountAnalysis

from own_voice_suppression.source_separation import (
    AsteroidConvTasNetWrapper,
    SpeechBrainSepFormerWrapper,
    DiartWrapper,
    WavLMVerifier,
    MODEL_OPTIONS as SEP_MODEL_OPTIONS,
)

# Add 'wavlm' to the list of models that can be analyzed
ALL_MODEL_OPTIONS = SEP_MODEL_OPTIONS + ["wavlm"]
ANALYSIS_DURATIONS_MS = {"default": 100, "wavlm": 1000}

def get_model_and_input(model_type: str, device: torch.device):
    """
    Loads the specified model and creates a dummy input tensor for analysis.

    Returns:
        A tuple containing the core nn.Module and a suitable dummy input tensor.
    """
    print(f"Loading model '{model_type}' for analysis...")
    duration_ms = ANALYSIS_DURATIONS_MS.get(model_type, ANALYSIS_DURATIONS_MS["default"])
    
    if model_type == "convtasnet":
        wrapper = AsteroidConvTasNetWrapper(device)
        model = wrapper.model
        input_sr = wrapper.NATIVE_SR
        dummy_input = torch.randn(1, int(input_sr * (duration_ms / 1000.0))).to(device)
    elif model_type == "sepformer":
        wrapper = SpeechBrainSepFormerWrapper(device)
        # The SepformerSeparation class is an nn.Module itself.
        model = wrapper.model
        input_sr = wrapper.NATIVE_SR
        # The underlying model's forward takes (batch, samples)
        dummy_input = torch.randn(1, int(input_sr * (duration_ms / 1000.0))).to(device)
    elif model_type == "diart":
        wrapper = DiartWrapper(device)
        model = wrapper.segmentation
        input_sr = wrapper.NATIVE_SR
        dummy_input = torch.randn(1, 1, int(input_sr * (duration_ms / 1000.0))).to(device)
    elif model_type == "wavlm":
        wrapper = WavLMVerifier(device)
        model = wrapper.model
        input_sr = 16000  # WavLM native SR
        dummy_wav_np = torch.randn(int(input_sr * (duration_ms / 1000.0))).numpy()
        inputs = wrapper.processor(dummy_wav_np, sampling_rate=input_sr, return_tensors="pt", padding=True).to(device)
        dummy_input = inputs.input_values
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model, dummy_input


def analyze_model(model_type: str):
    """
    Performs the analysis for the given model type and prints the results.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running analysis on {device}")

    model, dummy_input = get_model_and_input(model_type, device)
    model.eval()

    # Use state_dict to count parameters, as model.parameters() can be unreliable
    # for complex models where requires_grad might not be set as expected in eval mode.
    state_dict = model.state_dict()
    num_params = sum(p.numel() for p in state_dict.values())

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp:
        torch.save(model.state_dict(), tmp.name)
        model_size_mb = os.path.getsize(tmp.name) / (1024 * 1024)
    os.remove(tmp.name)

    tflops = "N/A"
    try:
        flops_analyzer = FlopCountAnalysis(model, dummy_input)
        total_flops = flops_analyzer.total()

        # If a different duration was used for analysis (e.g., for wavlm),
        # scale the result back to a 100ms equivalent.
        analysis_duration_ms = ANALYSIS_DURATIONS_MS.get(
            model_type, ANALYSIS_DURATIONS_MS["default"]
        )
        scaling_factor = analysis_duration_ms / 100.0
        
        scaled_flops = total_flops / scaling_factor
        tflops = f"{scaled_flops / 1e12:.4f}"
        
    except Exception as e:
        print(f"Could not calculate FLOPS automatically: {e}")
        print("FLOPS calculation will be skipped.")

    print("\n" + "="*40)
    print(f"Model Analysis Report: '{model_type}'")
    print("="*40)
    print(f"  - Number of Parameters: {num_params / 1e6:.2f} M")
    print(f"  - Model Size (on disk): {model_size_mb:.2f} MB")
    print(f"  - TFLOPs (for 100ms audio): {tflops}")

    analysis_duration_ms = ANALYSIS_DURATIONS_MS.get(
        model_type, ANALYSIS_DURATIONS_MS["default"]
    )
    if analysis_duration_ms != 100:
        print(f"    (Calculated with {analysis_duration_ms}ms input and scaled)")

    print("="*40 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
        Analyze model size, parameters, and TFLOPs for separation and verifier models.
        Calculates metrics for a 100ms audio input at the model's native sample rate.
        """
    )
    parser.add_argument(
        "--model-type",
        type=str,
        required=True,
        choices=ALL_MODEL_OPTIONS,
        help="The model to analyze.",
    )
    args = parser.parse_args()

    analyze_model(args.model_type)

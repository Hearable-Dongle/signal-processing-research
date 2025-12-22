import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_target_presence(log, output_path, model_type):
    """Plots target speaker presence over time."""
    times = [d["time"] for d in log]
    dets = [d["detected"] for d in log]
    
    plt.figure(figsize=(10, 4))
    plt.fill_between(times, dets, step="pre", alpha=0.4, color='red', label="Target Detected")
    plt.plot(times, dets, drawstyle="steps", color='red')
    plt.ylim(-0.1, 1.1)
    plt.title(f"Target Presence ({model_type})")
    plt.xlabel("Time (s)")
    plt.yticks([0, 1], ["Absent", "Present"])
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_source_amplitudes(log, output_path, model_type):
    """Plots separated source amplitudes (RMS) over time."""
    times = [d["time"] for d in log]
    s1_rms = [d["s1_rms"] for d in log]
    s2_rms = [d["s2_rms"] for d in log]

    plt.figure(figsize=(10, 4))
    plt.plot(times, s1_rms, label="Source 1 Amplitude")
    plt.plot(times, s2_rms, label="Source 2 Amplitude")
    plt.title(f"Source Amplitudes ({model_type})")
    plt.xlabel("Time (s)")
    plt.ylabel("RMS Amplitude")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_waveform_comparison(
    estimated_wav: torch.Tensor,
    ground_truth_wav: torch.Tensor,
    sr: int,
    output_path: str
):
    """
    Plots two waveforms on top of each other for comparison.
    """
    # Ensure tensors are on CPU and are 1D numpy arrays
    est_np = estimated_wav.cpu().numpy().squeeze()
    gt_np = ground_truth_wav.cpu().numpy().squeeze()
    
    # Create time axis
    num_samples = len(est_np)
    time_axis = np.linspace(0, num_samples / sr, num=num_samples)
    
    fig, axes = plt.subplots(2, 1, figsize=(15, 6), sharex=True, sharey=True)
    
    # Plot Estimated Waveform
    axes[0].plot(time_axis, est_np, label="Estimated Background", color="dodgerblue")
    axes[0].set_title("Estimated Background")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, linestyle='--', alpha=0.6)
    
    # Plot Ground Truth Waveform
    axes[1].plot(time_axis, gt_np, label="Ground Truth Background", color="green")
    axes[1].set_title("Ground Truth Background")
    axes[1].set_xlabel("Time (s)")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, linestyle='--', alpha=0.6)
    
    plt.suptitle("Waveform Comparison", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make room for suptitle
    plt.savefig(output_path)
    plt.close()


def plot_denoising_waveforms(
    noisy_wav: torch.Tensor,
    denoised_wav: torch.Tensor,
    ground_truth_wav: torch.Tensor,
    sr: int,
    output_path: str
):
    """
    Plots three waveforms (noisy, denoised, ground truth) for comparison.
    """
    noisy_np = noisy_wav.cpu().numpy().squeeze()
    denoised_np = denoised_wav.cpu().numpy().squeeze()
    gt_np = ground_truth_wav.cpu().numpy().squeeze()
    
    min_len = min(len(noisy_np), len(denoised_np), len(gt_np))
    noisy_np = noisy_np[:min_len]
    denoised_np = denoised_np[:min_len]
    gt_np = gt_np[:min_len]
    
    time_axis = np.linspace(0, min_len / sr, num=min_len)
    
    fig, axes = plt.subplots(3, 1, figsize=(15, 9), sharex=True, sharey=True)
    
    axes[0].plot(time_axis, noisy_np, label="Noisy Input", color="gray")
    axes[0].set_title("Noisy Input")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, linestyle='--', alpha=0.6)
    
    axes[1].plot(time_axis, denoised_np, label="Denoised Output", color="dodgerblue")
    axes[1].set_title("Denoised Output")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, linestyle='--', alpha=0.6)
    
    axes[2].plot(time_axis, gt_np, label="Ground Truth Speech", color="green")
    axes[2].set_title("Ground Truth Speech")
    axes[2].set_xlabel("Time (s)")
    axes[2].legend(loc="upper right")
    axes[2].grid(True, linestyle='--', alpha=0.6)
    
    plt.suptitle("Denoising Waveform Comparison", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path)
    plt.close()
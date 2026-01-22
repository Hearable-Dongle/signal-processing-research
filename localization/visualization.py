import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_source_comparison(true_doas_deg, est_doas_deg, output_dir: Path):
    """
    Plots true vs estimated DOAs on a polar plot.
    
    Args:
        true_doas_deg: List of true DOAs in degrees.
        est_doas_deg: List of estimated DOAs in degrees.
        output_dir: Directory to save the plot.
    """
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='polar')
    
    # Convert to radians
    true_rad = np.radians(true_doas_deg)
    est_rad = np.radians(est_doas_deg)
    
    # Plot True Sources
    for i, angle in enumerate(true_rad):
        label = 'True Source' if i == 0 else None
        # Plot a line from center to edge
        ax.plot([angle, angle], [0, 1], color='green', linewidth=3, alpha=0.7, label=label)
        # Add marker at the end
        ax.scatter([angle], [1], color='green', s=100, zorder=10)
        
    # Plot Estimated Sources
    for i, angle in enumerate(est_rad):
        label = 'Estimated' if i == 0 else None
        ax.plot([angle, angle], [0, 0.9], color='red', linestyle='--', linewidth=2, label=label)
        ax.scatter([angle], [0.9], color='red', marker='x', s=100, zorder=10)

    # Configure grid and labels
    ax.set_theta_zero_location("E") # 0 degrees at East (standard Cartesian)
    # Actually, in standard mic array geometry 0 is usually x-axis. 
    # arctan2(y, x) -> 0 is x-axis. 
    # Polar plot default 0 is East. So this matches.
    
    ax.set_yticks([])
    ax.set_title("Source Localization Results: True vs Estimated", pad=20)
    
    # Handle legend with unique labels
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right', bbox_to_anchor=(1.1, 1.1))
    
    output_path = output_dir / "doa_comparison.png"
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from .configure import Audio_Sources


def plot_mic_pos(mic_pos: NDArray[np.float64], output_dir: Path) -> None:

    x = mic_pos[0, :]
    y = mic_pos[1, :]

    mic_count = x.shape[0]

    fig = plt.figure()  

    ax = fig.add_subplot(111)

    # Plot microphones
    ax.scatter(x, y, c="red", marker="o")  

    for mic in range(mic_count):
        # Add channel index with small offset to prevent overlap with microphone marker
        ax.text(  
            x[mic] + 0.005,
            y[mic],
            f"{mic}",
            fontsize=10,
            color="black",
        )

    ax.set_xlabel("X position (m)") 
    ax.set_ylabel("Y position (m)")

    ax.set_aspect("equal", adjustable="box")

    ax.set_title("Microphone Layout (2D Top-Down View)")

    plt.grid(visible=True)

    image_dir = output_dir / "images"
    if not image_dir.exists():
        image_dir.mkdir(parents=True)

    plt.savefig(image_dir / "mic_layout.png")  
    plt.close()


def plot_room_pos(
    room_dim: NDArray[np.float64],
    mic_loc: NDArray[np.float64],
    sources: list[Audio_Sources],
    output_dir: Path,
) -> None:

    # Verify number of room dimensions
    if len(room_dim) != 3:
        raise ValueError("Room does not contain the required number of dimensions")

    fig = plt.figure()

    ax = fig.add_subplot(111, projection="3d")

    length, width, height = room_dim

    x_edge = [0, length]
    y_edge = [0, width]
    z_edge = [0, height]

    # Draw x, y axis edge
    for x_val in x_edge:
        for y_val in y_edge:
            ax.plot([x_val, x_val], [y_val, y_val], [0, height], "k") 

    # Draw x, z axis edge
    for x_val in x_edge:
        for z_val in z_edge:
            ax.plot([x_val, x_val], [0, width], [z_val, z_val], "k")  

    # Draw y, z axis edge
    for y_val in y_edge:
        for z_val in z_edge:
            ax.plot([0, length], [y_val, y_val], [z_val, z_val], "k")  

    if len(mic_loc) != 3:
        raise ValueError("Microphone location does not contain the required number of dimensions")

    ax.scatter(mic_loc[0], mic_loc[1], mic_loc[2], color="r", label="Mics")

    for source in sources:
        if len(source.loc) != 3:
            raise ValueError("Source location does not contain the required number of dimensions")

        ax.scatter(  
            source.loc[0],
            source.loc[1],
            source.loc[2],  
            color="b" if source.classification == "signal" else "g",
            label=source.classification.capitalize(),
        )

    ax.set_xlabel("X position (m)") 
    ax.set_ylabel("Y position (m)") 
    ax.set_zlabel("Z position (m)") 

    plt.legend()

    image_dir = output_dir / "images"
    if not image_dir.exists():
        image_dir.mkdir(parents=True)

    plt.savefig(image_dir / "room_layout.png")  

    plt.close()


def plot_history(data: dict[str, list[np.float64]], output_dir: Path) -> None:

    plt.figure()  

    for label, (history, kwargs) in data.items():
        plt.semilogy(history, label=label, **kwargs)

    plt.xlabel("Iteration")
    plt.ylabel("Noise Power")
    plt.grid(visible=True)
    plt.title("Convergence History")
    plt.legend()

    image_dir = output_dir / "images"
    if not image_dir.exists():
        image_dir.mkdir(parents=True)

    plt.savefig(image_dir / "convergence.png")

    plt.close()


def plot_beam_pattern(
    name: str,
    weights: NDArray[np.complex128],
    mic_pos: NDArray[np.float64],
    freq: float,
    sound_speed: float,
    output_dir: Path,
    angle_count: int = 360,
) -> None:
    pos = mic_pos.T
    pos_xy = pos[:, :2]

    k = 2.0 * np.pi * freq / sound_speed

    angles = np.linspace(0, 2 * np.pi, angle_count, endpoint=False)

    pattern = np.zeros(angle_count, dtype=np.complex128)

    for angle_idx, angle in enumerate(angles):
        direction = np.array([np.cos(angle), np.sin(angle)])

        phase = -1j * k * (pos_xy @ direction)

        steering_vec = np.exp(phase)

        pattern[angle_idx] = np.conj(weights) @ steering_vec

    # Normalize to 0 dB max
    pattern_db = 20 * np.log10(np.abs(pattern) / np.max(np.abs(pattern)))

    plt.figure()

    ax = plt.subplot(111, projection="polar")

    ax.plot(angles, pattern_db)

    ax.set_rlim(-40, 0)

    image_dir = output_dir / "images"
    if not image_dir.exists():
        image_dir.mkdir(parents=True)

    plt.savefig(image_dir / f"{name}.png")

    plt.close()

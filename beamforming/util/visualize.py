from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from .configure import Audio_Sources


def plot_mic_pos(mic_pos: NDArray[np.float64], output_dir: Path) -> None:

    # Determine x and y positions of microphones
    x = mic_pos[0, :]
    y = mic_pos[1, :]

    # Extract microphone count
    mic_count = x.shape[0]

    # Create figure
    fig = plt.figure()  # type: ignore[reportUnknownMemberType]

    # Add 2D subplot with one row and one column
    ax = fig.add_subplot(111)

    # Plot microphones
    ax.scatter(x, y, c="red", marker="o")  # type: ignore[reportUnknownMemberType]

    # Label each mic with its channel index
    for mic in range(mic_count):
        # Add channel index with small offset to prevent overlap with microphone marker
        ax.text(  # type: ignore[reportUnknownMemberType]
            x[mic] + 0.005,
            y[mic],
            f"{mic}",
            fontsize=10,
            color="black",
        )

    # Set axis labels
    ax.set_xlabel("X position (m)")  # type: ignore[reportUnknownMemberType]
    ax.set_ylabel("Y position (m)")  # type: ignore[reportUnknownMemberType]

    # Set figure aspect
    ax.set_aspect("equal", adjustable="box")

    # Set figure title
    ax.set_title("Microphone Layout (2D Top-Down View)")  # type: ignore[reportUnknownMemberType]

    # Turn on figure grid
    plt.grid(visible=True)  # type: ignore[reportUnknownMemberType]

    # Create image directory if it does not exist
    image_dir = output_dir / "images"
    if not image_dir.exists():
        image_dir.mkdir(parents=True)

    # Save plot to file
    plt.savefig(image_dir / "mic_layout.png")  # type: ignore[reportUnknownMemberType]

    # Close plot to prevent display
    plt.close()


def plot_room_pos(
    room_dim: NDArray[np.float64],
    mic_loc: NDArray[np.float64],
    sources: list[Audio_Sources],
    output_dir: Path,
) -> None:

    # Verify number of room dimensions
    if len(room_dim) != 3:
        # Print error message
        msg = "Room does not contain the required number of dimensions"
        raise ValueError(msg)

    # Create figure
    fig = plt.figure()  # type: ignore[reportUnknownMemberType]

    # Add 3D subplot with one row and one column
    ax = fig.add_subplot(111, projection="3d")

    # Extract room dimensions
    length, width, height = room_dim

    # Defined room edges
    x_edge = [0, length]
    y_edge = [0, width]
    z_edge = [0, height]

    # Draw x, y axis edge
    for x_val in x_edge:
        for y_val in y_edge:
            ax.plot([x_val, x_val], [y_val, y_val], [0, height], "k")  # type: ignore[reportUnknownMemberType]

    # Draw x, z axis edge
    for x_val in x_edge:
        for z_val in z_edge:
            ax.plot([x_val, x_val], [0, width], [z_val, z_val], "k")  # type: ignore[reportUnknownMemberType]

    # Draw y, z axis edge
    for y_val in y_edge:
        for z_val in z_edge:
            ax.plot([0, length], [y_val, y_val], [z_val, z_val], "k")  # type: ignore[reportUnknownMemberType]

    # Verify number of microphone dimensions
    if len(mic_loc) != 3:
        # Print error message
        msg = "Microphone location does not contain the required number of dimensions"
        raise ValueError(msg)

    # Plot microphone array
    ax.scatter(mic_loc[0], mic_loc[1], mic_loc[2], color="r", label="Mics")  # type: ignore[reportUnknownMemberType]

    # Iterate through sources
    for source in sources:
        # Verify number of microphone dimensions
        if len(source.loc) != 3:
            # Print error message
            msg = "Source location does not contain the required number of dimensions"
            raise ValueError(msg)

        # Plot sources
        ax.scatter(  # type: ignore[reportUnknownMemberType]
            source.loc[0],
            source.loc[1],
            source.loc[2],  # type: ignore[reportUnknownMemberType]
            color="b" if source.classification == "signal" else "g",
            label=source.classification.capitalize(),
        )

    # Set axis labels
    ax.set_xlabel("X position (m)")  # type: ignore[reportUnknownMemberType]
    ax.set_ylabel("Y position (m)")  # type: ignore[reportUnknownMemberType]
    ax.set_zlabel("Z position (m)")  # type: ignore[reportUnknownMemberType]

    # Add legend to figure
    plt.legend()  # type: ignore[reportUnknownMemberType]

    # Create image directory if it does not exist
    image_dir = output_dir / "images"
    if not image_dir.exists():
        image_dir.mkdir(parents=True)

    # Save plot to file
    plt.savefig(image_dir / "room_layout.png")  # type: ignore[reportUnknownMemberType]

    # Close plot to prevent display
    plt.close()


def plot_history(data: dict[str, list[np.float64]], output_dir: Path) -> None:

    # Create figure
    plt.figure()  # type: ignore[reportUnknownMemberType]

    # Iterate through data
    for label, (history, kwargs) in data.items():
        # Plot history
        plt.semilogy(history, label=label, **kwargs)  # type: ignore[reportUnknownMemberType]

    # Set axis labels
    plt.xlabel("Iteration")  # type: ignore[reportUnknownMemberType]
    plt.ylabel("Noise Power")  # type: ignore[reportUnknownMemberType]

    # Enable plot grid
    plt.grid(visible=True)  # type: ignore[reportUnknownMemberType]

    # Set figure title
    plt.title("Convergence History")  # type: ignore[reportUnknownMemberType]

    # Add legend to figure
    plt.legend()  # type: ignore[reportUnknownMemberType]

    # Create image directory if it does not exist
    image_dir = output_dir / "images"
    if not image_dir.exists():
        image_dir.mkdir(parents=True)

    # Save plot to file
    plt.savefig(image_dir / "convergence.png")  # type: ignore[reportUnknownMemberType]

    # Close plot to prevent display
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
    # Determine x and ymicrophone positions
    pos = mic_pos.T
    pos_xy = pos[:, :2]

    # Determine wave number
    k = 2.0 * np.pi * freq / sound_speed

    # Compute individual angles
    angles = np.linspace(0, 2 * np.pi, angle_count, endpoint=False)

    # Initialize pattern
    pattern = np.zeros(angle_count, dtype=np.complex128)

    # Iterate through angles
    for angle_idx, angle in enumerate(angles):
        # Compute unit direction vector in the horizontal plane
        direction = np.array([np.cos(angle), np.sin(angle)])

        # Get plane-wave phase at each mic
        phase = -1j * k * (pos_xy @ direction)

        # Create steering vector for look direction
        steering_vec = np.exp(phase)

        # Compute array response to steering vector
        pattern[angle_idx] = np.conj(weights) @ steering_vec

    # Normalize to 0 dB max
    pattern_db = 20 * np.log10(np.abs(pattern) / np.max(np.abs(pattern)))

    # Create figure
    plt.figure()  # type: ignore[reportUnknownMemberType]

    # Add 2D polar subplot with one row and one column
    ax = plt.subplot(111, projection="polar")  # type: ignore[reportUnknownMemberType]

    # Plot pattern
    ax.plot(angles, pattern_db)  # type: ignore[reportUnknownMemberType]

    # Set rim on -40 dB polar coordinates
    ax.set_rlim(-40, 0)  # type: ignore[reportUnknownMemberType]

    # Create image directory if it does not exist
    image_dir = output_dir / "images"
    if not image_dir.exists():
        image_dir.mkdir(parents=True)

    # Save plot to file
    plt.savefig(image_dir / f"{name}.png")  # type: ignore[reportUnknownMemberType]

    # Close plot to prevent display
    plt.close()

from enum import Enum

import numpy as np
import pyroomacoustics as pra
from numpy.typing import NDArray


class MicType(Enum):
    CIRCULAR = 0
    LINEAR = 1
    PLANAR = 2


def sim_mic(
    mic_count: int,
    mic_loc: np.ndarray,
    mic_spacing: float,
    mic_type: MicType = MicType.CIRCULAR,
    mic_fs: int = 16000,
) -> tuple[pra.MicrophoneArray, NDArray[np.float64]]:

    match mic_type:
        case MicType.CIRCULAR:
            # Define mic positions for circular array in xy-plane
            angles = np.linspace(0, 2 * np.pi, mic_count, endpoint=False)
            mic_pos = np.array([
                mic_spacing * np.cos(angles),
                mic_spacing * np.sin(angles),
                np.zeros(mic_count),
            ])

        case MicType.LINEAR:
            # Define mic positions for linear array along x-axis
            mic_pos = np.array([
                [i * mic_spacing - (mic_count - 1) * mic_spacing / 2, 0, 0]
                for i in range(mic_count)
            ]).T

        case MicType.PLANAR:
            # Define mic positions for planar array in xy-plane
            num_mics_side = int(np.sqrt(mic_count))
            x_coords = np.linspace(
                -(num_mics_side - 1) * mic_spacing / 2,
                (num_mics_side - 1) * mic_spacing / 2,
                num_mics_side,
            )
            y_coords = np.linspace(
                -(num_mics_side - 1) * mic_spacing / 2,
                (num_mics_side - 1) * mic_spacing / 2,
                num_mics_side,
            )
            xx, yy = np.meshgrid(x_coords, y_coords)
            mic_pos = np.vstack((
                xx.ravel(),
                yy.ravel(),
                np.zeros(num_mics_side * num_mics_side),
            ))

        case _:
            # Print error message
            err_msg = f'Array type "{mic_type}" not recognized'
            raise ValueError(err_msg)

    # Determine absolute positions of microphones
    mic_pos_abs = mic_pos + mic_loc.reshape(3, 1)

    # Return simulated microphone array and microphone layout
    return pra.MicrophoneArray(mic_pos_abs, fs=mic_fs), mic_pos


def sim_room(room_dim: list[int], fs: int, reflection_count: int) -> pra.ShoeBox:

    # Return simulated room
    return pra.ShoeBox(
        room_dim,
        fs=fs,
        max_order=reflection_count,
    )

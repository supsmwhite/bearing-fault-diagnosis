"""Read CWRU .mat files and extract Drive End DE_time vibration signals."""

from pathlib import Path

import numpy as np
from scipy.io import loadmat


def find_de_key(mat_dict: dict, expected_source_file_id: int = None) -> str:
    """
    Find the DE_time variable name from a dictionary returned by scipy.io.loadmat.
    """
    de_keys = [
        key
        for key in mat_dict
        if not key.startswith("__") and "DE_time" in key
    ]

    if expected_source_file_id is not None:
        expected_prefix = f"X{expected_source_file_id:03d}"
        expected_de_keys = [
            key
            for key in de_keys
            if key.startswith(expected_prefix)
        ]

        if len(expected_de_keys) == 1:
            return expected_de_keys[0]
        if len(expected_de_keys) > 1:
            raise ValueError(
                f"Multiple DE_time variables found for {expected_prefix}: {expected_de_keys}"
            )

    if len(de_keys) > 1:
        raise ValueError(
            "Multiple DE_time variables found: "
            f"{de_keys}. Pass expected_source_file_id to select the correct one."
        )
    if not de_keys:
        raise KeyError("No DE_time variable found in .mat file")

    return de_keys[0]


def load_de_signal(mat_path: str, expected_source_file_id: int = None) -> np.ndarray:
    """
    Read one CWRU .mat file and return a one-dimensional DE signal.
    """
    mat_dict = loadmat(Path(mat_path))
    de_key = find_de_key(
        mat_dict,
        expected_source_file_id=expected_source_file_id,
    )
    signal = np.asarray(mat_dict[de_key]).squeeze()

    if signal.ndim != 1:
        raise ValueError(
            f"DE signal must be one-dimensional after squeeze, got shape {signal.shape}"
        )

    return signal

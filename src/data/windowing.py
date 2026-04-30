"""Split one-dimensional vibration signals into fixed-length window samples."""

import numpy as np


def zscore_window(window: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Apply z-score normalization to a single window.
    """
    mean = window.mean()
    std = window.std()
    return ((window - mean) / (std + eps)).astype(np.float32)


def sliding_windows(
    signal: np.ndarray,
    window_size: int = 1024,
    stride: int = 512,
    normalize: bool = True,
) -> tuple:
    """
    Split a one-dimensional signal into fixed-length windows.

    Returns:
        windows: shape = (num_windows, window_size)
        start_indices: shape = (num_windows,)
    """
    signal = np.asarray(signal)

    if signal.ndim != 1:
        raise ValueError(f"Signal must be one-dimensional, got shape {signal.shape}")
    if len(signal) < window_size:
        raise ValueError(
            f"Signal length {len(signal)} is shorter than window_size {window_size}"
        )

    windows = []
    start_indices = []

    for start in range(0, len(signal) - window_size + 1, stride):
        window = signal[start:start + window_size]
        if normalize:
            window = zscore_window(window)
        else:
            window = window.astype(np.float32)

        windows.append(window)
        start_indices.append(start)

    return (
        np.stack(windows).astype(np.float32),
        np.asarray(start_indices, dtype=np.int64),
    )


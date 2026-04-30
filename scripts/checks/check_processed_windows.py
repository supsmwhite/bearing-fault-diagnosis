"""Check processed npz window data, labels, and metadata for correctness."""

from collections import Counter
from pathlib import Path

import numpy as np


NPZ_PATH = Path("data/processed/cwru_12k_de_windows.npz")


def print_counts(name: str, values: np.ndarray) -> None:
    counts = Counter(values.tolist())
    print(f"{name} unique values and counts: {dict(counts)}")


def main() -> None:
    data = np.load(NPZ_PATH, allow_pickle=False)

    X = data["X"]
    y = data["y"]
    load_hp = data["load_hp"]
    class_name = data["class_name"]
    file_name = data["file_name"]
    start_index = data["start_index"]

    print(f"Keys: {list(data.keys())}")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"X dtype: {X.dtype}")
    print(f"y dtype: {y.dtype}")
    print_counts("load_hp", load_hp)
    print_counts("y", y)
    print_counts("class_name", class_name)
    print(f"Has NaN: {np.isnan(X).any()}")
    print(f"Has inf: {np.isinf(X).any()}")

    print("First 5 samples:")
    for index in range(min(5, X.shape[0])):
        print(
            f"  {index}: "
            f"file_name={file_name[index]}, "
            f"class_name={class_name[index]}, "
            f"label={y[index]}, "
            f"load_hp={load_hp[index]}, "
            f"start_index={start_index[index]}"
        )


if __name__ == "__main__":
    main()


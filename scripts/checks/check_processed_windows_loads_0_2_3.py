"""Check processed CWRU windows for load 0/2/3."""

from collections import Counter
from pathlib import Path

import numpy as np


PROCESSED_PATH = Path("data/processed/cwru_12k_de_windows_loads_0_2_3.npz")
REQUIRED_KEYS = ["X", "y", "load_hp", "class_name", "file_name", "start_index"]


def main() -> None:
    data = np.load(PROCESSED_PATH, allow_pickle=False)
    for key in REQUIRED_KEYS:
        assert key in data.files, f"Missing key: {key}"

    X = data["X"]
    y = data["y"]
    load_hp = data["load_hp"]
    class_name = data["class_name"]
    file_name = data["file_name"]
    start_index = data["start_index"]

    lengths = [len(X), len(y), len(load_hp), len(class_name), len(file_name), len(start_index)]
    assert len(set(lengths)) == 1, f"Length mismatch: {lengths}"

    nan_count = int(np.isnan(X).sum())
    inf_count = int(np.isinf(X).sum())
    load_counts = Counter(load_hp.tolist())
    y_counts = Counter(y.tolist())
    class_counts = Counter(class_name.tolist())

    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"load_hp unique values and counts: {dict(sorted(load_counts.items()))}")
    print(f"y unique values and counts: {dict(sorted(y_counts.items()))}")
    print(f"class_name unique values and counts: {dict(sorted(class_counts.items()))}")
    print(f"NaN count: {nan_count}")
    print(f"inf count: {inf_count}")

    assert set(load_counts.keys()) == {0, 2, 3}, load_counts
    assert set(y_counts.keys()) == set(range(10)), y_counts
    assert nan_count == 0, f"NaN count must be 0, got {nan_count}"
    assert inf_count == 0, f"inf count must be 0, got {inf_count}"

    print("processed windows loads 0/2/3 check passed.")


if __name__ == "__main__":
    main()

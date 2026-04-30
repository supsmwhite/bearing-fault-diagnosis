"""Check CWRU 0 hp -> 3 hp split integrity."""

from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np


PROCESSED_PATH = Path("data/processed/cwru_12k_de_windows_loads_0_2_3.npz")
SPLIT_PATH = Path("data/processed/cwru_12k_de_splits_0to3.npz")
EXPECTED_LOADS = {
    "train_source_indices": 0,
    "val_source_indices": 0,
    "train_target_unlabeled_indices": 3,
    "test_target_indices": 3,
}
NUM_CLASSES = 10


def distribution(values: np.ndarray) -> dict:
    return dict(sorted(Counter(values.tolist()).items()))


def print_split(name: str, indices: np.ndarray, data) -> None:
    print(f"split name: {name}")
    print(f"  sample count: {len(indices)}")
    print(f"  load_hp distribution: {distribution(data['load_hp'][indices])}")
    print(f"  label distribution: {distribution(data['y'][indices])}")
    print(f"  class_name distribution: {distribution(data['class_name'][indices])}")


def overlap_count(left: np.ndarray, right: np.ndarray) -> int:
    return len(set(left.tolist()) & set(right.tolist()))


def main() -> None:
    data = np.load(PROCESSED_PATH, allow_pickle=False)
    splits_file = np.load(SPLIT_PATH, allow_pickle=False)
    num_samples = len(data["y"])

    splits = {
        name: splits_file[name].astype(np.int64)
        for name in EXPECTED_LOADS
    }

    for name, indices in splits.items():
        assert len(indices) > 0, f"{name} is empty"
        assert indices.min() >= 0 and indices.max() < num_samples, f"{name} has invalid index"
        expected_load = EXPECTED_LOADS[name]
        assert set(data["load_hp"][indices].tolist()) == {expected_load}, f"{name} load mismatch"
        assert set(data["y"][indices].tolist()) == set(range(NUM_CLASSES)), f"{name} missing labels"
        print_split(name, indices, data)

    for left_name, right_name in combinations(splits.keys(), 2):
        count = overlap_count(splits[left_name], splits[right_name])
        print(f"overlap {left_name} vs {right_name}: {count}")
        assert count == 0, f"{left_name} and {right_name} overlap by {count}"

    print("0 hp -> 3 hp split check passed.")


if __name__ == "__main__":
    main()

"""Create CWRU 0 hp -> 3 hp cross-load split."""

from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np


PROCESSED_PATH = Path("data/processed/cwru_12k_de_windows_loads_0_2_3.npz")
OUTPUT_PATH = Path("data/processed/cwru_12k_de_splits_0to3.npz")
SOURCE_LOAD_HP = 0
TARGET_LOAD_HP = 3
NUM_CLASSES = 10


def sorted_class_indices(data, load_hp_value: int, label: int) -> np.ndarray:
    mask = (data["load_hp"] == load_hp_value) & (data["y"] == label)
    indices = np.where(mask)[0]
    order = sorted(
        indices.tolist(),
        key=lambda index: (str(data["file_name"][index]), int(data["start_index"][index])),
    )
    return np.asarray(order, dtype=np.int64)


def split_by_class(data, load_hp_value: int, train_ratio: float) -> tuple:
    train_indices = []
    holdout_indices = []
    for label in range(NUM_CLASSES):
        indices = sorted_class_indices(data, load_hp_value, label)
        if len(indices) == 0:
            raise ValueError(f"No samples found for load={load_hp_value}, label={label}")
        split_at = int(len(indices) * train_ratio)
        if split_at <= 0 or split_at >= len(indices):
            raise ValueError(
                f"Invalid split for load={load_hp_value}, label={label}, samples={len(indices)}"
            )
        train_indices.append(indices[:split_at])
        holdout_indices.append(indices[split_at:])
    return (
        np.concatenate(train_indices).astype(np.int64),
        np.concatenate(holdout_indices).astype(np.int64),
    )


def distribution(values: np.ndarray) -> dict:
    return dict(sorted(Counter(values.tolist()).items()))


def print_split(name: str, indices: np.ndarray, data) -> None:
    print(f"{name}: {len(indices)}")
    print(f"  load_hp distribution: {distribution(data['load_hp'][indices])}")
    print(f"  label distribution: {distribution(data['y'][indices])}")


def overlap_count(left: np.ndarray, right: np.ndarray) -> int:
    return len(set(left.tolist()) & set(right.tolist()))


def main() -> None:
    data = np.load(PROCESSED_PATH, allow_pickle=False)
    train_source_indices, val_source_indices = split_by_class(
        data,
        SOURCE_LOAD_HP,
        train_ratio=0.8,
    )
    train_target_unlabeled_indices, test_target_indices = split_by_class(
        data,
        TARGET_LOAD_HP,
        train_ratio=0.7,
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUTPUT_PATH,
        train_source_indices=train_source_indices,
        val_source_indices=val_source_indices,
        train_target_unlabeled_indices=train_target_unlabeled_indices,
        test_target_indices=test_target_indices,
    )

    splits = {
        "train_source_indices": train_source_indices,
        "val_source_indices": val_source_indices,
        "train_target_unlabeled_indices": train_target_unlabeled_indices,
        "test_target_indices": test_target_indices,
    }
    for name, indices in splits.items():
        print_split(name, indices, data)

    print("split overlap counts:")
    for left_name, right_name in combinations(splits.keys(), 2):
        print(f"  {left_name} vs {right_name}: {overlap_count(splits[left_name], splits[right_name])}")

    print(f"split path: {OUTPUT_PATH}")
    print("0 hp -> 3 hp split created.")


if __name__ == "__main__":
    main()

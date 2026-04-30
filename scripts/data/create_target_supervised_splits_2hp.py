"""Create target-supervised 2 hp split indices for upper-bound diagnostic experiments."""

from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np


PROCESSED_PATH = Path("data/processed/cwru_12k_de_windows.npz")
SPLIT_PATH = Path("data/processed/cwru_12k_de_splits_target_supervised_2hp.npz")


def summarize_split(name: str, indices: np.ndarray, load_hp: np.ndarray, y: np.ndarray) -> None:
    print(f"{name}: {len(indices)}")
    print(f"  load_hp distribution: {dict(Counter(load_hp[indices].tolist()))}")
    print(f"  label distribution: {dict(Counter(y[indices].tolist()))}")


def sorted_class_indices(indices: np.ndarray, y: np.ndarray, file_name: np.ndarray, start_index: np.ndarray) -> list:
    grouped = []
    for label in sorted(np.unique(y[indices]).tolist()):
        label_indices = indices[y[indices] == label]
        order = np.lexsort((start_index[label_indices], file_name[label_indices]))
        grouped.append(label_indices[order])
    return grouped


def main() -> None:
    data = np.load(PROCESSED_PATH, allow_pickle=False)
    y = data["y"]
    load_hp = data["load_hp"]
    file_name = data["file_name"]
    start_index = data["start_index"]

    target_indices = np.where(load_hp == 2)[0]
    train_parts = []
    test_parts = []

    for label_indices in sorted_class_indices(target_indices, y, file_name, start_index):
        split_at = int(len(label_indices) * 0.70)
        train_parts.append(label_indices[:split_at])
        test_parts.append(label_indices[split_at:])

    target_train_labeled_indices = np.concatenate(train_parts).astype(np.int64)
    target_test_indices = np.concatenate(test_parts).astype(np.int64)

    np.savez(
        SPLIT_PATH,
        target_train_labeled_indices=target_train_labeled_indices,
        target_test_indices=target_test_indices,
    )

    summarize_split("target_train_labeled", target_train_labeled_indices, load_hp, y)
    summarize_split("target_test", target_test_indices, load_hp, y)

    split_map = {
        "target_train_labeled_indices": target_train_labeled_indices,
        "target_test_indices": target_test_indices,
    }
    for left_name, right_name in combinations(split_map.keys(), 2):
        overlap = set(split_map[left_name].tolist()) & set(split_map[right_name].tolist())
        print(f"{left_name} vs {right_name} overlap: {len(overlap)}")
    print(f"Saved path: {SPLIT_PATH}")


if __name__ == "__main__":
    main()


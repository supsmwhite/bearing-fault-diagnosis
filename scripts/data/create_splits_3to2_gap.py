"""Create 3 hp to 2 hp split indices with discarded gaps between adjacent window splits."""

from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np


PROCESSED_PATH = Path("data/processed/cwru_12k_de_windows.npz")
SPLIT_PATH = Path("data/processed/cwru_12k_de_splits_3to2_gap.npz")
FORMAL_SPLITS = [
    "train_source_indices",
    "val_source_indices",
    "train_target_unlabeled_indices",
    "test_target_indices",
]


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


def split_source(label_indices: np.ndarray) -> tuple:
    n = len(label_indices)
    train_end = int(n * 0.75)
    gap_end = int(n * 0.80)
    return label_indices[:train_end], label_indices[gap_end:], label_indices[train_end:gap_end]


def split_target(label_indices: np.ndarray) -> tuple:
    n = len(label_indices)
    train_end = int(n * 0.65)
    gap_end = int(n * 0.70)
    return label_indices[:train_end], label_indices[gap_end:], label_indices[train_end:gap_end]


def print_overlap_check(split_map: dict) -> None:
    for left_name, right_name in combinations(FORMAL_SPLITS, 2):
        overlap = set(split_map[left_name].tolist()) & set(split_map[right_name].tolist())
        print(f"{left_name} vs {right_name} overlap: {len(overlap)}")


def main() -> None:
    data = np.load(PROCESSED_PATH, allow_pickle=False)
    y = data["y"]
    load_hp = data["load_hp"]
    file_name = data["file_name"]
    start_index = data["start_index"]

    source_indices = np.where(load_hp == 3)[0]
    target_indices = np.where(load_hp == 2)[0]

    train_source_parts = []
    val_source_parts = []
    target_train_parts = []
    target_test_parts = []
    gap_parts = []

    for label_indices in sorted_class_indices(source_indices, y, file_name, start_index):
        train_part, val_part, gap_part = split_source(label_indices)
        train_source_parts.append(train_part)
        val_source_parts.append(val_part)
        gap_parts.append(gap_part)

    for label_indices in sorted_class_indices(target_indices, y, file_name, start_index):
        train_part, test_part, gap_part = split_target(label_indices)
        target_train_parts.append(train_part)
        target_test_parts.append(test_part)
        gap_parts.append(gap_part)

    split_map = {
        "train_source_indices": np.concatenate(train_source_parts).astype(np.int64),
        "val_source_indices": np.concatenate(val_source_parts).astype(np.int64),
        "train_target_unlabeled_indices": np.concatenate(target_train_parts).astype(np.int64),
        "test_target_indices": np.concatenate(target_test_parts).astype(np.int64),
        "discarded_gap_indices": np.concatenate(gap_parts).astype(np.int64),
    }

    np.savez(SPLIT_PATH, **split_map)

    for name in FORMAL_SPLITS:
        summarize_split(name, split_map[name], load_hp, y)
    print(f"discarded_gap_indices: {len(split_map['discarded_gap_indices'])}")
    print_overlap_check(split_map)
    print(f"Saved path: {SPLIT_PATH}")


if __name__ == "__main__":
    main()


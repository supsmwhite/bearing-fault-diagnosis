"""Create fixed 3 hp to 2 hp train/validation/test split index files for CWRU windows."""

from collections import Counter
from pathlib import Path

import numpy as np


PROCESSED_PATH = Path("data/processed/cwru_12k_de_windows.npz")
SPLIT_PATH = Path("data/processed/cwru_12k_de_splits_3to2.npz")


def summarize_split(name: str, indices: np.ndarray, load_hp: np.ndarray, y: np.ndarray) -> None:
    print(f"{name}: {len(indices)}")
    print(f"  load_hp distribution: {dict(Counter(load_hp[indices].tolist()))}")
    print(f"  class distribution: {dict(Counter(y[indices].tolist()))}")


def split_by_class(indices: np.ndarray, y: np.ndarray, first_ratio: float) -> tuple:
    first_parts = []
    second_parts = []

    for label in sorted(np.unique(y[indices]).tolist()):
        label_indices = indices[y[indices] == label]
        split_at = int(len(label_indices) * first_ratio)
        first_parts.append(label_indices[:split_at])
        second_parts.append(label_indices[split_at:])

    return np.concatenate(first_parts), np.concatenate(second_parts)


def main() -> None:
    data = np.load(PROCESSED_PATH, allow_pickle=False)
    y = data["y"]
    load_hp = data["load_hp"]

    source_indices = np.where(load_hp == 3)[0]
    target_indices = np.where(load_hp == 2)[0]

    train_source_indices, val_source_indices = split_by_class(
        source_indices,
        y,
        first_ratio=0.8,
    )
    train_target_unlabeled_indices, test_target_indices = split_by_class(
        target_indices,
        y,
        first_ratio=0.7,
    )

    np.savez(
        SPLIT_PATH,
        train_source_indices=train_source_indices.astype(np.int64),
        val_source_indices=val_source_indices.astype(np.int64),
        train_target_unlabeled_indices=train_target_unlabeled_indices.astype(np.int64),
        test_target_indices=test_target_indices.astype(np.int64),
    )

    summarize_split("train_source", train_source_indices, load_hp, y)
    summarize_split("val_source", val_source_indices, load_hp, y)
    summarize_split(
        "train_target_unlabeled",
        train_target_unlabeled_indices,
        load_hp,
        y,
    )
    summarize_split("test_target", test_target_indices, load_hp, y)
    print(f"Saved path: {SPLIT_PATH}")


if __name__ == "__main__":
    main()


"""Check the 3 hp to 2 hp gap split and verify discarded gap indices are excluded."""

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
ALL_SPLITS = FORMAL_SPLITS + ["discarded_gap_indices"]


def print_distribution(name: str, indices: np.ndarray, load_hp: np.ndarray, y: np.ndarray) -> None:
    print(f"{name}: {len(indices)}")
    print(f"  load_hp distribution: {dict(Counter(load_hp[indices].tolist()))}")
    print(f"  label distribution: {dict(Counter(y[indices].tolist()))}")


def assert_single_load(name: str, indices: np.ndarray, load_hp: np.ndarray, expected_load: int) -> None:
    actual_loads = set(load_hp[indices].tolist())
    if actual_loads != {expected_load}:
        raise ValueError(f"{name} expected load_hp {expected_load}, got {actual_loads}")
    print(f"{name} only contains load_hp == {expected_load}: True")


def main() -> None:
    data = np.load(PROCESSED_PATH, allow_pickle=False)
    splits = np.load(SPLIT_PATH, allow_pickle=False)
    y = data["y"]
    load_hp = data["load_hp"]

    print(f"Split keys: {list(splits.keys())}")
    missing_keys = [name for name in ALL_SPLITS if name not in splits]
    if missing_keys:
        raise KeyError(f"Missing split keys: {missing_keys}")

    split_indices = {name: splits[name].astype(np.int64) for name in ALL_SPLITS}

    for name in ALL_SPLITS:
        print_distribution(name, split_indices[name], load_hp, y)

    assert_single_load("train_source", split_indices["train_source_indices"], load_hp, 3)
    assert_single_load("val_source", split_indices["val_source_indices"], load_hp, 3)
    assert_single_load(
        "train_target_unlabeled",
        split_indices["train_target_unlabeled_indices"],
        load_hp,
        2,
    )
    assert_single_load("test_target", split_indices["test_target_indices"], load_hp, 2)

    for left_name, right_name in combinations(FORMAL_SPLITS, 2):
        overlap = set(split_indices[left_name].tolist()) & set(split_indices[right_name].tolist())
        print(f"{left_name} vs {right_name} overlap: {len(overlap)}")
        if overlap:
            raise ValueError(f"Split overlap found between {left_name} and {right_name}")

    formal_union = set()
    for name in FORMAL_SPLITS:
        formal_union.update(split_indices[name].tolist())
    gap_overlap = formal_union & set(split_indices["discarded_gap_indices"].tolist())
    print(f"discarded_gap_indices overlap with formal splits: {len(gap_overlap)}")
    if gap_overlap:
        raise ValueError("discarded_gap_indices must not appear in formal splits")


if __name__ == "__main__":
    main()


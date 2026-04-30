"""Check fixed CWRU split files and verify Dataset reads split samples correctly."""

import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import CWRUWindowDataset  # noqa: E402


PROCESSED_PATH = PROJECT_ROOT / "data/processed/cwru_12k_de_windows.npz"
SPLIT_PATH = PROJECT_ROOT / "data/processed/cwru_12k_de_splits_3to2.npz"
SPLIT_NAMES = [
    "train_source_indices",
    "val_source_indices",
    "train_target_unlabeled_indices",
    "test_target_indices",
]


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
    missing_keys = [name for name in SPLIT_NAMES if name not in splits]
    if missing_keys:
        raise KeyError(f"Missing split keys: {missing_keys}")
    print("All required splits exist: True")

    split_indices = {
        name: splits[name].astype(np.int64)
        for name in SPLIT_NAMES
    }

    for name, indices in split_indices.items():
        print_distribution(name, indices, load_hp, y)

    assert_single_load("train_source", split_indices["train_source_indices"], load_hp, 3)
    assert_single_load("val_source", split_indices["val_source_indices"], load_hp, 3)
    assert_single_load(
        "train_target_unlabeled",
        split_indices["train_target_unlabeled_indices"],
        load_hp,
        2,
    )
    assert_single_load("test_target", split_indices["test_target_indices"], load_hp, 2)

    for left_name, right_name in combinations(SPLIT_NAMES, 2):
        overlap = set(split_indices[left_name].tolist()) & set(split_indices[right_name].tolist())
        print(f"{left_name} vs {right_name} overlap: {len(overlap)}")
        if overlap:
            raise ValueError(f"Split overlap found between {left_name} and {right_name}")

    for name, indices in split_indices.items():
        dataset = CWRUWindowDataset(
            str(PROCESSED_PATH),
            indices=indices,
            return_meta=True,
        )
        x, label, meta = dataset[0]
        print(
            f"{name} first sample: "
            f"x shape={tuple(x.shape)}, y={int(label)}, meta={meta}"
        )


if __name__ == "__main__":
    main()

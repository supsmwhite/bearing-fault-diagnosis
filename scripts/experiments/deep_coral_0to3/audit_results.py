"""Audit Deep CORAL 0 hp -> 3 hp predictions."""

import csv
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


CONFIG_PATH = PROJECT_ROOT / "configs/experiments/deep_coral_0to3.yaml"


def overlap_count(left: np.ndarray, right: np.ndarray) -> int:
    return len(set(left.tolist()) & set(right.tolist()))


def read_prediction_rows(path: Path) -> list:
    with path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def check_predictions(tag: str, path: Path, test_indices: np.ndarray, processed_data) -> None:
    rows = read_prediction_rows(path)
    sample_indices = np.asarray([int(row["sample_index"]) for row in rows], dtype=np.int64)
    y_true = np.asarray([int(row["y_true"]) for row in rows], dtype=np.int64)
    load_hp = np.asarray([int(row["load_hp"]) for row in rows], dtype=np.int64)
    assert len(rows) == len(test_indices), f"{tag}: row count mismatch"
    assert len(set(sample_indices.tolist())) == len(sample_indices), f"{tag}: duplicate sample_index"
    assert set(sample_indices.tolist()) == set(test_indices.tolist()), f"{tag}: sample_index mismatch"
    assert np.array_equal(y_true, processed_data["y"][sample_indices].astype(np.int64)), f"{tag}: y_true mismatch"
    assert set(load_hp.tolist()) == {3}, f"{tag}: load_hp must all be 3"
    print(f"{tag} prediction csv passed.")


def main() -> None:
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    processed_path = PROJECT_ROOT / config["data"]["processed_npz"]
    split_path = PROJECT_ROOT / config["data"]["split_npz"]
    experiment_dir = PROJECT_ROOT / config["outputs"]["experiment_dir"]
    processed_data = np.load(processed_path, allow_pickle=False)
    splits_file = np.load(split_path, allow_pickle=False)
    split_names = [
        config["data"]["source_train_split"],
        config["data"]["source_val_split"],
        config["data"]["target_train_split"],
        config["data"]["target_test_split"],
    ]
    splits = {name: splits_file[name].astype(np.int64) for name in split_names}

    print("Split check:")
    for left_name, right_name in combinations(split_names, 2):
        count = overlap_count(splits[left_name], splits[right_name])
        print(f"  {left_name} vs {right_name}: {count}")
        assert count == 0, f"{left_name} and {right_name} overlap"
    assert set(processed_data["load_hp"][splits[config["data"]["source_train_split"]]].tolist()) == {0}
    assert set(processed_data["load_hp"][splits[config["data"]["source_val_split"]]].tolist()) == {0}
    assert set(processed_data["load_hp"][splits[config["data"]["target_train_split"]]].tolist()) == {3}
    assert set(processed_data["load_hp"][splits[config["data"]["target_test_split"]]].tolist()) == {3}
    print("  load checks passed.")

    print("Prediction check:")
    test_indices = splits[config["data"]["target_test_split"]]
    check_predictions("best", experiment_dir / "logs/cross_load_predictions_best.csv", test_indices, processed_data)
    check_predictions("final", experiment_dir / "logs/cross_load_predictions_final.csv", test_indices, processed_data)
    print("Deep CORAL 0->3 audit passed.")


if __name__ == "__main__":
    main()

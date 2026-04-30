"""Build CWRU windows with train-stat-only global normalization."""

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.cwru_reader import load_de_signal  # noqa: E402
from src.data.windowing import sliding_windows  # noqa: E402


CONFIG_PATH = PROJECT_ROOT / "configs/dataset/cwru_minimal_files.yaml"
SPLIT_PATH = PROJECT_ROOT / "data/processed/cwru_12k_de_splits_3to2.npz"
OUTPUT_PATH = PROJECT_ROOT / "data/processed/cwru_12k_de_windows_train_stat_norm.npz"
WINDOW_SIZE = 1024
STRIDE = 512
EPS = 1e-8

LABEL_MAP = {
    "Normal": 0,
    "IR007": 1,
    "B007": 2,
    "OR007@6": 3,
    "IR014": 4,
    "B014": 5,
    "OR014@6": 6,
    "IR021": 7,
    "B021": 8,
    "OR021@6": 9,
}


def build_raw_windows():
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    raw_dir = PROJECT_ROOT / config["dataset"]["raw_dir"]
    all_windows = []
    all_labels = []
    all_loads = []
    all_class_names = []
    all_file_names = []
    all_start_indices = []

    for item in config["files"]:
        file_name = item["save_name"]
        class_name = item["class_name"]
        load_hp = item["load_hp"]
        source_file_id = item["source_file_id"]
        label = LABEL_MAP[class_name]
        mat_path = raw_dir / file_name

        signal = load_de_signal(
            str(mat_path),
            expected_source_file_id=source_file_id,
        )
        windows, start_indices = sliding_windows(
            signal,
            window_size=WINDOW_SIZE,
            stride=STRIDE,
            normalize=False,
        )
        num_windows = windows.shape[0]

        all_windows.append(windows)
        all_labels.append(np.full(num_windows, label, dtype=np.int64))
        all_loads.append(np.full(num_windows, load_hp, dtype=np.int64))
        all_class_names.append(np.full(num_windows, class_name))
        all_file_names.append(np.full(num_windows, file_name))
        all_start_indices.append(start_indices)

    return {
        "X": np.concatenate(all_windows, axis=0).astype(np.float32),
        "y": np.concatenate(all_labels, axis=0).astype(np.int64),
        "load_hp": np.concatenate(all_loads, axis=0).astype(np.int64),
        "class_name": np.concatenate(all_class_names, axis=0),
        "file_name": np.concatenate(all_file_names, axis=0),
        "start_index": np.concatenate(all_start_indices, axis=0).astype(np.int64),
    }


def main() -> None:
    raw_data = build_raw_windows()
    splits = np.load(SPLIT_PATH, allow_pickle=False)
    norm_indices = np.concatenate(
        [
            splits["train_source_indices"].astype(np.int64),
            splits["train_target_unlabeled_indices"].astype(np.int64),
        ]
    )

    norm_values = raw_data["X"][norm_indices].astype(np.float64)
    norm_mean = float(norm_values.mean())
    norm_std = float(norm_values.std())
    X = ((raw_data["X"] - norm_mean) / (norm_std + EPS)).astype(np.float32)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUTPUT_PATH,
        X=X,
        y=raw_data["y"],
        load_hp=raw_data["load_hp"],
        class_name=raw_data["class_name"],
        file_name=raw_data["file_name"],
        start_index=raw_data["start_index"],
    )

    print("Train-stat normalization processed data built.")
    print(f"norm_mean: {norm_mean:.12f}")
    print(f"norm_std: {norm_std:.12f}")
    print(f"processed path: {OUTPUT_PATH}")
    print(f"Total windows: {X.shape[0]}")
    print(f"Load distribution: {dict(Counter(raw_data['load_hp'].tolist()))}")
    print(f"Class distribution: {dict(Counter(raw_data['class_name'].tolist()))}")


if __name__ == "__main__":
    main()

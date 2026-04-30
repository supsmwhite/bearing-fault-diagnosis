"""Build processed CWRU windows for load 0/2/3 raw files."""

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.cwru_reader import load_de_signal  # noqa: E402
from src.data.windowing import sliding_windows  # noqa: E402


CONFIG_PATH = PROJECT_ROOT / "configs/dataset/cwru_minimal_files.yaml"
OUTPUT_PATH = PROJECT_ROOT / "data/processed/cwru_12k_de_windows_loads_0_2_3.npz"
TARGET_LOADS = {0, 2, 3}
WINDOW_SIZE = 1024
STRIDE = 512
NORMALIZE = True

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


def main() -> None:
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    raw_dir = PROJECT_ROOT / config["dataset"]["raw_dir"]
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    all_windows = []
    all_labels = []
    all_loads = []
    all_class_names = []
    all_file_names = []
    all_start_indices = []

    items = [item for item in config["files"] if item["load_hp"] in TARGET_LOADS]
    for item in items:
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
            normalize=NORMALIZE,
        )
        num_windows = windows.shape[0]

        all_windows.append(windows)
        all_labels.append(np.full(num_windows, label, dtype=np.int64))
        all_loads.append(np.full(num_windows, load_hp, dtype=np.int64))
        all_class_names.append(np.full(num_windows, class_name))
        all_file_names.append(np.full(num_windows, file_name))
        all_start_indices.append(start_indices)

        print(f"File: {file_name}")
        print(f"  class_name: {class_name}")
        print(f"  label: {label}")
        print(f"  load_hp: {load_hp}")
        print(f"  signal length: {len(signal)}")
        print(f"  number of windows: {num_windows}")

    X = np.concatenate(all_windows, axis=0).astype(np.float32)
    y = np.concatenate(all_labels, axis=0).astype(np.int64)
    load_hp = np.concatenate(all_loads, axis=0).astype(np.int64)
    class_name = np.concatenate(all_class_names, axis=0)
    file_name = np.concatenate(all_file_names, axis=0)
    start_index = np.concatenate(all_start_indices, axis=0).astype(np.int64)

    np.savez(
        OUTPUT_PATH,
        X=X,
        y=y,
        load_hp=load_hp,
        class_name=class_name,
        file_name=file_name,
        start_index=start_index,
    )

    print(f"processed path: {OUTPUT_PATH}")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"load_hp distribution: {dict(Counter(load_hp.tolist()))}")
    print(f"label distribution: {dict(Counter(y.tolist()))}")
    print("processed loads 0/2/3 built.")


if __name__ == "__main__":
    main()

"""Verify that src/data/cwru_reader.py can stably read DE signals from all raw CWRU files."""

import sys
from pathlib import Path

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.cwru_reader import load_de_signal  # noqa: E402


RAW_DIR = PROJECT_ROOT / "data/raw/cwru_12k_de"
CONFIG_PATH = PROJECT_ROOT / "configs/dataset/cwru_minimal_files.yaml"


def load_source_file_id_map() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    return {
        item["save_name"]: item["source_file_id"]
        for item in config["files"]
    }


def main() -> None:
    source_file_id_map = load_source_file_id_map()
    mat_files = sorted(RAW_DIR.glob("*.mat"))
    success_count = 0
    failed_count = 0

    for mat_path in mat_files:
        source_file_id = source_file_id_map.get(mat_path.name)

        try:
            signal = load_de_signal(
                str(mat_path),
                expected_source_file_id=source_file_id,
            )
            success_count += 1

            print(f"File: {mat_path.name}")
            print(f"  source_file_id: {source_file_id}")
            print(f"  signal shape: {signal.shape}")
            print(f"  dtype: {signal.dtype}")
            print(f"  min: {np.min(signal)}")
            print(f"  max: {np.max(signal)}")
            print(f"  mean: {np.mean(signal)}")
            print(f"  std: {np.std(signal)}")
        except Exception as error:
            failed_count += 1
            print(f"[FAIL] {mat_path.name} {error}")

    print(f"Expected files: {len(mat_files)}")
    print(f"Successfully read files: {success_count}")
    print(f"Failed files: {failed_count}")


if __name__ == "__main__":
    main()

"""Check CWRU raw .mat files for load 0/2/3 coverage and DE_time readability."""

import sys
from collections import Counter
from pathlib import Path

import yaml
from scipy.io import loadmat


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.cwru_reader import find_de_key  # noqa: E402


CONFIG_PATH = PROJECT_ROOT / "configs/dataset/cwru_minimal_files.yaml"
TARGET_LOADS = {0, 2, 3}


def inspect_file(path: Path, source_file_id: int) -> int:
    mat_data = loadmat(path)
    de_key = find_de_key(mat_data, expected_source_file_id=source_file_id)
    signal = mat_data[de_key].squeeze()
    if signal.ndim != 1 or signal.size == 0:
        raise ValueError(f"{path} has invalid DE_time signal shape: {signal.shape}")
    return int(signal.size)


def main() -> None:
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    raw_dir = PROJECT_ROOT / config["dataset"]["raw_dir"]
    items = [item for item in config["files"] if item["load_hp"] in TARGET_LOADS]
    expected_count = len(items)
    found_count = 0
    missing = []
    load_counts = Counter()
    rows = []

    for item in items:
        path = raw_dir / item["save_name"]
        load_counts[item["load_hp"]] += 1
        if not path.exists():
            missing.append(item["save_name"])
            continue

        signal_length = inspect_file(path, item["source_file_id"])
        found_count += 1
        rows.append((item, signal_length))

    print(f"expected files: {expected_count}")
    print(f"found files: {found_count}")
    print(f"missing files: {len(missing)}")
    print(f"load_hp distribution: {dict(sorted(load_counts.items()))}")

    if missing:
        print("missing file names:")
        for name in missing:
            print(f"  - {name}")

    print("file details:")
    for item, signal_length in rows:
        print(
            "  "
            f"logical_name={item['logical_name']}, "
            f"save_name={item['save_name']}, "
            f"load_hp={item['load_hp']}, "
            f"class_name={item['class_name']}, "
            f"signal length={signal_length}"
        )

    assert expected_count == 30, f"expected files must be 30, got {expected_count}"
    assert found_count == 30, f"found files must be 30, got {found_count}"
    assert not missing, f"missing files: {missing}"
    assert load_counts == Counter({0: 10, 2: 10, 3: 10}), load_counts

    print("raw files load 0/2/3 check passed.")


if __name__ == "__main__":
    main()

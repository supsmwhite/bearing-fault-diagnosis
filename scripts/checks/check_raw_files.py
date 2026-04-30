"""Check whether required CWRU raw .mat files have been manually placed."""

from pathlib import Path

import yaml


CONFIG_PATH = Path("configs/dataset/cwru_minimal_files.yaml")


def main() -> None:
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    raw_dir = Path(config["dataset"]["raw_dir"])
    expected_names = [
        f"{item['logical_name']}{config['dataset']['file_extension']}"
        for item in config["files"]
    ]

    found_files = []
    missing_files = []

    for filename in expected_names:
        file_path = raw_dir / filename
        if file_path.exists():
            found_files.append(filename)
        else:
            missing_files.append(filename)

    print("Found files:")
    for filename in found_files:
        print(f"  - {filename}")

    print("Missing files:")
    for filename in missing_files:
        print(f"  - {filename}")

    print(f"Expected files: {len(expected_names)}")
    print(f"Found files: {len(found_files)}")
    print(f"Missing files: {len(missing_files)}")


if __name__ == "__main__":
    main()

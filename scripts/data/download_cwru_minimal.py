"""Download the minimal .mat file set required for CWRU reproduction from the official CWRU website."""

from pathlib import Path
from urllib.error import HTTPError
from urllib.error import URLError
from urllib.request import urlretrieve


RAW_DIR = Path("data/raw/cwru_12k_de")

FILES = [
    {
        "logical_name": "Normal_2",
        "save_name": "Normal_2.mat",
        "url": "https://engineering.case.edu/sites/default/files/99.mat",
        "load_hp": 2,
        "class_name": "Normal",
    },
    {
        "logical_name": "Normal_3",
        "save_name": "Normal_3.mat",
        "url": "https://engineering.case.edu/sites/default/files/100.mat",
        "load_hp": 3,
        "class_name": "Normal",
    },
    {
        "logical_name": "IR007_2",
        "save_name": "IR007_2.mat",
        "url": "https://engineering.case.edu/sites/default/files/107.mat",
        "load_hp": 2,
        "class_name": "IR007",
    },
    {
        "logical_name": "IR007_3",
        "save_name": "IR007_3.mat",
        "url": "https://engineering.case.edu/sites/default/files/108.mat",
        "load_hp": 3,
        "class_name": "IR007",
    },
    {
        "logical_name": "B007_2",
        "save_name": "B007_2.mat",
        "url": "https://engineering.case.edu/sites/default/files/120.mat",
        "load_hp": 2,
        "class_name": "B007",
    },
    {
        "logical_name": "B007_3",
        "save_name": "B007_3.mat",
        "url": "https://engineering.case.edu/sites/default/files/121.mat",
        "load_hp": 3,
        "class_name": "B007",
    },
    {
        "logical_name": "OR007@6_2",
        "save_name": "OR007@6_2.mat",
        "url": "https://engineering.case.edu/sites/default/files/132.mat",
        "load_hp": 2,
        "class_name": "OR007@6",
    },
    {
        "logical_name": "OR007@6_3",
        "save_name": "OR007@6_3.mat",
        "url": "https://engineering.case.edu/sites/default/files/133.mat",
        "load_hp": 3,
        "class_name": "OR007@6",
    },
    {
        "logical_name": "IR014_2",
        "save_name": "IR014_2.mat",
        "url": "https://engineering.case.edu/sites/default/files/171.mat",
        "load_hp": 2,
        "class_name": "IR014",
    },
    {
        "logical_name": "IR014_3",
        "save_name": "IR014_3.mat",
        "url": "https://engineering.case.edu/sites/default/files/172.mat",
        "load_hp": 3,
        "class_name": "IR014",
    },
    {
        "logical_name": "B014_2",
        "save_name": "B014_2.mat",
        "url": "https://engineering.case.edu/sites/default/files/187.mat",
        "load_hp": 2,
        "class_name": "B014",
    },
    {
        "logical_name": "B014_3",
        "save_name": "B014_3.mat",
        "url": "https://engineering.case.edu/sites/default/files/188.mat",
        "load_hp": 3,
        "class_name": "B014",
    },
    {
        "logical_name": "OR014@6_2",
        "save_name": "OR014@6_2.mat",
        "url": "https://engineering.case.edu/sites/default/files/199.mat",
        "load_hp": 2,
        "class_name": "OR014@6",
    },
    {
        "logical_name": "OR014@6_3",
        "save_name": "OR014@6_3.mat",
        "url": "https://engineering.case.edu/sites/default/files/200.mat",
        "load_hp": 3,
        "class_name": "OR014@6",
    },
    {
        "logical_name": "IR021_2",
        "save_name": "IR021_2.mat",
        "url": "https://engineering.case.edu/sites/default/files/211.mat",
        "load_hp": 2,
        "class_name": "IR021",
    },
    {
        "logical_name": "IR021_3",
        "save_name": "IR021_3.mat",
        "url": "https://engineering.case.edu/sites/default/files/212.mat",
        "load_hp": 3,
        "class_name": "IR021",
    },
    {
        "logical_name": "B021_2",
        "save_name": "B021_2.mat",
        "url": "https://engineering.case.edu/sites/default/files/224.mat",
        "load_hp": 2,
        "class_name": "B021",
    },
    {
        "logical_name": "B021_3",
        "save_name": "B021_3.mat",
        "url": "https://engineering.case.edu/sites/default/files/225.mat",
        "load_hp": 3,
        "class_name": "B021",
    },
    {
        "logical_name": "OR021@6_2",
        "save_name": "OR021@6_2.mat",
        "url": "https://engineering.case.edu/sites/default/files/236.mat",
        "load_hp": 2,
        "class_name": "OR021@6",
    },
    {
        "logical_name": "OR021@6_3",
        "save_name": "OR021@6_3.mat",
        "url": "https://engineering.case.edu/sites/default/files/237.mat",
        "load_hp": 3,
        "class_name": "OR021@6",
    },
]


def download_file(url: str, target_path: Path) -> None:
    temp_path = target_path.with_suffix(target_path.suffix + ".tmp")
    try:
        urlretrieve(url, temp_path)
        temp_path.replace(target_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    completed_count = 0
    failed_count = 0

    for item in FILES:
        save_name = item["save_name"]
        target_path = RAW_DIR / save_name

        if target_path.exists():
            completed_count += 1
            print(f"[SKIP] {save_name} already exists")
            continue

        try:
            download_file(item["url"], target_path)
            completed_count += 1
            print(f"[OK] {save_name}")
        except (HTTPError, URLError, OSError) as error:
            failed_count += 1
            print(f"[FAIL] {save_name} {error}")

    print(f"Expected files: {len(FILES)}")
    print(f"Downloaded/skipped files: {completed_count}")
    print(f"Failed files: {failed_count}")


if __name__ == "__main__":
    main()


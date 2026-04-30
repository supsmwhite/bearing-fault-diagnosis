"""Download and verify CWRU 12k Drive End load=0 raw .mat files."""

import sys
from pathlib import Path
from urllib.error import HTTPError
from urllib.error import URLError
from urllib.request import urlretrieve

import yaml
from scipy.io import loadmat


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.cwru_reader import find_de_key  # noqa: E402


CONFIG_PATH = PROJECT_ROOT / "configs/dataset/cwru_minimal_files.yaml"


def download_file(url: str, target_path: Path) -> None:
    temp_path = target_path.with_suffix(target_path.suffix + ".tmp")
    try:
        urlretrieve(url, temp_path)
        temp_path.replace(target_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def verify_mat_file(path: Path, source_file_id: int) -> None:
    try:
        mat_data = loadmat(path)
        de_key = find_de_key(mat_data, expected_source_file_id=source_file_id)
    except Exception as error:
        raise ValueError(f"{path} is not a valid readable CWRU .mat file: {error}") from error

    signal = mat_data[de_key].squeeze()
    if signal.ndim != 1 or signal.size == 0:
        raise ValueError(f"{path} has invalid DE_time signal shape: {signal.shape}")


def main() -> None:
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    raw_dir = PROJECT_ROOT / config["dataset"]["raw_dir"]
    raw_dir.mkdir(parents=True, exist_ok=True)
    load0_items = [item for item in config["files"] if item["load_hp"] == 0]

    for item in load0_items:
        target_path = raw_dir / item["save_name"]
        if target_path.exists():
            status = "skipped"
        else:
            try:
                download_file(item["url"], target_path)
                status = "downloaded"
            except (HTTPError, URLError, OSError) as error:
                raise RuntimeError(f"Failed to download {item['save_name']}: {error}") from error

        verify_mat_file(target_path, item["source_file_id"])
        print(f"{item['logical_name']}: {status}, verified")

    print("load 0 raw files downloaded and verified.")


if __name__ == "__main__":
    main()

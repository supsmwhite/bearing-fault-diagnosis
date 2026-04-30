"""Check that the 1D-CNN baseline can forward one CWRUWindowDataset batch into 10-class logits."""

import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import CWRUWindowDataset  # noqa: E402
from src.models.cnn1d import CNN1D  # noqa: E402


PROCESSED_PATH = PROJECT_ROOT / "data/processed/cwru_12k_de_windows.npz"
SPLIT_PATH = PROJECT_ROOT / "data/processed/cwru_12k_de_splits_3to2.npz"
BATCH_SIZE = 32


def count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def main() -> None:
    splits = np.load(SPLIT_PATH, allow_pickle=False)
    train_source_indices = splits["train_source_indices"]

    dataset = CWRUWindowDataset(
        str(PROCESSED_PATH),
        indices=train_source_indices,
        return_meta=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    x, y = next(iter(loader))
    model = CNN1D(num_classes=10)

    with torch.no_grad():
        logits = model(x)

    print(f"x shape: {tuple(x.shape)}")
    print(f"y shape: {tuple(y.shape)}")
    print(f"logits shape: {tuple(logits.shape)}")
    print(f"logits dtype: {logits.dtype}")
    print(f"model parameter count: {count_parameters(model)}")
    print(f"Has NaN: {torch.isnan(logits).any().item()}")
    print(f"Has inf: {torch.isinf(logits).any().item()}")


if __name__ == "__main__":
    main()

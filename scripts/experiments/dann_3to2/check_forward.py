"""Check DANN forward pass on one source batch and one target batch for the 3 hp to 2 hp task."""

import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import CWRUWindowDataset  # noqa: E402
from src.models.dann import DANN  # noqa: E402


PROCESSED_PATH = PROJECT_ROOT / "data/processed/cwru_12k_de_windows.npz"
SPLIT_PATH = PROJECT_ROOT / "data/processed/cwru_12k_de_splits_3to2.npz"
BATCH_SIZE = 32


def count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def has_nan_or_inf(*tensors) -> tuple:
    has_nan = any(torch.isnan(tensor).any().item() for tensor in tensors)
    has_inf = any(torch.isinf(tensor).any().item() for tensor in tensors)
    return has_nan, has_inf


def main() -> None:
    splits = np.load(SPLIT_PATH, allow_pickle=False)
    train_source_indices = splits["train_source_indices"]
    train_target_unlabeled_indices = splits["train_target_unlabeled_indices"]

    source_dataset = CWRUWindowDataset(
        str(PROCESSED_PATH),
        indices=train_source_indices,
        return_meta=False,
    )
    target_dataset = CWRUWindowDataset(
        str(PROCESSED_PATH),
        indices=train_target_unlabeled_indices,
        return_meta=False,
    )

    source_loader = DataLoader(source_dataset, batch_size=BATCH_SIZE, shuffle=False)
    target_loader = DataLoader(target_dataset, batch_size=BATCH_SIZE, shuffle=False)

    source_x, source_y = next(iter(source_loader))
    target_x, target_y = next(iter(target_loader))

    model = DANN(num_classes=10, grl_lambda=1.0)

    source_class_logits, source_domain_logits = model(source_x)
    target_class_logits, target_domain_logits = model(target_x)
    has_nan, has_inf = has_nan_or_inf(
        source_class_logits,
        source_domain_logits,
        target_class_logits,
        target_domain_logits,
    )

    print("source:")
    print(f"  source_x shape: {tuple(source_x.shape)}")
    print(f"  source_y shape: {tuple(source_y.shape)}")
    print(f"  source_class_logits shape: {tuple(source_class_logits.shape)}")
    print(f"  source_domain_logits shape: {tuple(source_domain_logits.shape)}")

    print("target:")
    print(f"  target_x shape: {tuple(target_x.shape)}")
    print(f"  target_y shape: {tuple(target_y.shape)}")
    print(f"  target_class_logits shape: {tuple(target_class_logits.shape)}")
    print(f"  target_domain_logits shape: {tuple(target_domain_logits.shape)}")

    print("model:")
    print(f"  total parameter count: {count_parameters(model)}")
    print(f"  has_nan: {has_nan}")
    print(f"  has_inf: {has_inf}")


if __name__ == "__main__":
    main()


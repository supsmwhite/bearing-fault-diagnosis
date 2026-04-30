"""Check Deep CORAL forward pass, CORAL loss, and backward pass."""

import sys
from pathlib import Path

import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.deep_coral import DeepCORAL1D  # noqa: E402
from src.train.train_deep_coral import coral_loss  # noqa: E402


def main() -> None:
    torch.manual_seed(42)

    model = DeepCORAL1D(num_classes=10)
    source_x = torch.randn(8, 1, 1024)
    target_x = torch.randn(8, 1, 1024)
    source_y = torch.randint(0, 10, (8,))

    source_logits, source_features = model(source_x)
    target_logits, target_features = model(target_x)

    assert source_logits.shape == (8, 10)
    assert target_logits.shape == (8, 10)
    assert source_features.shape == (8, 128)
    assert target_features.shape == (8, 128)

    criterion = nn.CrossEntropyLoss()
    class_loss = criterion(source_logits, source_y)
    coral = coral_loss(source_features, target_features)
    total_loss = class_loss + coral
    total_loss.backward()

    print(f"class_logits shape: {tuple(source_logits.shape)}")
    print(f"source_features shape: {tuple(source_features.shape)}")
    print(f"target_features shape: {tuple(target_features.shape)}")
    print(f"class_loss: {class_loss.item():.6f}")
    print(f"coral_loss: {coral.item():.6f}")
    print(f"total_loss: {total_loss.item():.6f}")
    print("Deep CORAL forward/loss/backward check passed.")


if __name__ == "__main__":
    main()

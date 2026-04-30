"""Train source-only CNN1D baseline on train_source and validate on val_source."""

import csv
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import CWRUWindowDataset  # noqa: E402
from src.models.cnn1d import CNN1D  # noqa: E402
from src.train.train_baseline import evaluate  # noqa: E402
from src.train.train_baseline import train_one_epoch  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402


PROCESSED_PATH = PROJECT_ROOT / "data/processed/cwru_12k_de_windows.npz"
SPLIT_PATH = PROJECT_ROOT / "data/processed/cwru_12k_de_splits_3to2.npz"
EXPERIMENT_DIR = PROJECT_ROOT / "outputs/experiments/cnn1d_source_3to2"
CHECKPOINT_PATH = EXPERIMENT_DIR / "checkpoints/cnn1d_source_best.pt"
LOG_PATH = EXPERIMENT_DIR / "logs/train_log.csv"

TRAIN_CONFIG = {
    "batch_size": 64,
    "epochs": 20,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "num_classes": 10,
    "seed": 42,
}


def compute_class_weights(y: np.ndarray, indices: np.ndarray, num_classes: int) -> torch.Tensor:
    labels = y[indices]
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    if np.any(counts == 0):
        missing = np.where(counts == 0)[0].tolist()
        raise ValueError(f"Missing classes in train_source labels: {missing}")

    weights = counts.sum() / (num_classes * counts)
    return torch.as_tensor(weights, dtype=torch.float32)


def write_log(rows: list) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "epoch",
        "train_loss",
        "train_accuracy",
        "train_macro_f1",
        "val_loss",
        "val_accuracy",
        "val_macro_f1",
    ]
    with LOG_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    set_seed(TRAIN_CONFIG["seed"])
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    processed = np.load(PROCESSED_PATH, allow_pickle=False)
    splits = np.load(SPLIT_PATH, allow_pickle=False)

    train_source_indices = splits["train_source_indices"]
    val_source_indices = splits["val_source_indices"]

    train_dataset = CWRUWindowDataset(
        str(PROCESSED_PATH),
        indices=train_source_indices,
        return_meta=False,
    )
    val_dataset = CWRUWindowDataset(
        str(PROCESSED_PATH),
        indices=val_source_indices,
        return_meta=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAIN_CONFIG["batch_size"],
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=TRAIN_CONFIG["batch_size"],
        shuffle=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN1D(num_classes=TRAIN_CONFIG["num_classes"]).to(device)
    class_weights = compute_class_weights(
        processed["y"],
        train_source_indices,
        TRAIN_CONFIG["num_classes"],
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=TRAIN_CONFIG["learning_rate"],
        weight_decay=TRAIN_CONFIG["weight_decay"],
    )

    best_epoch = 0
    best_val_macro_f1 = -1.0
    log_rows = []

    for epoch in range(1, TRAIN_CONFIG["epochs"] + 1):
        train_loss, train_acc, train_f1 = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )
        val_loss, val_acc, val_f1 = evaluate(
            model,
            val_loader,
            criterion,
            device,
        )

        log_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "train_macro_f1": train_f1,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "val_macro_f1": val_f1,
            }
        )

        print(
            f"Epoch {epoch:03d}/{TRAIN_CONFIG['epochs']:03d} | "
            f"train_loss={train_loss:.6f} train_acc={train_acc:.6f} train_f1={train_f1:.6f} | "
            f"val_loss={val_loss:.6f} val_acc={val_acc:.6f} val_f1={val_f1:.6f}"
        )

        if val_f1 > best_val_macro_f1:
            best_epoch = epoch
            best_val_macro_f1 = val_f1
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_val_macro_f1": best_val_macro_f1,
                    "train_config": TRAIN_CONFIG,
                },
                CHECKPOINT_PATH,
            )

    write_log(log_rows)

    print(f"Best epoch: {best_epoch}")
    print(f"Best val macro-F1: {best_val_macro_f1:.6f}")
    print(f"Checkpoint path: {CHECKPOINT_PATH}")
    print(f"Log path: {LOG_PATH}")


if __name__ == "__main__":
    main()

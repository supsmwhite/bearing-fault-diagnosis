"""Train Deep CORAL with train-stat-only normalized windows."""

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
from src.models.deep_coral import DeepCORAL1D  # noqa: E402
from src.train.train_deep_coral import evaluate_source_deep_coral  # noqa: E402
from src.train.train_deep_coral import train_one_epoch_deep_coral  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402


PROCESSED_PATH = PROJECT_ROOT / "data/processed/cwru_12k_de_windows_train_stat_norm.npz"
SPLIT_PATH = PROJECT_ROOT / "data/processed/cwru_12k_de_splits_3to2.npz"
EXPERIMENT_DIR = PROJECT_ROOT / "outputs/experiments/deep_coral_3to2_train_stat_norm"
CHECKPOINT_PATH = EXPERIMENT_DIR / "checkpoints/deep_coral_best.pt"
FINAL_CHECKPOINT_PATH = EXPERIMENT_DIR / "checkpoints/deep_coral_final.pt"
LOG_PATH = EXPERIMENT_DIR / "logs/train_log.csv"

BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
CORAL_LOSS_WEIGHT = 1.0
SEED = 42
NUM_CLASSES = 10


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
        "total_loss",
        "class_loss",
        "coral_loss",
        "source_train_accuracy",
        "source_train_macro_f1",
        "val_loss",
        "val_accuracy",
        "val_macro_f1",
    ]
    with LOG_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    set_seed(SEED)
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)

    processed = np.load(PROCESSED_PATH, allow_pickle=False)
    splits = np.load(SPLIT_PATH, allow_pickle=False)

    source_train_indices = splits["train_source_indices"]
    source_val_indices = splits["val_source_indices"]
    target_train_indices = splits["train_target_unlabeled_indices"]

    source_train_loader = DataLoader(
        CWRUWindowDataset(str(PROCESSED_PATH), indices=source_train_indices),
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
    )
    source_val_loader = DataLoader(
        CWRUWindowDataset(str(PROCESSED_PATH), indices=source_val_indices),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    target_train_loader = DataLoader(
        CWRUWindowDataset(str(PROCESSED_PATH), indices=target_train_indices),
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepCORAL1D(num_classes=NUM_CLASSES).to(device)
    class_weights = compute_class_weights(processed["y"], source_train_indices, NUM_CLASSES).to(device)
    class_criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    train_config = {
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "coral_loss_weight": CORAL_LOSS_WEIGHT,
        "seed": SEED,
        "use_class_weight": True,
    }

    best_epoch = 0
    best_val_macro_f1 = -1.0
    log_rows = []

    for epoch in range(1, EPOCHS + 1):
        total_loss, class_loss, coral, train_acc, train_f1 = train_one_epoch_deep_coral(
            model,
            source_train_loader,
            target_train_loader,
            class_criterion,
            optimizer,
            device,
            coral_loss_weight=CORAL_LOSS_WEIGHT,
        )
        val_loss, val_acc, val_f1 = evaluate_source_deep_coral(
            model,
            source_val_loader,
            class_criterion,
            device,
        )
        log_rows.append(
            {
                "epoch": epoch,
                "total_loss": total_loss,
                "class_loss": class_loss,
                "coral_loss": coral,
                "source_train_accuracy": train_acc,
                "source_train_macro_f1": train_f1,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "val_macro_f1": val_f1,
            }
        )
        print(
            f"Epoch {epoch:03d}/{EPOCHS:03d} | total_loss={total_loss:.6f} "
            f"class_loss={class_loss:.6f} coral_loss={coral:.6f} "
            f"source_train_acc={train_acc:.6f} source_train_f1={train_f1:.6f} | "
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
                    "train_config": train_config,
                },
                CHECKPOINT_PATH,
            )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": EPOCHS,
            "best_val_macro_f1": best_val_macro_f1,
            "train_config": train_config,
        },
        FINAL_CHECKPOINT_PATH,
    )
    write_log(log_rows)

    print(f"Best epoch: {best_epoch}")
    print(f"Best source-val macro-F1: {best_val_macro_f1:.6f}")
    print(f"Best checkpoint path: {CHECKPOINT_PATH}")
    print(f"Final checkpoint path: {FINAL_CHECKPOINT_PATH}")
    print(f"Log path: {LOG_PATH}")


if __name__ == "__main__":
    main()

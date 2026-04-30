"""Train source-only CNN1D baseline for 0 hp -> 3 hp."""

import csv
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
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


CONFIG_PATH = PROJECT_ROOT / "configs/experiments/cnn1d_source_0to3.yaml"


def compute_class_weights(y: np.ndarray, indices: np.ndarray, num_classes: int) -> torch.Tensor:
    labels = y[indices]
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    if np.any(counts == 0):
        missing = np.where(counts == 0)[0].tolist()
        raise ValueError(f"Missing classes in train_source labels: {missing}")
    weights = counts.sum() / (num_classes * counts)
    return torch.as_tensor(weights, dtype=torch.float32)


def write_log(log_path: Path, rows: list) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "epoch",
        "train_loss",
        "train_accuracy",
        "train_macro_f1",
        "val_loss",
        "val_accuracy",
        "val_macro_f1",
    ]
    with log_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    training_config = config["training"]
    data_config = config["data"]
    output_config = config["outputs"]
    experiment_config = config["experiment"]
    set_seed(training_config["seed"])

    processed_path = PROJECT_ROOT / data_config["processed_npz"]
    split_path = PROJECT_ROOT / data_config["split_npz"]
    checkpoint_path = PROJECT_ROOT / output_config["checkpoint"]
    log_path = PROJECT_ROOT / output_config["train_log"]
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    processed = np.load(processed_path, allow_pickle=False)
    splits = np.load(split_path, allow_pickle=False)
    train_source_indices = splits[data_config["train_split"]]
    val_source_indices = splits[data_config["val_split"]]

    train_loader = DataLoader(
        CWRUWindowDataset(str(processed_path), indices=train_source_indices, return_meta=False),
        batch_size=training_config["batch_size"],
        shuffle=True,
    )
    val_loader = DataLoader(
        CWRUWindowDataset(str(processed_path), indices=val_source_indices, return_meta=False),
        batch_size=training_config["batch_size"],
        shuffle=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN1D(num_classes=experiment_config["num_classes"]).to(device)
    class_weights = None
    if training_config.get("use_class_weight", True):
        class_weights = compute_class_weights(
            processed["y"],
            train_source_indices,
            experiment_config["num_classes"],
        ).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training_config["learning_rate"],
        weight_decay=training_config["weight_decay"],
    )

    best_epoch = 0
    best_val_macro_f1 = -1.0
    log_rows = []

    for epoch in range(1, training_config["epochs"] + 1):
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
            f"Epoch {epoch:03d}/{training_config['epochs']:03d} | "
            f"train_loss={train_loss:.6f} train_accuracy={train_acc:.6f} "
            f"train_macro_f1={train_f1:.6f} | "
            f"val_loss={val_loss:.6f} val_accuracy={val_acc:.6f} "
            f"val_macro_f1={val_f1:.6f}"
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
                    "train_config": training_config,
                },
                checkpoint_path,
            )

    write_log(log_path, log_rows)
    print(f"Best epoch: {best_epoch}")
    print(f"Best val macro-F1: {best_val_macro_f1:.6f}")
    print(f"Checkpoint path: {checkpoint_path}")
    print(f"Log path: {log_path}")


if __name__ == "__main__":
    main()

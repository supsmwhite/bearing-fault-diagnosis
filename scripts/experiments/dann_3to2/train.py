"""Train DANN for 3 hp to 2 hp adaptation without using target test labels."""

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
from src.models.dann import DANN  # noqa: E402
from src.train.train_dann import evaluate_source  # noqa: E402
from src.train.train_dann import train_one_epoch_dann  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402


CONFIG_PATH = PROJECT_ROOT / "configs/experiments/dann_3to2.yaml"


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
        "total_loss",
        "class_loss",
        "domain_loss",
        "source_train_accuracy",
        "source_train_macro_f1",
        "domain_accuracy",
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

    set_seed(training_config["seed"])

    processed_path = PROJECT_ROOT / data_config["processed_npz"]
    split_path = PROJECT_ROOT / data_config["split_npz"]
    checkpoint_path = PROJECT_ROOT / output_config["checkpoint"]
    final_checkpoint_path = PROJECT_ROOT / output_config["final_checkpoint"]
    log_path = PROJECT_ROOT / output_config["train_log"]
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    processed = np.load(processed_path, allow_pickle=False)
    splits = np.load(split_path, allow_pickle=False)

    source_train_indices = splits[data_config["source_train_split"]]
    source_val_indices = splits[data_config["source_val_split"]]
    target_train_indices = splits[data_config["target_train_split"]]

    source_train_dataset = CWRUWindowDataset(
        str(processed_path),
        indices=source_train_indices,
        return_meta=False,
    )
    source_val_dataset = CWRUWindowDataset(
        str(processed_path),
        indices=source_val_indices,
        return_meta=False,
    )
    target_train_dataset = CWRUWindowDataset(
        str(processed_path),
        indices=target_train_indices,
        return_meta=False,
    )

    source_train_loader = DataLoader(
        source_train_dataset,
        batch_size=training_config["batch_size"],
        shuffle=True,
    )
    source_val_loader = DataLoader(
        source_val_dataset,
        batch_size=training_config["batch_size"],
        shuffle=False,
    )
    target_train_loader = DataLoader(
        target_train_dataset,
        batch_size=training_config["batch_size"],
        shuffle=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DANN(
        num_classes=config["experiment"]["num_classes"],
        grl_lambda=training_config["grl_lambda"],
    ).to(device)

    class_weights = compute_class_weights(
        processed["y"],
        source_train_indices,
        config["experiment"]["num_classes"],
    ).to(device)
    class_criterion = nn.CrossEntropyLoss(weight=class_weights)
    domain_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training_config["learning_rate"],
        weight_decay=training_config["weight_decay"],
    )

    best_epoch = 0
    best_val_macro_f1 = -1.0
    log_rows = []

    for epoch in range(1, training_config["epochs"] + 1):
        (
            total_loss,
            class_loss,
            domain_loss,
            source_train_acc,
            source_train_f1,
            domain_acc,
        ) = train_one_epoch_dann(
            model,
            source_train_loader,
            target_train_loader,
            class_criterion,
            domain_criterion,
            optimizer,
            device,
            domain_loss_weight=training_config["domain_loss_weight"],
        )
        val_loss, val_acc, val_f1 = evaluate_source(
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
                "domain_loss": domain_loss,
                "source_train_accuracy": source_train_acc,
                "source_train_macro_f1": source_train_f1,
                "domain_accuracy": domain_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "val_macro_f1": val_f1,
            }
        )

        print(
            f"Epoch {epoch:03d}/{training_config['epochs']:03d} | "
            f"total_loss={total_loss:.6f} class_loss={class_loss:.6f} "
            f"domain_loss={domain_loss:.6f} source_train_acc={source_train_acc:.6f} "
            f"source_train_f1={source_train_f1:.6f} domain_acc={domain_acc:.6f} | "
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
                    "train_config": training_config,
                },
                checkpoint_path,
            )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": training_config["epochs"],
            "best_val_macro_f1": best_val_macro_f1,
            "train_config": training_config,
        },
        final_checkpoint_path,
    )
    write_log(log_path, log_rows)

    print(f"Best epoch: {best_epoch}")
    print(f"Best val macro-F1: {best_val_macro_f1:.6f}")
    print(f"Best checkpoint path: {checkpoint_path}")
    print(f"Final checkpoint path: {final_checkpoint_path}")
    print(f"Log path: {log_path}")


if __name__ == "__main__":
    main()


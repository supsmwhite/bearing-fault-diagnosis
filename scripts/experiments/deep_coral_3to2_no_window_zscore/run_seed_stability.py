"""Run 3-seed stability check for no-window-zscore Deep CORAL."""

import csv
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support
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


PROCESSED_PATH = PROJECT_ROOT / "data/processed/cwru_12k_de_windows_no_window_zscore.npz"
SPLIT_PATH = PROJECT_ROOT / "data/processed/cwru_12k_de_splits_3to2.npz"
SUMMARY_PATH = PROJECT_ROOT / "outputs/experiments/deep_coral_3to2_no_window_zscore_seed_stability_summary.csv"

SEEDS = [42, 43, 44]
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
CORAL_LOSS_WEIGHT = 1.0
USE_CLASS_WEIGHT = True
NUM_CLASSES = 10
IR007_LABEL = 1
IR014_LABEL = 4
CLASS_NAMES = [
    "Normal",
    "IR007",
    "B007",
    "OR007@6",
    "IR014",
    "B014",
    "OR014@6",
    "IR021",
    "B021",
    "OR021@6",
]


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
        "coral_loss",
        "source_train_accuracy",
        "source_train_macro_f1",
        "val_loss",
        "val_accuracy",
        "val_macro_f1",
    ]
    with log_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def evaluate_target(model, dataloader, device) -> dict:
    model.eval()
    y_true_batches = []
    y_pred_batches = []

    for x, y in dataloader:
        x = x.to(device)
        class_logits, _features = model(x)
        y_pred_batches.append(torch.argmax(class_logits, dim=1).cpu().numpy())
        y_true_batches.append(y.cpu().numpy())

    y_true = np.concatenate(y_true_batches, axis=0)
    y_pred = np.concatenate(y_pred_batches, axis=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "y_true": y_true,
        "y_pred": y_pred,
    }


def save_predictions(path: Path, sample_indices: np.ndarray, metrics: dict, processed_data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "sample_index",
                "y_true",
                "y_pred",
                "load_hp",
                "class_name",
                "file_name",
                "start_index",
            ],
        )
        writer.writeheader()
        for row_id, sample_index in enumerate(sample_indices):
            writer.writerow(
                {
                    "sample_index": int(sample_index),
                    "y_true": int(metrics["y_true"][row_id]),
                    "y_pred": int(metrics["y_pred"][row_id]),
                    "load_hp": int(processed_data["load_hp"][sample_index]),
                    "class_name": str(processed_data["class_name"][sample_index]),
                    "file_name": str(processed_data["file_name"][sample_index]),
                    "start_index": int(processed_data["start_index"][sample_index]),
                }
            )


def save_metrics(path: Path, checkpoint_path: Path, checkpoint: dict, metrics: dict) -> None:
    cm = confusion_matrix(metrics["y_true"], metrics["y_pred"], labels=np.arange(NUM_CLASSES))
    report = classification_report(
        metrics["y_true"],
        metrics["y_pred"],
        labels=np.arange(NUM_CLASSES),
        target_names=CLASS_NAMES,
        zero_division=0,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        file.write(f"checkpoint_path: {checkpoint_path}\n")
        file.write(f"checkpoint_epoch: {checkpoint.get('epoch')}\n")
        file.write(f"checkpoint_best_val_macro_f1: {checkpoint.get('best_val_macro_f1')}\n")
        file.write(f"target_accuracy: {metrics['accuracy']:.6f}\n")
        file.write(f"target_macro_f1: {metrics['macro_f1']:.6f}\n")
        file.write("\nconfusion_matrix:\n")
        file.write(np.array2string(cm))
        file.write("\n\nclassification_report:\n")
        file.write(report)


def ir014_ir007_summary(metrics: dict) -> dict:
    y_true = metrics["y_true"]
    y_pred = metrics["y_pred"]
    precision, recall, f1, _support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=np.arange(NUM_CLASSES),
        zero_division=0,
    )
    return {
        "ir014_precision": float(precision[IR014_LABEL]),
        "ir014_recall": float(recall[IR014_LABEL]),
        "ir014_f1": float(f1[IR014_LABEL]),
        "ir014_to_ir007_count": int(((y_true == IR014_LABEL) & (y_pred == IR007_LABEL)).sum()),
        "ir007_precision": float(precision[IR007_LABEL]),
        "ir007_recall": float(recall[IR007_LABEL]),
    }


def load_checkpoint_metrics(
    checkpoint_path: Path,
    test_loader,
    test_target_indices: np.ndarray,
    processed_data,
    device,
    tag: str,
) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = DeepCORAL1D(num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    metrics = evaluate_target(model, test_loader, device)
    experiment_dir = checkpoint_path.parents[1]
    save_predictions(
        experiment_dir / f"logs/cross_load_predictions_{tag}.csv",
        test_target_indices,
        metrics,
        processed_data,
    )
    save_metrics(
        experiment_dir / f"logs/cross_load_test_metrics_{tag}.txt",
        checkpoint_path,
        checkpoint,
        metrics,
    )
    return metrics


def run_seed(seed: int, processed_data, splits, device) -> dict:
    set_seed(seed)
    experiment_dir = PROJECT_ROOT / f"outputs/experiments/deep_coral_3to2_no_window_zscore_seed{seed}"
    checkpoint_path = experiment_dir / "checkpoints/deep_coral_best.pt"
    final_checkpoint_path = experiment_dir / "checkpoints/deep_coral_final.pt"
    log_path = experiment_dir / "logs/train_log.csv"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    source_train_indices = splits["train_source_indices"]
    source_val_indices = splits["val_source_indices"]
    target_train_indices = splits["train_target_unlabeled_indices"]
    test_target_indices = splits["test_target_indices"]

    source_train_dataset = CWRUWindowDataset(str(PROCESSED_PATH), indices=source_train_indices)
    source_val_dataset = CWRUWindowDataset(str(PROCESSED_PATH), indices=source_val_indices)
    target_train_dataset = CWRUWindowDataset(str(PROCESSED_PATH), indices=target_train_indices)
    test_target_dataset = CWRUWindowDataset(str(PROCESSED_PATH), indices=test_target_indices)

    source_train_loader = DataLoader(source_train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    source_val_loader = DataLoader(source_val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    target_train_loader = DataLoader(target_train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    test_target_loader = DataLoader(test_target_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = DeepCORAL1D(num_classes=NUM_CLASSES).to(device)
    class_weights = None
    if USE_CLASS_WEIGHT:
        class_weights = compute_class_weights(
            processed_data["y"],
            source_train_indices,
            NUM_CLASSES,
        ).to(device)
    class_criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    train_config = {
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "coral_loss_weight": CORAL_LOSS_WEIGHT,
        "seed": seed,
        "use_class_weight": USE_CLASS_WEIGHT,
        "processed_npz": str(PROCESSED_PATH),
        "split_npz": str(SPLIT_PATH),
    }
    best_epoch = 0
    best_val_macro_f1 = -1.0
    log_rows = []

    for epoch in range(1, EPOCHS + 1):
        (
            total_loss,
            class_loss,
            coral,
            source_train_acc,
            source_train_f1,
        ) = train_one_epoch_deep_coral(
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
                "source_train_accuracy": source_train_acc,
                "source_train_macro_f1": source_train_f1,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "val_macro_f1": val_f1,
            }
        )
        print(
            f"Seed {seed} | Epoch {epoch:03d}/{EPOCHS:03d} | "
            f"total_loss={total_loss:.6f} class_loss={class_loss:.6f} "
            f"coral_loss={coral:.6f} source_train_acc={source_train_acc:.6f} "
            f"source_train_f1={source_train_f1:.6f} | "
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
                checkpoint_path,
            )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": EPOCHS,
            "best_val_macro_f1": best_val_macro_f1,
            "train_config": train_config,
        },
        final_checkpoint_path,
    )
    write_log(log_path, log_rows)

    best_metrics = load_checkpoint_metrics(
        checkpoint_path,
        test_target_loader,
        test_target_indices,
        processed_data,
        device,
        "best",
    )
    final_metrics = load_checkpoint_metrics(
        final_checkpoint_path,
        test_target_loader,
        test_target_indices,
        processed_data,
        device,
        "final",
    )
    final_focus = ir014_ir007_summary(final_metrics)

    row = {
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_macro_f1,
        "best_target_accuracy": best_metrics["accuracy"],
        "best_target_macro_f1": best_metrics["macro_f1"],
        "final_target_accuracy": final_metrics["accuracy"],
        "final_target_macro_f1": final_metrics["macro_f1"],
        "final_ir014_precision": final_focus["ir014_precision"],
        "final_ir014_recall": final_focus["ir014_recall"],
        "final_ir014_f1": final_focus["ir014_f1"],
        "final_ir014_to_ir007_count": final_focus["ir014_to_ir007_count"],
        "final_ir007_precision": final_focus["ir007_precision"],
        "final_ir007_recall": final_focus["ir007_recall"],
    }

    print(f"Seed {seed} summary:")
    print(f"  best epoch: {best_epoch}")
    print(f"  best source-val macro-F1: {best_val_macro_f1:.6f}")
    print(f"  best target accuracy: {best_metrics['accuracy']:.6f}")
    print(f"  best target macro-F1: {best_metrics['macro_f1']:.6f}")
    print(f"  final target accuracy: {final_metrics['accuracy']:.6f}")
    print(f"  final target macro-F1: {final_metrics['macro_f1']:.6f}")
    print(
        "  final IR014: "
        f"precision={row['final_ir014_precision']:.6f}, "
        f"recall={row['final_ir014_recall']:.6f}, "
        f"F1={row['final_ir014_f1']:.6f}, "
        f"IR014->IR007={row['final_ir014_to_ir007_count']}"
    )
    print(
        "  final IR007: "
        f"precision={row['final_ir007_precision']:.6f}, "
        f"recall={row['final_ir007_recall']:.6f}"
    )
    return row


def write_summary(rows: list) -> None:
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "seed",
        "best_epoch",
        "best_val_macro_f1",
        "best_target_accuracy",
        "best_target_macro_f1",
        "final_target_accuracy",
        "final_target_macro_f1",
        "final_ir014_precision",
        "final_ir014_recall",
        "final_ir014_f1",
        "final_ir014_to_ir007_count",
        "final_ir007_precision",
        "final_ir007_recall",
    ]
    with SUMMARY_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    processed_data = np.load(PROCESSED_PATH, allow_pickle=False)
    splits = np.load(SPLIT_PATH, allow_pickle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rows = [run_seed(seed, processed_data, splits, device) for seed in SEEDS]
    write_summary(rows)

    final_macro_f1 = np.asarray([row["final_target_macro_f1"] for row in rows], dtype=np.float64)
    final_ir014_recall = np.asarray([row["final_ir014_recall"] for row in rows], dtype=np.float64)
    final_ir014_to_ir007 = np.asarray(
        [row["final_ir014_to_ir007_count"] for row in rows],
        dtype=np.float64,
    )

    print("No-window-zscore Deep CORAL seed stability summary")
    print(f"final macro-F1 mean: {final_macro_f1.mean():.6f}")
    print(f"final macro-F1 std: {final_macro_f1.std(ddof=0):.6f}")
    print(f"final IR014 recall mean: {final_ir014_recall.mean():.6f}")
    print(f"final IR014 recall std: {final_ir014_recall.std(ddof=0):.6f}")
    print(f"final IR014 -> IR007 count mean: {final_ir014_to_ir007.mean():.6f}")
    print(f"final IR014 -> IR007 count std: {final_ir014_to_ir007.std(ddof=0):.6f}")
    print(f"Summary path: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()

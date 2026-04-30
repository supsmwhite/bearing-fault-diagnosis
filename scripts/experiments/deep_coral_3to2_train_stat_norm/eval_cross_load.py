"""Evaluate train-stat-norm Deep CORAL checkpoints."""

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
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import CWRUWindowDataset  # noqa: E402
from src.models.deep_coral import DeepCORAL1D  # noqa: E402


PROCESSED_PATH = PROJECT_ROOT / "data/processed/cwru_12k_de_windows_train_stat_norm.npz"
SPLIT_PATH = PROJECT_ROOT / "data/processed/cwru_12k_de_splits_3to2.npz"
EXPERIMENT_DIR = PROJECT_ROOT / "outputs/experiments/deep_coral_3to2_train_stat_norm"
BATCH_SIZE = 64
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


@torch.no_grad()
def evaluate_model(model, loader, device) -> dict:
    model.eval()
    y_true_batches = []
    y_pred_batches = []
    for x, y in loader:
        x = x.to(device)
        logits, _features = model(x)
        y_pred_batches.append(torch.argmax(logits, dim=1).cpu().numpy())
        y_true_batches.append(y.cpu().numpy())
    y_true = np.concatenate(y_true_batches, axis=0)
    y_pred = np.concatenate(y_pred_batches, axis=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "y_true": y_true,
        "y_pred": y_pred,
    }


def focus_metrics(metrics: dict) -> dict:
    precision, recall, f1, _support = precision_recall_fscore_support(
        metrics["y_true"],
        metrics["y_pred"],
        labels=np.arange(NUM_CLASSES),
        zero_division=0,
    )
    return {
        "ir014_recall": float(recall[IR014_LABEL]),
        "ir014_f1": float(f1[IR014_LABEL]),
        "ir014_to_ir007": int(((metrics["y_true"] == IR014_LABEL) & (metrics["y_pred"] == IR007_LABEL)).sum()),
        "ir007_precision": float(precision[IR007_LABEL]),
        "ir007_recall": float(recall[IR007_LABEL]),
    }


def save_predictions(path: Path, sample_indices: np.ndarray, metrics: dict, processed_data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["sample_index", "y_true", "y_pred", "load_hp", "class_name", "file_name", "start_index"],
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


def evaluate_checkpoint(tag: str, checkpoint_path: Path, loader, test_indices, processed_data, device) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = DeepCORAL1D(num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    metrics = evaluate_model(model, loader, device)
    save_predictions(EXPERIMENT_DIR / f"logs/cross_load_predictions_{tag}.csv", test_indices, metrics, processed_data)
    save_metrics(EXPERIMENT_DIR / f"logs/cross_load_test_metrics_{tag}.txt", checkpoint_path, checkpoint, metrics)
    focus = focus_metrics(metrics)
    print(f"Deep CORAL train-stat-norm {tag}:")
    print(f"  target accuracy: {metrics['accuracy']:.6f}")
    print(f"  target macro-F1: {metrics['macro_f1']:.6f}")
    print(f"  IR014 recall: {focus['ir014_recall']:.6f}")
    print(f"  IR014 F1: {focus['ir014_f1']:.6f}")
    print(f"  IR014 -> IR007 count: {focus['ir014_to_ir007']}")
    print(f"  IR007 precision: {focus['ir007_precision']:.6f}")
    print(f"  IR007 recall: {focus['ir007_recall']:.6f}")
    return metrics


def main() -> None:
    processed_data = np.load(PROCESSED_PATH, allow_pickle=False)
    splits = np.load(SPLIT_PATH, allow_pickle=False)
    test_indices = splits["test_target_indices"]
    loader = DataLoader(CWRUWindowDataset(str(PROCESSED_PATH), indices=test_indices), batch_size=BATCH_SIZE, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evaluate_checkpoint("best", EXPERIMENT_DIR / "checkpoints/deep_coral_best.pt", loader, test_indices, processed_data, device)
    evaluate_checkpoint("final", EXPERIMENT_DIR / "checkpoints/deep_coral_final.pt", loader, test_indices, processed_data, device)


if __name__ == "__main__":
    main()

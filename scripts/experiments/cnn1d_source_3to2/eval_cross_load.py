"""Evaluate the trained source-only CNN1D checkpoint on the target 2 hp test split."""

import csv
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import CWRUWindowDataset  # noqa: E402
from src.eval.evaluate_classification import evaluate_classification  # noqa: E402
from src.models.cnn1d import CNN1D  # noqa: E402


PROCESSED_PATH = PROJECT_ROOT / "data/processed/cwru_12k_de_windows.npz"
SPLIT_PATH = PROJECT_ROOT / "data/processed/cwru_12k_de_splits_3to2.npz"
EXPERIMENT_DIR = PROJECT_ROOT / "outputs/experiments/cnn1d_source_3to2"
CHECKPOINT_PATH = EXPERIMENT_DIR / "checkpoints/cnn1d_source_best.pt"
METRICS_PATH = EXPERIMENT_DIR / "logs/cross_load_test_metrics.txt"
PREDICTIONS_PATH = EXPERIMENT_DIR / "logs/cross_load_predictions.csv"
CONFUSION_MATRIX_PATH = EXPERIMENT_DIR / "figures/cross_load_confusion_matrix.png"
BATCH_SIZE = 64
NUM_CLASSES = 10
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


def safe_load_checkpoint(checkpoint_path, device):
    """
    Load a trusted local checkpoint with compatibility for PyTorch weights_only behavior.
    """
    try:
        return torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True,
        )
    except Exception:
        print(
            "weights_only=True failed, falling back to weights_only=False "
            "for trusted local checkpoint."
        )
        return torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )


def load_checkpoint(model: torch.nn.Module, device: torch.device) -> dict:
    checkpoint = safe_load_checkpoint(CHECKPOINT_PATH, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


def save_predictions(
    sample_indices: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    processed_data,
) -> None:
    PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_index",
        "y_true",
        "y_pred",
        "load_hp",
        "class_name",
        "file_name",
        "start_index",
    ]

    with PREDICTIONS_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row_id, sample_index in enumerate(sample_indices):
            writer.writerow(
                {
                    "sample_index": int(sample_index),
                    "y_true": int(y_true[row_id]),
                    "y_pred": int(y_pred[row_id]),
                    "load_hp": int(processed_data["load_hp"][sample_index]),
                    "class_name": str(processed_data["class_name"][sample_index]),
                    "file_name": str(processed_data["file_name"][sample_index]),
                    "start_index": int(processed_data["start_index"][sample_index]),
                }
            )


def save_confusion_matrix_figure(cm: np.ndarray) -> None:
    CONFUSION_MATRIX_PATH.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 7))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title("Cross-load Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(CLASS_NAMES))
    plt.xticks(tick_marks, CLASS_NAMES, rotation=45, ha="right")
    plt.yticks(tick_marks, CLASS_NAMES)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")

    threshold = cm.max() / 2.0 if cm.max() > 0 else 0
    for row in range(cm.shape[0]):
        for col in range(cm.shape[1]):
            color = "white" if cm[row, col] > threshold else "black"
            plt.text(col, row, str(cm[row, col]), ha="center", va="center", color=color)

    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH, dpi=150)
    plt.close()


def save_metrics_text(metrics: dict, cm: np.ndarray, report: str, checkpoint: dict) -> None:
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with METRICS_PATH.open("w", encoding="utf-8") as file:
        file.write(f"checkpoint_path: {CHECKPOINT_PATH}\n")
        file.write(f"checkpoint_epoch: {checkpoint.get('epoch')}\n")
        file.write(f"checkpoint_best_val_macro_f1: {checkpoint.get('best_val_macro_f1')}\n")
        file.write(f"target_accuracy: {metrics['accuracy']:.6f}\n")
        file.write(f"target_macro_f1: {metrics['macro_f1']:.6f}\n")
        file.write("\nconfusion_matrix:\n")
        file.write(np.array2string(cm))
        file.write("\n\nclassification_report:\n")
        file.write(report)


def main() -> None:
    processed_data = np.load(PROCESSED_PATH, allow_pickle=False)
    splits = np.load(SPLIT_PATH, allow_pickle=False)
    test_target_indices = splits["test_target_indices"]

    test_dataset = CWRUWindowDataset(
        str(PROCESSED_PATH),
        indices=test_target_indices,
        return_meta=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN1D(num_classes=NUM_CLASSES).to(device)
    checkpoint = load_checkpoint(model, device)

    metrics = evaluate_classification(model, test_loader, device)
    y_true = metrics["y_true"]
    y_pred = metrics["y_pred"]

    cm = confusion_matrix(y_true, y_pred, labels=np.arange(NUM_CLASSES))
    report = classification_report(
        y_true,
        y_pred,
        labels=np.arange(NUM_CLASSES),
        target_names=CLASS_NAMES,
        zero_division=0,
    )

    save_predictions(test_target_indices, y_true, y_pred, processed_data)
    save_confusion_matrix_figure(cm)
    save_metrics_text(metrics, cm, report, checkpoint)

    test_loads = processed_data["load_hp"][test_target_indices]
    test_labels = processed_data["y"][test_target_indices]

    print(f"Checkpoint path: {CHECKPOINT_PATH}")
    print("Test split name: test_target_indices")
    print(f"Test samples: {len(test_target_indices)}")
    print(f"Test load distribution: {dict(Counter(test_loads.tolist()))}")
    print(f"Test label distribution: {dict(Counter(test_labels.tolist()))}")
    print(f"Target accuracy: {metrics['accuracy']:.6f}")
    print(f"Target macro-F1: {metrics['macro_f1']:.6f}")
    print(f"Metrics path: {METRICS_PATH}")
    print(f"Predictions path: {PREDICTIONS_PATH}")
    print(f"Confusion matrix path: {CONFUSION_MATRIX_PATH}")


if __name__ == "__main__":
    main()

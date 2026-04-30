"""Evaluate source-only CNN1D 0 hp -> 3 hp checkpoint on target test."""

import csv
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import CWRUWindowDataset  # noqa: E402
from src.eval.evaluate_classification import evaluate_classification  # noqa: E402
from src.models.cnn1d import CNN1D  # noqa: E402


CONFIG_PATH = PROJECT_ROOT / "configs/experiments/cnn1d_source_0to3.yaml"
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
    try:
        return torch.load(checkpoint_path, map_location=device, weights_only=True)
    except Exception:
        print(
            "weights_only=True failed, falling back to weights_only=False "
            "for trusted local checkpoint."
        )
        return torch.load(checkpoint_path, map_location=device, weights_only=False)


def save_predictions(path: Path, sample_indices: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, processed_data) -> None:
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
                    "y_true": int(y_true[row_id]),
                    "y_pred": int(y_pred[row_id]),
                    "load_hp": int(processed_data["load_hp"][sample_index]),
                    "class_name": str(processed_data["class_name"][sample_index]),
                    "file_name": str(processed_data["file_name"][sample_index]),
                    "start_index": int(processed_data["start_index"][sample_index]),
                }
            )


def save_confusion_matrix(path: Path, cm: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 7))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title("CNN1D Source-only 0->3 Confusion Matrix")
    plt.colorbar()
    ticks = np.arange(len(CLASS_NAMES))
    plt.xticks(ticks, CLASS_NAMES, rotation=45, ha="right")
    plt.yticks(ticks, CLASS_NAMES)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    threshold = cm.max() / 2.0 if cm.max() > 0 else 0
    for row in range(cm.shape[0]):
        for col in range(cm.shape[1]):
            color = "white" if cm[row, col] > threshold else "black"
            plt.text(col, row, str(cm[row, col]), ha="center", va="center", color=color)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_metrics(path: Path, checkpoint_path: Path, checkpoint: dict, metrics: dict, cm: np.ndarray, report: str) -> None:
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


def main() -> None:
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    processed_path = PROJECT_ROOT / config["data"]["processed_npz"]
    split_path = PROJECT_ROOT / config["data"]["split_npz"]
    checkpoint_path = PROJECT_ROOT / config["outputs"]["checkpoint"]
    metrics_path = PROJECT_ROOT / config["outputs"]["cross_load_metrics"]
    predictions_path = PROJECT_ROOT / config["outputs"]["predictions"]
    confusion_matrix_path = PROJECT_ROOT / config["outputs"]["confusion_matrix"]
    processed_data = np.load(processed_path, allow_pickle=False)
    splits = np.load(split_path, allow_pickle=False)
    test_target_indices = splits[config["data"]["test_split"]]

    test_loader = DataLoader(
        CWRUWindowDataset(str(processed_path), indices=test_target_indices, return_meta=False),
        batch_size=config["training"]["batch_size"],
        shuffle=False,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN1D(num_classes=config["experiment"]["num_classes"]).to(device)
    checkpoint = safe_load_checkpoint(checkpoint_path, device)
    model.load_state_dict(checkpoint["model_state_dict"])

    metrics = evaluate_classification(model, test_loader, device)
    y_true = metrics["y_true"]
    y_pred = metrics["y_pred"]
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(config["experiment"]["num_classes"]))
    report = classification_report(
        y_true,
        y_pred,
        labels=np.arange(config["experiment"]["num_classes"]),
        target_names=CLASS_NAMES,
        zero_division=0,
    )

    save_predictions(predictions_path, test_target_indices, y_true, y_pred, processed_data)
    save_confusion_matrix(confusion_matrix_path, cm)
    save_metrics(metrics_path, checkpoint_path, checkpoint, metrics, cm, report)

    test_loads = processed_data["load_hp"][test_target_indices]
    test_labels = processed_data["y"][test_target_indices]

    print("CNN1D source-only 0->3:")
    print(f"  source load: {config['experiment']['source_load_hp']}")
    print(f"  target load: {config['experiment']['target_load_hp']}")
    print(f"  test samples: {len(test_target_indices)}")
    print(f"  test load distribution: {dict(Counter(test_loads.tolist()))}")
    print(f"  test label distribution: {dict(Counter(test_labels.tolist()))}")
    print(f"  target accuracy: {metrics['accuracy']:.6f}")
    print(f"  target macro-F1: {metrics['macro_f1']:.6f}")
    print(f"  metrics path: {metrics_path}")
    print(f"  predictions path: {predictions_path}")
    print(f"  confusion matrix path: {confusion_matrix_path}")


if __name__ == "__main__":
    main()

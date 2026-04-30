"""Evaluate CNN1D source-only gap-split checkpoint on the target 2 hp test split."""

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


CONFIG_PATH = PROJECT_ROOT / "configs/experiments/cnn1d_source_3to2_gap.yaml"
CLASS_NAMES = ["Normal", "IR007", "B007", "OR007@6", "IR014", "B014", "OR014@6", "IR021", "B021", "OR021@6"]


def safe_load_checkpoint(path: Path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except Exception:
        print("weights_only=True failed, falling back to weights_only=False for trusted local checkpoint.")
        return torch.load(path, map_location=device, weights_only=False)


def save_predictions(path: Path, sample_indices: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["sample_index", "y_true", "y_pred", "load_hp", "class_name", "file_name", "start_index"])
        writer.writeheader()
        for row_id, sample_index in enumerate(sample_indices):
            writer.writerow(
                {
                    "sample_index": int(sample_index),
                    "y_true": int(y_true[row_id]),
                    "y_pred": int(y_pred[row_id]),
                    "load_hp": int(data["load_hp"][sample_index]),
                    "class_name": str(data["class_name"][sample_index]),
                    "file_name": str(data["file_name"][sample_index]),
                    "start_index": int(data["start_index"][sample_index]),
                }
            )


def save_confusion_matrix(path: Path, cm: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 7))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar()
    ticks = np.arange(len(CLASS_NAMES))
    plt.xticks(ticks, CLASS_NAMES, rotation=45, ha="right")
    plt.yticks(ticks, CLASS_NAMES)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main() -> None:
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    processed_path = PROJECT_ROOT / config["data"]["processed_npz"]
    split_path = PROJECT_ROOT / config["data"]["split_npz"]
    outputs = config["outputs"]
    data = np.load(processed_path, allow_pickle=False)
    splits = np.load(split_path, allow_pickle=False)
    test_indices = splits[config["data"]["test_split"]]

    loader = DataLoader(CWRUWindowDataset(str(processed_path), indices=test_indices), batch_size=config["training"]["batch_size"], shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN1D(num_classes=config["experiment"]["num_classes"]).to(device)
    checkpoint = safe_load_checkpoint(PROJECT_ROOT / outputs["checkpoint"], device)
    model.load_state_dict(checkpoint["model_state_dict"])
    metrics = evaluate_classification(model, loader, device)

    cm = confusion_matrix(metrics["y_true"], metrics["y_pred"], labels=np.arange(len(CLASS_NAMES)))
    report = classification_report(metrics["y_true"], metrics["y_pred"], labels=np.arange(len(CLASS_NAMES)), target_names=CLASS_NAMES, zero_division=0)
    metrics_path = PROJECT_ROOT / outputs["cross_load_metrics"]
    predictions_path = (PROJECT_ROOT / outputs["experiment_dir"]) / "logs/cross_load_predictions.csv"
    figure_path = (PROJECT_ROOT / outputs["experiment_dir"]) / "figures/cross_load_confusion_matrix.png"
    save_predictions(predictions_path, test_indices, metrics["y_true"], metrics["y_pred"], data)
    save_confusion_matrix(figure_path, cm)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        f"target_accuracy: {metrics['accuracy']:.6f}\n"
        f"target_macro_f1: {metrics['macro_f1']:.6f}\n\n"
        f"confusion_matrix:\n{np.array2string(cm)}\n\nclassification_report:\n{report}",
        encoding="utf-8",
    )
    print(f"Test samples: {len(test_indices)}")
    print(f"Test load distribution: {dict(Counter(data['load_hp'][test_indices].tolist()))}")
    print(f"Test label distribution: {dict(Counter(data['y'][test_indices].tolist()))}")
    print(f"Target accuracy: {metrics['accuracy']:.6f}")
    print(f"Target macro-F1: {metrics['macro_f1']:.6f}")
    print(f"Metrics path: {metrics_path}")
    print(f"Predictions path: {predictions_path}")
    print(f"Confusion matrix path: {figure_path}")


if __name__ == "__main__":
    main()

